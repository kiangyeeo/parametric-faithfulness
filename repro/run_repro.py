"""Reproduction entry point with local datasets and configurable result paths.

This wrapper keeps the original unlearning logic in ``unlearn.py`` but fixes
the small 2x2 reproduction settings used by this repository:

- use local OpenBookQA / StrategyQA subsets;
- use config-controlled N_VERIFY / N_UNLEARN;
- expose the retain KL coefficient used by lambda experiments;
- allow result roots outside final_results so experiments do not overwrite
  the reproduction baseline.
"""

import argparse
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import dataload
from repro.local_datasets import LOCAL_DATASETS

dataload.DATASETS.update(LOCAL_DATASETS)

import unlearn
from repro import config as cfg


def build_args(
    short_model,
    dataset,
    lr,
    smoke=False,
    rt_lambda=1.0,
    results_dir=None,
    n_unlearn=None,
    n_verify=None,
    epochs=None,
    log_suffix=None,
    mmlu=0,
    method=None,
    strategy=None,
    stepwise=None,
    temperature=None,
    seed=None,
    pos=None,
    ff2=None,
    atomic=False,
):
    model_id = cfg.MODELS[short_model]

    # Every knob falls back to its repro/config.py default when not given.
    return argparse.Namespace(
        model_name=model_id,
        dataset=dataset,
        method=method if method is not None else cfg.METHOD,
        strategy=strategy if strategy is not None else cfg.STRATEGY,
        stepwise=stepwise if stepwise is not None else cfg.STEPWISE,
        temperature=temperature if temperature is not None else cfg.TEMPERATURE,
        seed=seed if seed is not None else cfg.SEED,
        epochs=epochs if epochs is not None else (2 if smoke else cfg.EPOCHS),
        lr=lr,
        rt_lambda=rt_lambda,
        new_cot=False,
        pos=pos if pos is not None else cfg.POS_FILTER,
        ff2=ff2 if ff2 is not None else cfg.FF2_ONLY,
        ablation=smoke,
        mmlu=mmlu,
        gsm=0,
        results_dir=results_dir,
        n_unlearn=n_unlearn,
        n_verify=n_verify,
        log_suffix=log_suffix,
        atomic=atomic,
    )


def patched_main(args):
    import gc
    import random

    import numpy as np
    import torch
    from data import load_or_generate_dataset_cots, model_name_dict
    from huggingface_hub import login
    from transformers import AutoTokenizer as TOK
    from util import set_random_seed

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        login(hf_token)
    else:
        print("[warn] HF_TOKEN is not set; gated models must be cached locally.")

    set_random_seed(args.seed)

    tokenizer = TOK.from_pretrained(args.model_name)
    if "Phi" in args.model_name:
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.eos_token

    dh = dataload.DATASETS[args.dataset]
    cot_data = load_or_generate_dataset_cots(
        model_id=args.model_name,
        tokenizer=tokenizer,
        dataset_id=args.dataset,
        force_generate=args.new_cot,
        sentencize=(args.strategy == "sentencize"),
        temperature=args.temperature,
        seed=args.seed,
        atomic=args.atomic,
    )
    random.shuffle(cot_data)

    n_verify = args.n_verify if args.n_verify is not None else cfg.N_VERIFY
    n_unlearn = args.n_unlearn if args.n_unlearn is not None else cfg.N_UNLEARN
    if args.ablation and args.n_unlearn is None:
        n_unlearn = 5
    if args.mmlu: # mmlu
        n_unlearn = args.mmlu

    cots_train, cots_verify = cot_data[:-n_verify], cot_data[-n_verify:]

    mod = args.model_name.split("/")[-1]
    short_model = model_name_dict[mod]

    root_name = args.results_dir

    if root_name is None:
        if args.ablation:
            root_name = cfg.SMOKE_DIR
        elif args.mmlu:
            root_name = "mmlu_results"   
        else:
            root_name = cfg.RESULTS_DIR

    resdir = f"{root_name}/{args.dataset}/{short_model}/"
    os.makedirs(resdir, exist_ok=True)

    log_suffix = args.log_suffix or ""
    logfile_name = (
        f"{args.method}_{args.strategy}_s={args.stepwise}"
        f"_lr={args.lr}{log_suffix}_rs={args.seed}_pos={args.pos}_ff2={args.ff2}.out"
    )

    ids = unlearn.load_ids(resdir + logfile_name, stepwise=args.stepwise)
    print(f"Ids so far: {len(ids)}")

    for idx, target in enumerate(cots_train[:n_unlearn]):
        n_steps = len(target["segmented_cot"]) if args.stepwise else 1
        for step_idx in range(n_steps):
            check_id = target["id"]
            if args.stepwise:
                check_id = f"{check_id}_{step_idx}"
            if check_id in ids:
                continue

            instance_info = {
                "id": target["id"],
                "question": target["question"],
                "step_idx": step_idx,
                "options": target["options"],
                "correct": target["correct_letter"],
                "initial_cot": target["cot"],
                "initial_cot_probs": target["cot_probs"],
                "initial_probs": target["nocot_probs"],
                "prediction": int(np.argmax(target["nocot_probs"])),
                "cot_prediction": int(np.argmax(target["cot_probs"])),
                "rt_lambda": args.rt_lambda,
            }
            if args.stepwise:
                instance_info["cot_step"] = target["segmented_cot"][step_idx]
                instance_info["segmented_cot"] = target["segmented_cot"]

            ret = unlearn.unlearn_single(
                args.model_name,
                tokenizer,
                args,
                target,
                step_idx,
                cots_train,
                cots_verify,
                dh,
                idx,
            )
            if ret["unlearning_results"] is None:
                continue

            instance_info["unlearning_results"] = ret["unlearning_results"]
            if args.mmlu:
                instance_info["mmlu_results"] = ret.get("mmlu_results")
            unlearn.store(instance_info, resdir + logfile_name)
            del instance_info
            gc.collect()
            torch.cuda.empty_cache()


def make_cli():
    parser = argparse.ArgumentParser(
        description="Run the 2x2 reproduction. Flags default to repro/config.py."
    )
    parser.add_argument("--short_model", choices=list(cfg.MODELS), required=True)
    parser.add_argument("--dataset", choices=cfg.DATASETS, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--mmlu", type=int, default=0)
    parser.add_argument("--rt_lambda", type=float, default=1.0)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--n_unlearn", type=int, default=None)
    parser.add_argument("--n_verify", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--log_suffix", type=str, default=None)
    # Knobs that used to require editing repro/config.py. None => use config default.
    parser.add_argument("--method", type=str, default=None,
                        help=f"Unlearning loss (default config: {cfg.METHOD}).")
    parser.add_argument("--strategy", type=str, default=None,
                        help=f"CoT segmentation strategy (default config: {cfg.STRATEGY}).")
    parser.add_argument("--temperature", type=float, default=None,
                        help=f"CoT generation temperature (default config: {cfg.TEMPERATURE}).")
    parser.add_argument("--seed", type=int, default=None,
                        help=f"Random seed (default config: {cfg.SEED}).")
    parser.add_argument("--stepwise", action=argparse.BooleanOptionalAction, default=None,
                        help=f"Unlearn one CoT step at a time (default config: {cfg.STEPWISE}).")
    parser.add_argument("--pos", action=argparse.BooleanOptionalAction, default=None,
                        help=f"Filter out function words (default config: {cfg.POS_FILTER}).")
    parser.add_argument("--ff2", action=argparse.BooleanOptionalAction, default=None,
                        help=f"Tune only the FF2 (down_proj) layer (default config: {cfg.FF2_ONLY}).")
    parser.add_argument("--atomic", action="store_true",
                        help="Use atomic-statement segmentation for CoT steps.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run 5 examples x 2 epochs unless overridden.",
    )
    return parser


if __name__ == "__main__":
    cli_args = make_cli().parse_args()
    run_args = build_args(
        cli_args.short_model,
        cli_args.dataset,
        cli_args.lr,
        smoke=cli_args.smoke,
        mmlu=cli_args.mmlu,
        rt_lambda=cli_args.rt_lambda,
        results_dir=cli_args.results_dir,
        n_unlearn=cli_args.n_unlearn,
        n_verify=cli_args.n_verify,
        epochs=cli_args.epochs,
        log_suffix=cli_args.log_suffix,
        method=cli_args.method,
        strategy=cli_args.strategy,
        stepwise=cli_args.stepwise,
        temperature=cli_args.temperature,
        seed=cli_args.seed,
        pos=cli_args.pos,
        ff2=cli_args.ff2,
        atomic=cli_args.atomic,
    )
    patched_main(run_args)
