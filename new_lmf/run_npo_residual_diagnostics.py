"""Diagnostics for the NPO sequence-level underconstraint hypothesis.

This runner reruns a small NPO+KL pilot and records whether NPO has already
suppressed the target sequence while forget-token logits remain confident.
It writes sidecar diagnostics only; it does not replace the main .out metrics.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from repro import config as cfg


METHOD = "npo_KL"
DIAG_DIR = "new_lmf/npo_residual_diagnostics"


def diag_path(args):
    name = (
        f"{METHOD}_{cfg.STRATEGY}_s={cfg.STEPWISE}_lr={args.lr}"
        f"_beta={args.beta:g}_rs={args.seed}_pos={not args.no_pos}"
        f"_ff2={not args.no_ff2}.diagnostics.jsonl"
    )
    return Path(args.diagnostics_dir) / args.dataset / args.short_model / name


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def load_done(path):
    if not path.exists():
        return set()
    done = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                done.add(f"{row['id']}_{row['step_idx']}")
    return done


def scalar(x):
    return float(x.detach().cpu().float().item())


def masked_mean(values, mask):
    m = mask.float()
    return (values * m).sum() / m.sum().clamp(min=1)


def masked_max(values, mask):
    selected = values.masked_select(mask).detach().float()
    return None if selected.numel() == 0 else float(selected.max().cpu().item())


def positive_delta_concentration(token_delta, mask):
    import torch

    values = token_delta.masked_select(mask).detach().float()
    positive = values[values > 0]
    if positive.numel() == 0:
        return 0.0, 0.0, 0.0

    sorted_vals = torch.sort(positive, descending=True).values
    total = sorted_vals.sum().clamp(min=1e-12)
    top1_share = sorted_vals[0] / total
    k = max(1, int(0.2 * sorted_vals.numel()))
    top20_share = sorted_vals[:k].sum() / total
    positive_frac = positive.numel() / max(1, values.numel())
    return float(top1_share.cpu().item()), float(top20_share.cpu().item()), float(positive_frac)


def compute_diagnostics(model, oracle, batch, beta, grad_threshold, max_prob_threshold):
    import torch
    import torch.nn.functional as F

    was_training = model.training
    model.eval()
    oracle.eval()

    try:
        with torch.no_grad():
            forget_inputs, retain_inputs = batch
            input_ids, labels, attention_mask = forget_inputs
            current_out = model(input_ids, labels=labels, attention_mask=attention_mask)
            oracle_out = oracle(input_ids, labels=labels, attention_mask=attention_mask)

            current_logits = current_out.logits[:, :-1, :].contiguous().float()
            oracle_logits = oracle_out.logits[:, :-1, :].contiguous().float()
            y = labels[:, 1:].contiguous()
            mask = y.ne(-100)
            safe_y = y.masked_fill(~mask, 0)

            current_lp = F.log_softmax(current_logits, dim=-1)
            oracle_lp = F.log_softmax(oracle_logits, dim=-1)
            current_token_nll = -current_lp.gather(-1, safe_y.unsqueeze(-1)).squeeze(-1)
            oracle_token_nll = -oracle_lp.gather(-1, safe_y.unsqueeze(-1)).squeeze(-1)
            token_delta = current_token_nll - oracle_token_nll

            delta = (token_delta * mask.float()).sum()
            grad_factor = 2.0 * torch.sigmoid(-beta * delta)
            npo_success = torch.sigmoid(beta * delta)
            probs = current_lp.exp()
            margin = current_logits.max(dim=-1).values - current_logits.mean(dim=-1)
            max_prob = probs.max(dim=-1).values
            entropy = -(probs * current_lp).sum(dim=-1)
            target_prob = probs.gather(-1, safe_y.unsqueeze(-1)).squeeze(-1)

            retain_ids, retain_labels, retain_attention = retain_inputs
            retain_oracle = oracle(retain_ids, labels=retain_labels, attention_mask=retain_attention)
            retain_current = model(retain_ids, labels=retain_labels, attention_mask=retain_attention)
            r_oracle_lp = F.log_softmax(retain_oracle.logits[:, :-1, :].contiguous().float(), dim=-1)
            r_current_lp = F.log_softmax(retain_current.logits[:, :-1, :].contiguous().float(), dim=-1)
            rmask = retain_labels[:, 1:].contiguous().ne(-100)
            retain_kl = F.kl_div(r_current_lp, r_oracle_lp, reduction="none", log_target=True).sum(dim=-1)

            top1_share, top20_share, positive_frac = positive_delta_concentration(token_delta, mask)
            max_prob_mean = scalar(masked_mean(max_prob, mask))
            saturated = scalar(grad_factor) < grad_threshold
            residual_high_conf = max_prob_mean > max_prob_threshold

            return {
                "forget_valid_tokens": int(mask.sum().detach().cpu().item()),
                "npo_delta": scalar(delta),
                "npo_success_sigmoid": scalar(npo_success),
                "npo_grad_factor": scalar(grad_factor),
                "npo_saturated": int(saturated),
                "residual_high_conf": int(residual_high_conf),
                "saturated_and_high_conf": int(saturated and residual_high_conf),
                "forget_margin_mean": scalar(masked_mean(margin, mask)),
                "forget_margin_max": masked_max(margin, mask),
                "forget_max_softmax_prob_mean": max_prob_mean,
                "forget_max_softmax_prob_max": masked_max(max_prob, mask),
                "forget_entropy_mean": scalar(masked_mean(entropy, mask)),
                "target_token_prob_mean": scalar(masked_mean(target_prob, mask)),
                "target_token_nll_mean": scalar(masked_mean(current_token_nll, mask)),
                "token_delta_mean": scalar(masked_mean(token_delta, mask)),
                "positive_token_delta_fraction": positive_frac,
                "positive_delta_top1_share": top1_share,
                "positive_delta_top20pct_share": top20_share,
                "retain_valid_tokens": int(rmask.sum().detach().cpu().item()),
                "retain_kl": scalar(masked_mean(retain_kl, rmask)),
            }
    finally:
        model.train(was_training)


def train_one_target(model_id, tokenizer, args, target, step_idx, cots_train):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM as CLM

    import unlearn
    from data import FRCollator, cot_to_otfd

    model = CLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
        device_map="auto",
    )
    oracle = CLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
        device_map="auto",
    )
    collator = FRCollator(tokenizer, device=model.device)
    dataset = cot_to_otfd(
        target,
        cots_train,
        tokenizer,
        strategy=cfg.STRATEGY,
        stepwise=cfg.STEPWISE,
        step_idx=step_idx,
        pos=not args.no_pos,
    )

    n_targets = dataset.num_targets()
    if n_targets <= 2:
        print(f"skip {target['id']}_{step_idx}: too few targets ({n_targets})")
        del collator, dataset, model, oracle
        gc.collect()
        torch.cuda.empty_cache()
        return None

    if not args.no_ff2:
        for name, param in model.named_parameters():
            param.requires_grad = "mlp.down_proj.weight" in name

    loader = DataLoader(dataset, batch_size=1, collate_fn=collator, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = unlearn.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=args.epochs * len(dataset),
    )

    diagnostics = {
        0: compute_diagnostics(
            model, oracle, collator([dataset[0]]), args.beta,
            args.grad_threshold, args.max_prob_threshold
        )
    }
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        for batch in loader:
            loss = unlearn.compute_loss(
                model,
                oracle,
                batch,
                loss_type=METHOD,
                beta=args.beta,
                KL_coeff=args.rt_lambda,
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        diagnostics[epoch + 1] = compute_diagnostics(
            model, oracle, collator([dataset[0]]), args.beta,
            args.grad_threshold, args.max_prob_threshold
        )

    del loader, optimizer, scheduler, collator, dataset, model, oracle
    gc.collect()
    torch.cuda.empty_cache()
    return diagnostics


def run(args):
    import dataload
    from huggingface_hub import login
    from transformers import AutoTokenizer as TOK

    from repro.local_datasets import LOCAL_DATASETS

    dataload.DATASETS.update(LOCAL_DATASETS)
    from data import load_or_generate_dataset_cots
    from util import set_random_seed

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if os.environ.get("HF_TOKEN"):
        login(os.environ["HF_TOKEN"])
    else:
        print("[warn] HF_TOKEN is not set; gated models must be cached locally.")

    set_random_seed(args.seed)
    model_id = cfg.MODELS[args.short_model]
    tokenizer = TOK.from_pretrained(model_id, trust_remote_code=args.trust_remote_code)
    tokenizer.pad_token = tokenizer.unk_token if "Phi" in model_id else tokenizer.eos_token

    cot_data = load_or_generate_dataset_cots(
        model_id=model_id,
        tokenizer=tokenizer,
        dataset_id=args.dataset,
        force_generate=args.new_cot,
        sentencize=(cfg.STRATEGY == "sentencize"),
        temperature=cfg.TEMPERATURE,
        seed=args.seed,
        atomic=False,
    )
    random.shuffle(cot_data)
    cots_train = cot_data[:-args.n_verify]
    out = diag_path(args)
    done = load_done(out)
    print(f"diagnostics path: {out}")
    print(
        f"n_unlearn={args.n_unlearn}, n_verify={args.n_verify}, epochs={args.epochs}, "
        f"done={len(done)}"
    )

    for target in cots_train[:args.n_unlearn]:
        for step_idx in range(len(target["segmented_cot"]) if cfg.STEPWISE else 1):
            key = f"{target['id']}_{step_idx}"
            if key in done:
                continue
            diagnostics = train_one_target(model_id, tokenizer, args, target, step_idx, cots_train)
            if diagnostics is None:
                continue
            append_jsonl(out, {
                "id": target["id"],
                "question": target["question"],
                "step_idx": step_idx,
                "dataset": args.dataset,
                "short_model": args.short_model,
                "method": METHOD,
                "lr": args.lr,
                "beta": args.beta,
                "rt_lambda": args.rt_lambda,
                "grad_threshold": args.grad_threshold,
                "max_prob_threshold": args.max_prob_threshold,
                "pos": not args.no_pos,
                "ff2": not args.no_ff2,
                "cot_step": target["segmented_cot"][step_idx],
                "diagnostics_results": diagnostics,
            })
            done.add(key)
    summarize(out)


def summarize(jsonl_path, csv_path=None):
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        print(f"No diagnostics file found: {jsonl_path}")
        return None

    csv_path = Path(csv_path) if csv_path else jsonl_path.with_suffix(".summary.csv")
    groups = {}
    metrics = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            for epoch, vals in row["diagnostics_results"].items():
                key = (
                    row["dataset"], row["short_model"], row["lr"],
                    row["beta"], row["rt_lambda"], int(epoch)
                )
                group = groups.setdefault(key, {"n": 0, "sum": defaultdict(float), "cnt": defaultdict(int)})
                group["n"] += 1
                for name, value in vals.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        metrics.add(name)
                        group["sum"][name] += float(value)
                        group["cnt"][name] += 1

    metric_cols = sorted(metrics)
    fields = ["dataset", "short_model", "lr", "beta", "rt_lambda", "epoch", "n_records"] + metric_cols
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key in sorted(groups, key=lambda x: (x[0], x[1], x[5])):
            dataset, short_model, lr, beta, rt_lambda, epoch = key
            group = groups[key]
            row = {
                "dataset": dataset,
                "short_model": short_model,
                "lr": lr,
                "beta": beta,
                "rt_lambda": rt_lambda,
                "epoch": epoch,
                "n_records": group["n"],
            }
            row.update({
                m: group["sum"][m] / group["cnt"][m]
                if group["cnt"].get(m) else ""
                for m in metric_cols
            })
            writer.writerow(row)
    print(f"Wrote diagnostics summary: {csv_path}")
    return csv_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--short_model", choices=list(cfg.MODELS))
    parser.add_argument("--dataset", choices=cfg.DATASETS)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--rt_lambda", type=float, default=1.0)
    parser.add_argument("--n_unlearn", type=int, default=5)
    parser.add_argument("--n_verify", type=int, default=cfg.N_VERIFY)
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument("--grad_threshold", type=float, default=0.05)
    parser.add_argument("--max_prob_threshold", type=float, default=0.5)
    parser.add_argument("--diagnostics_dir", default=DIAG_DIR)
    parser.add_argument("--new_cot", action="store_true")
    parser.add_argument("--no_pos", action="store_true")
    parser.add_argument("--no_ff2", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--summarize")
    parser.add_argument("--summary_csv")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.summarize:
        summarize(args.summarize, args.summary_csv)
        return

    missing = [x for x in ("short_model", "dataset", "lr") if getattr(args, x) is None]
    if missing:
        raise SystemExit(f"Missing required run arguments: {', '.join('--' + m for m in missing)}")

    if args.dry_run:
        path = diag_path(args)
        print(f"diagnostics path: {path}")
        print(f"summary path: {path.with_suffix('.summary.csv')}")
        print(
            f"default size: n_unlearn={args.n_unlearn}, n_verify={args.n_verify}, "
            f"epochs={args.epochs}, beta={args.beta}"
        )
        return

    run(args)


if __name__ == "__main__":
    main()
