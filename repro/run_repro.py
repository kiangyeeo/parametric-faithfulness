"""
repro/run_repro.py

复现 entry point。做了三件事：
 1. 把源仓库的 DATASETS 全局字典替换成本地的 LOCAL_DATASETS；
 2. 修掉源代码两个会卡住的 bug：args.atomic 没注册、N_unlearn 默认 250；
 3. 提供 --smoke 选项，便于先用 5 条快速跑通整条 pipeline 再放大。

调用方式（在仓库根目录，**不是** 在 repro/ 里）：
    # 单组合
    python -m repro.run_repro --short_model Phi-3 --dataset openbook --lr 1e-4
    # smoke test（每个组合只跑 5 条 × 2 epoch，验证 pipeline）
    python -m repro.run_repro --short_model Phi-3 --dataset openbook --lr 1e-4 --smoke
    # 批量跑全部 2x2（建议放 shell/sbatch 脚本里串行/并行调度）
    bash repro/run_all.sh

注意：第一次跑某个 (model, dataset) 时会触发 CoT 生成，~10-30 min/组合。
       生成结果会写到 final_cot/{dataset}/{model}_s=1001_t=0.0_cots.jsonl，
       之后所有 run 都共享这份缓存。
"""
import argparse
import os
import sys

# 让 import 找得到仓库根
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ===== 关键一步：在 import unlearn 之前替换 DATASETS =====
# unlearn.py 顶部是 `from dataload import DATASETS`，是赋值不是引用，
# 我们需要在 dataload 模块上原地改字典。
import dataload
from repro.local_datasets import LOCAL_DATASETS
dataload.DATASETS.update(LOCAL_DATASETS)   # 覆盖 openbook / sqa 两个 key
# =========================================================

import unlearn
from repro import config as cfg


def build_args(short_model, dataset, lr, smoke=False):
    """构造 unlearn.main 期望的 argparse.Namespace。

    避开了 sys.argv，所以这个函数也方便从 notebook 里调。
    """
    model_id = cfg.MODELS[short_model]

    ns = argparse.Namespace(
        model_name  = model_id,
        dataset     = dataset,
        method      = cfg.METHOD,
        strategy    = cfg.STRATEGY,
        stepwise    = cfg.STEPWISE,
        temperature = cfg.TEMPERATURE,
        seed        = cfg.SEED,
        epochs      = 2 if smoke else cfg.EPOCHS,
        lr          = lr,
        new_cot     = False,           # 用缓存好的 CoT
        pos         = cfg.POS_FILTER,
        ff2         = cfg.FF2_ONLY,
        ablation    = smoke,           # 让源代码走 ablation 分支（N=30）
        mmlu        = 0,
        gsm         = 0,
        # 源仓库漏注册的 flag —— 必须手动塞进 Namespace
        atomic      = False,
    )
    return ns


def patched_main(args):
    """复制 unlearn.main 的逻辑，但用我们 config 里的 N_VERIFY/N_UNLEARN。

    源 main 把 N_verify=20 / N_unlearn=250 写死，与 50 条样本不兼容。
    这里把整个 main 重写一遍，逻辑保持一致，只换数字 + 注入 HF token。
    """
    import gc, json, random
    import numpy as np
    import torch
    from transformers import AutoTokenizer as TOK
    from data import load_or_generate_dataset_cots, model_name_dict
    from util import set_random_seed
    from huggingface_hub import login

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # 从环境变量取 HF token，不要硬编码到代码里
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        login(hf_token)
    else:
        print("[warn] HF_TOKEN 未设置，gated 模型（Llama）会拉取失败")

    set_random_seed(args.seed)

    tokenizer = TOK.from_pretrained(args.model_name)
    if "Phi" in args.model_name:
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.eos_token

    DH = dataload.DATASETS[args.dataset]  # 这里拿到的就是 Local* 子类
    cot_data = load_or_generate_dataset_cots(
        model_id=args.model_name, tokenizer=tokenizer,
        dataset_id=args.dataset, force_generate=args.new_cot,
        sentencize=(args.strategy == "sentencize"),
        temperature=args.temperature, seed=args.seed, atomic=args.atomic,
    )
    random.shuffle(cot_data)

    # 关键改动：N_verify / N_unlearn 从 config 来
    N_verify  = cfg.N_VERIFY
    N_unlearn = cfg.N_UNLEARN
    if args.ablation:  # smoke
        N_unlearn = 5

    cots_train, cots_verify = cot_data[:-N_verify], cot_data[-N_verify:]

    mod = args.model_name.split("/")[-1]
    short_model = model_name_dict[mod]

    root_name = cfg.SMOKE_DIR if args.ablation else cfg.RESULTS_DIR
    resdir = f"{root_name}/{args.dataset}/{short_model}/"
    os.makedirs(resdir, exist_ok=True)
    logfile_name = (
        f"{args.method}_{args.strategy}_s={args.stepwise}"
        f"_lr={args.lr}_rs={args.seed}_pos={args.pos}_ff2={args.ff2}.out"
    )

    ids = unlearn.load_ids(resdir + logfile_name, stepwise=args.stepwise)
    print(f"Ids so far: {len(ids)}")

    for idx, target in enumerate(cots_train[:N_unlearn]):
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
            }
            if args.stepwise:
                instance_info["cot_step"] = target["segmented_cot"][step_idx]
                instance_info["segmented_cot"] = target["segmented_cot"]

            ret = unlearn.unlearn_single(
                args.model_name, tokenizer, args, target, step_idx,
                cots_train, cots_verify, DH, idx,
            )
            if ret["unlearning_results"] is None:
                continue
            instance_info["unlearning_results"] = ret["unlearning_results"]
            unlearn.store(instance_info, resdir + logfile_name)
            del instance_info
            gc.collect()
            torch.cuda.empty_cache()


def make_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--short_model", choices=list(cfg.MODELS), required=True)
    p.add_argument("--dataset", choices=cfg.DATASETS, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--smoke", action="store_true",
                    help="跑 5 条 × 2 epoch 验证 pipeline，结果写 smoke_results/")
    return p


if __name__ == "__main__":
    cli_args = make_cli().parse_args()
    run_args = build_args(
        cli_args.short_model, cli_args.dataset, cli_args.lr, smoke=cli_args.smoke,
    )
    patched_main(run_args)
