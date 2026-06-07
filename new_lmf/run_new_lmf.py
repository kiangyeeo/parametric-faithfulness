"""Run the NPO+KL+LMF extension without modifying ``unlearn.py``.

The loss keeps NPO as the main forgetting objective, keeps the original
NPO+KL retain term for comparability, and adds a small LMF forget-token
regularizer:

    loss = NPO + rt_lambda * KL + lmf_coeff * LMF
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

METHOD = "npo_KL_lmf"
DEFAULT_RESULTS_DIR = "new_lmf/final_results"


def _lmf_loss_from_logits(logits, labels):
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()
    mask = shift_labels.ne(-100)
    margin = shift_logits.max(dim=-1).values - shift_logits.mean(dim=-1)
    return (margin.square() * mask).sum() / mask.sum().clamp(min=1)


def hybrid_loss(
    model,
    oracle_model,
    inputs,
    ref_policy="fine_tuned",
    beta=0.1,
    npo_coeff=1.0,
    KL_coeff=1.0,
    lmf_coeff=0.03,
    return_outputs=False,
):
    import torch
    import torch.nn.functional as F
    import unlearn

    if ref_policy != "fine_tuned":
        raise NotImplementedError("Only fine_tuned reference policy is supported.")

    forget_inputs, retain_inputs = inputs
    input_ids, labels, attention_mask = forget_inputs
    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)

    forget_loss_current = unlearn.get_batch_loss(outputs.logits, labels)
    with torch.no_grad():
        oracle_forget = oracle_model(input_ids, labels=labels, attention_mask=attention_mask)
        forget_loss_oracle = unlearn.get_batch_loss(oracle_forget.logits, labels)
    neg_log_ratios = forget_loss_current - forget_loss_oracle
    npo_loss = -F.logsigmoid(beta * neg_log_ratios).mean() * 2 / beta
    lmf_loss = _lmf_loss_from_logits(outputs.logits, labels)

    retain_ids, retain_labels, retain_attention = retain_inputs
    with torch.no_grad():
        retain_oracle = oracle_model(
            retain_ids,
            labels=retain_labels,
            attention_mask=retain_attention,
        )
    retain_current = model(retain_ids, labels=retain_labels, attention_mask=retain_attention)
    oracle_log_probs = F.log_softmax(retain_oracle.logits, dim=-1)
    current_log_probs = F.log_softmax(retain_current.logits, dim=-1)
    retain_loss = F.kl_div(
        current_log_probs.view(-1, retain_oracle.logits.shape[-1]),
        oracle_log_probs.view(-1, retain_oracle.logits.shape[-1]),
        reduction="batchmean",
        log_target=True,
    )

    loss = npo_coeff * npo_loss + KL_coeff * retain_loss + lmf_coeff * lmf_loss
    return (loss, outputs) if return_outputs else loss


def output_path(args):
    log_suffix = f"_lmf={args.lmf_coeff:g}"
    if args.rt_lambda != 1.0:
        log_suffix += f"_lambda={args.rt_lambda:g}"
    name = (
        f"{METHOD}_sentencize_s=True_lr={args.lr}{log_suffix}"
        f"_rs={args.seed}_pos=True_ff2=True.out"
    )
    return Path(args.results_dir) / args.dataset / args.short_model / name


def run_one(args):
    from repro.run_repro import build_args, patched_main
    import unlearn

    old_compute_loss = unlearn.compute_loss

    def patched_compute_loss(*loss_args, **loss_kwargs):
        if loss_kwargs.get("loss_type") == METHOD:
            loss_kwargs.pop("loss_type")
            return hybrid_loss(*loss_args, lmf_coeff=args.lmf_coeff, **loss_kwargs)
        return old_compute_loss(*loss_args, **loss_kwargs)

    run_args = build_args(
        short_model=args.short_model,
        dataset=args.dataset,
        lr=args.lr,
        smoke=False,
        rt_lambda=args.rt_lambda,
        results_dir=args.results_dir,
        n_unlearn=args.n_unlearn,
        n_verify=args.n_verify,
        epochs=args.epochs,
        log_suffix=f"_lmf={args.lmf_coeff:g}" + ("" if args.rt_lambda == 1.0 else f"_lambda={args.rt_lambda:g}"),
    )
    run_args.method = METHOD
    run_args.seed = args.seed

    try:
        unlearn.compute_loss = patched_compute_loss
        patched_main(run_args)
    finally:
        unlearn.compute_loss = old_compute_loss


def parse_args():
    from repro import config as cfg

    parser = argparse.ArgumentParser()
    parser.add_argument("--short_model", choices=list(cfg.MODELS), required=True)
    parser.add_argument("--dataset", choices=cfg.DATASETS, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--lmf_coeff", type=float, default=0.03)
    parser.add_argument("--rt_lambda", type=float, default=1.0)
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--n_unlearn", type=int, default=30)
    parser.add_argument("--n_verify", type=int, default=cfg.N_VERIFY)
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    if args.dry_run:
        print(output_path(args))
        print(
            f"method={METHOD}, lr={args.lr}, lmf_coeff={args.lmf_coeff}, "
            f"n_unlearn={args.n_unlearn}, n_verify={args.n_verify}, epochs={args.epochs}"
        )
        return
    run_one(args)


if __name__ == "__main__":
    main()
