"""Run a small NPO+KL+LMF coefficient ablation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from repro import config as cfg
from new_lmf.run_new_lmf import run_one


DEFAULT_RESULTS_DIR = "new_lmf/ablation"
DEFAULT_COEFFS = [0.01, 0.03, 0.1, 0.3]


def iter_runs(args):
    for short_model, dataset, base_lr in cfg.RUNS:
        if args.short_model and short_model != args.short_model:
            continue
        if args.dataset and dataset != args.dataset:
            continue
        lrs = args.lr or [base_lr]
        coeffs = args.lmf_coeff or DEFAULT_COEFFS
        for lr in lrs:
            for coeff in coeffs:
                yield short_model, dataset, lr, coeff


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--short_model", choices=list(cfg.MODELS))
    parser.add_argument("--dataset", choices=cfg.DATASETS)
    parser.add_argument("--lr", type=float, action="append")
    parser.add_argument("--lmf_coeff", type=float, action="append")
    parser.add_argument("--rt_lambda", type=float, default=1.0)
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--n_unlearn", type=int, default=30)
    parser.add_argument("--n_verify", type=int, default=cfg.N_VERIFY)
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    runs = list(iter_runs(args))
    print(f"planned runs: {len(runs)}")
    for short_model, dataset, lr, coeff in runs:
        print(f"{short_model}\t{dataset}\tlr={lr}\tlmf_coeff={coeff}")
    if args.dry_run:
        return

    for short_model, dataset, lr, coeff in runs:
        run_args = argparse.Namespace(
            short_model=short_model,
            dataset=dataset,
            lr=lr,
            lmf_coeff=coeff,
            rt_lambda=args.rt_lambda,
            results_dir=args.results_dir,
            n_unlearn=args.n_unlearn,
            n_verify=args.n_verify,
            epochs=args.epochs,
            seed=args.seed,
        )
        print("=" * 72)
        print(f"NPO+KL+LMF: {short_model} x {dataset}, lr={lr}, lmf_coeff={coeff}")
        print("=" * 72)
        run_one(run_args)


if __name__ == "__main__":
    main()
