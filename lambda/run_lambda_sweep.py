"""Run the lambda retain-regularization sweep.

Examples:
  python lambda/run_lambda_sweep.py --smoke --models Phi-3 --datasets openbook --lambdas 0.0 1.0 3.0
  CUDA_VISIBLE_DEVICES=0 python lambda/run_lambda_sweep.py --models LLaMA-3-3B
  CUDA_VISIBLE_DEVICES=1 python lambda/run_lambda_sweep.py --models Phi-3
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from repro import config as cfg


DEFAULT_LAMBDAS = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]


def lambda_label(value):
    value = float(value)
    if value == 0.0:
        return "0.0"
    if value.is_integer():
        return f"{value:.1f}"
    return f"{value:g}"


def lr_for(short_model, dataset):
    for run_model, run_dataset, lr in cfg.RUNS:
        if run_model == short_model and run_dataset == dataset:
            return lr
    raise ValueError(f"No learning rate configured for {short_model}/{dataset}")


def run_one(args, short_model, dataset, rt_lambda):
    label = lambda_label(rt_lambda)
    run_kind = "smoke" if args.smoke else "full"
    results_dir = Path(args.results_root)
    if args.smoke:
        results_dir = results_dir / "smoke"
    results_dir = results_dir / f"lambda={label}"

    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{run_kind}_{short_model}_{dataset}_lambda={label}.log"

    cmd = [
        sys.executable,
        "-m",
        "repro.run_repro",
        "--short_model",
        short_model,
        "--dataset",
        dataset,
        "--lr",
        str(lr_for(short_model, dataset)),
        "--rt_lambda",
        str(rt_lambda),
        "--results_dir",
        str(results_dir),
        "--log_suffix",
        f"_lambda={label}",
    ]
    if args.smoke:
        cmd.append("--smoke")
    if args.n_unlearn is not None:
        cmd.extend(["--n_unlearn", str(args.n_unlearn)])
    if args.n_verify is not None:
        cmd.extend(["--n_verify", str(args.n_verify)])
    if args.epochs is not None:
        cmd.extend(["--epochs", str(args.epochs)])

    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"

    print(f"[run] {short_model}/{dataset} lambda={label}")
    print(f"[log] {log_path}")
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write("\n" + "=" * 80 + "\n")
        log_file.write(" ".join(cmd) + "\n")
        log_file.flush()
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if proc.returncode != 0:
        raise SystemExit(f"Run failed for {short_model}/{dataset} lambda={label}; see {log_path}")


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(cfg.MODELS), choices=list(cfg.MODELS))
    parser.add_argument("--datasets", nargs="+", default=cfg.DATASETS, choices=cfg.DATASETS)
    parser.add_argument("--lambdas", nargs="+", type=float, default=DEFAULT_LAMBDAS)
    parser.add_argument("--results_root", default="lambda/results")
    parser.add_argument("--logs_dir", default="lambda/logs")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--n_unlearn", type=int, default=None)
    parser.add_argument("--n_verify", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    return parser


def main():
    args = make_parser().parse_args()
    for short_model in args.models:
        for dataset in args.datasets:
            if (short_model, dataset) not in {(m, d) for m, d, _ in cfg.RUNS}:
                continue
            for rt_lambda in args.lambdas:
                run_one(args, short_model, dataset, rt_lambda)


if __name__ == "__main__":
    main()
