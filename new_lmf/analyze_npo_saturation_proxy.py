"""Posthoc NPO saturation proxy from existing .out files.

The saved .out files do not contain logits, so they cannot prove residual
high-margin tokens. They do contain target CoT step log probabilities across
epochs. We use those to estimate

    delta_proxy = -log p_t + log p_0 = log p_0 - log p_t
    grad_factor = 2 * sigmoid(-beta * delta_proxy)

This checks whether the sequence-level NPO objective enters the low-gradient
region often enough to make the underconstraint worth testing with logits.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "final_results"
DEFAULT_OUTPUT = ROOT / "new_lmf" / "npo_saturation_proxy_summary.csv"


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def first_scalar(value):
    if isinstance(value, list):
        return first_scalar(value[0])
    return float(value)


def epoch_items(unlearning):
    return sorted(unlearning.items(), key=lambda item: int(item[0]))


def parse_path(path, root):
    rel = path.relative_to(root)
    return rel.parts[0], rel.parts[1], path.name


def iter_rows(results_dir, beta, grad_threshold):
    for path in Path(results_dir).glob("*/*/npo_KL_*.out"):
        dataset, model, filename = parse_path(path, Path(results_dir))
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                unlearning = row.get("unlearning_results") or {}
                if "0" not in unlearning:
                    continue
                logp0 = first_scalar(unlearning["0"]["cot_step_prob"])
                for epoch, vals in epoch_items(unlearning):
                    logpt = first_scalar(vals["cot_step_prob"])
                    delta = logp0 - logpt
                    grad = 2.0 * sigmoid(-beta * delta)
                    yield {
                        "dataset": dataset,
                        "model": model,
                        "filename": filename,
                        "id": row.get("id"),
                        "step_idx": row.get("step_idx"),
                        "epoch": int(epoch),
                        "beta": beta,
                        "logp0": logp0,
                        "logpt": logpt,
                        "delta_proxy": delta,
                        "npo_grad_factor_proxy": grad,
                        "npo_saturated_proxy": int(grad < grad_threshold),
                    }


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row["dataset"], row["model"], row["epoch"])
        group = groups.setdefault(key, defaultdict(float))
        group["n_records"] += 1
        for name in ("delta_proxy", "npo_grad_factor_proxy", "npo_saturated_proxy"):
            group[name] += row[name]

    out = []
    for (dataset, model, epoch), group in sorted(groups.items()):
        n = group["n_records"]
        out.append({
            "dataset": dataset,
            "model": model,
            "epoch": epoch,
            "n_records": int(n),
            "delta_proxy_mean": group["delta_proxy"] / n,
            "npo_grad_factor_proxy_mean": group["npo_grad_factor_proxy"] / n,
            "npo_saturated_proxy_rate": group["npo_saturated_proxy"] / n,
        })
    return out


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "model",
        "epoch",
        "n_records",
        "delta_proxy_mean",
        "npo_grad_factor_proxy_mean",
        "npo_saturated_proxy_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_table(rows):
    print("dataset,model,epoch,n,delta_mean,grad_mean,saturated_rate")
    for row in rows:
        print(
            f"{row['dataset']},{row['model']},{row['epoch']},"
            f"{row['n_records']},{row['delta_proxy_mean']:.3f},"
            f"{row['npo_grad_factor_proxy_mean']:.3f},"
            f"{row['npo_saturated_proxy_rate']:.3f}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output_csv", default=DEFAULT_OUTPUT)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--grad_threshold", type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = list(iter_rows(args.results_dir, args.beta, args.grad_threshold))
    summary = summarize(rows)
    write_csv(summary, args.output_csv)
    print_table(summary)
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
