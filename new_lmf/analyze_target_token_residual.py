"""Analyze whether NPO saturation leaves the original target token confident.

This is stricter than the residual-confidence diagnostic: high max-softmax can
mean probability moved to another token. Here we ask whether the original
forget target token itself remains high after NPO has saturated.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "new_lmf" / "npo_residual_diagnostics"
DEFAULT_OUTPUT = ROOT / "new_lmf" / "target_token_residual_summary.csv"


def metric(row, epoch, name):
    return float(row["diagnostics_results"][str(epoch)][name])


def final_epoch(row):
    return max(int(e) for e in row["diagnostics_results"])


def load_rows(input_dir):
    for path in Path(input_dir).glob("*/*/*.diagnostics.jsonl"):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    row["_path"] = str(path)
                    yield row


def summarize(rows, abs_thresholds, relative_thresholds):
    groups = {}
    for row in rows:
        epoch = final_epoch(row)
        key = (row["dataset"], row["short_model"], row["lr"], row["beta"], row["rt_lambda"], epoch)
        group = groups.setdefault(key, {"n": 0, "sum": {}, "cnt": {}})
        group["n"] += 1

        p0 = metric(row, 0, "target_token_prob_mean")
        pf = metric(row, epoch, "target_token_prob_mean")
        maxpf = metric(row, epoch, "forget_max_softmax_prob_mean")
        sat = metric(row, epoch, "npo_saturated") > 0.5
        high_conf = metric(row, epoch, "residual_high_conf") > 0.5
        ratio = pf / max(p0, 1e-12)

        values = {
            "target_prob_epoch0_mean": p0,
            "target_prob_final_mean": pf,
            "target_prob_drop_mean": p0 - pf,
            "target_prob_ratio_final_over_initial": ratio,
            "max_softmax_final_mean": maxpf,
            "npo_saturated_rate": float(sat),
            "residual_high_conf_rate": float(high_conf),
            "saturated_and_high_conf_rate": float(sat and high_conf),
            "confident_replacement_target_le_0p3_rate": float(sat and high_conf and pf <= 0.3),
        }
        for threshold in abs_thresholds:
            key_name = f"saturated_and_target_prob_gt_{threshold:g}_rate".replace(".", "p")
            values[key_name] = float(sat and pf > threshold)
        for threshold in relative_thresholds:
            key_name = f"saturated_and_target_ratio_gt_{threshold:g}_rate".replace(".", "p")
            values[key_name] = float(sat and ratio > threshold)

        for name, value in values.items():
            group["sum"][name] = group["sum"].get(name, 0.0) + value
            group["cnt"][name] = group["cnt"].get(name, 0) + 1

    out = []
    for key, group in sorted(groups.items()):
        dataset, model, lr, beta, rt_lambda, epoch = key
        row = {
            "dataset": dataset,
            "model": model,
            "lr": lr,
            "beta": beta,
            "rt_lambda": rt_lambda,
            "epoch": epoch,
            "n_records": group["n"],
        }
        for name in sorted(group["sum"]):
            row[name] = group["sum"][name] / group["cnt"][name]
        out.append(row)
    return out


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    prefix = ["dataset", "model", "lr", "beta", "rt_lambda", "epoch", "n_records"]
    fields = prefix + [f for f in fields if f not in prefix]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_table(rows):
    cols = [
        "dataset",
        "model",
        "n_records",
        "target_prob_final_mean",
        "target_prob_ratio_final_over_initial",
        "saturated_and_target_prob_gt_0p5_rate",
        "saturated_and_target_prob_gt_0p3_rate",
        "saturated_and_high_conf_rate",
        "confident_replacement_target_le_0p3_rate",
    ]
    print(",".join(cols))
    for row in rows:
        vals = []
        for col in cols:
            value = row[col]
            vals.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        print(",".join(vals))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=DEFAULT_INPUT)
    parser.add_argument("--output_csv", default=DEFAULT_OUTPUT)
    parser.add_argument("--abs_thresholds", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--relative_thresholds", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    return parser.parse_args()


def main():
    args = parse_args()
    rows = summarize(list(load_rows(args.input_dir)), args.abs_thresholds, args.relative_thresholds)
    write_csv(rows, args.output_csv)
    print_table(rows)
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
