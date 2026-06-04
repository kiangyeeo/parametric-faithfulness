"""Aggregate lambda sweep results and generate report artifacts."""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from repro import config as cfg
from stats import average_mass_shift, make_stats, max_mass_shift


DEFAULT_LAMBDAS = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]


def lambda_label(value):
    value = float(value)
    if value == 0.0:
        return "0.0"
    if value.is_integer():
        return f"{value:.1f}"
    return f"{value:g}"


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def answer_change_rate(results):
    agreeing = [r for r in results if r["prediction"] == r["cot_prediction"]]
    changed = 0
    for item in agreeing:
        cot_pred = item["cot_prediction"]
        if any(
            step_result.get("prediction") is not None
            and step_result.get("prediction") != cot_pred
            for step_result in item.get("unlearning_results", {}).values()
        ):
            changed += 1
    rate = changed / len(agreeing) * 100.0 if agreeing else 0.0
    return len(agreeing), changed, rate


def safe_mean(values):
    values = [v for v in values if v is not None and not math.isnan(v)]
    return sum(values) / len(values) if values else 0.0


def parse_lambda_dir(path):
    for part in path.parts:
        if part.startswith("lambda="):
            return float(part.split("=", 1)[1])
    raise ValueError(f"Could not parse lambda from {path}")


def collect_rows(results_root, expected_lambdas):
    rows = []
    for path in sorted(results_root.glob("lambda=*/*/*/*.out")):
        rt_lambda = parse_lambda_dir(path)
        if expected_lambdas is not None and rt_lambda not in expected_lambdas:
            continue
        dataset = path.parent.parent.name
        model = path.parent.name
        results = load_jsonl(path)
        if not results:
            continue

        stats = make_stats(results)
        agreeing, changed, change_rate = answer_change_rate(results)
        avg_mass = safe_mean([average_mass_shift(r) for r in results])
        max_mass = safe_mean([max_mass_shift(r) for r in results])

        rows.append(
            {
                "lambda": rt_lambda,
                "lambda_label": lambda_label(rt_lambda),
                "dataset": dataset,
                "model": model,
                "faithfulness": float(stats["faithfulness"]),
                "efficacy": float(stats["efficacy"]),
                "specificity": float(stats["specificity"]),
                "answer_change_rate": float(change_rate),
                "agreeing_instances": int(agreeing),
                "changed_instances": int(changed),
                "average_mass_shift": float(avg_mass),
                "max_mass_shift": float(max_mass),
                "n_instances": int(stats["n_instances"]),
                "n_cot_steps": int(stats["n_cot_steps"]),
                "filename": path.name,
                "path": str(path),
            }
        )
    add_baseline_deltas(rows)
    return rows


def add_baseline_deltas(rows):
    baseline = {}
    for row in rows:
        if row["lambda"] == 1.0:
            baseline[(row["dataset"], row["model"])] = row

    for row in rows:
        base = baseline.get((row["dataset"], row["model"]))
        for metric in ["faithfulness", "efficacy", "specificity", "answer_change_rate"]:
            key = f"delta_{metric}"
            row[key] = row[metric] - base[metric] if base else None


def write_json_artifacts(rows, output_dir):
    analysis = {}
    answer_summary = {
        "description": (
            "% of step-level rows where unlearning a reasoning step changes "
            "the model answer, measured only where no-CoT and CoT predictions agree."
        ),
        "unlearning_steps": {},
    }

    for row in rows:
        key = f"lambda={row['lambda_label']}/{row['dataset']}_{row['model']}"
        analysis[key] = row
        answer_summary["unlearning_steps"][key] = {
            "agreeing_instances": row["agreeing_instances"],
            "changed_instances": row["changed_instances"],
            "change_rate_percent": round(row["answer_change_rate"], 2),
        }

    with open(output_dir / "analysis_summary.json", "w", encoding="utf-8") as outfile:
        json.dump(analysis, outfile, indent=2)

    with open(output_dir / "answer_change_rate_summary.json", "w", encoding="utf-8") as outfile:
        json.dump(answer_summary, outfile, indent=2)


def write_csv(rows, output_dir):
    fields = [
        "lambda_label",
        "model",
        "dataset",
        "faithfulness",
        "efficacy",
        "specificity",
        "answer_change_rate",
        "delta_faithfulness",
        "delta_efficacy",
        "delta_specificity",
        "delta_answer_change_rate",
        "average_mass_shift",
        "max_mass_shift",
        "n_instances",
        "n_cot_steps",
        "agreeing_instances",
        "changed_instances",
        "filename",
        "path",
    ]
    with open(output_dir / "summary_table.csv", "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["model"], r["dataset"], r["lambda"])):
            writer.writerow({field: row.get(field) for field in fields})


def best_tradeoff(rows, tolerance=3.0):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["dataset"])].append(row)

    decisions = {}
    for key, items in grouped.items():
        baseline = next((r for r in items if r["lambda"] == 1.0), None)
        if baseline is None:
            decisions[key] = None
            continue
        min_specificity = baseline["specificity"] - tolerance
        candidates = [r for r in items if r["specificity"] >= min_specificity]
        candidates = sorted(candidates, key=lambda r: (r["faithfulness"], r["specificity"]), reverse=True)
        decisions[key] = candidates[0] if candidates else None
    return decisions


def write_report(rows, output_dir, expected_lambdas):
    decisions = best_tradeoff(rows)
    expected = {(lambda_label(l), model, dataset) for l in expected_lambdas for model, dataset, _ in cfg.RUNS}
    observed = {(row["lambda_label"], row["model"], row["dataset"]) for row in rows}
    missing = sorted(expected - observed)

    lines = [
        "# Lambda Sweep Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Experiment Design",
        "",
        "- Objective: evaluate how scaling the retain regularizer in "
        "L(theta)=L_NPO,beta(theta)+lambda*K_RT(theta) changes the "
        "faithfulness/specificity tradeoff.",
        "- Hypothesis: lambda=0.0 should maximize forgetting pressure but may "
        "damage retained behavior; larger lambda values should better preserve "
        "specificity while potentially weakening step-level intervention.",
        "- Scope: the fixed 2x2 reproduction grid from repro/config.py, namely "
        "Phi-3 and LLaMA-3-3B on OpenBookQA and StrategyQA.",
        "- Lambda grid: " + ", ".join(lambda_label(v) for v in expected_lambdas) + ".",
        "",
        "## Completeness",
        "",
        f"- Expected result groups: {len(expected)}",
        f"- Observed result groups: {len(observed)}",
        f"- Missing result groups: {len(missing)}",
    ]
    if missing:
        lines.append("- Missing: " + ", ".join(f"{l}/{m}/{d}" for l, m, d in missing))

    lines.extend(
        [
            "",
            "## Core Metrics",
            "",
            "| Lambda | Model | Dataset | Faithfulness | Efficacy | Specificity | Answer Change |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(rows, key=lambda r: (r["model"], r["dataset"], r["lambda"])):
        lines.append(
            f"| {row['lambda_label']} | {row['model']} | {row['dataset']} | "
            f"{row['faithfulness']:.2f} | {row['efficacy']:.2f} | "
            f"{row['specificity']:.2f} | {row['answer_change_rate']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Baseline-Relative Interpretation",
            "",
            "Deltas are computed against lambda=1.0 within the same model and dataset.",
            "Positive faithfulness/efficacy deltas indicate stronger step-level "
            "intervention; positive specificity deltas indicate better retention "
            "on held-out examples.",
            "",
            "| Lambda | Model | Dataset | Delta Faithfulness | Delta Efficacy | Delta Specificity | Delta Answer Change |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(rows, key=lambda r: (r["model"], r["dataset"], r["lambda"])):
        def fmt(value):
            return "NA" if value is None else f"{value:.2f}"

        lines.append(
            f"| {row['lambda_label']} | {row['model']} | {row['dataset']} | "
            f"{fmt(row['delta_faithfulness'])} | {fmt(row['delta_efficacy'])} | "
            f"{fmt(row['delta_specificity'])} | {fmt(row['delta_answer_change_rate'])} |"
        )

    lines.extend(["", "## Recommended Lambda By Combination", ""])
    for (model, dataset), row in sorted(decisions.items()):
        if row is None:
            lines.append(f"- {model}/{dataset}: no recommendation because lambda=1.0 baseline is missing.")
        else:
            lines.append(
                f"- {model}/{dataset}: lambda={row['lambda_label']} "
                f"(faithfulness={row['faithfulness']:.2f}, specificity={row['specificity']:.2f})."
            )

    lines.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            "- Low lambda weakens retain regularization; it should be treated as a plasticity stress test, not automatically as the best setting.",
            "- High lambda strengthens retain regularization; it is useful when specificity is the primary constraint.",
            "- The selected trade-off uses the pre-declared rule: maximize faithfulness while staying within 3 specificity points of lambda=1.0.",
            "- Answer-change rate is diagnostic rather than the sole target metric; it can rise when unlearning disrupts final-answer behavior.",
            "- Missing groups indicate incomplete runs, failed jobs, or results written outside the expected lambda/results/lambda=<value>/ tree.",
        ]
    )

    with open(output_dir / "report_notes.md", "w", encoding="utf-8") as outfile:
        outfile.write("\n".join(lines) + "\n")


def write_figures(rows, output_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] Skipping figures: {exc}")
        return

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics = ["faithfulness", "efficacy", "specificity", "answer_change_rate"]
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["dataset"])].append(row)

    for metric in metrics:
        plt.figure(figsize=(8, 5))
        for (model, dataset), items in sorted(grouped.items()):
            items = sorted(items, key=lambda r: r["lambda"])
            plt.plot([r["lambda_label"] for r in items], [r[metric] for r in items], marker="o", label=f"{model}/{dataset}")
        plt.xlabel("lambda")
        plt.ylabel(metric.replace("_", " ").title())
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(figures_dir / f"{metric}_by_lambda.png", dpi=200)
        plt.close()

    plt.figure(figsize=(7, 5))
    for (model, dataset), items in sorted(grouped.items()):
        items = sorted(items, key=lambda r: r["lambda"])
        plt.plot([r["specificity"] for r in items], [r["faithfulness"] for r in items], marker="o", label=f"{model}/{dataset}")
    plt.xlabel("Specificity")
    plt.ylabel("Faithfulness")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figures_dir / "faithfulness_specificity_frontier.png", dpi=200)
    plt.close()


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="lambda/results")
    parser.add_argument("--output_dir", default="lambda")
    parser.add_argument("--lambdas", nargs="+", type=float, default=DEFAULT_LAMBDAS)
    parser.add_argument("--no_figures", action="store_true")
    return parser


def main():
    args = make_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(Path(args.results_root), set(args.lambdas))
    write_json_artifacts(rows, output_dir)
    write_csv(rows, output_dir)
    write_report(rows, output_dir, args.lambdas)
    if not args.no_figures:
        write_figures(rows, output_dir)
    print(f"Wrote {len(rows)} result summaries to {output_dir}")


if __name__ == "__main__":
    main()
