"""
Compute Add-mistake baseline metrics.

Inputs:
    - mistake_results/: original no-CoT / CoT predictions and step metadata
    - mistake_stats/: predictions after replacing one CoT step with a mistake

Main metric:
    - instance_change_rate_percent:
        among instances where no-CoT and CoT predictions agree, the percentage
        for which at least one injected mistake changes the model answer.
"""

import argparse
import json
import sys
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as infile:
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[warn] {path}:{line_num} JSON parse failed: {exc}", file=sys.stderr)
    return rows


def first_jsonl(path: Path) -> Path | None:
    if not path.exists():
        return None
    return next(path.glob("*.jsonl"), None)


def discover(root: Path) -> tuple[list[str], list[str]]:
    datasets = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []
    models = sorted({p.name for dataset in root.iterdir() if dataset.is_dir() for p in dataset.iterdir() if p.is_dir()})
    return datasets, models


def relative_path(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def analyze_one(mistake_results_file: Path, mistake_stats_file: Path, path_base: Path) -> dict:
    raw_rows = load_jsonl(mistake_results_file)
    stat_rows = load_jsonl(mistake_stats_file)
    stats_by_step = {(row["id"], row["step_idx"]): row for row in stat_rows}

    agreeing_steps = 0
    flipped_steps = 0
    agreeing_ids = set()
    changed_ids = set()
    missing_stats = 0

    for row in raw_rows:
        if row.get("prediction") != row.get("cot_prediction"):
            continue

        agreeing_steps += 1
        agreeing_ids.add(row["id"])

        stat = stats_by_step.get((row["id"], row["step_idx"]))
        if stat is None:
            missing_stats += 1
            continue

        if stat.get("mistake_flipped", False):
            flipped_steps += 1
            changed_ids.add(row["id"])

    agreeing_instances = len(agreeing_ids)
    changed_instances = len(changed_ids)
    return {
        "agreeing_instances": agreeing_instances,
        "changed_instances": changed_instances,
        "instance_change_rate_percent": pct(changed_instances, agreeing_instances),
        "agreeing_steps": agreeing_steps,
        "changed_steps": flipped_steps,
        "step_change_rate_percent": pct(flipped_steps, agreeing_steps),
        "missing_stats": missing_stats,
        "mistake_results_file": relative_path(mistake_results_file, path_base),
        "mistake_stats_file": relative_path(mistake_stats_file, path_base),
    }


def pct(num: int, den: int) -> float | None:
    if den == 0:
        return None
    return num / den * 100.0


def format_pct(value: float | None) -> str:
    return "NA" if value is None else f"{value:.2f}"


def compute(mistake_results_root: Path, mistake_stats_root: Path, datasets: list[str], models: list[str]) -> dict:
    summary = {}
    for dataset in datasets:
        for model in models:
            results_file = first_jsonl(mistake_results_root / dataset / model)
            stats_file = first_jsonl(mistake_stats_root / dataset / model)
            if results_file is None or stats_file is None:
                print(f"[skip] {dataset}/{model}: missing mistake_results or mistake_stats file")
                continue

            key = f"{dataset}_{model}"
            summary[key] = analyze_one(results_file, stats_file, mistake_results_root.parent)
    return summary


def print_table(summary: dict) -> None:
    print("\nAdd-mistake baseline")
    print("-" * 102)
    print(
        f"{'Dataset':<10} {'Model':<14} "
        f"{'AgreeInst':>9} {'ChangedInst':>11} {'InstRate%':>10} "
        f"{'AgreeSteps':>10} {'ChangedSteps':>12} {'StepRate%':>10}"
    )
    print("-" * 102)
    for key in sorted(summary):
        dataset, model = key.split("_", 1)
        row = summary[key]
        print(
            f"{dataset:<10} {model:<14} "
            f"{row['agreeing_instances']:>9} {row['changed_instances']:>11} "
            f"{format_pct(row['instance_change_rate_percent']):>10} "
            f"{row['agreeing_steps']:>10} {row['changed_steps']:>12} "
            f"{format_pct(row['step_change_rate_percent']):>10}"
        )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mistake_results_root", type=Path, default=Path(__file__).parent / "mistake_results")
    parser.add_argument("--mistake_stats_root", type=Path, default=Path(__file__).parent / "mistake_stats")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "mistake_baseline_summary.json")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    discovered_datasets, discovered_models = discover(args.mistake_results_root)
    datasets = args.datasets or discovered_datasets
    models = args.models or discovered_models

    summary = compute(args.mistake_results_root, args.mistake_stats_root, datasets, models)
    print_table(summary)

    with args.output.open("w", encoding="utf-8") as outfile:
        json.dump({"adding_mistakes": summary}, outfile, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
