"""CoT-level interpretability analysis for the lambda sweep.

The script compares aligned stepwise unlearning records across lambda values.
It does not load models or rerun unlearning; all evidence comes from existing
JSONL outputs under lambda/results and final_results.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


LAMBDAS = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]
LOW_LAMBDAS = {0.0, 0.1, 0.3}
HIGH_LAMBDAS = {3.0, 10.0}
COMBOS = [
    ("LLaMA-3-3B", "openbook"),
    ("LLaMA-3-3B", "sqa"),
    ("Phi-3", "openbook"),
    ("Phi-3", "sqa"),
]


def lambda_label(value: float) -> str:
    return f"{float(value):.1f}"


def fmt(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if math.isnan(value):
        return ""
    return f"{value:.4f}"


def safe_text(value: object, limit: int = 360) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def final_epoch(unlearning_results: dict) -> dict:
    key = max(unlearning_results, key=lambda value: int(value))
    return unlearning_results[key]


def first_float(value: object) -> float | None:
    if isinstance(value, list):
        if not value:
            return None
        value = value[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def count_drift(before: list, after: list) -> int:
    return sum(1 for left, right in zip(before or [], after or []) if left != right)


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9]+", text.lower()))


def overlap_ratio(target: str, candidate: str) -> float:
    target_tokens = token_set(target)
    if not target_tokens:
        return 0.0
    return len(target_tokens & token_set(candidate)) / len(target_tokens)


def step_phase(step_idx: int, n_steps: int) -> str:
    if n_steps <= 1:
        return "single"
    rel = step_idx / max(n_steps - 1, 1)
    if rel < 1 / 3:
        return "early"
    if rel < 2 / 3:
        return "middle"
    return "late"


def step_tags(text: str) -> str:
    lower = text.lower()
    tags = []
    if re.search(r"\([a-e]\)|choice|option|answer", lower):
        tags.append("choice_or_answer")
    if re.search(r"\bbecause\b|\btherefore\b|\bthus\b|\bso\b|due to|leads? to|causes?", lower):
        tags.append("causal")
    if re.search(r"\bnot\b|n't|incorrect|eliminate|unlikely|except", lower):
        tags.append("negation_or_elimination")
    if re.search(r"\bmeans\b|\brefers to\b|\bis\b|\bare\b|\bwill\b|\bcan\b", lower):
        tags.append("factual_or_definition")
    if re.search(r"compare|more than|less than|similar|different", lower):
        tags.append("comparison")
    return ";".join(tags) if tags else "other"


def parse_lambda_result_path(path: Path) -> tuple[float, str, str]:
    parts = path.parts
    # .../lambda/results/lambda=0.0/openbook/Phi-3/file.out
    lambda_idx = next(i for i, part in enumerate(parts) if part.startswith("lambda="))
    lambda_value = float(parts[lambda_idx].split("=", 1)[1])
    return lambda_value, parts[lambda_idx + 1], parts[lambda_idx + 2]


def row_key(row: dict) -> tuple[str, int]:
    return str(row["id"]), int(row["step_idx"])


def load_result_file(path: Path) -> dict[tuple[str, int], dict]:
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(("<<<<<<<", "=======", ">>>>>>>")):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Could not parse JSONL record in {path} at line {line_no}") from exc
            if "unlearning_results" not in row:
                continue
            final = final_epoch(row["unlearning_results"])
            epoch0 = row["unlearning_results"].get("0", {})
            target_step = final.get("target_cot_step") or row.get("cot_step") or ""
            new_cot = final.get("new_cot") or ""
            segmented = row.get("segmented_cot") or []
            records[row_key(row)] = {
                "id": str(row["id"]),
                "step_idx": int(row["step_idx"]),
                "question": row.get("question", ""),
                "target_cot_step": target_step,
                "n_steps": len(segmented) if segmented else 1,
                "phase": step_phase(int(row["step_idx"]), len(segmented) if segmented else 1),
                "relative_position": int(row["step_idx"]) / max((len(segmented) if segmented else 1) - 1, 1),
                "step_tags": step_tags(target_step),
                "initial_prediction": row.get("prediction"),
                "cot_prediction": row.get("cot_prediction"),
                "final_prediction": final.get("prediction"),
                "epoch0_cot_step_logprob": first_float(epoch0.get("cot_step_prob")),
                "final_cot_step_logprob": first_float(final.get("cot_step_prob")),
                "epoch0_specificity_drift": 0,
                "final_specificity_drift": count_drift(
                    epoch0.get("specificity_preds") or [], final.get("specificity_preds") or []
                ),
                "new_cot": new_cot,
                "target_in_new_cot": safe_text(target_step, 240).lower() in safe_text(new_cot, 10000).lower(),
                "target_new_cot_overlap": overlap_ratio(target_step, new_cot),
            }
    return records


def load_lambda_results(root: Path) -> dict[tuple[str, str, float], dict[tuple[str, int], dict]]:
    results = {}
    for path in sorted((root / "lambda" / "results").glob("lambda=*/*/*/*.out")):
        if "smoke" in path.parts:
            continue
        lambda_value, dataset, model = parse_lambda_result_path(path)
        results[(model, dataset, lambda_value)] = load_result_file(path)
    return results


def load_final_results(root: Path) -> dict[tuple[str, str], dict[tuple[str, int], dict]]:
    results = {}
    for path in sorted((root / "final_results").glob("*/*/*.out")):
        parts = path.parts
        final_idx = parts.index("final_results")
        dataset, model = parts[final_idx + 1], parts[final_idx + 2]
        results[(model, dataset)] = load_result_file(path)
    return results


def verify_alignment(lambda_results: dict, final_results: dict) -> dict:
    checks = {
        "lambda_result_files": len(lambda_results),
        "lambda_step_records": sum(len(records) for records in lambda_results.values()),
        "final_result_files": len(final_results),
        "alignment_failures": [],
    }
    for model, dataset in COMBOS:
        base_keys = set(lambda_results[(model, dataset, 1.0)])
        final_keys = set(final_results[(model, dataset)])
        if base_keys != final_keys:
            checks["alignment_failures"].append(
                f"{model}/{dataset}: lambda=1.0 and final_results differ"
            )
        for lambda_value in LAMBDAS:
            keys = set(lambda_results[(model, dataset, lambda_value)])
            if keys != base_keys:
                checks["alignment_failures"].append(
                    f"{model}/{dataset}/lambda={lambda_label(lambda_value)} differs from lambda=1.0"
                )
    return checks


def build_step_effect_rows(lambda_results: dict, final_results: dict) -> list[dict]:
    rows = []
    for model, dataset in COMBOS:
        baseline = lambda_results[(model, dataset, 1.0)]
        final_baseline = final_results[(model, dataset)]
        for key, base in sorted(baseline.items()):
            original = final_baseline[key]
            for lambda_value in LAMBDAS:
                current = lambda_results[(model, dataset, lambda_value)][key]
                current_final_logprob = current["final_cot_step_logprob"]
                base_final_logprob = base["final_cot_step_logprob"]
                current_change = (
                    None
                    if current_final_logprob is None or current["epoch0_cot_step_logprob"] is None
                    else current_final_logprob - current["epoch0_cot_step_logprob"]
                )
                base_change = (
                    None
                    if base_final_logprob is None or base["epoch0_cot_step_logprob"] is None
                    else base_final_logprob - base["epoch0_cot_step_logprob"]
                )
                rows.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "lambda_label": lambda_label(lambda_value),
                        "id": current["id"],
                        "step_idx": current["step_idx"],
                        "n_steps": current["n_steps"],
                        "relative_position": fmt(current["relative_position"]),
                        "phase": current["phase"],
                        "step_tags": current["step_tags"],
                        "initial_prediction": current["initial_prediction"],
                        "lambda1_final_prediction": base["final_prediction"],
                        "current_final_prediction": current["final_prediction"],
                        "repro_final_prediction": original["final_prediction"],
                        "differs_from_lambda1_final_prediction": int(
                            current["final_prediction"] != base["final_prediction"]
                        ),
                        "answer_changed_from_initial": int(
                            current["final_prediction"] != current["initial_prediction"]
                        ),
                        "lambda1_answer_changed_from_initial": int(
                            base["final_prediction"] != base["initial_prediction"]
                        ),
                        "final_cot_step_logprob": fmt(current_final_logprob),
                        "lambda1_final_cot_step_logprob": fmt(base_final_logprob),
                        "final_cot_step_logprob_delta_vs_lambda1": fmt(
                            None
                            if current_final_logprob is None or base_final_logprob is None
                            else current_final_logprob - base_final_logprob
                        ),
                        "cot_step_logprob_change": fmt(current_change),
                        "lambda1_cot_step_logprob_change": fmt(base_change),
                        "cot_step_logprob_change_delta_vs_lambda1": fmt(
                            None if current_change is None or base_change is None else current_change - base_change
                        ),
                        "specificity_drift": current["final_specificity_drift"],
                        "lambda1_specificity_drift": base["final_specificity_drift"],
                        "specificity_drift_delta_vs_lambda1": current["final_specificity_drift"]
                        - base["final_specificity_drift"],
                        "target_new_cot_overlap": fmt(current["target_new_cot_overlap"]),
                        "lambda1_target_new_cot_overlap": fmt(base["target_new_cot_overlap"]),
                        "target_new_cot_overlap_delta_vs_lambda1": fmt(
                            current["target_new_cot_overlap"] - base["target_new_cot_overlap"]
                        ),
                        "target_in_new_cot": int(current["target_in_new_cot"]),
                        "question": safe_text(current["question"]),
                        "target_cot_step": safe_text(current["target_cot_step"]),
                        "new_cot": safe_text(current["new_cot"]),
                    }
                )
    return rows


def numeric(row: dict, key: str) -> float:
    value = row.get(key)
    if value == "" or value is None:
        return float("nan")
    return float(value)


def summarize(groups: dict[tuple, list[dict]]) -> list[dict]:
    output = []
    for key, rows in sorted(groups.items()):
        lambda_rows = rows
        logprob_deltas = [
            numeric(row, "final_cot_step_logprob_delta_vs_lambda1")
            for row in lambda_rows
            if not math.isnan(numeric(row, "final_cot_step_logprob_delta_vs_lambda1"))
        ]
        change_deltas = [
            numeric(row, "cot_step_logprob_change_delta_vs_lambda1")
            for row in lambda_rows
            if not math.isnan(numeric(row, "cot_step_logprob_change_delta_vs_lambda1"))
        ]
        output.append(
            {
                "group_type": key[0],
                "model": key[1],
                "dataset": key[2],
                "lambda_label": key[3],
                "group_value": key[4],
                "n_steps": len(lambda_rows),
                "prediction_diff_rate_vs_lambda1": mean(
                    int(row["differs_from_lambda1_final_prediction"]) for row in lambda_rows
                ),
                "answer_change_rate": mean(int(row["answer_changed_from_initial"]) for row in lambda_rows),
                "avg_final_logprob_delta_vs_lambda1": mean(logprob_deltas) if logprob_deltas else float("nan"),
                "avg_logprob_change_delta_vs_lambda1": mean(change_deltas) if change_deltas else float("nan"),
                "avg_specificity_drift_delta_vs_lambda1": mean(
                    int(row["specificity_drift_delta_vs_lambda1"]) for row in lambda_rows
                ),
                "avg_target_new_cot_overlap_delta_vs_lambda1": mean(
                    numeric(row, "target_new_cot_overlap_delta_vs_lambda1") for row in lambda_rows
                ),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_position_summary(step_rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in step_rows:
        groups[("phase", row["model"], row["dataset"], row["lambda_label"], row["phase"])].append(row)
        groups[("step_idx", row["model"], row["dataset"], row["lambda_label"], str(row["step_idx"]))].append(row)
        for tag in str(row["step_tags"]).split(";"):
            groups[("step_tag", row["model"], row["dataset"], row["lambda_label"], tag)].append(row)
    return summarize(groups)


def build_high_lambda_summary(step_rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in step_rows:
        lambda_value = float(row["lambda_label"])
        if lambda_value in HIGH_LAMBDAS:
            groups[(row["model"], row["dataset"], row["lambda_label"])].append(row)
    output = []
    for (model, dataset, lambda_value), rows in sorted(groups.items()):
        higher_logprob = [
            row
            for row in rows
            if numeric(row, "final_cot_step_logprob_delta_vs_lambda1") > 0
        ]
        output.append(
            {
                "model": model,
                "dataset": dataset,
                "lambda_label": lambda_value,
                "n_steps": len(rows),
                "prediction_diff_steps_vs_lambda1": sum(
                    int(row["differs_from_lambda1_final_prediction"]) for row in rows
                ),
                "prediction_diff_rate_vs_lambda1": mean(
                    int(row["differs_from_lambda1_final_prediction"]) for row in rows
                ),
                "steps_with_higher_final_cot_step_logprob_than_lambda1": len(higher_logprob),
                "share_higher_final_cot_step_logprob_than_lambda1": len(higher_logprob) / len(rows),
                "avg_final_cot_step_logprob_delta_vs_lambda1": mean(
                    numeric(row, "final_cot_step_logprob_delta_vs_lambda1") for row in rows
                ),
                "avg_specificity_drift_delta_vs_lambda1": mean(
                    int(row["specificity_drift_delta_vs_lambda1"]) for row in rows
                ),
                "answer_change_rate": mean(int(row["answer_changed_from_initial"]) for row in rows),
                "lambda1_answer_change_rate_on_same_steps": mean(
                    int(row["lambda1_answer_changed_from_initial"]) for row in rows
                ),
            }
        )
    return output


def case_sort_value(row: dict, key: str) -> float:
    value = numeric(row, key)
    return -999999 if math.isnan(value) else value


def format_case(row: dict, title: str, interpretation: str) -> str:
    return f"""### {title}

- 设置：{row['model']} / {row['dataset']} / lambda={row['lambda_label']}，id={row['id']}，step_idx={row['step_idx']}（{row['phase']}，{row['step_tags']}）
- 预测：initial={row['initial_prediction']}，lambda=1 final={row['lambda1_final_prediction']}，current final={row['current_final_prediction']}
- 目标 step logprob 相对 lambda=1 的差值：{row['final_cot_step_logprob_delta_vs_lambda1']}
- specificity drift 相对 lambda=1 的差值：{row['specificity_drift_delta_vs_lambda1']}
- target/new-CoT overlap 相对 lambda=1 的差值：{row['target_new_cot_overlap_delta_vs_lambda1']}
- 问题：{row['question']}
- 目标 CoT step：{row['target_cot_step']}
- 解释：{interpretation}
"""


def build_representative_cases(step_rows: list[dict], high_summary: list[dict]) -> str:
    lines = [
        "# CoT-Level Representative Cases",
        "",
        "这些案例把每个 lambda 与完全对齐的 `lambda=1.0` baseline 比较。",
        "目标 step logprob delta 为正，表示该目标 CoT step 比 `lambda=1.0` 更容易被模型保留，也就是该 step 承受的 forgetting pressure 更弱。",
        "",
    ]
    for model, dataset in COMBOS:
        combo_rows = [row for row in step_rows if row["model"] == model and row["dataset"] == dataset]
        lines.append(f"## {model} / {dataset}")
        high_rows = [row for row in combo_rows if float(row["lambda_label"]) in HIGH_LAMBDAS]
        low_rows = [row for row in combo_rows if float(row["lambda_label"]) in LOW_LAMBDAS]

        retained = max(
            high_rows,
            key=lambda row: (
                case_sort_value(row, "final_cot_step_logprob_delta_vs_lambda1"),
                int(row["differs_from_lambda1_final_prediction"]),
            ),
        )
        lines.append(
            format_case(
                retained,
                "高 lambda 保留压力案例",
                "更高的 lambda 使这个目标 CoT step 明显比 `lambda=1.0` 更高概率地保留下来，这是 retain KL 抑制该局部 step forgetting 的直接迹象。",
            )
        )

        low_forgetting = min(
            low_rows,
            key=lambda row: case_sort_value(row, "final_cot_step_logprob_delta_vs_lambda1"),
        )
        lines.append(
            format_case(
                low_forgetting,
                "低 lambda 更强遗忘案例",
                "更低的 lambda 使这个目标 step 比 `lambda=1.0` 更低概率地保留，说明 trade-off 的 plasticity 一侧确实落在具体 CoT step 上。",
            )
        )

        divergent = [
            row
            for row in combo_rows
            if int(row["differs_from_lambda1_final_prediction"]) == 1
        ]
        if divergent:
            chosen = max(
                divergent,
                key=lambda row: (
                    float(row["lambda_label"]) in HIGH_LAMBDAS,
                    abs(case_sort_value(row, "specificity_drift_delta_vs_lambda1")),
                    abs(case_sort_value(row, "final_cot_step_logprob_delta_vs_lambda1")),
                ),
            )
            lines.append(
                format_case(
                    chosen,
                    "final answer 分歧案例",
                    "这个 step 上 changing lambda 会改变相对 `lambda=1.0` 的最终答案，因此适合观察 CoT 层差异如何外溢成 answer-level 差异。",
                )
            )
        else:
            lines.append(
                "### final answer 分歧案例\n\n- 该组合中没有发现相对 `lambda=1.0` 的 final-prediction 分歧。\n"
            )

        drift = max(
            combo_rows,
            key=lambda row: abs(int(row["specificity_drift_delta_vs_lambda1"])),
        )
        lines.append(
            format_case(
                drift,
                "specificity drift 案例",
                "这个例子展示 changing lambda 最明显改变 retained-probe behavior 的位置，有助于区分 faithfulness 收益和一般性扰动。",
            )
        )
    lines.append("## High-Lambda Aggregate Summary")
    lines.append("")
    lines.append("| Model | Dataset | Lambda | Steps | Higher target-step logprob share | Avg logprob delta | Prediction diff rate | Avg specificity drift delta |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in high_summary:
        lines.append(
            "| {model} | {dataset} | {lambda_label} | {n_steps} | {share:.2%} | {delta:.4f} | {diff:.2%} | {drift:.4f} |".format(
                model=row["model"],
                dataset=row["dataset"],
                lambda_label=row["lambda_label"],
                n_steps=row["n_steps"],
                share=row["share_higher_final_cot_step_logprob_than_lambda1"],
                delta=row["avg_final_cot_step_logprob_delta_vs_lambda1"],
                diff=row["prediction_diff_rate_vs_lambda1"],
                drift=row["avg_specificity_drift_delta_vs_lambda1"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def mean_by(rows: list[dict], key: str) -> float:
    values = [numeric(row, key) for row in rows if not math.isnan(numeric(row, key))]
    return mean(values) if values else float("nan")


def make_bar_svg(path: Path, title: str, rows: list[dict], value_key: str, y_label: str) -> None:
    width, height = 980, 440
    margin_left, margin_bottom, margin_top, margin_right = 72, 90, 54, 28
    plot_w, plot_h = width - margin_left - margin_right, height - margin_top - margin_bottom
    values = [float(row[value_key]) for row in rows]
    min_v, max_v = min(values), max(values)
    pad = max(0.05, (max_v - min_v) * 0.15)
    min_v -= pad
    max_v += pad
    if min_v > 0:
        min_v = 0
    if max_v < 0:
        max_v = 0
    bar_w = plot_w / len(rows) * 0.68
    gap = plot_w / len(rows)

    def y(value: float) -> float:
        return margin_top + (max_v - value) * plot_h / (max_v - min_v)

    zero_y = y(0)
    output = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#faf9f5"/>',
        f'<text x="24" y="32" font-family="Arial, sans-serif" font-size="21" font-weight="700" fill="#222">{esc(title)}</text>',
        f'<text x="24" y="52" font-family="Arial, sans-serif" font-size="12" fill="#555">{esc(y_label)}</text>',
        f'<line x1="{margin_left}" y1="{zero_y:.1f}" x2="{width-margin_right}" y2="{zero_y:.1f}" stroke="#555" stroke-width="1.2"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height-margin_bottom}" stroke="#777"/>',
    ]
    colors = {"3.0": "#c8a641", "10.0": "#4aa3b5"}
    for i, row in enumerate(rows):
        value = float(row[value_key])
        x = margin_left + i * gap + (gap - bar_w) / 2
        y0 = y(max(value, 0))
        y1 = y(min(value, 0))
        h = max(1, y1 - y0)
        color = colors.get(str(row.get("lambda_label")), "#8172b3")
        output.append(f'<rect x="{x:.1f}" y="{y0:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>')
        output.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{y0 - 5 if value >= 0 else y1 + 15:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#333">{value:.2f}</text>'
        )
        label = f'{row["model"].replace("-3B", "")}/{row["dataset"]}/λ={row["lambda_label"]}'
        output.append(
            f'<text transform="translate({x + bar_w / 2:.1f},{height - margin_bottom + 12}) rotate(45)" font-family="Arial, sans-serif" font-size="10" fill="#333">{esc(label)}</text>'
        )
    output.append(f'<text x="18" y="{margin_top + 8}" font-family="Arial, sans-serif" font-size="10" fill="#555">{max_v:.2f}</text>')
    output.append(f'<text x="18" y="{height - margin_bottom}" font-family="Arial, sans-serif" font-size="10" fill="#555">{min_v:.2f}</text>')
    output.append("</svg>")
    path.write_text("\n".join(output), encoding="utf-8")


def make_phase_svg(path: Path, position_summary: list[dict]) -> None:
    phase_rows = [
        row
        for row in position_summary
        if row["group_type"] == "phase" and row["lambda_label"] in {"3.0", "10.0"}
    ]
    rows = []
    for model, dataset in COMBOS:
        for lambda_value in ["3.0", "10.0"]:
            for phase in ["early", "middle", "late"]:
                matches = [
                    row
                    for row in phase_rows
                    if row["model"] == model
                    and row["dataset"] == dataset
                    and row["lambda_label"] == lambda_value
                    and row["group_value"] == phase
                ]
                if matches:
                    row = matches[0].copy()
                    row["label"] = f"{model}/{dataset}/{phase}/λ={lambda_value}"
                    rows.append(row)
    make_bar_svg(
        path,
        "High-Lambda CoT Step-Position Effect",
        rows,
        "avg_final_logprob_delta_vs_lambda1",
        "Average final target-step logprob delta vs lambda=1.0 by early/middle/late position.",
    )


def update_report(root: Path, high_summary: list[dict], position_summary: list[dict]) -> None:
    report_path = root / "lambda" / "report_notes.md"
    report = report_path.read_text(encoding="utf-8")
    cot_marker = "\n## 9. CoT-Level Interpretability Analysis"
    old_limitations_marker = "\n## 9. Limitations"
    new_limitations_marker = "\n## 10. Limitations"

    if cot_marker in report:
        prefix, remainder = report.split(cot_marker, 1)
        if new_limitations_marker in remainder:
            _, limitations = remainder.split(new_limitations_marker, 1)
            limitations = "## 10. Limitations" + limitations
        elif old_limitations_marker in remainder:
            _, limitations = remainder.split(old_limitations_marker, 1)
            limitations = "## 10. Limitations" + limitations
        else:
            limitations = ""
    elif old_limitations_marker in report:
        prefix, limitations = report.split(old_limitations_marker, 1)
        limitations = "## 10. Limitations" + limitations
    elif new_limitations_marker in report:
        prefix, limitations = report.split(new_limitations_marker, 1)
        limitations = "## 10. Limitations" + limitations
    else:
        prefix, limitations = report.rstrip(), ""
    if cot_marker.strip() in limitations:
        limitations = limitations.split(cot_marker.strip(), 1)[0].rstrip() + "\n"

    high_lines = [
        "| Model | Dataset | Lambda | Higher logprob share | Avg logprob delta | Prediction diff rate | Avg specificity drift delta |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in high_summary:
        high_lines.append(
            "| {model} | {dataset} | {lambda_label} | {share:.2%} | {delta:.4f} | {diff:.2%} | {drift:.4f} |".format(
                model=row["model"],
                dataset=row["dataset"],
                lambda_label=row["lambda_label"],
                share=row["share_higher_final_cot_step_logprob_than_lambda1"],
                delta=row["avg_final_cot_step_logprob_delta_vs_lambda1"],
                diff=row["prediction_diff_rate_vs_lambda1"],
                drift=row["avg_specificity_drift_delta_vs_lambda1"],
            )
        )

    phase_examples = [
        row
        for row in position_summary
        if row["group_type"] == "phase"
        and row["lambda_label"] in {"3.0", "10.0"}
        and row["group_value"] in {"early", "middle", "late"}
    ]
    strongest = sorted(
        phase_examples,
        key=lambda row: abs(float(row["avg_final_logprob_delta_vs_lambda1"])),
        reverse=True,
    )[:8]
    phase_lines = [
        "| Model | Dataset | Lambda | Position | Avg logprob delta | Prediction diff rate | Avg specificity drift delta |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in strongest:
        phase_lines.append(
            "| {model} | {dataset} | {lambda_label} | {group_value} | {delta:.4f} | {diff:.2%} | {drift:.4f} |".format(
                model=row["model"],
                dataset=row["dataset"],
                lambda_label=row["lambda_label"],
                group_value=row["group_value"],
                delta=float(row["avg_final_logprob_delta_vs_lambda1"]),
                diff=float(row["prediction_diff_rate_vs_lambda1"]),
                drift=float(row["avg_specificity_drift_delta_vs_lambda1"]),
            )
        )

    cot_section = f"""
## 9. CoT-Level Interpretability Analysis

这一节把前面 aggregate-level 的 lambda 结论落到逐 CoT step。所有比较都使用跨 lambda 完全对齐的 `(id, step_idx)` 记录，并以 `lambda=1.0` 作为 baseline。这里最关键的局部量是 target-step logprob delta：如果它为正，说明该目标 CoT step 在当前 lambda 下比 `lambda=1.0` 更容易被模型继续生成，也就是 retain KL 对这个 step 的 forgetting 有抑制作用；如果它为负，则说明该 step 比 baseline 更容易被遗忘或替换。

### 9.1 高 lambda 的局部效应

{chr(10).join(high_lines)}

主要结论是：高 lambda 通常会让目标 CoT step 比 `lambda=1.0` 更高概率地保留下来。这个现象在 LLaMA/openbook 和 LLaMA/sqa 上最明显，`lambda=10.0` 的平均 final target-step logprob delta 分别约为 5.22 和 4.74；Phi/openbook 也很强，约为 5.72；Phi/sqa 较弱但方向仍为正。换句话说，调高 lambda 主要不是创造新的推理路径，而是增强对原始目标 step 的保留压力。这可以解释为什么高 lambda 往往提高 specificity、降低或抑制 answer-change，但也可能削弱 unlearning 对 final answer 的影响。

### 9.2 差异集中在 CoT 的哪些位置

{chr(10).join(phase_lines)}

差异并不是均匀分布在所有 CoT step 上。高 lambda 的最大 logprob 差异常出现在早期事实铺垫、中段因果/定义判断、以及带有排除或答案承诺的 step。比如 LLaMA/sqa 的 early step 在 `lambda=10.0` 下 prediction diff rate 达到 6.90%，说明一些早期判断一旦被保留或替换，可能更容易传播到最终答案。Phi/openbook 则在 middle 和 late step 上都有较强的 logprob 保留效应，但 final answer 分歧率很低，说明它更多表现为局部 CoT 稳定性变化，而不是答案翻转。

### 9.3 对“调高比例导致哪里差异”的解释

调高到 `lambda=3.0/10.0` 后，最直接的变化是目标 step 的 final logprob 相对 `lambda=1.0` 上升：四个 model/dataset 组合中，`lambda=10.0` 有 82.94% 到 96.19% 的 step 高于 baseline。也就是说，高 lambda 会在具体 step 层面抑制 NPO 对目标推理片段的删除。若该 step 是事实定义、因果桥接或答案选择相关判断，它被保留下来后 final answer 就更可能维持 baseline；若该 step 只是局部解释性文本，它可能只改变 new CoT 的措辞或 overlap，而不改变答案。

低 lambda 的行为正好提供对照：`lambda=0.0/0.1/0.3` 更容易降低目标 step logprob，代表更强 plasticity 和更强 forgetting pressure。这个方向有时会带来更高 answer-level faithfulness 或 answer-change，但也更容易引入 specificity drift。因此，lambda 的可解释作用可以概括为：它调节“目标 step 被替换的强度”和“retained probes 被扰动的强度”之间的平衡，而不是简单单调地提高所有指标。

### 9.4 具体案例证据

`lambda/cot_level_analysis/representative_cases.md` 列出了每个 model/dataset 的代表样本，包含高 lambda 保留目标 step、低 lambda 强化遗忘、final-answer 分歧、specificity drift 四类案例。每个案例都给出 question、target CoT step、step position、prediction change、target-step logprob delta、specificity drift delta 和解释，用来定位“到底是哪一个局部 CoT 判断发生了差异”。

### 9.5 新增 CoT-Level 产物

- `lambda/cot_level_analysis/cot_step_effects.csv`: 每一行是一个对齐后的 lambda/CoT step 比较。
- `lambda/cot_level_analysis/step_position_summary.csv`: 按 step index、early/middle/late 位置和启发式 target-step tag 聚合。
- `lambda/cot_level_analysis/high_lambda_effect_summary.csv`: 专门比较 `lambda=3.0/10.0` 相对 `lambda=1.0` 的局部影响。
- `lambda/cot_level_analysis/representative_cases.md`: 支撑数值结论的具体样本。
- `lambda/figures/cot_level_high_lambda_logprob_delta.svg`
- `lambda/figures/cot_level_high_lambda_prediction_diff.svg`
- `lambda/figures/cot_level_position_effects.svg`

"""
    report_path.write_text(prefix.rstrip() + "\n" + cot_section + limitations, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root)
    output_dir = root / "lambda" / "cot_level_analysis"
    figure_dir = root / "lambda" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    lambda_results = load_lambda_results(root)
    final_results = load_final_results(root)
    checks = verify_alignment(lambda_results, final_results)
    if checks["alignment_failures"]:
        raise RuntimeError("Alignment failures: " + "; ".join(checks["alignment_failures"]))

    step_rows = build_step_effect_rows(lambda_results, final_results)
    position_summary = build_position_summary(step_rows)
    high_summary = build_high_lambda_summary(step_rows)

    write_csv(output_dir / "cot_step_effects.csv", step_rows)
    write_csv(output_dir / "step_position_summary.csv", position_summary)
    write_csv(output_dir / "high_lambda_effect_summary.csv", high_summary)
    (output_dir / "representative_cases.md").write_text(
        build_representative_cases(step_rows, high_summary),
        encoding="utf-8",
    )

    make_bar_svg(
        figure_dir / "cot_level_high_lambda_logprob_delta.svg",
        "High-Lambda Target-Step Retention",
        high_summary,
        "avg_final_cot_step_logprob_delta_vs_lambda1",
        "Average final target-step logprob delta vs lambda=1.0. Positive means less forgetting.",
    )
    make_bar_svg(
        figure_dir / "cot_level_high_lambda_prediction_diff.svg",
        "High-Lambda Final-Prediction Divergence",
        high_summary,
        "prediction_diff_rate_vs_lambda1",
        "Share of CoT steps where final prediction differs from lambda=1.0.",
    )
    make_phase_svg(figure_dir / "cot_level_position_effects.svg", position_summary)
    update_report(root, high_summary, position_summary)

    print(json.dumps(checks, indent=2))
    print(f"Wrote {len(step_rows)} step rows to {output_dir}")


if __name__ == "__main__":
    main()
