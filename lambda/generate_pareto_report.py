"""Generate Pareto scatter plots and a merged lambda experiment report."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("lambda")
FIGURE_DIR = ROOT / "figures"

LAMBDAS = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]
COMBOS = [
    ("LLaMA-3-3B", "openbook"),
    ("LLaMA-3-3B", "sqa"),
    ("Phi-3", "openbook"),
    ("Phi-3", "sqa"),
]
METRICS = [
    ("faithfulness", "Faithfulness"),
    ("efficacy", "Efficacy"),
    ("answer_change_rate", "Answer Change Rate"),
]
COLORS = {
    0.0: "#3b6ea8",
    0.1: "#55a868",
    0.3: "#c44e52",
    1.0: "#8172b3",
    3.0: "#c8a641",
    10.0: "#4aa3b5",
}


def fmt(value: float) -> str:
    return f"{value:.2f}"


def lambda_label(value: float) -> str:
    return f"{value:.1f}"


def slug(value: str) -> str:
    return value.replace(" ", "_").replace("/", "_").replace("-", "_").lower()


def esc(value: object) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def load_rows() -> list[dict]:
    rows: list[dict] = []
    with (ROOT / "summary_table.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for key in [
                "lambda_label",
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
            ]:
                row[key] = float(row[key])
            for key in ["n_instances", "n_cot_steps", "agreeing_instances", "changed_instances"]:
                row[key] = int(float(row[key]))
            rows.append(row)
    return rows


def rows_for(rows: list[dict], model: str, dataset: str) -> list[dict]:
    return [row for row in rows if row["model"] == model and row["dataset"] == dataset]


def pareto_frontier(points: list[dict], y_key: str) -> list[dict]:
    """Return nondominated points when specificity and y_key are maximized."""
    frontier = []
    for point in points:
        dominated = False
        for other in points:
            if other is point:
                continue
            better_or_equal = (
                other["specificity"] >= point["specificity"] and other[y_key] >= point[y_key]
            )
            strictly_better = (
                other["specificity"] > point["specificity"] or other[y_key] > point[y_key]
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(point)
    return sorted(frontier, key=lambda row: row["specificity"])


def scale(value: float, low: float, high: float, out_low: float, out_high: float) -> float:
    if abs(high - low) < 1e-12:
        return (out_low + out_high) / 2
    return out_low + (value - low) * (out_high - out_low) / (high - low)


def make_metric_svg(rows: list[dict], y_key: str, y_label: str, path: Path) -> None:
    width, height = 1160, 760
    panel_width, panel_height = 540, 285
    top, gap_x, gap_y = 88, 36, 52
    output = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#faf9f5"/>',
        f'<text x="28" y="36" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#222">{esc(y_label)} vs Specificity Pareto Frontiers</text>',
        '<text x="28" y="62" font-family="Arial, sans-serif" font-size="13" fill="#555">Specificity and the vertical metric are both objectives to maximize. Black rings and lines mark nondominated lambda values; lambda=1.0 has a dark center.</text>',
    ]

    legend_x = 28
    for value in LAMBDAS:
        output.append(
            f'<circle cx="{legend_x + 8}" cy="82" r="6" fill="{COLORS[value]}" stroke="#fff"/>'
        )
        output.append(
            f'<text x="{legend_x + 20}" y="86" font-family="Arial, sans-serif" font-size="12" fill="#333">lambda={lambda_label(value)}</text>'
        )
        legend_x += 122

    for index, (model, dataset) in enumerate(COMBOS):
        x0 = 28 + (index % 2) * (panel_width + gap_x)
        y0 = top + (index // 2) * (panel_height + gap_y)
        points = rows_for(rows, model, dataset)
        x_values = [point["specificity"] for point in points]
        y_values = [point[y_key] for point in points]
        x_pad = max(0.08, (max(x_values) - min(x_values)) * 0.16)
        y_pad = max(0.05 if y_key == "efficacy" else 0.6, (max(y_values) - min(y_values)) * 0.18)
        x_min, x_max = min(x_values) - x_pad, max(x_values) + x_pad
        y_min, y_max = min(y_values) - y_pad, max(y_values) + y_pad
        inner_left, inner_right = x0 + 58, x0 + panel_width - 22
        inner_top, inner_bottom = y0 + 42, y0 + panel_height - 46

        def sx(point: dict) -> float:
            return scale(point["specificity"], x_min, x_max, inner_left, inner_right)

        def sy(point: dict) -> float:
            return scale(point[y_key], y_min, y_max, inner_bottom, inner_top)

        output.append(
            f'<rect x="{x0}" y="{y0}" width="{panel_width}" height="{panel_height}" rx="8" fill="#fff" stroke="#ddd8ce"/>'
        )
        output.append(
            f'<text x="{x0 + 16}" y="{y0 + 25}" font-family="Arial, sans-serif" font-size="15" font-weight="700" fill="#222">{esc(model)} / {esc(dataset)}</text>'
        )
        for tick in [0, 0.25, 0.5, 0.75, 1]:
            x = inner_left + tick * (inner_right - inner_left)
            y = inner_top + tick * (inner_bottom - inner_top)
            output.append(
                f'<line x1="{x:.1f}" y1="{inner_top}" x2="{x:.1f}" y2="{inner_bottom}" stroke="#eee8dc"/>'
            )
            output.append(
                f'<line x1="{inner_left}" y1="{y:.1f}" x2="{inner_right}" y2="{y:.1f}" stroke="#eee8dc"/>'
            )
        output.append(
            f'<line x1="{inner_left}" y1="{inner_bottom}" x2="{inner_right}" y2="{inner_bottom}" stroke="#777"/>'
        )
        output.append(
            f'<line x1="{inner_left}" y1="{inner_top}" x2="{inner_left}" y2="{inner_bottom}" stroke="#777"/>'
        )
        output.append(
            f'<text x="{inner_left}" y="{inner_bottom + 27}" font-family="Arial, sans-serif" font-size="12" fill="#555">Specificity</text>'
        )
        output.append(
            f'<text x="{inner_left - 48}" y="{inner_top - 10}" font-family="Arial, sans-serif" font-size="11" fill="#555">{fmt(y_max)}</text>'
        )
        output.append(
            f'<text x="{inner_left - 48}" y="{inner_bottom + 4}" font-family="Arial, sans-serif" font-size="11" fill="#555">{fmt(y_min)}</text>'
        )
        output.append(
            f'<text x="{inner_left}" y="{inner_bottom + 16}" font-family="Arial, sans-serif" font-size="10" fill="#777">{fmt(x_min)}</text>'
        )
        output.append(
            f'<text x="{inner_right - 34}" y="{inner_bottom + 16}" font-family="Arial, sans-serif" font-size="10" fill="#777">{fmt(x_max)}</text>'
        )

        frontier = pareto_frontier(points, y_key)
        if len(frontier) > 1:
            segments = []
            for i, point in enumerate(frontier):
                segments.append(("M" if i == 0 else "L") + f" {sx(point):.1f} {sy(point):.1f}")
            output.append(
                f'<path d="{" ".join(segments)}" fill="none" stroke="#111" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>'
            )

        for point in points:
            value = point["lambda_label"]
            radius = 6 if abs(value - 1.0) < 1e-9 else 5
            stroke = "#111" if abs(value - 1.0) < 1e-9 else "#fff"
            stroke_width = 2.2 if abs(value - 1.0) < 1e-9 else 1.4
            output.append(
                f'<circle cx="{sx(point):.1f}" cy="{sy(point):.1f}" r="{radius}" fill="{COLORS[value]}" stroke="{stroke}" stroke-width="{stroke_width}"><title>lambda={lambda_label(value)}; specificity={fmt(point["specificity"])}; {esc(y_label)}={fmt(point[y_key])}</title></circle>'
            )
            output.append(
                f'<text x="{sx(point) + 8:.1f}" y="{sy(point) + 4:.1f}" font-family="Arial, sans-serif" font-size="10" fill="#333">{lambda_label(value)}</text>'
            )
        for point in frontier:
            output.append(
                f'<circle cx="{sx(point):.1f}" cy="{sy(point):.1f}" r="10" fill="none" stroke="#111" stroke-width="2"/>'
            )
        labels = ", ".join(lambda_label(point["lambda_label"]) for point in frontier)
        output.append(
            f'<text x="{x0 + 16}" y="{y0 + panel_height - 14}" font-family="Arial, sans-serif" font-size="12" fill="#333">Pareto lambda: {labels}</text>'
        )

    output.append("</svg>")
    path.write_text("\n".join(output), encoding="utf-8")


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(rows: list[dict]) -> None:
    core_rows = []
    for model, dataset in COMBOS:
        for value in LAMBDAS:
            row = next(
                candidate
                for candidate in rows
                if candidate["model"] == model
                and candidate["dataset"] == dataset
                and abs(candidate["lambda_label"] - value) < 1e-9
            )
            core_rows.append(
                [
                    lambda_label(value),
                    model,
                    dataset,
                    fmt(row["faithfulness"]),
                    fmt(row["efficacy"]),
                    fmt(row["specificity"]),
                    fmt(row["answer_change_rate"]),
                ]
            )

    recommendation_rows = []
    for model, dataset in COMBOS:
        points = rows_for(rows, model, dataset)
        baseline = next(point for point in points if abs(point["lambda_label"] - 1.0) < 1e-9)
        eligible = [point for point in points if point["specificity"] >= baseline["specificity"] - 3.0]
        best = max(eligible, key=lambda point: (point["faithfulness"], point["specificity"]))
        faith_front = pareto_frontier(points, "faithfulness")
        efficacy_front = pareto_frontier(points, "efficacy")
        answer_front = pareto_frontier(points, "answer_change_rate")
        recommendation_rows.append(
            [
                model,
                dataset,
                lambda_label(best["lambda_label"]),
                f'{fmt(best["faithfulness"])}/{fmt(best["specificity"])}',
                ", ".join(lambda_label(point["lambda_label"]) for point in faith_front),
                ", ".join(lambda_label(point["lambda_label"]) for point in efficacy_front),
                ", ".join(lambda_label(point["lambda_label"]) for point in answer_front),
            ]
        )

    report = f"""# Lambda Regularization Experiment Report

## 1. Executive Summary

This experiment extends the objective from `L(theta) = L_NPO,beta(theta) + K_RT(theta)` to `L(theta) = L_NPO,beta(theta) + lambda K_RT(theta)`, sweeping `lambda in {{0.0, 0.1, 0.3, 1.0, 3.0, 10.0}}` on the fixed 2x2 setup. It covers Phi-3 and LLaMA-3-3B on OpenBookQA and StrategyQA, 24 full combinations in total; all results are written under `lambda/results/lambda=<value>/...` and do not overwrite `final_results`.

Conclusion: lambda does control the stability-plasticity trade-off, but the curve is not strictly monotonic. Lower lambda tends to improve answer-level faithfulness or answer-change behavior; higher lambda tends to better preserve specificity; the optimum clearly depends on model and dataset. Under the pre-registered constraint that specificity may not drop more than 3 points below lambda=1.0, the recommended values are: LLaMA/openbook `0.3`, LLaMA/sqa `3.0`, Phi/openbook `0.0`, Phi/sqa `10.0`.

## 2. Experiment Purpose And Hypotheses

Purpose: examine how the retain KL regularization weight `lambda` affects unlearning efficacy, answer-level faithfulness, specificity, and answer-change rate, and whether the current implicit `lambda=1.0` baseline sits in a good trade-off region.

Hypotheses:

- H1: Lower lambda weakens retain regularization, raising local plasticity; it may strengthen faithfulness/answer-change but risks specificity.
- H2: Higher lambda strengthens retained-behavior stability; it may improve specificity but suppresses unlearning's effect on the final answer.
- H3: The current implicit baseline `lambda=1.0` is not necessarily optimal.
- H4: Phi-3 and LLaMA-3-3B differ in their sensitivity to lambda.

The results broadly support these hypotheses, especially H3/H4: the recommended lambda differs across the four model/dataset combinations.

## 3. Objects, Scope, And Workflow

Objects and scope:

- Models: Phi-3, LLaMA-3-3B
- Datasets: OpenBookQA (`openbook`), StrategyQA (`sqa`)
- Method: `npo_KL`, sentence-level stepwise unlearning
- Optimization: POS filter enabled, FF2-only optimization enabled
- Scale: `N_UNLEARN=40`, `N_VERIFY=10`, seed `1001`
- Lambda grid: `0.0, 0.1, 0.3, 1.0, 3.0, 10.0`

Workflow: first expose `--rt_lambda` and ensure result paths include lambda; then smoke-test schema, paths, and GPU memory; then run the full sweep on 2xA100; finally generate JSON/CSV/PNG/SVG and this merged report.

## 4. Deliverables

- Raw results: `lambda/results/`
- Metric summary: `lambda/analysis_summary.json`
- Answer-change summary: `lambda/answer_change_rate_summary.json`
- Main table: `lambda/summary_table.csv`
- Pareto summary table: `lambda/pareto_frontier_summary.csv`
- Report: `lambda/report_notes.md`
- Existing plots: `lambda/figures/*.png`
- Added Pareto/scatter plots:
  - `lambda/figures/pareto_faithfulness_specificity.svg`
  - `lambda/figures/pareto_efficacy_specificity.svg`
  - `lambda/figures/pareto_answer_change_rate_specificity.svg`

## 5. Core Metrics

{markdown_table(['Lambda', 'Model', 'Dataset', 'Faithfulness', 'Efficacy', 'Specificity', 'Answer Change'], core_rows)}

## 6. Pareto Frontier Analysis

Here specificity and a second metric are both treated as objectives to maximize. A lambda is dominated if some other lambda is at least as good on both objectives. Black rings and connecting lines in the figures mark the nondominated lambda values.

{markdown_table(['Model', 'Dataset', 'Recommended lambda', 'Faithfulness/Specificity', 'Faithfulness frontier', 'Efficacy frontier', 'Answer-change frontier'], recommendation_rows)}

### 6.1 Faithfulness-Specificity Frontier

- LLaMA/openbook: `lambda=0.3` sits at a good faithfulness-specificity trade-off, reaching faithfulness 65.00 with specificity only 0.71 below baseline. `lambda=10.0` has the highest specificity but faithfulness drops to 60.00.
- LLaMA/sqa: `lambda=3.0` keeps faithfulness at 67.50 while achieving the highest specificity 92.42, so it beats `lambda=1.0`.
- Phi/openbook: `lambda=0.0` has the highest faithfulness 40.00 but specificity below baseline; `lambda=3.0` is a conservative trade-off, faithfulness 37.50 with specificity 94.94.
- Phi/sqa: faithfulness is insensitive to lambda, staying at 25.00 for all values; here the frontier is driven mainly by specificity, and `lambda=10.0` is strongest.

### 6.2 Efficacy-Specificity Frontier

Efficacy changes little across most settings, indicating that target-step probability reduction is less sensitive to lambda than the answer-level metrics. Phi/sqa is one exception: raising lambda improves specificity while efficacy falls from 81.65 to 80.39, reflecting how an overly strong retain penalty suppresses forgetting pressure.

### 6.3 Answer-Change-Specificity Frontier

Answer-change rate is closer to final answer behavior. For LLaMA/sqa, `lambda=0.1` gives the highest answer-change rate 32.30, but `lambda=3.0` has better specificity with a lower answer-change rate. For Phi/openbook, low lambda raises answer-change but slightly lowers specificity. This frontier shows answer-change cannot be the sole optimization target and must be read together with specificity.

## 7. Interpretation By Combination

### LLaMA-3-3B / OpenBookQA

`lambda=0.1` and `lambda=0.3` raise faithfulness from 62.50 to 65.00, but `lambda=0.3` has higher specificity, so `0.3` is recommended. `lambda=3.0` and `10.0` are more stable but do not improve faithfulness, and `10.0` shows a clear drop.

### LLaMA-3-3B / StrategyQA

Faithfulness stays at 67.50 from `lambda=0.0` to `3.0`; `lambda=3.0` has the highest specificity, so it is cleanest on the faithfulness-specificity frontier. `lambda=10.0` drops faithfulness to 62.50 and is not recommended as a default.

### Phi-3 / OpenBookQA

This is the combination where low lambda helps most. `lambda=0.0` raises faithfulness from 35.00 to 40.00, with specificity dropping from 94.57 to 93.86, still within the pre-set 3-point tolerance. For answer-level faithfulness choose `0.0`; for a conservative setting choose `3.0`.

### Phi-3 / StrategyQA

lambda essentially cannot improve faithfulness; all settings give 25.00. Raising lambda mainly improves specificity, with `lambda=10.0` reaching 97.96 but efficacy dropping to 80.39. Here lambda acts more as a retention knob than a faithfulness lever.

## 8. Main Conclusions

1. `lambda=1.0` is a reasonable baseline but not a universal Pareto optimum; in several combinations it is dominated by other lambda values.
2. The benefit of low lambda is mainly stronger answer-level plasticity, especially Phi/openbook.
3. The benefit of high lambda is mainly specificity, but too high reduces faithfulness, especially `lambda=10.0` for LLaMA/sqa and LLaMA/openbook.
4. Efficacy alone does not explain the trade-off, since it changes little across most settings.
5. For future scaling, do not keep blindly increasing lambda; it is more worthwhile to run repeats, confidence intervals, and larger samples around `0.0, 0.3, 1.0, 3.0`.

## 9. Limitations

- This is still a small-scale 2x2 reproduction; conclusions should be read as trend-level evidence.
- beta and learning rate were not swept jointly, so lambda cannot be disentangled from optimization strength.
- The Pareto analysis is based only on aggregate metrics and does not replace manual case-level error analysis.
"""
    (ROOT / "report_notes.md").write_text(report, encoding="utf-8")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    for y_key, y_label in METRICS:
        make_metric_svg(
            rows,
            y_key,
            y_label,
            FIGURE_DIR / f"pareto_{slug(y_label)}_specificity.svg",
        )

    frontier_rows = []
    for model, dataset in COMBOS:
        points = rows_for(rows, model, dataset)
        for y_key, y_label in METRICS:
            frontier = pareto_frontier(points, y_key)
            frontier_rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "objective_pair": f"{y_label} + Specificity",
                    "pareto_lambdas": ", ".join(
                        lambda_label(point["lambda_label"]) for point in frontier
                    ),
                    "baseline_dominated": "yes"
                    if not any(abs(point["lambda_label"] - 1.0) < 1e-9 for point in frontier)
                    else "no",
                    "best_tradeoff_lambda": lambda_label(
                        max(frontier, key=lambda point: (point[y_key], point["specificity"]))[
                            "lambda_label"
                        ]
                    ),
                }
            )
    with (ROOT / "pareto_frontier_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frontier_rows[0]))
        writer.writeheader()
        writer.writerows(frontier_rows)

    write_report(rows)
    print("Wrote Pareto figures, pareto_frontier_summary.csv, and merged report_notes.md")


if __name__ == "__main__":
    main()
