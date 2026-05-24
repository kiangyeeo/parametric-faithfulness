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

本实验将目标函数从 `L(theta) = L_NPO,beta(theta) + K_RT(theta)` 扩展为 `L(theta) = L_NPO,beta(theta) + lambda K_RT(theta)`，在固定 2x2 设置上扫描 `lambda in {{0.0, 0.1, 0.3, 1.0, 3.0, 10.0}}`。实验覆盖 Phi-3 与 LLaMA-3-3B，在 OpenBookQA 与 StrategyQA 上共 24 个正式组合；全部结果已写入 `lambda/results/lambda=<value>/...`，没有覆盖 `final_results`。

结论是：lambda 确实控制 stability-plasticity trade-off，但曲线并不严格单调。较低 lambda 通常更容易提高 answer-level faithfulness 或 answer-change 行为，较高 lambda 通常更保护 specificity；不过最优点明显依赖模型和数据集。以预先设定的“specificity 不比 lambda=1.0 低超过 3 个百分点”为约束，推荐值为：LLaMA/openbook 取 `0.3`，LLaMA/sqa 取 `3.0`，Phi/openbook 取 `0.0`，Phi/sqa 取 `10.0`。

## 2. Experiment Purpose And Hypotheses

实验目的：检验 retain KL regularization 权重 `lambda` 如何影响 unlearning efficacy、answer-level faithfulness、specificity 与 answer change rate，并判断当前隐式 `lambda=1.0` baseline 是否处在较优 trade-off 区间。

实验假设：

- H1: 较低 lambda 减弱 retain regularization，会提高局部可塑性，可能增强 faithfulness/answer change，但带来 specificity 风险。
- H2: 较高 lambda 增强 retained behavior 稳定性，可能提高 specificity，但压制 unlearning 对 final answer 的影响。
- H3: 当前 implicit baseline `lambda=1.0` 不一定是最优值。
- H4: Phi-3 与 LLaMA-3-3B 对 lambda 的敏感性不同。

结果总体支持这些假设，尤其 H3/H4：四个 model/dataset 组合的推荐 lambda 并不一致。

## 3. Objects, Scope, And Workflow

实验对象与范围：

- Models: Phi-3, LLaMA-3-3B
- Datasets: OpenBookQA (`openbook`), StrategyQA (`sqa`)
- Method: `npo_KL`, sentence-level stepwise unlearning
- Optimization: POS filter enabled, FF2-only optimization enabled
- Scale: `N_UNLEARN=40`, `N_VERIFY=10`, seed `1001`
- Lambda grid: `0.0, 0.1, 0.3, 1.0, 3.0, 10.0`

流程：先暴露 `--rt_lambda` 并确保结果路径包含 lambda；再 smoke test 验证 schema、路径和显存；随后在 2xA100 上完成完整 sweep；最后生成 JSON/CSV/PNG/SVG 和本综合报告。

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

这里把 specificity 和另一个指标同时视为要最大化的目标。如果某个 lambda 在两个目标上都不优于另一个 lambda，则它被视为 dominated。图中的黑色圆环和连线表示 nondominated lambda。

{markdown_table(['Model', 'Dataset', 'Recommended lambda', 'Faithfulness/Specificity', 'Faithfulness frontier', 'Efficacy frontier', 'Answer-change frontier'], recommendation_rows)}

### 6.1 Faithfulness-Specificity Frontier

- LLaMA/openbook: `lambda=0.3` 位于较好的 faithfulness-specificity 折中点，faithfulness 达到 65.00，specificity 只比 baseline 低 0.71。`lambda=10.0` specificity 最高但 faithfulness 下降到 60.00。
- LLaMA/sqa: `lambda=3.0` 在保持 faithfulness 67.50 的同时获得最高 specificity 92.42，因此比 `lambda=1.0` 更优。
- Phi/openbook: `lambda=0.0` faithfulness 最高 40.00，但 specificity 低于 baseline；`lambda=3.0` 是保守折中，faithfulness 37.50 且 specificity 94.94。
- Phi/sqa: faithfulness 对 lambda 不敏感，所有 lambda 都是 25.00；此时前沿主要由 specificity 决定，`lambda=10.0` 最强。

### 6.2 Efficacy-Specificity Frontier

Efficacy 在大部分设置中变化很小，说明 target-step probability reduction 对 lambda 不如 answer-level 指标敏感。Phi/sqa 是例外之一，高 lambda 提高 specificity 的同时 efficacy 从 81.65 降到 80.39，体现了过强 retain penalty 对 forgetting pressure 的抑制。

### 6.3 Answer-Change-Specificity Frontier

Answer-change rate 更接近最终 answer behavior。LLaMA/sqa 的 `lambda=0.1` 带来最高 answer-change rate 32.30，但 `lambda=3.0` specificity 更好且 answer-change rate 较低。Phi/openbook 的低 lambda 提高 answer-change，但同时略降 specificity。这个前沿说明 answer-change 不能单独作为最优目标，需要和 specificity 联合解释。

## 7. Interpretation By Combination

### LLaMA-3-3B / OpenBookQA

`lambda=0.1` 和 `lambda=0.3` 将 faithfulness 从 62.50 提高到 65.00，但 `lambda=0.3` 的 specificity 更高，因此推荐 `0.3`。`lambda=3.0` 和 `10.0` 更稳定，但 faithfulness 没有提升，`10.0` 还出现明显下降。

### LLaMA-3-3B / StrategyQA

`lambda=0.0` 到 `3.0` 的 faithfulness 都保持 67.50；`lambda=3.0` specificity 最高，所以它在 faithfulness-specificity 前沿上最干净。`lambda=10.0` faithfulness 降到 62.50，不建议作为默认设置。

### Phi-3 / OpenBookQA

这是低 lambda 最有用的组合。`lambda=0.0` 把 faithfulness 从 35.00 提高到 40.00，specificity 从 94.57 降到 93.86，仍在预设 3 个百分点容忍范围内。如果追求 answer-level faithfulness，可以选 `0.0`；如果偏保守，可以选 `3.0`。

### Phi-3 / StrategyQA

lambda 基本不能提高 faithfulness，所有设置都是 25.00。提高 lambda 主要提升 specificity，`lambda=10.0` 达到 97.96，但 efficacy 下降到 80.39。因此这里 lambda 更像 retention knob，而不是 faithfulness 改善手段。

## 8. Main Conclusions

1. `lambda=1.0` 是合理 baseline，但不是 Pareto 意义下的普遍最优点；在多个组合中它被其他 lambda 支配。
2. 低 lambda 的收益主要体现在更强 answer-level 可塑性，尤其 Phi/openbook。
3. 高 lambda 的收益主要体现在 specificity，但过高会降低 faithfulness，尤其 LLaMA/sqa 和 LLaMA/openbook 的 `lambda=10.0`。
4. Efficacy 本身不足以解释 trade-off，因为它在多数设置中变化很小。
5. 后续如果扩大实验，不建议继续盲目加大 lambda；更值得围绕 `0.0, 0.3, 1.0, 3.0` 做重复实验、置信区间和更大样本。

## 9. Limitations

- 本实验仍是小规模 2x2 reproduction，结论应视为趋势性证据。
- 没有同时 sweep beta 和 learning rate，因此无法分离 lambda 与优化强度的交互。
- Pareto 分析只基于汇总指标，没有替代人工 case-level error analysis。
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
