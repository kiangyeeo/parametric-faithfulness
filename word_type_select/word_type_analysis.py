"""
Comprehensive statistical analysis of word-type unlearning results.

Analyses:
1. Word type forgetting effect differences (which word types flip predictions more easily?)
2. Efficacy vs Specificity trade-off
3. Random control vs word type groups (ruling out token-count-driven effects)
4. Skip rate analysis by word type
5. Cross-type instance analysis (which word type has the largest impact on CoT faithfulness)
6. New CoT qualitative analysis (how CoTs change after unlearning different word types)

Uses statistical methods: correlation analysis, rank tests, effect size, bootstrap CI.
"""

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path("/inspire/hdd/project/fdu-aidake-cfff/public/wanyizhou/measuring cot/parametric-faithfulness")
CSV_PATH = BASE / "table_all_word_types.csv"
RESULTS_DIR = BASE / "final_results"
OUT_DIR = BASE / "analysis_figures"
OUT_DIR.mkdir(exist_ok=True)

# ── Load CSV summary ──────────────────────────────────────────────────────
with open(CSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    csv_rows = list(reader)

# Convert numeric fields
for r in csv_rows:
    for k in ["ff_hard(%)", "ff_soft", "specificity(%)", "efficacy(%)"]:
        r[k] = float(r[k])

# ── Load per-instance data from .out files ────────────────────────────────
def load_out_file(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def parse_word_type(filename):
    """Extract word type from filename like npo_KL_..._wt=action_..."""
    for token in filename.replace(".out", "").split("_"):
        if token.startswith("wt="):
            return token.split("=", 1)[1]
    # sentencize_s=True means all_content
    if "sentencize_s=True" in filename:
        return "all_content"
    return None

# Gather all .out files (excluding .ipynb_checkpoints)
all_data = {}  # (dataset, word_type) -> list of instance rows
skip_counts = {}  # (dataset, word_type) -> number of skipped instances
total_instances = {}  # (dataset,) -> total instances (from all_content)

for dataset_dir in RESULTS_DIR.iterdir():
    if not dataset_dir.is_dir():
        continue
    dataset = dataset_dir.name
    for model_dir in dataset_dir.iterdir():
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue
        model = model_dir.name
        if model != "LLaMA-3-3B":
            continue
        for fpath in model_dir.iterdir():
            if not fpath.name.endswith(".out") or ".ipynb_checkpoints" in str(fpath):
                continue
            wt = parse_word_type(fpath.name)
            if wt is None:
                continue
            key = (dataset, wt)
            all_data[key] = load_out_file(fpath)
            # Count skipped
            skip_path = fpath.with_name(fpath.name.replace(".out", "_skipped.jsonl"))
            n_skipped = 0
            if skip_path.exists():
                with open(skip_path, "r") as sf:
                    for line in sf:
                        if line.strip():
                            n_skipped += 1
            skip_counts[key] = n_skipped

# Total instances per dataset (from all_content)
for (dataset, wt), rows in all_data.items():
    if wt == "all_content":
        total_instances[dataset] = len({r.get("id", i) for i, r in enumerate(rows)})

# ── Color scheme ──────────────────────────────────────────────────────────
WORD_TYPES_ORDERED = ["action", "attribute", "entity", "function", "modifier", "random", "all_content"]
COLORS = {
    "action": "#e74c3c",
    "attribute": "#3498db",
    "entity": "#2ecc71",
    "function": "#9b59b6",
    "modifier": "#f39c12",
    "random": "#95a5a6",
    "all_content": "#1abc9c",
}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})

# ══════════════════════════════════════════════════════════════════════════
# ANALYSIS 1: Word type forgetting effect differences
# ══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("ANALYSIS 1: Word Type Forgetting Effect Differences")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax_idx, dataset in enumerate(["openbook", "sqa"]):
    ax = axes[ax_idx]
    wt_order = [wt for wt in WORD_TYPES_ORDERED if wt != "all_content"]
    ff_hard_vals = []
    ff_soft_vals = []
    labels = []
    for wt in wt_order:
        key = (dataset, wt)
        matching = [r for r in csv_rows if r["dataset"] == dataset and r["word_type"] == wt]
        if matching:
            ff_hard_vals.append(matching[0]["ff_hard(%)"])
            ff_soft_vals.append(matching[0]["ff_soft"])
            labels.append(wt)
        else:
            ff_hard_vals.append(0)
            ff_soft_vals.append(0)
            labels.append(wt)

    x = np.arange(len(labels))
    width = 0.35
    bars1 = ax.bar(x - width / 2, ff_hard_vals, width, label="FF-Hard (%)",
                   color=[COLORS[wt] for wt in labels], alpha=0.8, edgecolor="white")
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width / 2, ff_soft_vals, width, label="FF-Soft",
                    color=[COLORS[wt] for wt in labels], alpha=0.5, edgecolor="black",
                    hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("FF-Hard (%)", fontweight="bold")
    ax2.set_ylabel("FF-Soft", fontweight="bold")
    ax.set_title(f"{dataset.upper()} - FF-Hard & FF-Soft by Word Type")

    # Add value labels
    for bar, val in zip(bars1, ff_hard_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, ff_soft_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=7)

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

plt.tight_layout()
plt.savefig(OUT_DIR / "fig1_ff_hard_soft_by_word_type.png", bbox_inches="tight")
plt.close()

# Statistical test: Kruskal-Wallis on per-instance ff_hard across word types
print("\nPer-instance FF-Hard comparison (Kruskal-Wallis H test):")
for dataset in ["openbook", "sqa"]:
    groups = {}
    for wt in WORD_TYPES_ORDERED:
        if wt == "all_content":
            continue
        key = (dataset, wt)
        if key not in all_data:
            continue
        instance_flipped = {}
        for row in all_data[key]:
            iid = row.get("id", str(row.get("question", "")))
            pred0 = int(row.get("prediction", -1))
            cot_pred0 = int(row.get("cot_prediction", -1))
            if pred0 != cot_pred0:
                continue
            ur = row.get("unlearning_results", {})
            keys_sorted = sorted(ur.keys(), key=lambda k: int(k) if k.isdigit() else k)
            flipped = False
            for k in keys_sorted[1:]:
                item = ur[k]
                pred = item.get("prediction")
                if pred is not None and int(pred) != pred0:
                    flipped = True
                    break
            instance_flipped[iid] = instance_flipped.get(iid, False) or flipped
        groups[wt] = [1 if v else 0 for v in instance_flipped.values()]

    group_vals = [groups[wt] for wt in groups if len(groups[wt]) > 0]
    group_labels = [wt for wt in groups if len(groups[wt]) > 0]
    if len(group_vals) >= 2:
        H, p = stats.kruskal(*group_vals)
        print(f"  {dataset}: H={H:.3f}, p={p:.4f}")
        # Pairwise Mann-Whitney U tests with Bonferroni correction
        print(f"  Pairwise comparisons (Mann-Whitney U, Bonferroni-corrected):")
        n_pairs = len(group_labels) * (len(group_labels) - 1) // 2
        for i in range(len(group_labels)):
            for j in range(i + 1, len(group_labels)):
                u, p_raw = stats.mannwhitneyu(group_vals[i], group_vals[j], alternative="two-sided")
                p_adj = min(p_raw * n_pairs, 1.0)
                sig = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "n.s."
                print(f"    {group_labels[i]} vs {group_labels[j]}: U={u:.1f}, p_adj={p_adj:.4f} {sig}")

# Rank ordering
print("\nFF-Hard ranking (easiest to hardest to flip):")
for dataset in ["openbook", "sqa"]:
    wt_ff = []
    for wt in WORD_TYPES_ORDERED:
        if wt == "all_content":
            continue
        matching = [r for r in csv_rows if r["dataset"] == dataset and r["word_type"] == wt]
        if matching:
            wt_ff.append((wt, matching[0]["ff_hard(%)"]))
    wt_ff.sort(key=lambda x: -x[1])
    rank_str = " > ".join([f"{wt}({v:.1f}%)" for wt, v in wt_ff])
    print(f"  {dataset}: {rank_str}")

# ══════════════════════════════════════════════════════════════════════════
# ANALYSIS 2: Efficacy vs Specificity trade-off
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ANALYSIS 2: Efficacy vs Specificity Trade-off")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax_idx, dataset in enumerate(["openbook", "sqa"]):
    ax = axes[ax_idx]
    eff_vals = []
    spec_vals = []
    labels = []
    for wt in WORD_TYPES_ORDERED:
        matching = [r for r in csv_rows if r["dataset"] == dataset and r["word_type"] == wt]
        if matching:
            eff_vals.append(matching[0]["efficacy(%)"])
            spec_vals.append(matching[0]["specificity(%)"])
            labels.append(wt)

    for i, (e, s, wt) in enumerate(zip(eff_vals, spec_vals, labels)):
        ax.scatter(e, s, s=150, c=COLORS[wt], label=wt, edgecolors="black", linewidth=0.5, zorder=5)
        offset_x = 1.0 if wt != "function" else -3.0
        offset_y = 0.3
        ax.annotate(wt, (e, s), textcoords="offset points", xytext=(offset_x, offset_y),
                    fontsize=9, fontweight="bold")

    # Pearson & Spearman correlation
    if len(eff_vals) >= 3:
        pearson_r, pearson_p = stats.pearsonr(eff_vals, spec_vals)
        spearman_r, spearman_p = stats.spearmanr(eff_vals, spec_vals)
        ax.set_title(f"{dataset.upper()}\nPearson r={pearson_r:.3f} (p={pearson_p:.3f}), "
                     f"Spearman ρ={spearman_r:.3f} (p={spearman_p:.3f})")

    ax.set_xlabel("Efficacy (%)")
    ax.set_ylabel("Specificity (%)")
    ax.grid(True, alpha=0.3)

    # Add ideal region annotation
    ax.axhline(y=95, color="green", linestyle="--", alpha=0.3, label="95% specificity")
    ax.axvline(x=70, color="red", linestyle="--", alpha=0.3, label="70% efficacy")

plt.tight_layout()
plt.savefig(OUT_DIR / "fig2_efficacy_vs_specificity.png", bbox_inches="tight")
plt.close()

# Compute trade-off score: harmonic mean of efficacy and specificity (both as fractions)
print("\nEfficacy-Specificity trade-off (harmonic mean):")
for dataset in ["openbook", "sqa"]:
    tradeoff = []
    for wt in WORD_TYPES_ORDERED:
        matching = [r for r in csv_rows if r["dataset"] == dataset and r["word_type"] == wt]
        if matching:
            e = matching[0]["efficacy(%)"] / 100.0
            s = matching[0]["specificity(%)"] / 100.0
            hm = 2 * e * s / (e + s) if (e + s) > 0 else 0
            tradeoff.append((wt, hm, matching[0]["efficacy(%)"], matching[0]["specificity(%)"]))
    tradeoff.sort(key=lambda x: -x[1])
    for wt, hm, eff, spec in tradeoff:
        print(f"  {dataset} {wt:12s}: HM={hm:.4f}  (eff={eff:.1f}%, spec={spec:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════
# ANALYSIS 3: Random control vs word type groups
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ANALYSIS 3: Random Control vs Word Type Groups")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
metrics = ["ff_hard(%)", "ff_soft", "specificity(%)", "efficacy(%)"]
metric_labels = ["FF-Hard (%)", "FF-Soft", "Specificity (%)", "Efficacy (%)"]

for m_idx, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
    ax = axes[m_idx // 2][m_idx % 2]
    for d_idx, dataset in enumerate(["openbook", "sqa"]):
        random_val = None
        wt_vals = {}
        for wt in WORD_TYPES_ORDERED:
            if wt in ("random", "all_content"):
                continue
            matching = [r for r in csv_rows if r["dataset"] == dataset and r["word_type"] == wt]
            if matching:
                wt_vals[wt] = matching[0][metric]
                if wt == "random":
                    random_val = matching[0][metric]
        # Get random
        matching_r = [r for r in csv_rows if r["dataset"] == dataset and r["word_type"] == "random"]
        if matching_r:
            random_val = matching_r[0][metric]

        # Bar chart: each word type vs random baseline
        wts = [wt for wt in WORD_TYPES_ORDERED if wt not in ("random", "all_content")]
        vals = [wt_vals.get(wt, 0) for wt in wts]
        x = np.arange(len(wts))
        width = 0.35
        offset = -width / 2 if d_idx == 0 else width / 2
        color = "#3498db" if d_idx == 0 else "#e74c3c"
        ax.bar(x + offset, vals, width, label=f"{dataset}", color=color, alpha=0.7)
        if random_val is not None:
            ax.axhline(y=random_val, color=color, linestyle="--", alpha=0.6,
                       label=f"{dataset} random={random_val:.2f}")

    ax.set_xticks(np.arange(len(wts)))
    ax.set_xticklabels(wts, rotation=30, ha="right")
    ax.set_ylabel(mlabel)
    ax.set_title(f"{mlabel}: Word Types vs Random Baseline")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")

plt.tight_layout()
plt.savefig(OUT_DIR / "fig3_random_vs_word_types.png", bbox_inches="tight")
plt.close()

# Statistical test: each word type vs random (per-instance)
print("\nPer-instance FF-Hard: each word type vs random (Fisher's exact test):")
for dataset in ["openbook", "sqa"]:
    # Get random group per-instance data
    random_key = (dataset, "random")
    if random_key not in all_data:
        continue
    random_flipped = {}
    for row in all_data[random_key]:
        iid = row.get("id", str(row.get("question", "")))
        pred0 = int(row.get("prediction", -1))
        cot_pred0 = int(row.get("cot_prediction", -1))
        if pred0 != cot_pred0:
            continue
        ur = row.get("unlearning_results", {})
        keys_sorted = sorted(ur.keys(), key=lambda k: int(k) if k.isdigit() else k)
        flipped = False
        for k in keys_sorted[1:]:
            item = ur[k]
            pred = item.get("prediction")
            if pred is not None and int(pred) != pred0:
                flipped = True
                break
        random_flipped[iid] = random_flipped.get(iid, False) or flipped

    for wt in WORD_TYPES_ORDERED:
        if wt in ("random", "all_content"):
            continue
        key = (dataset, wt)
        if key not in all_data:
            continue
        wt_flipped = {}
        for row in all_data[key]:
            iid = row.get("id", str(row.get("question", "")))
            pred0 = int(row.get("prediction", -1))
            cot_pred0 = int(row.get("cot_prediction", -1))
            if pred0 != cot_pred0:
                continue
            ur = row.get("unlearning_results", {})
            keys_sorted = sorted(ur.keys(), key=lambda k: int(k) if k.isdigit() else k)
            flipped = False
            for k in keys_sorted[1:]:
                item = ur[k]
                pred = item.get("prediction")
                if pred is not None and int(pred) != pred0:
                    flipped = True
                    break
            wt_flipped[iid] = wt_flipped.get(iid, False) or flipped

        # Build 2x2 contingency table for common instances
        common_ids = set(wt_flipped.keys()) & set(random_flipped.keys())
        if not common_ids:
            continue
        # Paired McNemar test
        b = sum(1 for iid in common_ids if wt_flipped[iid] and not random_flipped[iid])  # wt flipped, random not
        c = sum(1 for iid in common_ids if not wt_flipped[iid] and random_flipped[iid])  # random flipped, wt not
        a = sum(1 for iid in common_ids if wt_flipped[iid] and random_flipped[iid])
        d = sum(1 for iid in common_ids if not wt_flipped[iid] and not random_flipped[iid])

        # McNemar's test
        if b + c > 0:
            mcnemar_stat = (abs(b - c) - 1) ** 2 / (b + c) if b + c > 0 else 0
            mcnemar_p = 1 - stats.chi2.cdf(mcnemar_stat, 1)
        else:
            mcnemar_p = 1.0

        # Effect size: odds ratio
        if b > 0 and c > 0:
            odds_ratio = b / c
        elif b > 0:
            odds_ratio = float("inf")
        elif c > 0:
            odds_ratio = 0.0
        else:
            odds_ratio = 1.0

        wt_rate = sum(wt_flipped.values()) / len(wt_flipped) * 100 if wt_flipped else 0
        rand_rate = sum(random_flipped.values()) / len(random_flipped) * 100 if random_flipped else 0
        sig = "***" if mcnemar_p < 0.001 else "**" if mcnemar_p < 0.01 else "*" if mcnemar_p < 0.05 else "n.s."
        direction = "↑" if wt_rate > rand_rate else "↓"
        print(f"  {dataset} {wt:10s} vs random: wt={wt_rate:.1f}%, rand={rand_rate:.1f}% {direction} "
              f"McNemar p={mcnemar_p:.4f} {sig}, OR={odds_ratio:.2f}")

# ══════════════════════════════════════════════════════════════════════════
# ANALYSIS 4: Skip rate analysis
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ANALYSIS 4: Skip Rate Analysis by Word Type")
print("=" * 70)

fig, ax = plt.subplots(figsize=(10, 5))

skip_data = {}
for dataset in ["openbook", "sqa"]:
    total = total_instances.get(dataset, 230)
    for wt in WORD_TYPES_ORDERED:
        if wt == "all_content":
            continue
        key = (dataset, wt)
        n_out = len(all_data.get(key, []))
        n_skipped = skip_counts.get(key, 0)
        # Each instance can have multiple steps, so total steps = n_out + n_skipped
        # But we want instance-level skip rate
        # Actually, skipped means the step was skipped because it didn't contain that word type
        # n_out is the number of step-level results, n_skipped is the number of skipped steps
        total_steps = n_out + n_skipped
        skip_rate = n_skipped / total_steps * 100 if total_steps > 0 else 0
        skip_data[(dataset, wt)] = {
            "n_results": n_out,
            "n_skipped": n_skipped,
            "total_steps": total_steps,
            "skip_rate": skip_rate,
        }

# Plot
wts = [wt for wt in WORD_TYPES_ORDERED if wt not in ("all_content",)]
x = np.arange(len(wts))
width = 0.35
for d_idx, dataset in enumerate(["openbook", "sqa"]):
    rates = [skip_data.get((dataset, wt), {}).get("skip_rate", 0) for wt in wts]
    color = "#3498db" if d_idx == 0 else "#e74c3c"
    offset = -width / 2 if d_idx == 0 else width / 2
    bars = ax.bar(x + offset, rates, width, label=dataset, color=color, alpha=0.8)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(wts, rotation=30, ha="right")
ax.set_ylabel("Skip Rate (%)")
ax.set_title("Step Skip Rate by Word Type\n(Steps skipped because they don't contain the target word type)")
ax.legend()
ax.grid(True, alpha=0.2, axis="y")

plt.tight_layout()
plt.savefig(OUT_DIR / "fig4_skip_rate.png", bbox_inches="tight")
plt.close()

print("\nSkip rate details:")
for dataset in ["openbook", "sqa"]:
    print(f"\n  {dataset}:")
    for wt in wts:
        sd = skip_data.get((dataset, wt), {})
        print(f"    {wt:10s}: {sd.get('n_skipped', 0):3d} skipped / {sd.get('total_steps', 0):3d} total "
              f"= {sd.get('skip_rate', 0):.1f}% skip rate")

# Chi-square test for skip rate differences
print("\nChi-square test for skip rate differences across word types:")
for dataset in ["openbook", "sqa"]:
    observed = []
    wt_labels = []
    for wt in wts:
        sd = skip_data.get((dataset, wt), {})
        if sd.get("total_steps", 0) > 0:
            observed.append([sd.get("n_skipped", 0), sd.get("n_results", 0)])
            wt_labels.append(wt)
    if len(observed) >= 2:
        chi2, p, dof, expected = stats.chi2_contingency(observed)
        print(f"  {dataset}: χ²={chi2:.3f}, df={dof}, p={p:.6f}")

# ══════════════════════════════════════════════════════════════════════════
# ANALYSIS 5: Cross-type instance analysis
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ANALYSIS 5: Cross-type Instance Analysis")
print("=" * 70)

# For each instance that appears in multiple word types, compare which word type
# causes the most prediction flips
for dataset in ["openbook", "sqa"]:
    # Collect per-instance flip data across word types
    instance_flips = defaultdict(dict)  # iid -> {wt: flipped}
    instance_ff_soft = defaultdict(dict)  # iid -> {wt: ff_soft_value}

    for wt in WORD_TYPES_ORDERED:
        if wt == "all_content":
            continue
        key = (dataset, wt)
        if key not in all_data:
            continue
        for row in all_data[key]:
            iid = row.get("id", str(row.get("question", "")))
            pred0 = int(row.get("prediction", -1))
            cot_pred0 = int(row.get("cot_prediction", -1))
            if pred0 != cot_pred0:
                continue

            ur = row.get("unlearning_results", {})
            keys_sorted = sorted(ur.keys(), key=lambda k: int(k) if k.isdigit() else k)
            first = ur[keys_sorted[0]]
            probs0 = first.get("probs", [])
            if not probs0:
                probs0 = row.get("initial_probs", [])
            if not probs0:
                continue
            probs0 = [float(x) for x in probs0]
            if pred0 >= len(probs0):
                continue

            # Renormalize
            total = sum(probs0)
            base_probs = [p / total for p in probs0] if total > 0 else probs0
            base_mass = base_probs[pred0]

            flipped = False
            soft_deltas = []
            for k in keys_sorted[1:]:
                item = ur[k]
                pred = item.get("prediction")
                if pred is not None and int(pred) != pred0:
                    flipped = True
                probs = item.get("probs", [])
                if probs and pred0 < len(probs):
                    probs = [float(x) for x in probs]
                    t2 = sum(probs)
                    post_probs = [p / t2 for p in probs] if t2 > 0 else probs
                    soft_deltas.append(base_mass - post_probs[pred0])

            instance_flips[iid][wt] = flipped
            if soft_deltas:
                instance_ff_soft[iid][wt] = np.mean(soft_deltas)

    # Find instances that appear in at least 3 word types
    multi_type_instances = {iid: flips for iid, flips in instance_flips.items() if len(flips) >= 3}
    print(f"\n  {dataset}: {len(multi_type_instances)} instances appear in ≥3 word types")

    if not multi_type_instances:
        continue

    # For each instance, which word type causes the most flips?
    wt_flip_count = defaultdict(int)
    wt_soft_sum = defaultdict(float)
    wt_soft_count = defaultdict(int)
    for iid, flips in multi_type_instances.items():
        # Which types flipped this instance?
        for wt, flipped in flips.items():
            if flipped:
                wt_flip_count[wt] += 1
        # Which type had highest ff_soft?
        softs = instance_ff_soft.get(iid, {})
        if softs:
            max_wt = max(softs, key=softs.get)
            wt_soft_sum[max_wt] += softs[max_wt]
            wt_soft_count[max_wt] += 1

    print(f"  {dataset}: Which word type most frequently flips multi-type instances?")
    for wt in WORD_TYPES_ORDERED:
        if wt == "all_content":
            continue
        count = wt_flip_count.get(wt, 0)
        rate = count / len(multi_type_instances) * 100 if multi_type_instances else 0
        print(f"    {wt:10s}: flipped {count}/{len(multi_type_instances)} instances ({rate:.1f}%)")

    # Friedman test (non-parametric repeated measures) on ff_soft across word types
    # for instances that have data for all word types
    common_wts = [wt for wt in WORD_TYPES_ORDERED if wt != "all_content"]
    complete_instances = []
    for iid in multi_type_instances:
        softs = instance_ff_soft.get(iid, {})
        if all(wt in softs for wt in common_wts if (dataset, wt) in all_data):
            complete_instances.append(iid)

    if len(complete_instances) >= 5:
        available_wts = [wt for wt in common_wts if (dataset, wt) in all_data]
        matrix = np.array([[instance_ff_soft[iid][wt] for wt in available_wts]
                           for iid in complete_instances])
        try:
            friedman_stat, friedman_p = stats.friedmanchisquare(*[matrix[:, i] for i in range(matrix.shape[1])])
            print(f"\n  {dataset}: Friedman test on FF-Soft across word types (n={len(complete_instances)}):")
            print(f"    χ²={friedman_stat:.3f}, p={friedman_p:.4f}")

            # Nemenyi post-hoc: compare each pair
            if friedman_p < 0.05:
                print(f"    Significant! Pairwise Wilcoxon signed-rank tests (Bonferroni):")
                n_pairs = len(available_wts) * (len(available_wts) - 1) // 2
                for i in range(len(available_wts)):
                    for j in range(i + 1, len(available_wts)):
                        try:
                            w, p_raw = stats.wilcoxon(matrix[:, i], matrix[:, j])
                            p_adj = min(p_raw * n_pairs, 1.0)
                            sig = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "n.s."
                            mean_i = np.mean(matrix[:, i])
                            mean_j = np.mean(matrix[:, j])
                            print(f"      {available_wts[i]}({mean_i:.4f}) vs {available_wts[j]}({mean_j:.4f}): "
                                  f"W={w:.1f}, p_adj={p_adj:.4f} {sig}")
                        except Exception:
                            pass
        except Exception as e:
            print(f"  Friedman test failed: {e}")

    # Visualization: heatmap of flip patterns
    fig, ax = plt.subplots(figsize=(10, max(6, len(multi_type_instances) * 0.15)))
    available_wts = [wt for wt in WORD_TYPES_ORDERED if wt != "all_content" and (dataset, wt) in all_data]
    # Sort instances by number of flips
    sorted_iids = sorted(multi_type_instances.keys(),
                         key=lambda iid: -sum(1 for v in instance_flips[iid].values() if v))
    # Limit to top 50 for readability
    show_iids = sorted_iids[:50]

    matrix = np.zeros((len(show_iids), len(available_wts)))
    for i, iid in enumerate(show_iids):
        for j, wt in enumerate(available_wts):
            matrix[i, j] = 1 if instance_flips[iid].get(wt, False) else 0

    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(available_wts)))
    ax.set_xticklabels(available_wts, rotation=30, ha="right")
    ax.set_yticks(range(len(show_iids)))
    ax.set_yticklabels([f"#{iid}" for iid in show_iids], fontsize=6)
    ax.set_xlabel("Word Type")
    ax.set_ylabel("Instance")
    ax.set_title(f"{dataset.upper()}: Flip Pattern Across Word Types (top 50 multi-type instances)\n"
                 f"Red=flipped, Green=not flipped")
    plt.colorbar(im, ax=ax, label="Flipped", shrink=0.8)

    plt.tight_layout()
    plt.savefig(OUT_DIR / f"fig5_flip_heatmap_{dataset}.png", bbox_inches="tight")
    plt.close()

# ══════════════════════════════════════════════════════════════════════════
# ANALYSIS 6: New CoT qualitative analysis
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ANALYSIS 6: New CoT Qualitative Analysis")
print("=" * 70)

# Analyze how CoTs change after unlearning different word types
# new_cot field in unlearning results

def analyze_cot_changes(dataset, wt, rows):
    """Analyze CoT changes after unlearning."""
    results = {
        "total": 0,
        "cot_changed": 0,
        "answer_changed": 0,
        "cot_length_change": [],
        "new_cot_samples": [],
    }

    for row in rows:
        pred0 = int(row.get("prediction", -1))
        cot_pred0 = int(row.get("cot_prediction", -1))
        if pred0 != cot_pred0:
            continue

        ur = row.get("unlearning_results", {})
        keys_sorted = sorted(ur.keys(), key=lambda k: int(k) if k.isdigit() else k)
        if len(keys_sorted) < 2:
            continue

        original_cot = row.get("initial_cot", "")
        results["total"] += 1

        # Look at last iteration
        last_key = keys_sorted[-1]
        last_item = ur[last_key]

        new_cot = last_item.get("new_cot", "")
        new_pred = last_item.get("prediction")

        if new_cot and original_cot:
            # Simple similarity: word overlap
            orig_words = set(original_cot.lower().split())
            new_words = set(new_cot.lower().split())
            if orig_words:
                overlap = len(orig_words & new_words) / len(orig_words)
            else:
                overlap = 0
            results["cot_changed"] += 1 if overlap < 0.8 else 0
            results["cot_length_change"].append(len(new_cot.split()) - len(original_cot.split()))

            if new_pred is not None and int(new_pred) != pred0:
                results["answer_changed"] += 1

            # Save samples
            if len(results["new_cot_samples"]) < 3 and overlap < 0.5:
                results["new_cot_samples"].append({
                    "id": row.get("id", ""),
                    "step_idx": row.get("step_idx", ""),
                    "original_cot": original_cot[:300],
                    "new_cot": new_cot[:300],
                    "overlap": overlap,
                    "pred_changed": int(new_pred) != pred0 if new_pred is not None else None,
                })

    return results

# Collect CoT change statistics
cot_analysis = {}
for dataset in ["openbook", "sqa"]:
    for wt in WORD_TYPES_ORDERED:
        if wt == "all_content":
            continue
        key = (dataset, wt)
        if key not in all_data:
            continue
        cot_analysis[key] = analyze_cot_changes(dataset, wt, all_data[key])

# Visualization: CoT change rate and answer change rate
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax_idx, dataset in enumerate(["openbook", "sqa"]):
    ax = axes[ax_idx]
    wts = [wt for wt in WORD_TYPES_ORDERED if wt != "all_content"]
    cot_change_rates = []
    answer_change_rates = []
    avg_length_changes = []

    for wt in wts:
        ca = cot_analysis.get((dataset, wt), {})
        total = ca.get("total", 1)
        cot_change_rates.append(ca.get("cot_changed", 0) / total * 100 if total > 0 else 0)
        answer_change_rates.append(ca.get("answer_changed", 0) / total * 100 if total > 0 else 0)
        lc = ca.get("cot_length_change", [])
        avg_length_changes.append(np.mean(lc) if lc else 0)

    x = np.arange(len(wts))
    width = 0.3
    ax.bar(x - width, cot_change_rates, width, label="CoT Changed (%)", color="#e74c3c", alpha=0.7)
    ax.bar(x, answer_change_rates, width, label="Answer Changed (%)", color="#3498db", alpha=0.7)
    ax2 = ax.twinx()
    ax2.bar(x + width, avg_length_changes, width, label="Avg Length Δ", color="#2ecc71", alpha=0.5)
    ax2.set_ylabel("Avg CoT Length Change (words)")

    ax.set_xticks(x)
    ax.set_xticklabels(wts, rotation=30, ha="right")
    ax.set_ylabel("Rate (%)")
    ax.set_title(f"{dataset.upper()}: CoT & Answer Change After Unlearning")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")

plt.tight_layout()
plt.savefig(OUT_DIR / "fig6_cot_changes.png", bbox_inches="tight")
plt.close()

# Print CoT change statistics
print("\nCoT change statistics:")
for dataset in ["openbook", "sqa"]:
    print(f"\n  {dataset}:")
    for wt in WORD_TYPES_ORDERED:
        if wt == "all_content":
            continue
        ca = cot_analysis.get((dataset, wt), {})
        total = ca.get("total", 1)
        cot_changed = ca.get("cot_changed", 0)
        ans_changed = ca.get("answer_changed", 0)
        lc = ca.get("cot_length_change", [])
        avg_lc = np.mean(lc) if lc else 0
        print(f"    {wt:10s}: CoT changed={cot_changed}/{total} ({cot_changed/total*100:.1f}%), "
              f"Answer changed={ans_changed}/{total} ({ans_changed/total*100:.1f}%), "
              f"Avg length Δ={avg_lc:.1f} words")

# Print some sample CoT changes
print("\nSample CoT changes (high divergence):")
for dataset in ["openbook", "sqa"]:
    print(f"\n  {dataset}:")
    for wt in WORD_TYPES_ORDERED:
        if wt == "all_content":
            continue
        ca = cot_analysis.get((dataset, wt), {})
        samples = ca.get("new_cot_samples", [])
        if samples:
            print(f"    {wt}:")
            for s in samples[:1]:
                print(f"      Instance #{s['id']}, step {s['step_idx']}, overlap={s['overlap']:.2f}, "
                      f"pred_changed={s['pred_changed']}")
                print(f"      Original: {s['original_cot'][:150]}...")
                print(f"      New:      {s['new_cot'][:150]}...")

# ══════════════════════════════════════════════════════════════════════════
# SUMMARY FIGURE: Combined dashboard
# ══════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)

# Panel A: FF-Hard by word type (both datasets)
ax1 = fig.add_subplot(gs[0, 0])
for d_idx, dataset in enumerate(["openbook", "sqa"]):
    wts = [wt for wt in WORD_TYPES_ORDERED if wt != "all_content"]
    vals = []
    for wt in wts:
        matching = [r for r in csv_rows if r["dataset"] == dataset and r["word_type"] == wt]
        vals.append(matching[0]["ff_hard(%)"] if matching else 0)
    x = np.arange(len(wts))
    width = 0.35
    offset = -width / 2 if d_idx == 0 else width / 2
    color = "#3498db" if d_idx == 0 else "#e74c3c"
    ax1.bar(x + offset, vals, width, label=dataset, color=color, alpha=0.8)
ax1.set_xticks(x)
ax1.set_xticklabels(wts, rotation=30, ha="right", fontsize=8)
ax1.set_ylabel("FF-Hard (%)")
ax1.set_title("(A) FF-Hard by Word Type")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.2, axis="y")

# Panel B: Efficacy vs Specificity scatter
ax2 = fig.add_subplot(gs[0, 1])
for dataset in ["openbook", "sqa"]:
    marker = "o" if dataset == "openbook" else "s"
    for wt in WORD_TYPES_ORDERED:
        matching = [r for r in csv_rows if r["dataset"] == dataset and r["word_type"] == wt]
        if matching:
            ax2.scatter(matching[0]["efficacy(%)"], matching[0]["specificity(%)"],
                       s=80, c=COLORS[wt], marker=marker, edgecolors="black", linewidth=0.5,
                       label=f"{dataset[:3]}-{wt}" if dataset == "openbook" else "")
ax2.set_xlabel("Efficacy (%)")
ax2.set_ylabel("Specificity (%)")
ax2.set_title("(B) Efficacy vs Specificity")
ax2.grid(True, alpha=0.3)

# Panel C: Skip rate
ax3 = fig.add_subplot(gs[0, 2])
for d_idx, dataset in enumerate(["openbook", "sqa"]):
    wts = [wt for wt in WORD_TYPES_ORDERED if wt not in ("all_content",)]
    rates = [skip_data.get((dataset, wt), {}).get("skip_rate", 0) for wt in wts]
    x = np.arange(len(wts))
    width = 0.35
    offset = -width / 2 if d_idx == 0 else width / 2
    color = "#3498db" if d_idx == 0 else "#e74c3c"
    ax3.bar(x + offset, rates, width, label=dataset, color=color, alpha=0.8)
ax3.set_xticks(x)
ax3.set_xticklabels(wts, rotation=30, ha="right", fontsize=8)
ax3.set_ylabel("Skip Rate (%)")
ax3.set_title("(C) Step Skip Rate")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.2, axis="y")

# Panel D: FF-Soft by word type
ax4 = fig.add_subplot(gs[1, 0])
for d_idx, dataset in enumerate(["openbook", "sqa"]):
    wts = [wt for wt in WORD_TYPES_ORDERED if wt != "all_content"]
    vals = []
    for wt in wts:
        matching = [r for r in csv_rows if r["dataset"] == dataset and r["word_type"] == wt]
        vals.append(matching[0]["ff_soft"] if matching else 0)
    x = np.arange(len(wts))
    width = 0.35
    offset = -width / 2 if d_idx == 0 else width / 2
    color = "#3498db" if d_idx == 0 else "#e74c3c"
    ax4.bar(x + offset, vals, width, label=dataset, color=color, alpha=0.8)
ax4.set_xticks(x)
ax4.set_xticklabels(wts, rotation=30, ha="right", fontsize=8)
ax4.set_ylabel("FF-Soft")
ax4.set_title("(D) FF-Soft by Word Type")
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.2, axis="y")

# Panel E: Efficacy by word type
ax5 = fig.add_subplot(gs[1, 1])
for d_idx, dataset in enumerate(["openbook", "sqa"]):
    wts = [wt for wt in WORD_TYPES_ORDERED if wt != "all_content"]
    vals = []
    for wt in wts:
        matching = [r for r in csv_rows if r["dataset"] == dataset and r["word_type"] == wt]
        vals.append(matching[0]["efficacy(%)"] if matching else 0)
    x = np.arange(len(wts))
    width = 0.35
    offset = -width / 2 if d_idx == 0 else width / 2
    color = "#3498db" if d_idx == 0 else "#e74c3c"
    ax5.bar(x + offset, vals, width, label=dataset, color=color, alpha=0.8)
ax5.set_xticks(x)
ax5.set_xticklabels(wts, rotation=30, ha="right", fontsize=8)
ax5.set_ylabel("Efficacy (%)")
ax5.set_title("(E) Efficacy by Word Type")
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.2, axis="y")

# Panel F: CoT change rate
ax6 = fig.add_subplot(gs[1, 2])
for d_idx, dataset in enumerate(["openbook", "sqa"]):
    wts = [wt for wt in WORD_TYPES_ORDERED if wt != "all_content"]
    vals = []
    for wt in wts:
        ca = cot_analysis.get((dataset, wt), {})
        total = ca.get("total", 1)
        vals.append(ca.get("answer_changed", 0) / total * 100 if total > 0 else 0)
    x = np.arange(len(wts))
    width = 0.35
    offset = -width / 2 if d_idx == 0 else width / 2
    color = "#3498db" if d_idx == 0 else "#e74c3c"
    ax6.bar(x + offset, vals, width, label=dataset, color=color, alpha=0.8)
ax6.set_xticks(x)
ax6.set_xticklabels(wts, rotation=30, ha="right", fontsize=8)
ax6.set_ylabel("Answer Change Rate (%)")
ax6.set_title("(F) Answer Change After Unlearning")
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.2, axis="y")

plt.savefig(OUT_DIR / "fig0_summary_dashboard.png", bbox_inches="tight")
plt.close()

print("\n" + "=" * 70)
print("All figures saved to:", OUT_DIR)
print("=" * 70)
print("\nGenerated files:")
for f in sorted(OUT_DIR.iterdir()):
    print(f"  {f.name}")
