"""CoT vocabulary-type statistics.

Walks the CoT text under ``final_cot``, POS-tags it with spaCy, buckets tokens
into the experiment's vocabulary-type groups (entity / attribute / action /
function / modifier / other), and writes per-cell and overall pie charts plus
summary tables.

Usage:
    python analyze_cot_vocab_stats.py

Outputs (under vocab_stats_output/):
    - cot_vocab_stats_summary.json          per-dataset summary
    - cot_vocab_stats_<dataset>_<model>.png per-cell pie chart
    - cot_vocab_stats_overall.png           overall pie chart
    - cot_vocab_stats_detail.csv            per-step detail rows
"""

import os
import json
import csv
import glob
from collections import defaultdict, OrderedDict

import spacy
import matplotlib
matplotlib.use('Agg')  # non-interactive backend (headless)
import matplotlib.pyplot as plt

# Vocabulary-type groups (kept in sync with segment_for_more_kinds_of_words.py).
WORD_TYPE_GROUPS = OrderedDict([
    ('entity',    {'NOUN', 'PROPN'}),       # nouns / proper nouns
    ('attribute', {'ADJ', 'NUM'}),          # adjectives / numbers
    ('action',    {'VERB'}),                # verbs
    ('modifier',  {'ADV'}),                 # adverbs
    ('function',  {'ADP', 'AUX', 'CCONJ',   # function words
                   'DET', 'PART', 'PRON',
                   'SCONJ'}),
])

# Pie-chart labels.
GROUP_DISPLAY = {
    'entity':    'Entity\n(NOUN/PROPN)',
    'attribute': 'Attribute\n(ADJ/NUM)',
    'action':    'Action\n(VERB)',
    'modifier':  'Modifier\n(ADV)',
    'function':  'Function\n(ADP/AUX/CCONJ/...)',
    'other':     'Other\n(SYM/PUNCT/...)',
}

# Pie-chart colors.
GROUP_COLORS = {
    'entity':    '#4285F4',   # blue
    'attribute': '#FBBC05',   # yellow
    'action':    '#EA4335',   # red
    'modifier':  '#9C27B0',   # purple
    'function':  '#607D8B',   # blue-grey
    'other':     '#BDBDBD',   # light grey
}


def classify_pos(pos_tag):
    """Return the vocabulary-type group for a POS tag, or 'other'."""
    for group_name, tag_set in WORD_TYPE_GROUPS.items():
        if pos_tag in tag_set:
            return group_name
    return 'other'


def load_cot_data(cot_dir):
    """Load every JSONL file under ``cot_dir``.

    Returns a list of dicts with file_path, dataset, model, items.
    """
    all_data = []
    pattern = os.path.join(cot_dir, '**', '*.jsonl')
    for fpath in sorted(glob.glob(pattern, recursive=True)):
        # Derive dataset and model from the path.
        parts = fpath.replace(cot_dir, '').strip(os.sep).split(os.sep)
        dataset = parts[0] if len(parts) >= 1 else 'unknown'
        model = os.path.basename(fpath).split('_s=')[0] if '_s=' in fpath else 'unknown'

        items = []
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))

        all_data.append({
            'file_path': fpath,
            'dataset': dataset,
            'model': model,
            'items': items,
        })
    return all_data


def analyze_cot_texts(items, nlp):
    """Count vocabulary-type groups over a list of CoT items.

    Returns total_tokens, group_counts, group_ratios, per_step_stats,
    per_instance_stats.
    """
    total_counts = defaultdict(int)
    per_step_stats = []
    per_instance_stats = []
    total_tokens = 0

    for item in items:
        # Prefer the segmented CoT; fall back to the full CoT.
        if 'segmented_cot' in item and item['segmented_cot']:
            cot_steps = item['segmented_cot']
        elif 'cot' in item:
            cot_steps = [item['cot']]
        else:
            continue

        instance_counts = defaultdict(int)
        instance_total = 0

        for step_idx, step_text in enumerate(cot_steps):
            doc = nlp(step_text)
            step_counts = defaultdict(int)
            step_total = 0

            for token in doc:
                if token.pos_ == 'SPACE':
                    continue
                group = classify_pos(token.pos_)
                step_counts[group] += 1
                total_counts[group] += 1
                instance_counts[group] += 1
                step_total += 1

            step_total_tokens = sum(step_counts.values())
            step_ratios = {g: c / step_total_tokens for g, c in step_counts.items()} if step_total_tokens > 0 else {}
            per_step_stats.append({
                'instance_id': item.get('id', 'unknown'),
                'step_idx': step_idx,
                'step_text': step_text[:80] + '...' if len(step_text) > 80 else step_text,
                'total_tokens': step_total_tokens,
                'group_counts': dict(step_counts),
                'group_ratios': {g: round(r, 4) for g, r in step_ratios.items()},
            })

            instance_total += step_total
            total_tokens += step_total

        inst_total_tokens = sum(instance_counts.values())
        inst_ratios = {g: c / inst_total_tokens for g, c in instance_counts.items()} if inst_total_tokens > 0 else {}
        per_instance_stats.append({
            'instance_id': item.get('id', 'unknown'),
            'total_tokens': inst_total_tokens,
            'group_counts': dict(instance_counts),
            'group_ratios': {g: round(r, 4) for g, r in inst_ratios.items()},
        })

    total_sum = sum(total_counts.values())
    group_ratios = {g: c / total_sum for g, c in total_counts.items()} if total_sum > 0 else {}

    return {
        'total_tokens': total_sum,
        'group_counts': dict(total_counts),
        'group_ratios': {g: round(r, 4) for g, r in group_ratios.items()},
        'per_step_stats': per_step_stats,
        'per_instance_stats': per_instance_stats,
    }


def plot_pie_chart(group_ratios, title, save_path, total_tokens=0):
    """Render a vocabulary-type distribution pie chart."""
    # Fixed order, 'other' last.
    ordered_groups = list(WORD_TYPE_GROUPS.keys()) + ['other']
    labels = []
    sizes = []
    colors = []
    explode = []

    for g in ordered_groups:
        ratio = group_ratios.get(g, 0)
        if ratio <= 0:
            continue
        labels.append(GROUP_DISPLAY.get(g, g))
        sizes.append(ratio)
        colors.append(GROUP_COLORS.get(g, '#999999'))
        # Nudge entity and action out slightly.
        explode.append(0.05 if g in ('entity', 'action') else 0)

    if not sizes:
        print(f"[warn] nothing to plot: {title}")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        explode=explode,
        autopct=lambda pct: f'{pct:.1f}%\n({int(round(pct/100.*sum(sizes)*total_tokens))})' if total_tokens > 0 else f'{pct:.1f}%',
        startangle=140,
        pctdistance=0.75,
        textprops={'fontsize': 10},
    )

    for autotext in autotexts:
        autotext.set_fontsize(9)

    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)

    ax.legend(
        wedges, [f'{GROUP_DISPLAY.get(g, g)}: {group_ratios.get(g, 0)*100:.1f}%'
                 for g in ordered_groups if group_ratios.get(g, 0) > 0],
        title='Word Type',
        loc='center left',
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=9,
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved pie chart: {save_path}")


def main():
    # ---- paths ----
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cot_dir = os.path.join(base_dir, 'final_cot')
    output_dir = os.path.join(base_dir, 'vocab_stats_output')
    os.makedirs(output_dir, exist_ok=True)

    # ---- spaCy ----
    print("Loading spaCy model...")
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

    # ---- load CoT data ----
    print(f"Scanning CoT directory: {cot_dir}")
    all_data = load_cot_data(cot_dir)
    print(f"  found {len(all_data)} data files")

    if not all_data:
        print("[error] no CoT data files found; check the path.")
        return

    # ---- analyze per file ----
    summary = {}
    all_detail_rows = []
    global_counts = defaultdict(int)
    global_total = 0

    for data_info in all_data:
        dataset = data_info['dataset']
        model = data_info['model']
        items = data_info['items']
        key = f"{dataset}/{model}"

        print(f"\nAnalyzing: {key} ({len(items)} items)")

        result = analyze_cot_texts(items, nlp)

        summary[key] = {
            'dataset': dataset,
            'model': model,
            'n_instances': len(items),
            'total_tokens': result['total_tokens'],
            'group_counts': result['group_counts'],
            'group_ratios': result['group_ratios'],
        }

        for g, c in result['group_counts'].items():
            global_counts[g] += c
        global_total += result['total_tokens']

        safe_name = f"cot_vocab_stats_{dataset}_{model}"
        plot_pie_chart(
            result['group_ratios'],
            f'CoT Word Type Distribution — {dataset} / {model}\n(total {result["total_tokens"]} tokens)',
            os.path.join(output_dir, f'{safe_name}.png'),
            total_tokens=result['total_tokens'],
        )

        for step_stat in result['per_step_stats']:
            row = {
                'dataset': dataset,
                'model': model,
                'instance_id': step_stat['instance_id'],
                'step_idx': step_stat['step_idx'],
                'total_tokens': step_stat['total_tokens'],
            }
            for g in list(WORD_TYPE_GROUPS.keys()) + ['other']:
                row[f'{g}_count'] = step_stat['group_counts'].get(g, 0)
                row[f'{g}_ratio'] = step_stat['group_ratios'].get(g, 0)
            all_detail_rows.append(row)

    # ---- overall pie chart ----
    global_ratios = {g: c / global_total for g, c in global_counts.items()} if global_total > 0 else {}
    plot_pie_chart(
        global_ratios,
        f'CoT Word Type Distribution — Overall\n(total {global_total} tokens, {len(all_data)} files)',
        os.path.join(output_dir, 'cot_vocab_stats_overall.png'),
        total_tokens=global_total,
    )

    # ---- write summary JSON ----
    summary['global'] = {
        'total_tokens': global_total,
        'group_counts': dict(global_counts),
        'group_ratios': {g: round(r, 4) for g, r in global_ratios.items()},
    }
    summary_path = os.path.join(output_dir, 'cot_vocab_stats_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSummary written: {summary_path}")

    # ---- write detail CSV ----
    if all_detail_rows:
        csv_path = os.path.join(output_dir, 'cot_vocab_stats_detail.csv')
        fieldnames = list(all_detail_rows[0].keys())
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_detail_rows)
        print(f"Detail written: {csv_path}")

    # ---- print summary table ----
    print("\n" + "=" * 80)
    print("  CoT Word Type Statistics")
    print("=" * 80)

    header = f"{'dataset/model':<30s} {'tokens':>8s}"
    for g in list(WORD_TYPE_GROUPS.keys()) + ['other']:
        header += f" {g:>10s}"
    print(header)
    print("-" * 80)

    for key, info in summary.items():
        if key == 'global':
            continue
        line = f"{key:<30s} {info['total_tokens']:>8d}"
        for g in list(WORD_TYPE_GROUPS.keys()) + ['other']:
            ratio = info['group_ratios'].get(g, 0)
            count = info['group_counts'].get(g, 0)
            line += f" {count:>5d}({ratio:.1%})"
        print(line)

    print("-" * 80)
    g_info = summary['global']
    line = f"{'overall':<30s} {g_info['total_tokens']:>8d}"
    for g in list(WORD_TYPE_GROUPS.keys()) + ['other']:
        ratio = g_info['group_ratios'].get(g, 0)
        count = g_info['group_counts'].get(g, 0)
        line += f" {count:>5d}({ratio:.1%})"
    print(line)
    print("=" * 80)

    # ---- per-dataset aggregation (across models) ----
    print("\nBy dataset:")
    dataset_agg = defaultdict(lambda: defaultdict(int))
    for key, info in summary.items():
        if key == 'global':
            continue
        ds = info['dataset']
        for g, c in info['group_counts'].items():
            dataset_agg[ds][g] += c

    for ds, counts in sorted(dataset_agg.items()):
        total = sum(counts.values())
        print(f"\n  [{ds}] total tokens: {total}")
        for g in list(WORD_TYPE_GROUPS.keys()) + ['other']:
            c = counts.get(g, 0)
            r = c / total if total > 0 else 0
            bar = '█' * max(1, int(r * 40))
            print(f"    {g:12s}: {c:6d} ({r:5.1%}) {bar}")

    print(f"\nAll outputs saved under: {output_dir}")


if __name__ == '__main__':
    main()
