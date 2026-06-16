"""
CoT 词汇类型统计分析脚本

遍历 final_cot 目录下所有 CoT 文本数据，使用 spaCy 进行词性标注，
按实验定义的词汇类型分组（entity/attribute/action/function/modifier/other）
统计各类词语的数量及占比，生成饼图和统计报告。

用法:
    python analyze_cot_vocab_stats.py

输出:
    - cot_vocab_stats_summary.json   各数据集的统计汇总
    - cot_vocab_stats_<dataset>_<model>.png   各数据集×模型的饼图
    - cot_vocab_stats_overall.png    全局汇总饼图
    - cot_vocab_stats_detail.csv     逐条明细表
"""

import os
import json
import csv
import glob
from collections import defaultdict, OrderedDict

import spacy
import matplotlib
matplotlib.use('Agg')  # 无GUI环境用非交互后端
import matplotlib.pyplot as plt

# ============================================================
# 词汇类型分组定义（与 segment.py 保持一致）
# ============================================================
WORD_TYPE_GROUPS = OrderedDict([
    ('entity',    {'NOUN', 'PROPN'}),       # 实体词：名词、专有名词
    ('attribute', {'ADJ', 'NUM'}),          # 属性词：形容词、数词
    ('action',    {'VERB'}),                # 动作词：动词
    ('modifier',  {'ADV'}),                 # 修饰词：副词
    ('function',  {'ADP', 'AUX', 'CCONJ',   # 虚词：介词、助动词、连词、
                   'DET', 'PART', 'PRON',   #       限定词、小品词、代词、
                   'SCONJ'}),               #       从属连词
])

# 饼图标签（英文，避免中文字体缺失）
GROUP_DISPLAY = {
    'entity':    'Entity\n(NOUN/PROPN)',
    'attribute': 'Attribute\n(ADJ/NUM)',
    'action':    'Action\n(VERB)',
    'modifier':  'Modifier\n(ADV)',
    'function':  'Function\n(ADP/AUX/CCONJ/...)',
    'other':     'Other\n(SYM/PUNCT/...)',
}

# 饼图配色
GROUP_COLORS = {
    'entity':    '#4285F4',   # 蓝
    'attribute': '#FBBC05',   # 黄
    'action':    '#EA4335',   # 红
    'modifier':  '#9C27B0',   # 紫
    'function':  '#607D8B',   # 灰蓝
    'other':     '#BDBDBD',   # 浅灰
}


def classify_pos(pos_tag):
    """根据 POS 标签返回词汇类型分组名。"""
    for group_name, tag_set in WORD_TYPE_GROUPS.items():
        if pos_tag in tag_set:
            return group_name
    return 'other'


def load_cot_data(cot_dir):
    """加载 final_cot 目录下所有 JSONL 文件。

    Returns:
        list of dict, 每项包含:
            - file_path: 文件路径
            - dataset: 数据集名
            - model: 模型名
            - items: 该文件中的所有 CoT 条目
    """
    all_data = []
    pattern = os.path.join(cot_dir, '**', '*.jsonl')
    for fpath in sorted(glob.glob(pattern, recursive=True)):
        # 从路径提取 dataset 和 model
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
    """对一组 CoT 条目进行词汇类型统计。

    Args:
        items: CoT 条目列表
        nlp: spaCy 模型

    Returns:
        dict: {
            'total_tokens': 总token数,
            'group_counts': {group: count},
            'group_ratios': {group: ratio},
            'per_step_stats': [{step_idx, group_counts, group_ratios}],
            'per_instance_stats': [{instance_id, group_counts, group_ratios}],
        }
    """
    total_counts = defaultdict(int)
    per_step_stats = []
    per_instance_stats = []
    total_tokens = 0

    for item in items:
        # 优先使用 segmented_cot（已分步），否则用完整 cot
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

    # 汇总比例
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
    """生成词汇类型分布饼图。"""
    # 按固定顺序排列，other 放最后
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
        # 实体词和动作词略微突出
        explode.append(0.05 if g in ('entity', 'action') else 0)

    if not sizes:
        print(f"[warn] 无数据可绘制: {title}")
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

    # 调整百分比文字大小
    for autotext in autotexts:
        autotext.set_fontsize(9)

    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)

    # 添加图例
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
    print(f"  饼图已保存: {save_path}")


def main():
    # ---- 配置路径 ----
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cot_dir = os.path.join(base_dir, 'final_cot')
    output_dir = os.path.join(base_dir, 'vocab_stats_output')
    os.makedirs(output_dir, exist_ok=True)

    # ---- 加载 spaCy ----
    print("加载 spaCy 模型...")
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

    # ---- 加载 CoT 数据 ----
    print(f"扫描 CoT 数据目录: {cot_dir}")
    all_data = load_cot_data(cot_dir)
    print(f"  找到 {len(all_data)} 个数据文件")

    if not all_data:
        print("[error] 未找到任何 CoT 数据文件，请检查路径。")
        return

    # ---- 逐文件分析 ----
    summary = {}
    all_detail_rows = []
    global_counts = defaultdict(int)
    global_total = 0

    for data_info in all_data:
        dataset = data_info['dataset']
        model = data_info['model']
        items = data_info['items']
        key = f"{dataset}/{model}"

        print(f"\n分析: {key} ({len(items)} 条)")

        result = analyze_cot_texts(items, nlp)

        # 汇总
        summary[key] = {
            'dataset': dataset,
            'model': model,
            'n_instances': len(items),
            'total_tokens': result['total_tokens'],
            'group_counts': result['group_counts'],
            'group_ratios': result['group_ratios'],
        }

        # 累积全局统计
        for g, c in result['group_counts'].items():
            global_counts[g] += c
        global_total += result['total_tokens']

        # 生成饼图
        safe_name = f"cot_vocab_stats_{dataset}_{model}"
        plot_pie_chart(
            result['group_ratios'],
            f'CoT Word Type Distribution — {dataset} / {model}\n(total {result["total_tokens"]} tokens)',
            os.path.join(output_dir, f'{safe_name}.png'),
            total_tokens=result['total_tokens'],
        )

        # 收集明细行
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

    # ---- 全局汇总饼图 ----
    global_ratios = {g: c / global_total for g, c in global_counts.items()} if global_total > 0 else {}
    plot_pie_chart(
        global_ratios,
        f'CoT Word Type Distribution — Overall\n(total {global_total} tokens, {len(all_data)} files)',
        os.path.join(output_dir, 'cot_vocab_stats_overall.png'),
        total_tokens=global_total,
    )

    # ---- 保存统计汇总 JSON ----
    summary['global'] = {
        'total_tokens': global_total,
        'group_counts': dict(global_counts),
        'group_ratios': {g: round(r, 4) for g, r in global_ratios.items()},
    }
    summary_path = os.path.join(output_dir, 'cot_vocab_stats_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n统计汇总已保存: {summary_path}")

    # ---- 保存明细 CSV ----
    if all_detail_rows:
        csv_path = os.path.join(output_dir, 'cot_vocab_stats_detail.csv')
        fieldnames = list(all_detail_rows[0].keys())
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_detail_rows)
        print(f"明细数据已保存: {csv_path}")

    # ---- 打印汇总表 ----
    print("\n" + "=" * 80)
    print("  CoT 词汇类型统计汇总")
    print("=" * 80)

    header = f"{'数据集/模型':<30s} {'总token':>8s}"
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
    line = f"{'全局汇总':<30s} {g_info['total_tokens']:>8d}"
    for g in list(WORD_TYPE_GROUPS.keys()) + ['other']:
        ratio = g_info['group_ratios'].get(g, 0)
        count = g_info['group_counts'].get(g, 0)
        line += f" {count:>5d}({ratio:.1%})"
    print(line)
    print("=" * 80)

    # ---- 按数据集汇总（跨模型聚合） ----
    print("\n按数据集汇总:")
    dataset_agg = defaultdict(lambda: defaultdict(int))
    for key, info in summary.items():
        if key == 'global':
            continue
        ds = info['dataset']
        for g, c in info['group_counts'].items():
            dataset_agg[ds][g] += c

    for ds, counts in sorted(dataset_agg.items()):
        total = sum(counts.values())
        print(f"\n  [{ds}] 总token: {total}")
        for g in list(WORD_TYPE_GROUPS.keys()) + ['other']:
            c = counts.get(g, 0)
            r = c / total if total > 0 else 0
            bar = '█' * max(1, int(r * 40))
            print(f"    {g:12s}: {c:6d} ({r:5.1%}) {bar}")

    print(f"\n所有输出文件保存在: {output_dir}")


if __name__ == '__main__':
    main()
