"""
compute_answer_change_rate.py

计算论文中 Table 2 的核心指标：
"% of instances where adding mistakes or unlearning a reasoning step
 changes the model's answer. Measured only on instances where no-CoT
 and CoT model predictions agree. Scores over 1% higher in bold."

该指标衡量的是：在 no-CoT 和 CoT 预测一致的样本中，
引入错误（adding mistakes）或遗忘推理步骤（unlearning）后，
模型答案发生改变的实例所占的百分比。

数据来源：
  - mistake_results/: 包含原始 no-CoT 预测(prediction)和 CoT 预测(cot_prediction)
  - mistake_stats/:   包含引入错误后的预测(mistake_prediction)和翻转标记(mistake_flipped)
  - final_results/:   包含遗忘推理步骤后的预测(unlearning_results)

计算方法：
  1. 从 mistake_results 筛选 prediction == cot_prediction 的实例
  2. 从 mistake_stats 匹配对应实例，统计 mistake_flipped == True 的数量
  3. 从 final_results 筛选 prediction == cot_prediction 的实例
  4. 检查 unlearning_results 中各步骤的 prediction 是否与 cot_prediction 不同
  5. 计算百分比 = (改变数 / 一致数) * 100
"""

import json
import os
import sys
from collections import defaultdict


BOLD_START = "\033[1m"
BOLD_END = "\033[0m"


def load_jsonl(filepath):
    """加载 JSONL 文件，跳过空行和格式错误的行"""
    results = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [警告] {filepath} 第 {line_num} 行 JSON 解析失败: {e}",
                      file=sys.stderr)
    return results


def analyze_mistakes(script_dir, datasets, models):
    """
    分析"添加错误"场景下的答案变化率。

    流程：
      1. 从 mistake_results 读取原始数据，筛选 prediction == cot_prediction 的实例
      2. 从 mistake_stats 读取错误引入后的结果，匹配相同 (id, step_idx)
      3. 统计 mistake_flipped == True 的数量
      4. 计算百分比
    """
    print("=" * 70)
    print("  Part A: Adding Mistakes — 答案变化率分析")
    print("=" * 70)

    results = {}

    for dataset in datasets:
        for model in models:
            key = f"{dataset}_{model}"

            # 路径
            mistake_results_path = os.path.join(
                script_dir, "mistake_results", dataset, model
            )
            mistake_stats_path = os.path.join(
                script_dir, "mistake_stats", dataset, model
            )

            # 查找文件
            mistake_results_file = None
            mistake_stats_file = None
            try:
                for fname in os.listdir(mistake_results_path):
                    if fname.endswith('.jsonl'):
                        mistake_results_file = os.path.join(
                            mistake_results_path, fname
                        )
                        break
                for fname in os.listdir(mistake_stats_path):
                    if fname.endswith('.jsonl'):
                        mistake_stats_file = os.path.join(
                            mistake_stats_path, fname
                        )
                        break
            except FileNotFoundError:
                print(f"  [跳过] {key}: 数据目录不存在")
                continue

            if mistake_results_file is None or mistake_stats_file is None:
                print(f"  [跳过] {key}: 未找到数据文件")
                continue

            # 加载数据
            raw_data = load_jsonl(mistake_results_file)
            stats_data = load_jsonl(mistake_stats_file)

            # 构建 mistake_stats 索引: (id, step_idx) -> mistake_flipped
            stats_index = {}
            for item in stats_data:
                stats_index[(item['id'], item['step_idx'])] = item

            # 筛选 no-CoT 和 CoT 预测一致的实例
            agreeing_instances = []
            for item in raw_data:
                if item['prediction'] == item['cot_prediction']:
                    agreeing_instances.append(item)

            # 统计错误引入后答案改变的实例
            flipped_count = 0
            total_agreeing = len(agreeing_instances)

            for item in agreeing_instances:
                match_key = (item['id'], item['step_idx'])
                if match_key in stats_index:
                    if stats_index[match_key].get('mistake_flipped', False):
                        flipped_count += 1

            # 计算百分比
            if total_agreeing > 0:
                change_rate = (flipped_count / total_agreeing) * 100
            else:
                change_rate = 0.0

            results[key] = {
                'agreeing': total_agreeing,
                'flipped': flipped_count,
                'change_rate': change_rate,
            }

            # 输出
            rate_str = f"{change_rate:.1f}%"
            if change_rate >= 1.0:
                rate_str = f"{BOLD_START}{rate_str}{BOLD_END}"

            print(f"\n  {model} / {dataset}")
            print(f"    no-CoT 与 CoT 预测一致的实例数: {total_agreeing}")
            print(f"    引入错误后答案改变的实例数:     {flipped_count}")
            print(f"    答案变化率:                     {rate_str}")

    return results


def analyze_unlearning(script_dir, datasets, models):
    """
    分析"遗忘推理步骤"场景下的答案变化率。

    流程：
      1. 从 final_results 读取数据，筛选 prediction == cot_prediction 的实例
      2. 对每个实例，检查 unlearning_results 中各步骤的 prediction
         是否与原始 cot_prediction 不同
      3. 只要任一步骤的遗忘导致预测改变，该实例即计为"改变"
      4. 计算百分比
    """
    print("\n" + "=" * 70)
    print("  Part B: Unlearning Reasoning Steps — 答案变化率分析")
    print("=" * 70)

    results = {}

    for dataset in datasets:
        for model in models:
            key = f"{dataset}_{model}"

            final_results_path = os.path.join(
                script_dir, "final_results", dataset, model
            )

            # 查找文件
            final_results_file = None
            try:
                for fname in os.listdir(final_results_path):
                    if fname.endswith('.out'):
                        final_results_file = os.path.join(
                            final_results_path, fname
                        )
                        break
            except FileNotFoundError:
                print(f"  [跳过] {key}: 数据目录不存在")
                continue

            if final_results_file is None:
                print(f"  [跳过] {key}: 未找到数据文件")
                continue

            # 加载数据
            raw_data = load_jsonl(final_results_file)

            # 筛选 no-CoT 和 CoT 预测一致的实例
            agreeing_instances = []
            for item in raw_data:
                if item['prediction'] == item['cot_prediction']:
                    agreeing_instances.append(item)

            # 统计遗忘后答案改变的实例
            # 只要任一步骤的遗忘导致预测改变，该实例即计为"改变"
            changed_count = 0
            total_agreeing = len(agreeing_instances)

            for item in agreeing_instances:
                cot_pred = item['cot_prediction']
                unlearning_results = item.get('unlearning_results', {})

                instance_changed = False
                for step_key, step_result in unlearning_results.items():
                    after_unlearn_pred = step_result.get('prediction', None)
                    if after_unlearn_pred is not None and after_unlearn_pred != cot_pred:
                        instance_changed = True
                        break

                if instance_changed:
                    changed_count += 1

            # 计算百分比
            if total_agreeing > 0:
                change_rate = (changed_count / total_agreeing) * 100
            else:
                change_rate = 0.0

            results[key] = {
                'agreeing': total_agreeing,
                'changed': changed_count,
                'change_rate': change_rate,
            }

            # 输出
            rate_str = f"{change_rate:.1f}%"
            if change_rate >= 1.0:
                rate_str = f"{BOLD_START}{rate_str}{BOLD_END}"

            print(f"\n  {model} / {dataset}")
            print(f"    no-CoT 与 CoT 预测一致的实例数: {total_agreeing}")
            print(f"    遗忘步骤后答案改变的实例数:     {changed_count}")
            print(f"    答案变化率:                     {rate_str}")

    return results


def print_summary_table(mistake_results, unlearn_results, datasets, models):
    """打印汇总表格，超过 1% 的数值以粗体显示"""

    def bold_if(v):
        s = f"{v:.1f}%"
        return f"{BOLD_START}{s}{BOLD_END}" if v >= 1.0 else s

    print("\n" + "=" * 70)
    print("  汇总表格: % of instances where adding mistakes or")
    print("  unlearning a reasoning step changes the model's answer")
    print("  (仅统计 no-CoT 与 CoT 预测一致的实例)")
    print("=" * 70)

    # 表头
    header = (
        f"{'Model':<16} {'Dataset':<10} "
        f"{'Mistake Agree':>14} {'Mistake Changed':>16} "
        f"{'Mistake Rate':>13} | "
        f"{'Unlearn Agree':>14} {'Unlearn Changed':>16} "
        f"{'Unlearn Rate':>13}"
    )
    sep = "-" * len(header)
    print(f"\n  {header}")
    print(f"  {sep}")

    for model in models:
        for dataset in datasets:
            mk = f"{dataset}_{model}"
            uk = f"{dataset}_{model}"

            m = mistake_results.get(mk, {})
            u = unlearn_results.get(uk, {})

            m_agree = m.get('agreeing', 0)
            m_changed = m.get('flipped', 0)
            m_rate = m.get('change_rate', 0.0)

            u_agree = u.get('agreeing', 0)
            u_changed = u.get('changed', 0)
            u_rate = u.get('change_rate', 0.0)

            m_rate_str = bold_if(m_rate) if m_rate >= 1.0 else f"{m_rate:.1f}%"
            u_rate_str = bold_if(u_rate) if u_rate >= 1.0 else f"{u_rate:.1f}%"

            # 在终端中，粗体 ANSI 码会影响对齐，这里用纯文本 + 标记
            # 实际粗体通过 ANSI 码实现
            print(
                f"  {model:<16} {dataset:<10} "
                f"{m_agree:>14} {m_changed:>16} "
                f"{m_rate_str:>20} | "
                f"{u_agree:>14} {u_changed:>16} "
                f"{u_rate_str:>20}"
            )

    print(f"\n  注: {BOLD_START}粗体{BOLD_END} 表示变化率 >= 1%")


def export_json_summary(mistake_results, unlearn_results, script_dir):
    """导出 JSON 格式的汇总结果"""
    summary = {
        "description": (
            "% of instances where adding mistakes or unlearning a reasoning "
            "step changes the model's answer. Measured only on instances where "
            "no-CoT and CoT model predictions agree."
        ),
        "adding_mistakes": {},
        "unlearning_steps": {},
    }

    for key, val in mistake_results.items():
        summary["adding_mistakes"][key] = {
            "agreeing_instances": val["agreeing"],
            "changed_instances": val["flipped"],
            "change_rate_percent": round(val["change_rate"], 2),
        }

    for key, val in unlearn_results.items():
        summary["unlearning_steps"][key] = {
            "agreeing_instances": val["agreeing"],
            "changed_instances": val["changed"],
            "change_rate_percent": round(val["change_rate"], 2),
        }

    output_path = os.path.join(script_dir, "answer_change_rate_summary.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  JSON 汇总已保存至: {output_path}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    datasets = ['openbook', 'sqa']
    models = ['LLaMA-3-3B', 'Phi-3']

    print("=" * 70)
    print("  论文指标计算: % of instances where adding mistakes or")
    print("  unlearning a reasoning step changes the model's answer")
    print("=" * 70)

    # Part A: 添加错误分析
    mistake_results = analyze_mistakes(script_dir, datasets, models)

    # Part B: 遗忘推理步骤分析
    unlearn_results = analyze_unlearning(script_dir, datasets, models)

    # 汇总表格
    print_summary_table(mistake_results, unlearn_results, datasets, models)

    # 导出 JSON
    export_json_summary(mistake_results, unlearn_results, script_dir)

    print("\n  分析完成。")


if __name__ == "__main__":
    main()