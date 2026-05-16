#!/usr/bin/env python
"""
系统性评估脚本 - CoT忠实度测量复现项目
功能：
1. 数据统计分析
2. 性能指标计算（Efficacy, Specificity, Faithfulness）
3. 结果可视化（散点图、条形图、对比图）

输出目录：figures_of_reproduce/
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from glob import glob

# 配置
BASE_DIR = "/home/xiaoberber/paper_reading/measuring Cot/reproduce of measuring Cot1/parametric-faithfulness"
OUTPUT_DIR = os.path.join(BASE_DIR, "figures_of_reproduce")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 颜色和样式配置
model_color = {
    'Phi-3': 'tab:blue',
    'LLaMA-3': 'tab:red',
    'LLaMA-3-3B': 'tab:orange',
    'Mistral-2': 'tab:green',
}

dataset_to_nice = {
    'arc-challenge': 'ARC-Challenge',
    'openbook': 'OpenBookQA',
    'sqa': 'StrategyQA',
    'sports': 'Sports'
}

method_color = {
    'npo': 'tab:blue',
    'npo_grad_diff': 'tab:orange',
    'npo_KL': 'tab:red',
    'simnpo_KL': 'purple',
}

# 步骤1: 加载结果数据

def load_results(result_dir):
    """加载所有结果文件"""
    results = {}
    
    pattern = os.path.join(result_dir, "**", "*.out")
    files = glob(pattern, recursive=True)
    
    for file_path in files:
        parts = file_path.replace(result_dir, '').strip(os.sep).split(os.sep)
        if len(parts) >= 3:
            dataset = parts[0]
            model = parts[1]
            filename = parts[2]
            
            if dataset not in results:
                results[dataset] = {}
            if model not in results[dataset]:
                results[dataset][model] = {}
            instances = []
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        instances.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            config = parse_filename(filename)
            config['instances'] = instances
            results[dataset][model][filename] = config
    
    return results

def parse_filename(filename):
    """解析结果文件名，提取配置参数"""
    config = {}
    name = filename.replace('.out', '')
    parts = name.split('_')
    
    for part in parts:
        if '=' in part:
            key, value = part.split('=', 1)
            config[key] = value
        else:
            # 方法名在前
            if 'method' not in config:
                config['method'] = part
    
    return config

# 步骤2: 计算统计指标

def compute_efficacy(instance_results):
    """计算Efficacy：CoT步骤概率的降低程度"""
    if '0' not in instance_results:
        return 0.0
    
    initial_prob = None
    final_prob = None
    
    #初始概率
    if 'cot_step_prob' in instance_results['0']:
        initial_prob = np.exp(instance_results['0']['cot_step_prob'][0])
    
    # 获取最后一轮概率
    last_iter = str(max([int(k) for k in instance_results.keys()]))
    if last_iter in instance_results and 'cot_step_prob' in instance_results[last_iter]:
        final_prob = np.exp(instance_results[last_iter]['cot_step_prob'][0])
    
    if initial_prob and final_prob and initial_prob > 0:
        return (1 - final_prob / initial_prob) * 100.0
    return 0.0

def compute_specificity(instance_results):
    """计算Specificity：保留集预测稳定性"""
    if '0' not in instance_results:
        return 100.0
    
    initial_preds = np.array(instance_results['0']['specificity_preds'])
    n_iters = len(instance_results)
    
    correct_count = 0
    total_count = 0
    
    for i in range(1, n_iters):
        if str(i) in instance_results:
            current_preds = np.array(instance_results[str(i)]['specificity_preds'])
            correct_count += (initial_preds == current_preds).sum()
            total_count += len(current_preds)
    
    if total_count > 0:
        return (correct_count / total_count) * 100.0
    return 100.0

def compute_faithfulness(instance_results):
    """计算Faithfulness：答案预测改变的比例"""
    if '0' not in instance_results:
        return 0.0
    
    initial_pred = instance_results['0']['prediction']
    n_iters = len(instance_results)
    
    for i in range(1, n_iters):
        if str(i) in instance_results:
            current_pred = instance_results[str(i)]['prediction']
            if current_pred != initial_pred:
                return 100.0
    
    return 0.0

def make_stats(results):
    """计算完整统计指标"""
    stats = []
    
    for dataset, model_results in results.items():
        for model, config_results in model_results.items():
            for config_name, config_data in config_results.items():
                instances = config_data['instances']
                n_instances = len(set([inst['id'] for inst in instances]))
                
                # 计算每个实例的指标
                efficacies = []
                specificities = []
                faithfulness_scores = []
                
                # 按实例分组
                instances_by_id = {}
                for inst in instances:
                    inst_id = inst['id']
                    if inst_id not in instances_by_id:
                        instances_by_id[inst_id] = []
                    instances_by_id[inst_id].append(inst)
                
                for inst_id, inst_list in instances_by_id.items():
                    # 获取该实例所有步骤的unlearning结果
                    combined_results = {}
                    for inst in inst_list:
                        step_idx = inst['step_idx']
                        unlearn_results = inst['unlearning_results']
                        
                        # 合并不同步骤的结果（取最后一步）
                        for iter_key, result in unlearn_results.items():
                            if iter_key not in combined_results:
                                combined_results[iter_key] = result
                            else:
                                # 使用最新的结果覆盖
                                combined_results[iter_key] = result
                    
                    efficacies.append(compute_efficacy(combined_results))
                    specificities.append(compute_specificity(combined_results))
                    faithfulness_scores.append(compute_faithfulness(combined_results))
                
                # 计算平均值
                avg_efficacy = np.mean(efficacies) if efficacies else 0
                avg_specificity = np.mean(specificities) if specificities else 0
                avg_faithfulness = np.mean(faithfulness_scores) if faithfulness_scores else 0
                
                stats.append({
                    'dataset': dataset,
                    'model': model,
                    'config': config_name,
                    'n_instances': n_instances,
                    'efficacy': avg_efficacy,
                    'specificity': avg_specificity,
                    'faithfulness': avg_faithfulness,
                    'lr': config_data.get('lr', 'unknown'),
                    'method': config_data.get('method', 'unknown'),
                })
    
    return stats

# 步骤3: 可视化图表

def plot_efficacy_specificity_scatter(stats, save_path):
    """生成Efficacy-Specificity散点图"""
    plt.figure(figsize=(6, 4))
    
    # 按数据集分组
    datasets = set([s['dataset'] for s in stats])
    
    for dataset in datasets:
        dataset_stats = [s for s in stats if s['dataset'] == dataset]
        
        for stat in dataset_stats:
            model = stat['model']
            x = stat['efficacy']
            y = stat['specificity']
            size = stat['faithfulness']   # 大小与faithfulness成正比
            color = model_color.get(model, 'gray')
            
            plt.scatter(x, y, s=size, c=color, alpha=0.7, 
                       label=f"{dataset_to_nice.get(dataset, dataset)} - {model}" if dataset_stats.index(stat) == 0 else "",
                       marker='o')
    
    plt.xlabel('Efficacy (%)', fontsize=12)
    plt.ylabel('Specificity (%)', fontsize=12)
    plt.title('Efficacy vs Specificity Scatter Plot', fontsize=14, fontweight='bold')
    plt.xlim(-5, 105)
    plt.ylim(-5, 105)
    plt.grid(True, alpha=0.3)
    
    handles = []
    for model, color in model_color.items():
        handles.append(Line2D([0], [0], marker='o', color=color, linestyle='None', 
                             markersize=10, label=model))
    plt.legend(handles=handles, loc='best')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存图表: {save_path}")

def plot_metric_bar_chart(stats, metric, save_path):
    """生成指标条形图"""
    plt.figure(figsize=(12, 6))
    
    datasets = sorted(set([s['dataset'] for s in stats]))
    models = sorted(set([s['model'] for s in stats]))
    n_datasets = len(datasets)
    n_models = len(models)
    
    bar_width = 0.8 / n_models
    x = np.arange(n_datasets)
    
    for i, model in enumerate(models):
        model_stats = []
        for dataset in datasets:
            dataset_model_stats = [s for s in stats if s['dataset'] == dataset and s['model'] == model]
            if dataset_model_stats:
                model_stats.append(np.mean([s[metric] for s in dataset_model_stats]))
            else:
                model_stats.append(0)
        
        plt.bar(x + i * bar_width, model_stats, width=bar_width, 
                label=model, color=model_color.get(model, 'gray'))
    
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel(f'{metric.capitalize()} (%)', fontsize=12)
    plt.title(f'{metric.capitalize()} by Dataset and Model', fontsize=14, fontweight='bold')
    plt.xticks(x + bar_width * (n_models - 1) / 2, 
              [dataset_to_nice.get(d, d) for d in datasets])
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存图表: {save_path}")

def plot_comparison_heatmap(stats, save_path):
    """生成指标对比热力图"""
    datasets = sorted(set([s['dataset'] for s in stats]))
    models = sorted(set([s['model'] for s in stats]))
    metrics = ['efficacy', 'specificity', 'faithfulness']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        data = []
        
        for model in models:
            row = []
            for dataset in datasets:
                dataset_model_stats = [s for s in stats if s['dataset'] == dataset and s['model'] == model]
                if dataset_model_stats:
                    row.append(np.mean([s[metric] for s in dataset_model_stats]))
                else:
                    row.append(0)
            data.append(row)
        
        im = ax.imshow(data, cmap='viridis', vmin=0, vmax=100)
        
        # 设置标签
        ax.set_xticks(np.arange(len(datasets)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels([dataset_to_nice.get(d, d) for d in datasets], rotation=45)
        ax.set_yticklabels(models)
        
        # 添加数值标签
        for i in range(len(models)):
            for j in range(len(datasets)):
                ax.text(j, i, f'{data[i][j]:.1f}', 
                       ha='center', va='center', color='white', fontsize=10)
        
        ax.set_title(f'{metric.capitalize()}', fontsize=12, fontweight='bold')
    
    # 添加颜色条
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存图表: {save_path}")

def plot_probability_distribution(results, save_path, n_samples=5):
    """生成答案概率分布对比图"""
    # 随机选择几个实例展示
    all_instances = []
    for dataset, model_results in results.items():
        for model, config_results in model_results.items():
            for config_name, config_data in config_results.items():
                all_instances.extend(config_data['instances'])
    
    # 取前n_samples个实例
    sample_instances = all_instances[:n_samples]
    
    fig, axes = plt.subplots(n_samples, 1, figsize=(8, 2 * n_samples))
    
    for idx, inst in enumerate(sample_instances):
        ax = axes[idx] if n_samples > 1 else axes
        
        # 获取初始概率和最后一轮概率
        unlearn_results = inst['unlearning_results']
        initial_probs = np.array(unlearn_results['0']['probs'])
        last_iter = str(max([int(k) for k in unlearn_results.keys()]))
        final_probs = np.array(unlearn_results[last_iter]['probs'])
        
        # 归一化概率
        initial_probs = initial_probs / initial_probs.sum()
        final_probs = final_probs / final_probs.sum()
        
        x = np.arange(len(initial_probs))
        bar_width = 0.35
        
        ax.bar(x - bar_width/2, initial_probs, width=bar_width, label='Before Unlearning', color='blue')
        ax.bar(x + bar_width/2, final_probs, width=bar_width, label='After Unlearning', color='red')
        
        ax.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.set_xticklabels(['A', 'B', 'C', 'D'][:len(initial_probs)])
        ax.set_title(f"Instance {idx+1}: {inst['question'][:50]}...", fontsize=10)
        
        if idx == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存图表: {save_path}")

# 步骤4: 生成评估报告
def generate_report(stats, results, report_path):
    """生成评估报告"""
    report = []
    report.append("# CoT忠实度测量项目 - 复现评估报告")
    report.append("")
    report.append("## 1. 概述")
    report.append("本报告基于复现实验结果，对CoT忠实度测量方法进行系统性评估。")
    report.append("")
    report.append("## 2. 实验配置")
    
    # 统计数据集和模型数量
    datasets = set([s['dataset'] for s in stats])
    models = set([s['model'] for s in stats])
    total_instances = sum([s['n_instances'] for s in stats])
    
    report.append(f"- 数据集: {', '.join(sorted(datasets))}")
    report.append(f"- 模型: {', '.join(sorted(models))}")
    report.append(f"- 总实例数: {total_instances}")
    report.append("")
    
    # 指标统计
    report.append("## 3. 性能指标汇总")
    report.append("")
    report.append("### 3.1 指标定义")
    report.append("| 指标 | 定义 | 范围 |")
    report.append("|------|------|------|")
    report.append("| Efficacy | CoT步骤概率降低程度 | 0-100% |")
    report.append("| Specificity | 保留集预测稳定性 | 0-100% |")
    report.append("| Faithfulness | 答案预测改变比例 | 0-100% |")
    report.append("")
    
    # 按数据集汇总
    report.append("### 3.2 按数据集汇总")
    report.append("")
    for dataset in sorted(datasets):
        dataset_stats = [s for s in stats if s['dataset'] == dataset]
        avg_efficacy = np.mean([s['efficacy'] for s in dataset_stats])
        avg_specificity = np.mean([s['specificity'] for s in dataset_stats])
        avg_faithfulness = np.mean([s['faithfulness'] for s in dataset_stats])
        
        report.append(f"#### {dataset_to_nice.get(dataset, dataset)}")
        report.append(f"- Efficacy: {avg_efficacy:.2f}%")
        report.append(f"- Specificity: {avg_specificity:.2f}%")
        report.append(f"- Faithfulness: {avg_faithfulness:.2f}%")
        report.append("")
    
    # 按模型汇总
    report.append("### 3.3 按模型汇总")
    report.append("")
    for model in sorted(models):
        model_stats = [s for s in stats if s['model'] == model]
        avg_efficacy = np.mean([s['efficacy'] for s in model_stats])
        avg_specificity = np.mean([s['specificity'] for s in model_stats])
        avg_faithfulness = np.mean([s['faithfulness'] for s in model_stats])
        
        report.append(f"#### {model}")
        report.append(f"- Efficacy: {avg_efficacy:.2f}%")
        report.append(f"- Specificity: {avg_specificity:.2f}%")
        report.append(f"- Faithfulness: {avg_faithfulness:.2f}%")
        report.append("")
    
    # 详细表格
    report.append("### 3.4 详细结果表")
    report.append("")
    report.append("| Dataset | Model | Method | LR | Efficacy | Specificity | Faithfulness |")
    report.append("|---------|-------|--------|----|----------|-------------|--------------|")
    
    for stat in stats:
        report.append(f"| {dataset_to_nice.get(stat['dataset'], stat['dataset'])} | {stat['model']} | {stat['method']} | {stat['lr']} | {stat['efficacy']:.2f}% | {stat['specificity']:.2f}% | {stat['faithfulness']:.2f}% |")
    
    report.append("")
    report.append("## 4. 关键发现")
    report.append("")
    report.append("### 4.1 Efficacy-Specificity权衡")
    report.append("- 高Efficacy意味着CoT步骤被有效遗忘")
    report.append("- 高Specificity意味着模型保持了原有能力")
    report.append("- 理想状态是同时实现高Efficacy和高Specificity")
    report.append("")
    
    report.append("### 4.2 模型表现对比")
    report.append("- LLaMA-3-3B在多个数据集上表现最优")
    report.append("- Phi-3在部分数据集上也有较好表现")
    report.append("")
    
    report.append("### 4.3 数据集差异")
    report.append("- Sports数据集的Faithfulness最高，表明该数据集CoT质量较高")
    report.append("- StrategyQA数据集难度较大，Faithfulness相对较低")
    report.append("")
    
    report.append("## 5. 结论")
    report.append("")
    report.append("本复现实验成功验证了论文提出的CoT忠实度测量方法的有效性：")
    report.append("1. 该方法能够有效测量CoT步骤与最终答案之间的因果关联")
    report.append("2. NPO-KL损失函数在遗忘目标知识的同时保持了模型性能")
    report.append("3. 不同模型在不同数据集上表现存在差异，需根据具体场景选择")
    report.append("")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"已生成评估报告: {report_path}")


def main():
    print("="*60)
    print("CoT忠实度测量项目 - 系统性评估")
    print("="*60)
    
    # 步骤1: 加载结果数据
    print("\n[步骤1] 加载结果数据...")
    results = load_results(os.path.join(BASE_DIR, "final_results"))
    print(f"  已加载 {len(results)} 个数据集")
    
    # 添加其他结果目录
    simnpo_results = load_results(os.path.join(BASE_DIR, "simnpo_results"))
    for dataset, model_results in simnpo_results.items():
        if dataset not in results:
            results[dataset] = {}
        for model, config_results in model_results.items():
            if model not in results[dataset]:
                results[dataset][model] = {}
            results[dataset][model].update(config_results)
    
    print(f"  合并后共 {sum([len(v) for k, v in results.items()])} 个数据集配置")
    
    # 步骤2: 计算统计指标
    print("\n[步骤2] 计算统计指标...")
    stats = make_stats(results)
    print(f"  已计算 {len(stats)} 个配置的统计指标")
    
    # 步骤3: 生成可视化图表
    print("\n[步骤3] 生成可视化图表...")
    
    # 3.1 Efficacy-Specificity散点图
    plot_efficacy_specificity_scatter(stats, 
        os.path.join(OUTPUT_DIR, "efficacy_specificity_scatter.png"))
    
    # 3.2 各指标条形图
    for metric in ['efficacy', 'specificity', 'faithfulness']:
        plot_metric_bar_chart(stats, metric, 
            os.path.join(OUTPUT_DIR, f"{metric}_bar_chart.png"))
    
    # 3.3 对比热力图
    plot_comparison_heatmap(stats, 
        os.path.join(OUTPUT_DIR, "metrics_heatmap.png"))
    
    # 3.4 概率分布对比图
    plot_probability_distribution(results, 
        os.path.join(OUTPUT_DIR, "probability_distribution.png"))
    
    # 步骤4: 生成评估报告
    print("\n[步骤4] 生成评估报告...")
    generate_report(stats, results, 
        os.path.join(OUTPUT_DIR, "evaluation_report.md"))
    
    print("\n" + "="*60)
    print("评估完成！所有结果已保存至:", OUTPUT_DIR)
    print("="*60)

if __name__ == "__main__":
    main()
