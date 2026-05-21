import json
import os
import numpy as np
from scipy import stats
from stats import make_stats, changed_prediction, compute_specificity, average_efficacy, average_mass_shift, max_mass_shift
from util import load_results, filter_for_agreement, filter_for_correctness

def analyze_single_file(filepath):
    """分析单个结果文件"""
    results = load_results(filepath)
    print(f"\n=== 分析文件: {filepath} ===")
    print(f"实例数量: {len(results)}")
    
    # 基础统计
    stats_dict = make_stats(results)
    print("\n核心指标:")
    print(f"  - 忠实度 (Faithfulness): {stats_dict['faithfulness']:.2f}%")
    print(f"  - 效能 (Efficacy): {stats_dict['efficacy']:.2f}%")
    print(f"  - 特异性 (Specificity): {stats_dict['specificity']:.2f}%")
    print(f"  - 实例数: {stats_dict['n_instances']}")
    print(f"  - CoT步数: {stats_dict['n_cot_steps']}")
    
    # 预测一致性分析
    agreement_results = filter_for_agreement(results)
    correct_results = filter_for_correctness(results)
    print(f"\n预测分析:")
    print(f"  - 初始预测与CoT预测一致: {len(agreement_results)}/{len(results)} ({len(agreement_results)/len(results)*100:.2f}%)")
    print(f"  - 预测正确且一致: {len(correct_results)}/{len(results)} ({len(correct_results)/len(results)*100:.2f}%)")
    
    # 质量位移分析
    mass_shifts = [average_mass_shift(r) for r in results]
    max_shifts = [max_mass_shift(r) for r in results]
    print(f"\n质量位移分析:")
    print(f"  - 平均质量位移: {np.mean(mass_shifts):.6f}")
    print(f"  - 最大质量位移: {np.mean(max_shifts):.6f}")
    print(f"  - 位移标准差: {np.std(mass_shifts):.6f}")
    
    # 统计显著性检验
    print(f"\n统计检验:")
    print(f"  - 质量位移t检验 (H0: mean=0):")
    t_stat, p_val = stats.ttest_1samp(mass_shifts, 0)
    print(f"    t={t_stat:.4f}, p={p_val:.4e}")
    
    return stats_dict

def analyze_all_results(base_dir):
    """分析所有结果文件"""
    all_stats = {}
    
    for dataset in os.listdir(base_dir):
        dataset_path = os.path.join(base_dir, dataset)
        if not os.path.isdir(dataset_path):
            continue
        
        for model in os.listdir(dataset_path):
            model_path = os.path.join(dataset_path, model)
            if not os.path.isdir(model_path):
                continue
            
            for filename in os.listdir(model_path):
                if not filename.endswith('.out'):
                    continue
                
                filepath = os.path.join(model_path, filename)
                key = f"{dataset}_{model}"
                
                try:
                    stats_dict = analyze_single_file(filepath)
                    stats_dict['filename'] = filename
                    all_stats[key] = stats_dict
                except Exception as e:
                    print(f"分析失败 {filepath}: {e}")
    
    return all_stats

def compare_results(stats_dict):
    """比较不同数据集和模型的结果"""
    print("\n" + "="*80)
    print("结果比较汇总")
    print("="*80)
    
    datasets = set(k.split('_')[0] for k in stats_dict.keys())
    models = set(k.split('_')[1] for k in stats_dict.keys())
    
    print(f"\n数据集: {', '.join(sorted(datasets))}")
    print(f"模型: {', '.join(sorted(models))}")
    
    # 按数据集分组
    print("\n按数据集统计:")
    for dataset in sorted(datasets):
        dataset_results = {k:v for k,v in stats_dict.items() if k.startswith(dataset)}
        if not dataset_results:
            continue
        
        faithfulness = [v['faithfulness'] for v in dataset_results.values()]
        efficacy = [v['efficacy'] for v in dataset_results.values()]
        specificity = [v['specificity'] for v in dataset_results.values()]
        
        print(f"\n{dataset}:")
        print(f"  忠实度: {np.mean(faithfulness):.2f}% ± {np.std(faithfulness):.2f}")
        print(f"  效能: {np.mean(efficacy):.2f}% ± {np.std(efficacy):.2f}")
        print(f"  特异性: {np.mean(specificity):.2f}% ± {np.std(specificity):.2f}")
    
    # 按模型分组
    print("\n按模型统计:")
    for model in sorted(models):
        model_results = {k:v for k,v in stats_dict.items() if k.endswith(model)}
        if not model_results:
            continue
        
        faithfulness = [v['faithfulness'] for v in model_results.values()]
        efficacy = [v['efficacy'] for v in model_results.values()]
        specificity = [v['specificity'] for v in model_results.values()]
        
        print(f"\n{model}:")
        print(f"  忠实度: {np.mean(faithfulness):.2f}% ± {np.std(faithfulness):.2f}")
        print(f"  效能: {np.mean(efficacy):.2f}% ± {np.std(efficacy):.2f}")
        print(f"  特异性: {np.mean(specificity):.2f}% ± {np.std(specificity):.2f}")

if __name__ == "__main__":
    # 使用脚本所在目录的绝对路径，避免工作目录问题
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "final_results")
    print("="*80)
    print("CoT忠实度评估分析")
    print("="*80)
    
    all_stats = analyze_all_results(base_dir)
    compare_results(all_stats)
    
    # 保存分析结果
    output_path = os.path.join(script_dir, 'analysis_summary.json')
    with open(output_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n分析结果已保存到 {output_path}")
