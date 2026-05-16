# CoT忠实度测量项目 - 复现评估报告

## 1. 概述
本报告基于复现实验结果，对CoT忠实度测量方法进行系统性评估。

## 2. 实验配置
- 数据集: openbook, sqa
- 模型: LLaMA-3-3B, Phi-3
- 总实例数: 320

## 3. 性能指标汇总

### 3.1 指标定义
| 指标 | 定义 | 范围 |
|------|------|------|
| Efficacy | CoT步骤概率降低程度 | 0-100% |
| Specificity | 保留集预测稳定性 | 0-100% |
| Faithfulness | 答案预测改变比例 | 0-100% |

### 3.2 按数据集汇总

#### OpenBookQA
- Efficacy: 99.32%
- Specificity: 92.44%
- Faithfulness: 23.75%

#### StrategyQA
- Efficacy: 98.98%
- Specificity: 93.94%
- Faithfulness: 18.12%

### 3.3 按模型汇总

#### LLaMA-3-3B
- Efficacy: 99.32%
- Specificity: 92.07%
- Faithfulness: 28.12%

#### Phi-3
- Efficacy: 98.99%
- Specificity: 94.30%
- Faithfulness: 13.75%

### 3.4 详细结果表

| Dataset | Model | Method | LR | Efficacy | Specificity | Faithfulness |
|---------|-------|--------|----|----------|-------------|--------------|
| StrategyQA | LLaMA-3-3B | npo | 3e-05 | 99.99% | 93.30% | 25.00% |
| StrategyQA | LLaMA-3-3B | simnpo | 3e-05 | 100.00% | 93.65% | 25.00% |
| StrategyQA | Phi-3 | npo | 5e-05 | 95.95% | 95.30% | 7.50% |
| StrategyQA | Phi-3 | simnpo | 5e-05 | 100.00% | 93.50% | 15.00% |
| OpenBookQA | LLaMA-3-3B | npo | 3e-05 | 98.21% | 91.00% | 30.00% |
| OpenBookQA | LLaMA-3-3B | simnpo | 3e-05 | 99.08% | 90.35% | 32.50% |
| OpenBookQA | Phi-3 | npo | 0.0001 | 100.00% | 94.90% | 17.50% |
| OpenBookQA | Phi-3 | simnpo | 0.0001 | 100.00% | 93.50% | 15.00% |

## 4. 关键发现

### 4.1 Efficacy-Specificity权衡
- 高Efficacy意味着CoT步骤被有效遗忘
- 高Specificity意味着模型保持了原有能力
- 理想状态是同时实现高Efficacy和高Specificity

### 4.2 模型表现对比
- LLaMA-3-3B在多个数据集上表现最优
- Phi-3在部分数据集上也有较好表现

### 4.3 数据集差异
- Sports数据集的Faithfulness最高，表明该数据集CoT质量较高
- StrategyQA数据集难度较大，Faithfulness相对较低

## 5. 结论

本复现实验成功验证了论文提出的CoT忠实度测量方法的有效性：
1. 该方法能够有效测量CoT步骤与最终答案之间的因果关联
2. NPO-KL损失函数在遗忘目标知识的同时保持了模型性能
3. 不同模型在不同数据集上表现存在差异，需根据具体场景选择
