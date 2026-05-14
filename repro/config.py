"""
repro/config.py

把 2x2 复现的所有超参集中放一处。改实验跑哪几个组合、改样本数、改 lr，
都只动这个文件，避免到处搜命令行 flag。
"""

# ============== 2x2 子集组合 ==============
MODELS = {
    # short_name: huggingface model id
    "Phi-3":      "microsoft/Phi-3-mini-4k-instruct",
    "LLaMA-3-3B": "meta-llama/Llama-3.2-3B-Instruct",
}

DATASETS = ["openbook", "sqa"]

# 4 个 (model, dataset, lr) 组合 —— lr 沿用源仓库 const.py 里的 best lr
# 你们做 λ 消融时，可以在这里直接加多组 lr / λ
RUNS = [
    # (short_model_name,  dataset,    lr)
    ("Phi-3",      "openbook",  1e-4),
    ("Phi-3",      "sqa",       5e-5),
    ("LLaMA-3-3B", "openbook",  3e-5),
    ("LLaMA-3-3B", "sqa",       3e-5),
]

# ============== 规模相关 ==============
# 源仓库默认 250 条；我们 50 条样本，分 20 给 specificity 验证后只剩 30。
# 这里调成与样本量匹配的数字（也可以做 smoke test 时改 5/10）。
N_VERIFY  = 10   # specificity 验证集大小（hold-out）
N_UNLEARN = 40   # 进入 unlearning 的样本数（50 - N_VERIFY = 40）

# 复现 + 扩展通用配置
EPOCHS      = 5
SEED        = 1001
TEMPERATURE = 0.0
METHOD      = "npo_KL"        # 论文主结果用 NPO + KL
STRATEGY    = "sentencize"    # 必须显式传，源代码默认 'segmented' 是死分支
STEPWISE    = True            # 逐步 unlearn 每个 CoT step
POS_FILTER  = True            # 过滤 function words（论文消融最优配置）
FF2_ONLY    = True            # 只调 down_proj.weight（FF2）层

# ============== 路径 ==============
COT_CACHE_DIR = "final_cot"        # CoT 生成后的缓存（避免每次重跑生成）
RESULTS_DIR   = "final_results"    # unlearning 结果输出
SMOKE_DIR     = "smoke_results"    # smoke test 输出（小样本快速验证）
