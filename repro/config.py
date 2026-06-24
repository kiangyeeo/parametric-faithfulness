"""Central defaults for the 2x2 reproduction.

These values are the defaults consumed by ``repro.run_repro``; every field can
be overridden on the command line (see ``run_repro.make_cli``), so editing this
file is only needed to change the standing baseline.
"""

# ============== 2x2 subset ==============
MODELS = {
    # short_name: huggingface model id
    "Phi-3":      "microsoft/Phi-3-mini-4k-instruct",
    "LLaMA-3-3B": "meta-llama/Llama-3.2-3B-Instruct",
}

DATASETS = ["openbook", "sqa"]

# (short_model, dataset, lr) cells. lr values are the best LRs from the
# upstream const.py:dataset_model_best_lr table.
RUNS = [
    # (short_model_name,  dataset,    lr)
    ("Phi-3",      "openbook",  1e-4),
    ("Phi-3",      "sqa",       5e-5),
    ("LLaMA-3-3B", "openbook",  3e-5),
    ("LLaMA-3-3B", "sqa",       3e-5),
]

# ============== Scale ==============
# Upstream uses 250 instances; our subset is 50, of which 20 are held out for
# specificity, leaving 30 for unlearning. Adjust for smoke tests (e.g. 5/10).
N_VERIFY  = 20    # held-out specificity set size
N_UNLEARN = 230   # instances entering unlearning

# Shared reproduction + extension settings
EPOCHS      = 5
SEED        = 1001
TEMPERATURE = 0.0
METHOD      = "npo_KL"        # paper main result: NPO + KL
STRATEGY    = "sentencize"    # must be passed explicitly; upstream default 'segmented' is a dead branch
STEPWISE    = True            # unlearn each CoT step in turn
POS_FILTER  = True            # filter out function words (best ablation config)
FF2_ONLY    = True            # tune only down_proj.weight (FF2)

# ============== Paths ==============
COT_CACHE_DIR = "final_cot"        # cached generated CoT/noCoT (avoids regeneration)
RESULTS_DIR   = "final_results"    # unlearning output
SMOKE_DIR     = "smoke_results"    # smoke-test output (small, fast sanity check)
