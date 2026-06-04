"""跑完 --mmlu 后，把每个 cell 的 ~50 个准确率汇总成 Table 1 的 Gen 数。"""
import glob, json, os
import numpy as np

BASE_MMLU = {  # base 模型 0-shot MMLU（来自 mmlu.py 注释）
    "LLaMA-3-3B": 0.604,
    "Phi-3": 0.699,
}

for path in sorted(glob.glob("mmlu_results/*/*/*.out")):
    parts = path.split(os.sep)
    dataset, model = parts[1], parts[2]
    accs = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("mmlu_results") is not None:
                accs.append(float(r["mmlu_results"]))
    if not accs:
        continue
    mean = np.mean(accs)
    base = BASE_MMLU.get(model)
    drift = f"{(mean - base) * 100:+.1f}pp vs base" if base else ""
    print(f"{model:12s} {dataset:10s}  Gen = {mean*100:5.1f}%  "
          f"(n={len(accs)})  {drift}")