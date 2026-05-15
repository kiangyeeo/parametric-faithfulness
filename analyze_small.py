import json
import glob
import numpy as np

files = glob.glob("simnpo final_results/openbook/*/simnpo_KL_sentencize*.out")
print(files)

for file in files:
    rows = [json.loads(line) for line in open(file)]

    flips = 0
    ff_soft = []

    for r in rows:
        pred0 = r["prediction"]
        results = r["unlearning_results"]
        last_key = str(max(map(int, results.keys())))
        pred_after = results[last_key]["prediction"]

        p0 = r["initial_probs"][pred0]
        p_after = results[last_key]["probs"][pred0]

        flips += int(pred_after != pred0)
        ff_soft.append(p0 - p_after)

    print("file:", file)
    print("num step results:", len(rows))
    print("step-level flip rate:", flips / len(rows))
    print("avg FF-SOFT:", float(np.mean(ff_soft)))