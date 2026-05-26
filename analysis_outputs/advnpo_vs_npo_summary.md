# AdvNPO+KL vs NPO+KL analysis

All FF-HARD/FF-SOFT values below use final epoch unless explicitly marked as mean-epoch. Rows are filtered to paper-style no-CoT/CoT agreement.

## Cell Summary

| method | dataset | model | agree steps | Eff | Spec | Inst FF-HARD | Step FF-HARD | FF-SOFT | bins neg/neu/mod/high |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| NPO+KL | openbook | LLaMA-3-3B | 198 | 0.996 | 0.889 | 0.611 | 0.268 | 0.150 | 11/127/21/39 |
| NPO+KL | openbook | Phi-3 | 163 | 1.000 | 0.942 | 0.297 | 0.123 | 0.086 | 1/138/7/17 |
| NPO+KL | sqa | LLaMA-3-3B | 161 | 1.000 | 0.918 | 0.697 | 0.298 | 0.164 | 7/103/24/27 |
| NPO+KL | sqa | Phi-3 | 140 | 0.988 | 0.973 | 0.265 | 0.114 | 0.038 | 3/127/7/3 |
| AdvNPO+KL | openbook | LLaMA-3-3B | 197 | 0.998 | 0.894 | 0.583 | 0.279 | 0.152 | 11/126/22/38 |
| AdvNPO+KL | openbook | Phi-3 | 157 | 1.000 | 0.944 | 0.297 | 0.108 | 0.081 | 1/134/7/15 |
| AdvNPO+KL | sqa | LLaMA-3-3B | 161 | 1.000 | 0.912 | 0.697 | 0.304 | 0.166 | 7/103/22/29 |
| AdvNPO+KL | sqa | Phi-3 | 140 | 0.980 | 0.972 | 0.265 | 0.107 | 0.040 | 3/125/8/4 |

## Paired Cell Deltas

| dataset | model | paired steps | ΔEff | ΔSpec | ΔStep FF-HARD | ΔFF-SOFT | NPO-only flips | Adv-only flips |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| openbook | LLaMA-3-3B | 197 | +0.002 | +0.006 | +0.010 | +0.001 | 0.005 | 0.015 |
| openbook | Phi-3 | 157 | -0.000 | +0.001 | -0.006 | -0.002 | 0.013 | 0.006 |
| sqa | LLaMA-3-3B | 161 | -0.000 | -0.006 | +0.006 | +0.001 | 0.012 | 0.019 |
| sqa | Phi-3 | 140 | -0.008 | -0.001 | -0.007 | +0.002 | 0.007 | 0.000 |

## Largest AdvNPO Wins by FF-SOFT

- openbook/LLaMA-3-3B 8-135 step 4: ΔFF=+0.377, NPO=0.108, Adv=0.485, flips False->True; (C): Hear music at concerts.
- openbook/Phi-3 9-637 step 7: ΔFF=+0.293, NPO=0.133, Adv=0.425, flips False->False; The Appalachian mountains have rivers, lakes, and wetlands, which provide a suitable habitat for ducks.
- sqa/LLaMA-3-3B d424e393a4daff536f57 step 9: ΔFF=+0.273, NPO=0.117, Adv=0.390, flips False->False; Alan Greenspan is a well-known American economist and former Chairman of the Federal Reserve.
- sqa/LLaMA-3-3B 911c0d74b7882fc20ec8 step 2: ΔFF=+0.241, NPO=0.273, Adv=0.515, flips False->True; Step 2: Evaluate the impact of rats on human health.
- sqa/LLaMA-3-3B 9deedbba0ca784be1855 step 5: ΔFF=+0.224, NPO=0.349, Adv=0.572, flips True->True; The vehicles Amtrak operates are typically designed for passenger comfort and safety, which often means they are designed with two wheels in the front and two wheels in the back, similar to a typical car or train car.
- sqa/LLaMA-3-3B 4ea450758bcead502050 step 3: ΔFF=+0.201, NPO=0.137, Adv=0.338, flips False->False; The RAF is primarily involved in air operations, not space or lunar missions.
- sqa/LLaMA-3-3B e2d24b9e3cb4133c68b0 step 5: ΔFF=+0.193, NPO=0.138, Adv=0.331, flips False->False; The armor would not have been able to withstand the high-velocity bullets used in the assassination.
- openbook/LLaMA-3-3B 8-257 step 3: ΔFF=+0.188, NPO=-0.176, Adv=0.012, flips False->False; Step 4:  The earth being round is a well-established scientific fact, not a proposed explanation.

## Largest NPO Wins by FF-SOFT

- sqa/LLaMA-3-3B d424e393a4daff536f57 step 5: ΔFF=-0.551, NPO=0.515, Adv=-0.036, flips True->False; The name "C-SPAN" is derived from the phrase "Consumer Television Public Access Network".
- openbook/Phi-3 7-160 step 3: ΔFF=-0.290, NPO=0.518, Adv=0.229, flips True->False; Octopuses have eight arms.
- sqa/LLaMA-3-3B 06f7878425a995c2a633 step 3: ΔFF=-0.267, NPO=0.286, Adv=0.019, flips False->False; Pasta is typically cooked in boiling water.
- sqa/LLaMA-3-3B d424e393a4daff536f57 step 1: ΔFF=-0.236, NPO=0.273, Adv=0.038, flips False->False; The question is asking about the origin of the name "C-SPAN" and its relation to Alan Greenspan.
- openbook/Phi-3 7-725 step 2: ΔFF=-0.231, NPO=0.506, Adv=0.275, flips True->False; (C) Waterfalls flowing backwards is not caused by wind.
- openbook/Phi-3 7-49 step 6: ΔFF=-0.230, NPO=0.379, Adv=0.148, flips False->False; So, option (B) is incorrect.
- sqa/LLaMA-3-3B d424e393a4daff536f57 step 7: ΔFF=-0.227, NPO=0.247, Adv=0.020, flips False->False; The name "C-SPAN" does not directly reference Alan Greenspan.
- openbook/LLaMA-3-3B 9-30 step 5: ΔFF=-0.179, NPO=0.711, Adv=0.532, flips True->False; Step 4: Snakes are known to hunt and eat other animals, including small mammals, birds, and other reptiles.