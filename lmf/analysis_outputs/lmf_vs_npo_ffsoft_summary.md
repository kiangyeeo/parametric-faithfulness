# LMF+KL vs NPO+KL FF-SOFT comparison

Common step pairs: 784. CSV: `lmf_vs_npo_ffsoft_steps.csv`.

Read report `(dataset, model, npo_good, npo_bad, lmf_good, lmf_bad)`:

- ('openbook', 'LLaMA-3-3B', 236, 0, 232, 0)
- ('openbook', 'Phi-3', 178, 0, 172, 0)
- ('sqa', 'LLaMA-3-3B', 210, 0, 210, 0)
- ('sqa', 'Phi-3', 177, 12, 170, 0)

| dataset | model | steps | NPO FF | LMF FF | LMF-NPO | NPO flip | LMF flip | NPO spec | LMF spec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| openbook | LLaMA-3-3B | 232 | 0.1276 | 0.2877 | +0.1601 | 29.3% | 40.5% | 90.3% | 88.9% |
| openbook | Phi-3 | 172 | 0.0887 | 0.3577 | +0.2690 | 15.1% | 36.6% | 94.7% | 79.3% |
| sqa | LLaMA-3-3B | 210 | 0.1195 | 0.1634 | +0.0439 | 26.2% | 30.5% | 92.3% | 92.8% |
| sqa | Phi-3 | 170 | 0.0353 | 0.0920 | +0.0566 | 10.0% | 10.0% | 97.2% | 93.1% |

## Largest LMF wins
- openbook/LLaMA-3-3B 9-229 step 2: diff=+0.9653, NPO=-0.0725, LMF=0.8928; (C) A car with gasoline would not affect a bat's ability to fly.
- openbook/Phi-3 7-49 step 12: diff=+0.9550, NPO=-0.0428, LMF=0.9122; So, option (D) is incorrect.
- openbook/LLaMA-3-3B 9-229 step 0: diff=+0.9414, NPO=-0.0669, LMF=0.8746; (A) A rainy sky would make flying more difficult, not easier.
- openbook/Phi-3 9-519 step 5: diff=+0.9185, NPO=-0.0110, LMF=0.9076; Option (A) friction: Friction can generate heat when two objects rub against each other.
- openbook/Phi-3 8-257 step 9: diff=+0.9150, NPO=0.0374, LMF=0.9524; Option (D) "Gravity causes objects to fall" is a scientific law, not a hypothesis.

## Largest NPO wins
- sqa/LLaMA-3-3B 9975870880bf73b8644e step 13: diff=-0.8884, NPO=0.5195, LMF=-0.3688; **Conclusion**: Given the diversity of religious beliefs and the lack of a universal view on Armageddon, it's reasonable to conclude that not all religions look forward to Armageddon.
- openbook/Phi-3 889 step 3: diff=-0.8627, NPO=0.6570, LMF=-0.2056; Among the given options, a rat (A) and a pig (D) are omnivores, meaning they eat both plants and animals.
- openbook/Phi-3 7-160 step 7: diff=-0.8343, NPO=0.7333, LMF=-0.1010; Octopuses can squirt ink as a defense mechanism.
- openbook/LLaMA-3-3B 266 step 1: diff=-0.8291, NPO=0.9291, LMF=0.0999; - The cornea is the transparent layer at the front of the eye that helps focus light.
- sqa/LLaMA-3-3B 2c71a9a0b7d73c151442 step 4: diff=-0.8287, NPO=0.6375, LMF=-0.1912; Step 5:  It is likely that the Spanish learned about maize cultivation from the Native American tribes they encountered during their conquest.
