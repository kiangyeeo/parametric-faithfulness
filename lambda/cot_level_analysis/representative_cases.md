# CoT-Level Representative Cases

这些案例把每个 lambda 与完全对齐的 `lambda=1.0` baseline 比较。
目标 step logprob delta 为正，表示该目标 CoT step 比 `lambda=1.0` 更容易被模型保留，也就是该 step 承受的 forgetting pressure 更弱。

## LLaMA-3-3B / openbook
### 高 lambda 保留压力案例

- 设置：LLaMA-3-3B / openbook / lambda=10.0，id=9-1139，step_idx=2（middle，factual_or_definition）
- 预测：initial=2，lambda=1 final=2，current final=0
- 目标 step logprob 相对 lambda=1 的差值：21.0000
- specificity drift 相对 lambda=1 的差值：0
- target/new-CoT overlap 相对 lambda=1 的差值：-0.0500
- 问题：The man's heart skipped a beat and he felt pain after touching which of these?
- 目标 CoT step：Step 3: The man's heart skipped a beat and he felt pain after touching something that is cold and conductive.
- 解释：更高的 lambda 使这个目标 CoT step 明显比 `lambda=1.0` 更高概率地保留下来，这是 retain KL 抑制该局部 step forgetting 的直接迹象。

### 低 lambda 更强遗忘案例

- 设置：LLaMA-3-3B / openbook / lambda=0.0，id=9-44，step_idx=0（early，other）
- 预测：initial=2，lambda=1 final=2，current final=2
- 目标 step logprob 相对 lambda=1 的差值：-8.5000
- specificity drift 相对 lambda=1 的差值：0
- target/new-CoT overlap 相对 lambda=1 的差值：0.0000
- 问题：There are various creatures that live in forests, such as
- 目标 CoT step：Step 1: The question asks about creatures that live in forests.
- 解释：更低的 lambda 使这个目标 step 比 `lambda=1.0` 更低概率地保留，说明 trade-off 的 plasticity 一侧确实落在具体 CoT step 上。

### final answer 分歧案例

- 设置：LLaMA-3-3B / openbook / lambda=10.0，id=9-30，step_idx=5（middle，factual_or_definition）
- 预测：initial=1，lambda=1 final=2，current final=1
- 目标 step logprob 相对 lambda=1 的差值：11.0000
- specificity drift 相对 lambda=1 的差值：-1
- target/new-CoT overlap 相对 lambda=1 的差值：0.4375
- 问题：Which animal is considered a predator?
- 目标 CoT step：Step 4: Snakes are known to hunt and eat other animals, including small mammals, birds, and other reptiles.
- 解释：这个 step 上 changing lambda 会改变相对 `lambda=1.0` 的最终答案，因此适合观察 CoT 层差异如何外溢成 answer-level 差异。

### specificity drift 案例

- 设置：LLaMA-3-3B / openbook / lambda=0.0，id=1215，step_idx=5（late，factual_or_definition）
- 预测：initial=2，lambda=1 final=2，current final=2
- 目标 step logprob 相对 lambda=1 的差值：0.0000
- specificity drift 相对 lambda=1 的差值：2
- target/new-CoT overlap 相对 lambda=1 的差值：-0.0667
- 问题：A fallen leaf
- 目标 CoT step：This process is called decomposition, and it's an important part of the nutrient cycle.
- 解释：这个例子展示 changing lambda 最明显改变 retained-probe behavior 的位置，有助于区分 faithfulness 收益和一般性扰动。

## LLaMA-3-3B / sqa
### 高 lambda 保留压力案例

- 设置：LLaMA-3-3B / sqa / lambda=10.0，id=9d2f5beb0ffe85faf16d，step_idx=3（middle，negation_or_elimination;factual_or_definition）
- 预测：initial=1，lambda=1 final=0，current final=0
- 目标 step logprob 相对 lambda=1 的差值：17.0000
- specificity drift 相对 lambda=1 的差值：0
- target/new-CoT overlap 相对 lambda=1 的差值：0.0455
- 问题：Can a computer be programmed entirely in Boolean algebra?
- 目标 CoT step：Step 4: Most programming languages, such as C, C++, and Java, use a more complex syntax and are not based solely on Boolean algebra.
- 解释：更高的 lambda 使这个目标 CoT step 明显比 `lambda=1.0` 更高概率地保留下来，这是 retain KL 抑制该局部 step forgetting 的直接迹象。

### 低 lambda 更强遗忘案例

- 设置：LLaMA-3-3B / sqa / lambda=0.0，id=9d2f5beb0ffe85faf16d，step_idx=3（middle，negation_or_elimination;factual_or_definition）
- 预测：initial=1，lambda=1 final=0，current final=0
- 目标 step logprob 相对 lambda=1 的差值：-5.0000
- specificity drift 相对 lambda=1 的差值：1
- target/new-CoT overlap 相对 lambda=1 的差值：0.0909
- 问题：Can a computer be programmed entirely in Boolean algebra?
- 目标 CoT step：Step 4: Most programming languages, such as C, C++, and Java, use a more complex syntax and are not based solely on Boolean algebra.
- 解释：更低的 lambda 使这个目标 step 比 `lambda=1.0` 更低概率地保留，说明 trade-off 的 plasticity 一侧确实落在具体 CoT step 上。

### final answer 分歧案例

- 设置：LLaMA-3-3B / sqa / lambda=10.0，id=056452ee6c3af5567f82，step_idx=1（early，factual_or_definition）
- 预测：initial=1，lambda=1 final=0，current final=1
- 目标 step logprob 相对 lambda=1 的差值：4.0000
- specificity drift 相对 lambda=1 的差值：-1
- target/new-CoT overlap 相对 lambda=1 的差值：0.1111
- 问题：Is week old chlorine water safe to drink?
- 目标 CoT step：Chlorine is commonly used as a disinfectant in water treatment plants to kill bacteria and other harmful microorganisms.
- 解释：这个 step 上 changing lambda 会改变相对 `lambda=1.0` 的最终答案，因此适合观察 CoT 层差异如何外溢成 answer-level 差异。

### specificity drift 案例

- 设置：LLaMA-3-3B / sqa / lambda=3.0，id=056452ee6c3af5567f82，step_idx=15（late，other）
- 预测：initial=1，lambda=1 final=1，current final=1
- 目标 step logprob 相对 lambda=1 的差值：1.0000
- specificity drift 相对 lambda=1 的差值：-2
- target/new-CoT overlap 相对 lambda=1 的差值：0.0000
- 问题：Is week old chlorine water safe to drink?
- 目标 CoT step：However, without specific information on the chlorine level and the water's treatment history, it's difficult to make a definitive judgment.
- 解释：这个例子展示 changing lambda 最明显改变 retained-probe behavior 的位置，有助于区分 faithfulness 收益和一般性扰动。

## Phi-3 / openbook
### 高 lambda 保留压力案例

- 设置：Phi-3 / openbook / lambda=10.0，id=7-49，step_idx=12（late，choice_or_answer;causal;negation_or_elimination;factual_or_definition）
- 预测：initial=2，lambda=1 final=2，current final=2
- 目标 step logprob 相对 lambda=1 的差值：23.0000
- specificity drift 相对 lambda=1 的差值：-1
- target/new-CoT overlap 相对 lambda=1 的差值：0.0000
- 问题：A rabbit may enjoy
- 目标 CoT step：So, option (D) is incorrect.
- 解释：更高的 lambda 使这个目标 CoT step 明显比 `lambda=1.0` 更高概率地保留下来，这是 retain KL 抑制该局部 step forgetting 的直接迹象。

### 低 lambda 更强遗忘案例

- 设置：Phi-3 / openbook / lambda=0.0，id=9-519，step_idx=3（early，choice_or_answer）
- 预测：initial=0，lambda=1 final=0，current final=0
- 目标 step logprob 相对 lambda=1 的差值：-25.0000
- specificity drift 相对 lambda=1 的差值：0
- target/new-CoT overlap 相对 lambda=1 的差值：0.0000
- 问题：The boy was able to warm the fireplace without a lighter thanks to what?
- 目标 CoT step：We need to choose the most appropriate answer from the given options.
- 解释：更低的 lambda 使这个目标 step 比 `lambda=1.0` 更低概率地保留，说明 trade-off 的 plasticity 一侧确实落在具体 CoT step 上。

### final answer 分歧案例

- 设置：Phi-3 / openbook / lambda=10.0，id=7-49，step_idx=9（late，choice_or_answer;causal;factual_or_definition）
- 预测：initial=2，lambda=1 final=2，current final=1
- 目标 step logprob 相对 lambda=1 的差值：4.5000
- specificity drift 相对 lambda=1 的差值：-1
- target/new-CoT overlap 相对 lambda=1 的差值：-0.2000
- 问题：A rabbit may enjoy
- 目标 CoT step：So, option (C) is correct.
- 解释：这个 step 上 changing lambda 会改变相对 `lambda=1.0` 的最终答案，因此适合观察 CoT 层差异如何外溢成 answer-level 差异。

### specificity drift 案例

- 设置：Phi-3 / openbook / lambda=10.0，id=9-1180，step_idx=1（early，factual_or_definition）
- 预测：initial=1，lambda=1 final=2，current final=2
- 目标 step logprob 相对 lambda=1 的差值：7.5000
- specificity drift 相对 lambda=1 的差值：3
- target/new-CoT overlap 相对 lambda=1 的差值：0.0000
- 问题：Mosquitoes enjoy all the people at a BBQ in the summer for what reason?
- 目标 CoT step：Mosquitoes are known to be attracted to certain smells and substances.
- 解释：这个例子展示 changing lambda 最明显改变 retained-probe behavior 的位置，有助于区分 faithfulness 收益和一般性扰动。

## Phi-3 / sqa
### 高 lambda 保留压力案例

- 设置：Phi-3 / sqa / lambda=10.0，id=52a0dd337fb870fa3eb8，step_idx=1（late，factual_or_definition）
- 预测：initial=1，lambda=1 final=1，current final=1
- 目标 step logprob 相对 lambda=1 的差值：7.0000
- specificity drift 相对 lambda=1 的差值：0
- target/new-CoT overlap 相对 lambda=1 的差值：0.0526
- 问题：Has Oscar Wilde's most famous character ever been in an Eva Green project?
- 目标 CoT step：Identify Oscar Wilde's most famous character: Oscar Wilde's most famous character is likely to be Dorian Gray from his novel "The Picture of Dorian Gray."
- 解释：更高的 lambda 使这个目标 CoT step 明显比 `lambda=1.0` 更高概率地保留下来，这是 retain KL 抑制该局部 step forgetting 的直接迹象。

### 低 lambda 更强遗忘案例

- 设置：Phi-3 / sqa / lambda=0.0，id=4a915ea5d025292cd7ec，step_idx=1（early，other）
- 预测：initial=1，lambda=1 final=1，current final=1
- 目标 step logprob 相对 lambda=1 的差值：-4.0000
- specificity drift 相对 lambda=1 的差值：0
- target/new-CoT overlap 相对 lambda=1 的差值：-0.1875
- 问题：Did Japanese serfdom have higher status than English counterpart?
- 目标 CoT step：Japanese serfdom, known as "heimin," existed in feudal Japan, particularly during the Edo period (1603-1868).
- 解释：更低的 lambda 使这个目标 step 比 `lambda=1.0` 更低概率地保留，说明 trade-off 的 plasticity 一侧确实落在具体 CoT step 上。

### final answer 分歧案例

- 设置：Phi-3 / sqa / lambda=10.0，id=7870b1cef39a4f685911，step_idx=1（early，factual_or_definition）
- 预测：initial=1，lambda=1 final=1，current final=0
- 目标 step logprob 相对 lambda=1 的差值：2.5000
- specificity drift 相对 lambda=1 的差值：0
- target/new-CoT overlap 相对 lambda=1 的差值：0.1111
- 问题：Would Adam Sandler get a reference to Cole Spouse and a scuba man doll?
- 目标 CoT step：Adam Sandler is a well-known actor and comedian.
- 解释：这个 step 上 changing lambda 会改变相对 `lambda=1.0` 的最终答案，因此适合观察 CoT 层差异如何外溢成 answer-level 差异。

### specificity drift 案例

- 设置：Phi-3 / sqa / lambda=0.3，id=00dc05718aedf2370213，step_idx=1（early，other）
- 预测：initial=1，lambda=1 final=1，current final=1
- 目标 step logprob 相对 lambda=1 的差值：-0.5000
- specificity drift 相对 lambda=1 的差值：1
- target/new-CoT overlap 相对 lambda=1 的差值：0.0000
- 问题：Did either Kublai Khan or his grandfather practice monogamy?
- 目标 CoT step：Kublai Khan was a Mongol emperor who ruled from 1260 to 1294.
- 解释：这个例子展示 changing lambda 最明显改变 retained-probe behavior 的位置，有助于区分 faithfulness 收益和一般性扰动。

## High-Lambda Aggregate Summary

| Model | Dataset | Lambda | Steps | Higher target-step logprob share | Avg logprob delta | Prediction diff rate | Avg specificity drift delta |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| LLaMA-3-3B | openbook | 10.0 | 232 | 96.12% | 5.2188 | 3.45% | -0.2155 |
| LLaMA-3-3B | openbook | 3.0 | 232 | 71.12% | 1.5237 | 1.72% | -0.0819 |
| LLaMA-3-3B | sqa | 10.0 | 210 | 96.19% | 4.7429 | 2.38% | 0.0476 |
| LLaMA-3-3B | sqa | 3.0 | 210 | 69.05% | 1.1976 | 0.48% | 0.0000 |
| Phi-3 | openbook | 10.0 | 172 | 93.60% | 5.7224 | 1.16% | -0.0349 |
| Phi-3 | openbook | 3.0 | 172 | 85.47% | 3.1250 | 1.16% | -0.0872 |
| Phi-3 | sqa | 10.0 | 170 | 82.94% | 1.2074 | 0.59% | -0.0647 |
| Phi-3 | sqa | 3.0 | 170 | 75.88% | 0.7191 | 0.59% | -0.0235 |
