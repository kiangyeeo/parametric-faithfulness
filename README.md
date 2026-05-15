# Reproducing FUR: A Closer Look at Parametric Faithfulness

## 📖 About the Project

This project is a group assignment for the [NLP & LLM course (Spring 2026) ](https://github.com/baojian/llm-26) .

This project aims to conduct an in-depth investigation into parametric faithfulness in the reasoning steps of large language model Chain of Thought (CoT). We primarily focus on the FUR (Faithfulness by Unlearning Reasoning steps) method proposed by [Tutek et al. (2025)  ](https://aclanthology.org/2025.emnlp-main.504.pdf), and our core work is divided into the following two parts:

* **Reproduction**：Reproduce the core workflow and key metrics of FUR on scaled-down $2\times2$ model-dataset subsets (Phi-3-mini / LLaMA-3.2-3B and OpenBookQA / StrategyQA).
* **Extension**：Conduct an in-depth investigation of the loss function for FUR, including an analysis of how the weight ( $\lambda$ ) for regularization affects model performance, as well as an attempt to replace the original approach with alternative anti-supervised learning objectives (such as SimNPO or Gradient Ascent) to verify the robustness of the original method.

## 🔗 Documents & Links

**Project Proposal:** [Group1_Proposal.pdf](./docs/Group1_Proposal.pdf)

**Presentation Slides:** [parametric-faithfulness.pptx](./presentation/parametric-faithfulness.pptx)

**Data:** [OBQA&SQA](./data)

**Reproduction Results:** [CoT&noCoT results](./final_cot),  [unlearning results](./final_results),  [Add-mistake baseline results](./mistake_stats)

## 🚀 Quick Start Reproduction

1. **Set up the environment and install packages.** 

   ```bash
   cd repro
   conda activate pf
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
   ```

   **Tips:**  If you are using Qizhi (http://qz.cfff.fudan.edu.cn/), you can directly use the environment we make by `conda activate pf`.

2. **Set your Huggingface Token. Make sure you have the access to use Llama-3.2-3B.**

   ```bash
   export HF_TOKEN=hf_xxxxxx 
   ```

   **Tips:** Apply for your access to visit Llama-3.2-3B in https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct. If  you are using Qizhi (http://qz.cfff.fudan.edu.cn/), you can ignore this part because we have download the model on the server.

3. **Run the CoT&noCoT baseline before unlearning, and use GPU smoke test.**

   ```bash
   SMOKE=1 bash repro/run_all.sh
   ```

   The CoT&noCoT results will be saved in `./final_cot`. 
   Smoke test results will be saved in `./smoke_results`.

4. **Run the unlearning experiment.**

   ```bash
   bash repro/run_all.sh
   ```

   Unlearn results will be saved in `./final_results`.

5. **Mistake injection and Gemini-3-flash baseline**

   ```bash
   set GEMINI_API_KEY=your_google_api_key
   jupyter notebook "Adding mistakes repro.ipynb"
   python mistakes_repro.py --short_model Phi-3 --dataset openbook
   ```
   
   The notebook reads `./final_cot` and writes injected mistakes to `./mistake_results`. `mistakes_repro.py` evaluates them and writes results to `./mistake_stats`. Change `--short_model` and `--dataset` for other 2$\times$2 combinations.
   
6. Compute scores

## 👨‍💻 Team Members

This project was completed collaboratively by the following five students from *School of Data Science, Fudan University*:

* Jialong Chen
* Tianle Chen
* Junyan Liu
* Kengyi Wang
* Wanyi Zhou

## ✅ Progress Tracker

### Phase 1 · Reproduction

- [x] Data processing
- [x] Environment setup 
- [x] Run CoT prompting pipeline, record baseline accuracy
- [x] Run no-CoT prompting pipeline as control
- [x] NPO-KL stepwise unlearning experiments
- [x] Implement Lanham mistake-injection baseline 
- [ ] Compute FF-HARD / FF-SOFT metrics
- [ ] Generate plots to compare and show the results

### Phase 2 · Extension

- [ ] Design $\lambda$ sweep range & run ablation
- [ ] Run one alternative method under the same setup
- [ ] Compare against NPO
- [ ] Integrate results from both extension tracks
- [ ] Final report completed

---

## 🗓️ Timeline

|   DDL    | Milestone                                       | Owner                               |
| :------: | :---------------------------------------------- | :---------------------------------- |
| April 26 | Determine project and communication             | All                                 |
| April 30 | Proposal and code&data                          | All                                 |
|  May 14  | Environment setup + CoT generation + unlearning | Tianle Chen / Kengyi Wang           |
|  May 16  | Add mistake baseline                            | Junyan Liu                          |
|  May 17  | Metrics + visualization                         | Wanyi Zhou                          |
|  May 21  | Midterm presentation                            | Jialong Chen                        |
|  May 24  | Reproduction summary                            | All                                 |
| June 07  | Extension: $\lambda$ ablation                   | Jialong Chen                        |
| June 07  | NPO replacement experiments                     | Junyan Liu, Tianle Chen, Wanyi Zhou |
| June 14  | Final report                                    | Kengyi Wang                         |
| June 25  | Modify and submit                               | All                                 |

## 🙏 Acknowledgments

This repository is an adaptation and extension based on the [Original Repository](https://github.com/technion-cs-nlp/parametric-faithfulness). For those interested, please refer to the original code. We would like to express our gratitude to the authors of the original paper, *[Measuring Chain of Thought Faithfulness by Unlearning Reasoning Steps](https://aclanthology.org/2025.emnlp-main.504.pdf)*, and their open-source repository. Their work provided a valuable reference baseline and essential code support for the reproduction and extension of this project.

