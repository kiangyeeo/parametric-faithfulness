# Reproducing FUR: A Closer Look at Parametric Faithfulness

## 📖 About the Project

This project is a group assignment for the [NLP & LLM course (Spring 2026) ](https://github.com/baojian/llm-26) .

This project aims to conduct an in-depth investigation into parametric faithfulness in the reasoning steps of large language model Chain of Thought (CoT). We primarily focus on the FUR (Faithfulness by Unlearning Reasoning steps) method proposed by [Tutek et al. (2025)  ](https://aclanthology.org/2025.emnlp-main.504.pdf), and our core work is divided into the following two parts:

* **Reproduction**：Reproduce the core workflow and key metrics of FUR on scaled-down $2\times2$ model-dataset subsets (Phi-3-mini / LLaMA-3.2-3B and OpenBookQA / StrategyQA).
* **Extension**：Conduct an in-depth investigation of the loss function for FUR, including an analysis of how the weight ( $\lambda$ ) for regularization affects model performance, as well as an attempt to replace the original approach with alternative anti-supervised learning objectives (such as SimNPO or Gradient Ascent) to verify the robustness of the original method.

## 🔗 Documents & Links

**Project Proposal**: [Group1_Proposal.pdf](./docs/Group1_Proposal.pdf)

**Data**: [OBQA&SQA](./data)

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
- [ ] Environment setup 
- [ ] Run CoT prompting pipeline, record baseline accuracy
- [ ] Run no-CoT prompting pipeline as control
- [ ] NPO-KL stepwise unlearning experiments
- [ ] Implement Lanham mistake-injection baseline 
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

|   DDL    | Milestone                                                | Owner                               |
| :------: | :------------------------------------------------------- | :---------------------------------- |
| April 26 | Determine project and communication                      | All                                 |
| April 30 | Proposal and code&data                                   | All                                 |
|  May 13  | Environment setup + CoT / no-CoT baseline on 10 demos    | Junyan Liu                          |
|  May 16  | NPO unlearning & mistake baseline (parallel) on 10 demos | Tianle Chen / Kengyi Wang           |
|  May 17  | Metrics + visualization                                  | Wanyi Zhou                          |
|  May 21  | Midterm presentation                                     | Jialong Chen                        |
|  May 24  | Reproduction workflow on the full data sample            | All                                 |
| June 07  | Extension: $\lambda$ ablation                            | Jialong Chen                        |
| June 07  | NPO replacement experiments                              | Junyan Liu, Tianle Chen, Wanyi Zhou |
| June 14  | Final report                                             | Kengyi Wang                         |
| June 25  | Modify and submit                                        | All                                 |

## 🙏 Acknowledgments

This repository is an adaptation and extension based on the [Original Repository](https://github.com/technion-cs-nlp/parametric-faithfulness). For those interested, please refer to the original code. We would like to express our gratitude to the authors of the original paper, *[Measuring Chain of Thought Faithfulness by Unlearning Reasoning Steps](https://aclanthology.org/2025.emnlp-main.504.pdf)*, and their open-source repository. Their work provided a valuable reference baseline and essential code support for the reproduction and extension of this project.

