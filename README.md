# AI-Generated Text Detection:
Multi-Task Learning And Low-Rank Adaptation For
Cross-Domain And Multiclass Classification

> **Paper**: *Advancements in AI-Generated Text Detection: Multi-Task Learning and Low-Rank Adaptation for Cross-Domain and Multiclass Classification*
> **Authors**: Anna Remizova, German Gritsai, Andrey Grabovoy
> **Date**: June 2025

## Overview

This project explores a hybrid approach for detecting AI-generated text using:

* **Multi-task learning (MTL)**
* **Low-Rank Adaptation (LoRA)**
* **Transformer architectures (DeBERTa, DistilRoBERTa)**

The model is optimized for:

* **Binary classification**: Human vs AI-generated text
* **Multiclass classification**: ChatGPT, Davinci, Cohere, Human

📊 Significant reductions in memory and training time are achieved with negligible loss (or even improvements) in performance.

📁 **Source dataset** [SemEval2024 Task 8 - Subtask B](https://github.com/mbzuai-nlp/SemEval2024-task8)


## Key Contributions

* Designed a **multihead DeBERTa model** to handle both classification tasks.
* Integrated **LoRA adapters** into classification heads to reduce computational load.
* Theoretically proved that **Rademacher complexity** of LoRA+MTL model is strictly lower than that of STL or base model.

## Architecture

* **Backbones**:

  * DistilRoBERTa (baseline & LoRA versions)
  * DeBERTa (multihead & LoRA variants)
* **LoRA** used to decompose weight updates as low-rank matrices (A \* B)
* **Multihead setup** for parallel training on binary + multiclass objectives

## Performance Summary

| Model                    | Task             | F1    | Precision | Recall |
| ------------------------ | ---------------- | ----- | --------- | ------ |
| Multihead DeBERTa        | Binary (H vs AI) | 0.971 | 0.977     | 0.970  |
| Multihead DeBERTa + LoRA | Binary (H vs AI) | 0.969 | 0.956     | 0.984  |
| Multihead DeBERTa        | Multiclass       | 0.896 | 0.895     | 0.893  |
| Multihead DeBERTa + LoRA | Multiclass       | 0.898 | 0.898     | 0.899  |

| **Model Name**                                                                 | **Runtime (sec)** |
|--------------------------------------------------------------------------------|-------------------|
| **Multiclass & Binary Classification, 5 epochs**                               |                   |
| DeBERTa (baseline)                                                             | 8205              |
| DeBERTa with LoRA                                                              | 4451              |
| **Previous experiments: Binary Classification (Human vs. AI)**                 |                   |
| DistilRoBERTa with LoRA (binary)                                               | 1616              |
| **Previous experiments: Multiclass Classification**                            |                   |
| DistilRoBERTa with LoRA                                                        | 3211              |
| DistilRoBERTa (baseline)                                                       | 4041              |

⏱ **Speed-up**: LoRA model trained **45,6% faster** than baseline, with **1-3% better metrics**

## Theoretical Guarantees

* Calculated lower empirical Rademacher complexity using LoRA + MTL
* Proven that Multitask + LoRA optimizes both generalization and speed

## Repository Contents

* `multitask_lora_model.py` — DeBERTa multihead model with LoRA
* `multitask_classification_model.py` — Baseline classification model setup
* `multitask_lora_notebook.ipynb` — Full training and evaluation notebook
* `Article.pdf` — Research paper with all theoretical and empirical results

## Getting Started

1. Clone the repo and install dependencies
2. Download and preprocess data from [SemEval2024 Task 8 Subtask B](https://github.com/mbzuai-nlp/SemEval2024-task8)
3. Run training notebook: `multitask_lora_notebook.ipynb`

## License

---

## Citation

If you use this code or article in your research, please cite:

```
@article{2025aidetectorloramultitask,
  title={Advancements in AI-Generated Text Detection: Multi-Task Learning and Low-Rank Adaptation},
  author={Remizova Anna, Gritsai German and Grabovoy Andrey},
  journal={Preprint},
  year={2025}
}
```
