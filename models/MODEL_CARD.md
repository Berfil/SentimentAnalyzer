# Model Card — KOKORO Sentiment Classifier

## Overview

| Field | Detail |
|---|---|
| **Author** | Filip Berndtsson |
| **Base model** | [`cardiffnlp/twitter-xlm-roberta-base-sentiment`](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment) |
| **Architecture** | XLM-RoBERTa (multilingual) |
| **Task** | 3-class sentiment classification: `negative` / `neutral` / `positive` |
| **Language** | Japanese (ja) |
| **Weights** | `models/finetuned_cardiffnlp/best/` |

---

## Why this base model?

`cardiffnlp/twitter-xlm-roberta-base-sentiment` was chosen because it was trained on multilingual Twitter data — short, informal, opinion-heavy text — which is structurally similar to Japanese news comment sections and review platforms. Standard Japanese BERT models (e.g. `cl-tohoku/bert-base-japanese`) were evaluated but performed worse on the informal, sometimes sarcastic tone of review platform comments.

---

## Training Data

Hand-labelled dataset of Japanese social media comments collected from Yahoo Japan News comment sections.

| Metric | Value |
|---|---|
| Total samples | 454 |
| Negative | 121 (26.7%) |
| Neutral | 214 (47.1%) |
| Positive | 119 (26.2%) |
| Labelled by | Filip Berndtsson |
| Source platforms | Yahoo Japan News |

Labels were assigned based on the overall sentiment of the comment toward the brand or topic being discussed, not the emotional tone of the writing in general (e.g. an angry comment complaining about a product is `negative` even if written humorously).

---

## Training Setup

| Hyperparameter | Value |
|---|---|
| Epochs | 5 (early stopping, patience=2) |
| Batch size (train) | 16 |
| Batch size (eval) | 32 |
| Learning rate | 2e-5 |
| Warmup ratio | 0.1 |
| Weight decay | 0.01 |
| Train / val split | 85% / 15% (stratified) |
| Loss function | Weighted cross-entropy (balanced class weights) |
| Best model selection | Highest macro F1 on validation set |
| Mixed precision | fp16 when GPU available |

Class weights were computed using sklearn's `balanced` strategy to counter the slight underrepresentation of positive and negative labels relative to neutral.

---

## Evaluation

Pre-fine-tuning, the base model agreed with human labels on **67.0%** (304/454) of the training set. After fine-tuning on the human-labelled split, the model is evaluated on the held-out 15% validation set using macro F1 as the primary metric (reported during training via `classification_report`).

To re-run evaluation:

```bash
python models/finetune.py --data data/labelled_comments.csv --output models/finetuned_cardiffnlp
```

---

## Limitations

- **Domain**: Trained exclusively on Yahoo Japan News comments. Performance may degrade on highly specialised vocabulary (e.g. medical, financial) or niche platforms.
- **Sarcasm**: Japanese sarcasm and indirect criticism are difficult to classify correctly and represent the main source of labelling errors.
- **Dataset size**: 454 samples is small. Accuracy improves noticeably with more labelled data — contributions welcome via `data/labelled_comments.csv`.
- **Neutral bias**: The dataset skews neutral (47%). The model may over-predict neutral for borderline cases.

---

## Reproducing the Fine-tune

```bash
# Install dependencies
pip install -r requirements.txt

# Run fine-tuning (saves best checkpoint to models/finetuned_cardiffnlp/best/)
python models/finetune.py \
  --data data/labelled_comments.csv \
  --output models/finetuned_cardiffnlp
```
