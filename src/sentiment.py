from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

FINETUNED_MODEL = "models/finetuned_cardiffnlp/best"

MODELS = {
    FINETUNED_MODEL: {
        "labels": {0: "negative", 1: "neutral", 2: "positive"},
    },
    "koheiduck/bert-japanese-finetuned-sentiment": {
        "labels": {0: "negative", 1: "positive"},
    },
    "cl-tohoku/bert-base-japanese-v3": {
        "labels": None,
        "skip": True,
    },
    "cardiffnlp/twitter-xlm-roberta-base-sentiment": {
        "labels": {0: "negative", 1: "neutral", 2: "positive"},
    },
    "lxyuan/distilbert-base-multilingual-cased-sentiments-student": {
        "labels": {0: "negative", 1: "neutral", 2: "positive"},
    },
}


def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        truncation=True,
        max_length=512,
    )
    return classifier


def predict(classifier, text: str) -> dict:
    result = classifier(text)[0]
    return {"label": result["label"], "score": round(result["score"], 4)}


def predict_batch(classifier, texts: list[str]) -> list[dict]:
    results = classifier(texts, batch_size=16)
    return [{"label": r["label"], "score": round(r["score"], 4)} for r in results]
