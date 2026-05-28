"""
Smoke tests for the sentiment model.
Tests that the model loads, returns valid structure, and handles edge cases.
Skipped automatically if fine-tuned weights are not present locally.
"""

import pytest
from pathlib import Path

WEIGHTS_PATH = Path("models/finetuned_cardiffnlp/best")
VALID_LABELS = {"positive", "neutral", "negative"}

requires_weights = pytest.mark.skipif(
    not WEIGHTS_PATH.exists(),
    reason="Fine-tuned weights not found — run models/finetune.py first",
)


@pytest.fixture(scope="module")
def classifier():
    from src.sentiment import load_model, FINETUNED_MODEL
    return load_model(FINETUNED_MODEL)


@requires_weights
def test_model_loads(classifier):
    assert classifier is not None


@requires_weights
def test_predict_returns_valid_label(classifier):
    from src.sentiment import predict
    result = predict(classifier, "このラーメンはとても美味しかったです！")
    assert "label" in result
    assert "score" in result
    assert result["label"] in VALID_LABELS
    assert 0.0 <= result["score"] <= 1.0


@requires_weights
def test_predict_batch_returns_correct_count(classifier):
    from src.sentiment import predict_batch
    texts = [
        "最高の体験でした。また来たいです。",   # likely positive
        "普通でした。特に感想はありません。",     # likely neutral
        "最悪でした。二度と行きません。",         # likely negative
    ]
    results = predict_batch(classifier, texts)
    assert len(results) == len(texts)
    for r in results:
        assert r["label"] in VALID_LABELS
        assert 0.0 <= r["score"] <= 1.0


@requires_weights
def test_predict_positive_text(classifier):
    from src.sentiment import predict
    result = predict(classifier, "素晴らしい商品です！大満足です！")
    assert result["label"] == "positive"


@requires_weights
def test_predict_negative_text(classifier):
    from src.sentiment import predict
    result = predict(classifier, "ひどい品質でした。お金の無駄です。")
    assert result["label"] == "negative"


@requires_weights
def test_predict_handles_long_text(classifier):
    from src.sentiment import predict
    long_text = "良い商品です。" * 100  # well over 512 tokens — tests truncation
    result = predict(classifier, long_text)
    assert result["label"] in VALID_LABELS


@requires_weights
def test_predict_handles_empty_string(classifier):
    from src.sentiment import predict
    result = predict(classifier, "")
    assert result["label"] in VALID_LABELS
