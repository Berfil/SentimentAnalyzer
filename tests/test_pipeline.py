"""
Integration smoke tests for the full scrape → classify pipeline.
Mocks collect_new_comments and the model so no network, browser, or GPU needed.
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

import src.analyse  # must be imported before patch() can resolve it

SAMPLE_DF = pd.DataFrame([
    {"article_url": "https://example.com/1", "article_title": "記事1", "comment": "とても良い商品です",  "source": "yahoo"},
    {"article_url": "https://example.com/2", "article_title": "記事2", "comment": "普通でした",          "source": "yahoo"},
    {"article_url": "https://example.com/3", "article_title": "記事3", "comment": "最悪でした",          "source": "yahoo"},
])

MOCK_PREDICTIONS = [
    {"label": "positive", "score": 0.92},
    {"label": "neutral",  "score": 0.78},
    {"label": "negative", "score": 0.88},
]


def test_pipeline_produces_output_csv(tmp_path):
    output = str(tmp_path / "output.csv")
    with patch("src.analyse.collect_new_comments", return_value=SAMPLE_DF.copy()), \
         patch("src.analyse.load_model", return_value=MagicMock()), \
         patch("src.analyse.predict_batch", return_value=MOCK_PREDICTIONS):
        src.analyse.run(keyword="テスト", source="yahoo", max_articles=3,
                        max_products=5, max_restaurants=5, output=output)

    assert Path(output).exists()
    df = pd.read_csv(output, encoding="utf-8-sig")
    assert len(df) == 3
    assert {"comment", "sentiment", "score"}.issubset(df.columns)


def test_pipeline_sentiment_values_are_valid(tmp_path):
    output = str(tmp_path / "output.csv")
    with patch("src.analyse.collect_new_comments", return_value=SAMPLE_DF.copy()), \
         patch("src.analyse.load_model", return_value=MagicMock()), \
         patch("src.analyse.predict_batch", return_value=MOCK_PREDICTIONS):
        src.analyse.run(keyword="テスト", source="yahoo", max_articles=3,
                        max_products=5, max_restaurants=5, output=output)

    df = pd.read_csv(output, encoding="utf-8-sig")
    assert set(df["sentiment"].unique()).issubset({"positive", "neutral", "negative"})


def test_pipeline_deduplicates_on_rerun(tmp_path):
    output = str(tmp_path / "output.csv")
    with patch("src.analyse.collect_new_comments", return_value=SAMPLE_DF.copy()), \
         patch("src.analyse.load_model", return_value=MagicMock()), \
         patch("src.analyse.predict_batch", return_value=MOCK_PREDICTIONS):
        src.analyse.run(keyword="テスト", source="yahoo", max_articles=3,
                        max_products=5, max_restaurants=5, output=output)
        # Second run with identical comments — should add nothing
        src.analyse.run(keyword="テスト", source="yahoo", max_articles=3,
                        max_products=5, max_restaurants=5, output=output)

    df = pd.read_csv(output, encoding="utf-8-sig")
    assert len(df) == 3  # still 3, not 6


def test_pipeline_stamps_keyword_and_date(tmp_path):
    output = str(tmp_path / "output.csv")
    with patch("src.analyse.collect_new_comments", return_value=SAMPLE_DF.copy()), \
         patch("src.analyse.load_model", return_value=MagicMock()), \
         patch("src.analyse.predict_batch", return_value=MOCK_PREDICTIONS):
        src.analyse.run(keyword="資生堂", source="yahoo", max_articles=3,
                        max_products=5, max_restaurants=5, output=output)

    df = pd.read_csv(output, encoding="utf-8-sig")
    assert "keyword" in df.columns
    assert "scraped_date" in df.columns
    assert (df["keyword"] == "資生堂").all()
