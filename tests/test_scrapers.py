"""
Unit tests for scraper parsing logic.
Uses lightweight HTML fixtures — no network calls, no browser required.
"""

import pytest
from unittest.mock import MagicMock, patch
from bs4 import BeautifulSoup


# ── Yahoo scraper ──────────────────────────────────────────────────────────────

def test_yahoo_comment_dataclass_fields():
    from src.scraper import Comment
    c = Comment(
        article_url="https://news.yahoo.co.jp/articles/abc",
        article_title="テスト記事",
        comment="コメントのテスト",
    )
    assert c.article_url.startswith("https://")
    assert c.comment == "コメントのテスト"


def test_yahoo_comment_dataclass_has_required_fields():
    from src.scraper import Comment
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(Comment)}
    assert {"article_url", "article_title", "comment"} == field_names


# ── Twitter scraper ────────────────────────────────────────────────────────────

def test_twitter_tweet_dataclass_fields():
    from src.scraper_twitter import Tweet
    t = Tweet(
        tweet_url="https://twitter.com/i/web/status/123",
        author="testuser",
        comment="任天堂の新作が楽しみ",
    )
    assert t.tweet_url.startswith("https://")
    assert t.author == "testuser"
    assert t.comment == "任天堂の新作が楽しみ"


def test_twitter_tweet_dataclass_has_required_fields():
    from src.scraper_twitter import Tweet
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(Tweet)}
    assert {"tweet_url", "author", "comment"} == field_names


def test_twitter_run_returns_empty_on_invalid_token():
    from src.scraper_twitter import run
    result = run(
        keyword="任天堂",
        bearer_token="invalid_token_xyz",
        max_tweets=10,
        output="data/_test_twitter.csv",
    )
    assert isinstance(result, list)
    assert len(result) == 0


# ── Analyse pipeline ───────────────────────────────────────────────────────────

def test_normalise_known_labels():
    from src.analyse import normalise
    assert normalise("positive") == "positive"
    assert normalise("neutral") == "neutral"
    assert normalise("negative") == "negative"


def test_normalise_numeric_labels():
    from src.analyse import normalise
    assert normalise("0") == "negative"
    assert normalise("1") == "neutral"
    assert normalise("2") == "positive"


def test_normalise_uppercase():
    from src.analyse import normalise
    assert normalise("POSITIVE") == "positive"
    assert normalise("NEGATIVE") == "negative"


def test_normalise_unknown_returns_lowercase():
    from src.analyse import normalise
    assert normalise("UNKNOWN") == "unknown"
