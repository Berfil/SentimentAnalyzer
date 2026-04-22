"""
Scrapes multiple sources for a keyword and runs sentiment analysis.

Usage:
    python src/analyse.py --keyword "モスバーガー" --source yahoo --max_articles 25
    python src/analyse.py --keyword "資生堂" --source cosme --max_products 5
    python src/analyse.py --keyword "モスバーガー" --source tabelog --max_restaurants 5
    python src/analyse.py --keyword "ソニー" --source kakaku --max_products 5
    python src/analyse.py --keyword "任天堂" --source twitter --max_tweets 100 --bearer_token YOUR_TOKEN
    python src/analyse.py --keyword "モスバーガー" --source all --max_articles 25 --max_restaurants 5
"""

import argparse
import io
import os
import sys
from datetime import date
from pathlib import Path
from collections import Counter

import pandas as pd

# Ensure UTF-8 output when called as a subprocess from the dashboard
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.scraper import run as scrape_yahoo
from src.scraper_cosme import run as scrape_cosme
from src.scraper_tabelog import run as scrape_tabelog
from src.scraper_kakaku import run as scrape_kakaku
from src.scraper_twitter import run as scrape_twitter
from src.sentiment import load_model, predict_batch, FINETUNED_MODEL

MODEL_NAME = FINETUNED_MODEL
LABEL_MAP = {"0": "negative", "1": "neutral", "2": "positive"}


def normalise(label: str) -> str:
    return LABEL_MAP.get(label, label.lower())


def print_summary(keyword: str, df: pd.DataFrame):
    counts = Counter(df["sentiment"])
    total = len(df)

    print(f"\n{'=' * 50}")
    print(f"Sentiment summary for: {keyword}")
    print(f"Comments analysed: {total}")
    print(f"{'=' * 50}")

    for label in ["positive", "neutral", "negative"]:
        n = counts.get(label, 0)
        pct = n / total * 100 if total else 0
        bar = "█" * int(pct / 3)
        print(f"  {label:<10}: {pct:5.1f}%  {bar}")

    print(f"\nTop negative comments (by confidence):")
    neg = df[df["sentiment"] == "negative"].nlargest(3, "score")
    for _, row in neg.iterrows():
        print(f"  [{row['score']:.2f}] {row['comment'][:80]}")

    print(f"\nTop positive comments (by confidence):")
    pos = df[df["sentiment"] == "positive"].nlargest(3, "score")
    for _, row in pos.iterrows():
        print(f"  [{row['score']:.2f}] {row['comment'][:80]}")


def collect_new_comments(keyword: str, source: str, max_articles: int, max_products: int, max_restaurants: int, max_tweets: int = 100, bearer_token: str | None = None) -> pd.DataFrame:
    frames = []

    if source in ("yahoo", "all"):
        tmp = "data/_yahoo_tmp.csv"
        scrape_yahoo(keyword=keyword, max_articles=max_articles, output=tmp)
        df = pd.read_csv(tmp, encoding="utf-8-sig")
        df["source"] = "yahoo"
        frames.append(df)
        Path(tmp).unlink(missing_ok=True)

    if source in ("cosme", "all"):
        tmp = "data/_cosme_tmp.csv"
        scrape_cosme(keyword=keyword, max_products=max_products, output=tmp)
        df = pd.read_csv(tmp, encoding="utf-8-sig")
        df = df.rename(columns={"product_url": "article_url", "product_name": "article_title"})
        df["source"] = "cosme"
        frames.append(df)
        Path(tmp).unlink(missing_ok=True)

    if source in ("tabelog", "all"):
        tmp = "data/_tabelog_tmp.csv"
        scrape_tabelog(keyword=keyword, max_restaurants=max_restaurants, output=tmp)
        df = pd.read_csv(tmp, encoding="utf-8-sig")
        df = df.rename(columns={"restaurant_url": "article_url", "restaurant_name": "article_title"})
        df["source"] = "tabelog"
        frames.append(df)
        Path(tmp).unlink(missing_ok=True)

    if source in ("kakaku", "all"):
        tmp = "data/_kakaku_tmp.csv"
        scrape_kakaku(keyword=keyword, max_products=max_products, output=tmp)
        df = pd.read_csv(tmp, encoding="utf-8-sig")
        df = df.rename(columns={"product_url": "article_url", "product_name": "article_title"})
        df["source"] = "kakaku"
        frames.append(df)
        Path(tmp).unlink(missing_ok=True)

    if source in ("twitter", "all"):
        token = bearer_token or os.environ.get("TWITTER_BEARER_TOKEN")
        if not token:
            print("Skipping Twitter: no bearer token provided (use --bearer_token or TWITTER_BEARER_TOKEN env var).")
        else:
            tmp = "data/_twitter_tmp.csv"
            scrape_twitter(keyword=keyword, bearer_token=token, max_tweets=max_tweets, output=tmp)
            df = pd.read_csv(tmp, encoding="utf-8-sig")
            df = df.rename(columns={"tweet_url": "article_url", "author": "article_title"})
            df["source"] = "twitter"
            frames.append(df)
            Path(tmp).unlink(missing_ok=True)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def run(keyword: str, source: str, max_articles: int, max_products: int, max_restaurants: int, output: str, max_tweets: int = 100, bearer_token: str | None = None):
    new_df = collect_new_comments(keyword, source, max_articles, max_products, max_restaurants, max_tweets, bearer_token)

    if new_df.empty:
        print("No comments collected.")
        return

    out_path = Path(output)
    if out_path.exists():
        existing_df = pd.read_csv(out_path, encoding="utf-8-sig")
        existing_comments = set(existing_df["comment"].tolist())
        new_df = new_df[~new_df["comment"].isin(existing_comments)]
        print(f"{len(new_df)} new comments after deduplication")
        if new_df.empty:
            print("No new comments to add.")
            return
    else:
        existing_df = pd.DataFrame()

    print(f"\nLoading model: {MODEL_NAME}")
    classifier = load_model(MODEL_NAME)

    print("Running sentiment analysis ...")
    results = predict_batch(classifier, new_df["comment"].tolist())

    new_df = new_df.copy()
    new_df["keyword"] = keyword
    new_df["scraped_date"] = date.today().isoformat()
    new_df["sentiment"] = [normalise(r["label"]) for r in results]
    new_df["score"] = [r["score"] for r in results]

    combined = pd.concat([existing_df, new_df], ignore_index=True)
    combined.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Results saved to {output} (total: {len(combined)} comments)")

    print_summary(keyword, new_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", required=True, help="Japanese search keyword e.g. 任天堂")
    parser.add_argument("--source", choices=["yahoo", "cosme", "tabelog", "kakaku", "twitter", "all"], default="yahoo")
    parser.add_argument("--max_articles", type=int, default=25, help="Yahoo: max articles to scrape")
    parser.add_argument("--max_products", type=int, default=5, help="@cosme/Kakaku: max products to scrape")
    parser.add_argument("--max_restaurants", type=int, default=5, help="Tabelog: max restaurant locations")
    parser.add_argument("--max_tweets", type=int, default=100, help="Twitter: max tweets to fetch")
    parser.add_argument("--bearer_token", default=os.environ.get("TWITTER_BEARER_TOKEN"), help="Twitter API v2 Bearer Token")
    parser.add_argument("--output", default="data/scraped_comments.csv")
    args = parser.parse_args()
    run(args.keyword, args.source, args.max_articles, args.max_products, args.max_restaurants, args.output, args.max_tweets, args.bearer_token)
