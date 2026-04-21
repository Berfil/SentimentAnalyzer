"""
Scrapes Yahoo! Japan News comments for a given search keyword.

Usage:
    python src/scraper.py --keyword "任天堂" --max_articles 5 --output data/scraped_comments.csv
"""

import argparse
import time
import re
import csv
from dataclasses import dataclass, fields
from urllib.parse import quote

import requests
import urllib3
from bs4 import BeautifulSoup

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en;q=0.9",
}

SEARCH_URL = "https://news.yahoo.co.jp/search?p={query}&ei=UTF-8"


@dataclass
class Comment:
    article_url: str
    article_title: str
    comment: str


def search_articles(keyword: str, max_articles: int) -> list[dict]:
    url = SEARCH_URL.format(query=quote(keyword))
    resp = requests.get(url, headers=HEADERS, timeout=10, verify=False)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    articles = []

    for a in soup.select("a[href]"):
        href = a["href"]
        # Yahoo News article links contain /articles/
        if "news.yahoo.co.jp/articles/" in href and href not in [x["url"] for x in articles]:
            title = a.get_text(strip=True)
            if title:
                articles.append({"url": href, "title": title})
        if len(articles) >= max_articles:
            break

    return articles


def scrape_comments(article_url: str, article_title: str) -> list[Comment]:
    # comments are on a separate page
    comment_url = article_url.rstrip("/") + "/comments"
    resp = requests.get(comment_url, headers=HEADERS, timeout=10, verify=False)

    if resp.status_code != 200:
        print(f"  Could not fetch comments ({resp.status_code}): {comment_url}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    comments = []

    # Yahoo Japan renders comment text in <p> tags whose class starts with "sc-169yn8p"
    # (styled-components hash — stable within a deployment)
    for el in soup.find_all("p", class_=re.compile(r"sc-169yn8p")):
        text = el.get_text(strip=True)
        if text and len(text) > 5:
            comments.append(Comment(
                article_url=article_url,
                article_title=article_title,
                comment=text,
            ))

    return comments


def run(keyword: str, max_articles: int, output: str):
    print(f"Searching Yahoo! Japan News for: {keyword}")
    articles = search_articles(keyword, max_articles)
    print(f"Found {len(articles)} articles")

    all_comments: list[Comment] = []

    for i, article in enumerate(articles, 1):
        print(f"  [{i}/{len(articles)}] {article['title'][:60]}")
        comments = scrape_comments(article["url"], article["title"])
        print(f"    → {len(comments)} comments")
        all_comments.extend(comments)
        time.sleep(1.5)  # polite crawl delay

    print(f"\nTotal comments collected: {len(all_comments)}")

    with open(output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in fields(Comment)])
        writer.writeheader()
        for c in all_comments:
            writer.writerow({
                "article_url": c.article_url,
                "article_title": c.article_title,
                "comment": c.comment,
            })

    print(f"Saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", required=True, help="Japanese search keyword")
    parser.add_argument("--max_articles", type=int, default=5)
    parser.add_argument("--output", default="data/scraped_comments.csv")
    args = parser.parse_args()
    run(args.keyword, args.max_articles, args.output)
