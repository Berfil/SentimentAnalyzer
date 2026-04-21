"""
Scrapes Kakaku.com (価格.com) product reviews for a given keyword using Playwright.

Usage:
    python src/scraper_kakaku.py --keyword "ソニー" --max_products 5 --output data/scraped_kakaku.csv
"""

import argparse
import csv
import io
import re
import sys
import time
from dataclasses import dataclass, fields
from urllib.parse import quote

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

SEARCH_URL = "https://search.kakaku.com/{keyword}/"
REVIEW_URL = "https://review.kakaku.com/review/{item_id}/"
ITEM_RE = re.compile(r"kakaku\.com/item/(K\d+)/")


@dataclass
class Review:
    product_url: str
    product_name: str
    comment: str


def clean_review_text(raw: str) -> str:
    # Remove date line e.g. "2026年2月13日 07:51 [2008939-1]"
    raw = re.sub(r"\d{4}年\d{1,2}月\d{1,2}日\s+\d{2}:\d{2}\s*\[[\d-]+\]\s*", "", raw)
    # Remove rating lines: "満足度\t5" etc.
    raw = re.sub(r"(満足度|操作性|表現力|携帯性|機能性|デザイン|コストパフォーマンス|食事|サービス|雰囲気|コスパ)\s*[\t\n]+[\d\.]+\s*", "", raw)
    # Remove score values left alone on lines
    raw = re.sub(r"^\d+(\.\d+)?\s*$", "", raw, flags=re.MULTILINE)
    # Strip leading/trailing whitespace from each line, then join
    lines = [l.strip() for l in raw.splitlines()]
    raw = "\n".join(l for l in lines if l)
    return raw.strip()


def get_item_name(page, item_url: str) -> str:
    try:
        page.goto(item_url, wait_until="load", timeout=15000)
    except PWTimeout:
        return item_url.split("/")[-2]
    title = page.title()
    # Title format: "価格.com - 製品名 価格比較"
    name = re.sub(r"\s*-\s*価格\.com.*$", "", title).strip()
    return name[:80] if name else item_url.split("/")[-2]


def scrape_item_reviews(page, item_id: str, product_name: str, product_url: str) -> list[Review]:
    review_url = REVIEW_URL.format(item_id=item_id)
    try:
        page.goto(review_url, wait_until="load", timeout=15000)
    except PWTimeout:
        return []

    raw_texts = page.eval_on_selector_all(
        ".reviewBoxWtInner",
        "els => els.map(e => e.innerText.trim())"
    )

    reviews = []
    for raw in raw_texts:
        text = clean_review_text(raw)
        if len(text) > 20:
            reviews.append(Review(
                product_url=product_url,
                product_name=product_name,
                comment=text,
            ))
    return reviews


def search_products(page, keyword: str, max_products: int) -> list[dict]:
    search_url = SEARCH_URL.format(keyword=quote(keyword))
    try:
        page.goto(search_url, wait_until="load", timeout=20000)
    except PWTimeout:
        return []

    links = page.eval_on_selector_all(
        "a[href*='/item/K']",
        "els => [...new Map(els.map(e => [e.href, e.innerText.trim()])).entries()].map(([href, text]) => ({href, text}))"
    )

    products = []
    seen = set()
    for link in links:
        href = link["href"]
        m = ITEM_RE.search(href)
        if not m or m.group(1) in seen:
            continue
        item_id = m.group(1)
        seen.add(item_id)
        name = link["text"][:80] or item_id
        products.append({"item_id": item_id, "url": f"https://kakaku.com/item/{item_id}/", "name": name})
        if len(products) >= max_products:
            break

    return products


def run(keyword: str, max_products: int, output: str):
    print(f"Searching Kakaku.com for: {keyword}")
    all_reviews: list[Review] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            locale="ja-JP",
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        )

        products = search_products(page, keyword, max_products)
        print(f"Found {len(products)} products")

        for i, product in enumerate(products, 1):
            # Get proper product name from item page
            name = get_item_name(page, product["url"])
            print(f"  [{i}/{len(products)}] {name[:60]}")
            reviews = scrape_item_reviews(page, product["item_id"], name, product["url"])
            print(f"    → {len(reviews)} reviews")
            all_reviews.extend(reviews)
            time.sleep(1.0)

        browser.close()

    print(f"\nTotal reviews collected: {len(all_reviews)}")

    with open(output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in fields(Review)])
        writer.writeheader()
        for r in all_reviews:
            writer.writerow({"product_url": r.product_url, "product_name": r.product_name, "comment": r.comment})

    print(f"Saved to {output}")


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", required=True)
    parser.add_argument("--max_products", type=int, default=5)
    parser.add_argument("--output", default="data/scraped_kakaku.csv")
    args = parser.parse_args()
    run(args.keyword, args.max_products, args.output)
