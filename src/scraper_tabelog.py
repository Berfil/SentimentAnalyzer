"""
Scrapes Tabelog restaurant reviews for a given brand/chain using Playwright.

Usage:
    python src/scraper_tabelog.py --keyword "モスバーガー" --max_restaurants 5 --output data/scraped_tabelog.csv
"""

import argparse
import csv
import re
import time
from dataclasses import dataclass, fields

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

SEARCH_URL = "https://tabelog.com/rstLst/?vs=1&sk={keyword}&sw={keyword}"


@dataclass
class Review:
    restaurant_url: str
    restaurant_name: str
    comment: str


def clean_review_text(raw: str) -> str:
    # strip username + 口コミ tag at the start (e.g. "グルメgirl\n口コミ" or "名前\n口コミN件...")
    raw = re.sub(r"^.{1,30}\n口コミ\S*\s*", "", raw)
    # strip username block: "名前 口コミN件 フォロワーN人"
    raw = re.sub(r".+?口コミ\d+件\s*フォロワー\d+人\s*", "", raw)
    # strip rating block: "4.0 ～￥999 1人 詳細 料理・味..."
    raw = re.sub(r"[\d.]+\s*[～￥\d,]+.*?詳細.*?(?=\d{4}/\d{2}訪問)", "", raw, flags=re.DOTALL)
    # strip visit date: "2024/10訪問 1 回目"
    raw = re.sub(r"\d{4}/\d{2}訪問\s*\d+\s*回目\s*", "", raw)
    # strip trailing social actions
    raw = re.sub(r"\n(いいね|お店を保存|口コミを|写真を|もっと見る).*$", "", raw, flags=re.DOTALL)
    return raw.strip()


def get_restaurant_urls(page, keyword: str, max_restaurants: int) -> list[dict]:
    url = SEARCH_URL.format(keyword=keyword)
    try:
        page.goto(url, wait_until="networkidle", timeout=15000)
    except PWTimeout:
        return []

    links = page.eval_on_selector_all(
        "a[href*='tabelog.com'][href*='/A']",
        "els => [...new Set(els.map(e => e.href))].filter(h => /[/]\\d{8}[/]$/.test(h))"
    )

    restaurants = []
    seen = set()
    for link in links:
        if link not in seen:
            seen.add(link)
            restaurants.append({"url": link, "name": ""})
        if len(restaurants) >= max_restaurants:
            break

    return restaurants


def scrape_restaurant_reviews(page, restaurant_url: str) -> tuple[str, list[Review]]:
    review_url = restaurant_url.rstrip("/") + "/dtlrvwlst/"
    try:
        page.goto(review_url, wait_until="networkidle", timeout=15000)
    except PWTimeout:
        return "", []

    # get restaurant name from page title
    name_el = page.query_selector("h2.rd-header__title-name, .rdheader-restaurant__name, h1[class*=restaurant], h2[class*=restaurant]")
    name = name_el.inner_text().strip() if name_el else restaurant_url

    raw_items = page.eval_on_selector_all(
        ".js-rvw-item-clickable-area",
        "els => els.map(e => e.innerText.trim())"
    )

    reviews = []
    for raw in raw_items:
        text = clean_review_text(raw)
        if len(text) > 20:
            reviews.append(Review(
                restaurant_url=restaurant_url,
                restaurant_name=name[:60],
                comment=text,
            ))

    return name, reviews


def run(keyword: str, max_restaurants: int, output: str):
    print(f"Searching Tabelog for: {keyword}")
    all_reviews: list[Review] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(locale="ja-JP", user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

        restaurants = get_restaurant_urls(page, keyword, max_restaurants)
        print(f"Found {len(restaurants)} restaurant locations")

        for i, restaurant in enumerate(restaurants, 1):
            name, reviews = scrape_restaurant_reviews(page, restaurant["url"])
            print(f"  [{i}/{len(restaurants)}] {name[:50]} → {len(reviews)} reviews")
            all_reviews.extend(reviews)
            time.sleep(1.0)

        browser.close()

    print(f"\nTotal reviews collected: {len(all_reviews)}")

    with open(output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in fields(Review)])
        writer.writeheader()
        for r in all_reviews:
            writer.writerow({"restaurant_url": r.restaurant_url, "restaurant_name": r.restaurant_name, "comment": r.comment})

    print(f"Saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", required=True)
    parser.add_argument("--max_restaurants", type=int, default=5)
    parser.add_argument("--output", default="data/scraped_tabelog.csv")
    args = parser.parse_args()
    run(args.keyword, args.max_restaurants, args.output)
