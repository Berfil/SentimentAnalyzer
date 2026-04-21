"""
Scrapes @cosme (cosme.net) product reviews for a given keyword using Playwright.

Usage:
    python src/scraper_cosme.py --keyword "資生堂" --max_products 5 --output data/scraped_cosme.csv
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

SEARCH_URL = "https://www.cosme.net/search/"


@dataclass
class Review:
    product_url: str
    product_name: str
    comment: str


def clean_review_text(raw: str) -> str:
    raw = re.sub(r"^\d+購入品リピート\s*", "", raw)
    raw = re.sub(r"^\d+購入品\s*", "", raw)
    raw = re.sub(r"^\d+モニター[^\n]*\n", "", raw)
    raw = re.sub(r"\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}\s*", "", raw)
    raw = re.sub(r"続きを読む.*$", "", raw, flags=re.DOTALL)
    raw = re.sub(r"購入場所.*$", "", raw, flags=re.DOTALL)
    raw = re.sub(r"効果.*$", "", raw, flags=re.DOTALL)
    return raw.strip()


def scrape_product_reviews(page, product_url: str, product_name: str) -> list[Review]:
    review_url = product_url.rstrip("/") + "/review/"
    try:
        page.goto(review_url, wait_until="networkidle", timeout=20000)
    except PWTimeout:
        return []

    raw_texts = page.eval_on_selector_all(
        ".review-sec .body, .post-review .body, [class*=review] [class*=body]",
        "els => els.map(e => e.innerText.trim())"
    )

    reviews = []
    for raw in raw_texts:
        text = clean_review_text(raw)
        if len(text) > 15:
            reviews.append(Review(
                product_url=product_url,
                product_name=product_name,
                comment=text,
            ))
    return reviews


def get_product_name(page, product_url: str) -> str:
    try:
        page.goto(product_url, wait_until="networkidle", timeout=15000)
    except PWTimeout:
        return product_url.split("/")[-2]

    name = page.eval_on_selector(
        "h1, [class*=product-name], [class*=item-name]",
        "el => el.innerText.trim()"
    ) if page.query_selector("h1") else ""
    return (name or product_url.split("/")[-2])[:60]


def _extract_products_from_page(page, max_products: int) -> list[dict]:
    links = page.eval_on_selector_all(
        "a[href*='/products/']",
        "els => [...new Map(els.map(e => [e.href, e.innerText.trim()])).entries()].map(([href, text]) => ({href, text}))"
    )
    products = []
    seen = set()
    for link in links:
        href = link["href"]
        if href and "/review" not in href and "/photo" not in href and "/qa" not in href \
                and href not in seen and re.search(r"/products/\d+/$", href):
            seen.add(href)
            name = link["text"] or href.split("/")[-2]
            products.append({"url": href, "name": name[:60]})
        if len(products) >= max_products:
            break
    return products


def search_products(page, keyword: str, max_products: int) -> list[dict]:
    # Navigate directly with query param to bypass anti-bot form submission
    search_url = f"{SEARCH_URL}?fw={quote(keyword)}"
    try:
        page.goto(search_url, wait_until="networkidle", timeout=20000)
    except PWTimeout:
        return []

    # Try to find product links on the search results page
    try:
        page.wait_for_selector("a[href*='/products/']", timeout=8000)
    except PWTimeout:
        pass

    products = _extract_products_from_page(page, max_products)

    # Fallback: search returned a brand page — navigate to brand's product list
    if not products:
        all_hrefs = page.eval_on_selector_all(
            "a[href*='/brands/']",
            "els => els.map(e => e.href)"
        )
        brand_urls = [h for h in all_hrefs if re.search(r"/brands/\d+/$", h)]
        if brand_urls:
            brand_url = brand_urls[0]
            print(f"  (search → brand page {brand_url})")
            try:
                page.goto(brand_url, wait_until="networkidle", timeout=15000)
                products = _extract_products_from_page(page, max_products)
            except PWTimeout:
                pass

    return products


def run(keyword: str, max_products: int, output: str):
    print(f"Searching @cosme for: {keyword}")
    all_reviews: list[Review] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            locale="ja-JP",
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        products = search_products(page, keyword, max_products)
        print(f"Found {len(products)} products")

        for i, product in enumerate(products, 1):
            print(f"  [{i}/{len(products)}] {product['name'][:50]}")
            reviews = scrape_product_reviews(page, product["url"], product["name"])
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
    parser.add_argument("--output", default="data/scraped_cosme.csv")
    args = parser.parse_args()
    run(args.keyword, args.max_products, args.output)
