"""
Scrapes 5ch.net threads for a given keyword using the DAT file API.

5ch DAT format (one post per line, fields separated by <>):
    name <> email <> date_and_id <> message_html <> thread_title (first line only)

Usage:
    python src/scraper_5ch.py --keyword "モスバーガー" --max_threads 5 --output data/scraped_5ch.csv
"""

import argparse
import re
import time
import csv
from dataclasses import dataclass, fields
from urllib.parse import quote

import requests
import urllib3
from bs4 import BeautifulSoup

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Monazilla/1.00",
    "Accept-Language": "ja,en;q=0.9",
}

SEARCH_URL = "https://search.5ch.net/index.php?q={query}&type=subject"


@dataclass
class Post:
    thread_url: str
    thread_title: str
    post_number: int
    comment: str


def search_threads(keyword: str, max_threads: int) -> list[dict]:
    url = SEARCH_URL.format(query=quote(keyword))
    resp = requests.get(url, headers=HEADERS, timeout=10, verify=False)
    resp.encoding = resp.apparent_encoding

    soup = BeautifulSoup(resp.text, "html.parser")
    threads = []

    for a in soup.select("a[href*='5ch.net']"):
        href = a["href"]
        # match thread URLs like https://board.5ch.net/test/read.cgi/board/threadid/
        if "/test/read.cgi/" in href and href not in [t["url"] for t in threads]:
            title = a.get_text(strip=True)
            if title:
                threads.append({"url": href, "title": title})
        if len(threads) >= max_threads:
            break

    return threads


def thread_url_to_dat_url(thread_url: str) -> str | None:
    # https://board.5ch.net/test/read.cgi/BOARD/THREADID/
    # → https://board.5ch.net/BOARD/dat/THREADID.dat
    match = re.match(
        r"(https://[\w-]+\.5ch\.net)/test/read\.cgi/([\w]+)/(\d+)", thread_url
    )
    if not match:
        return None
    base, board, thread_id = match.groups()
    return f"{base}/{board}/dat/{thread_id}.dat"


def parse_dat(dat_text: str, thread_url: str, thread_title: str) -> list[Post]:
    posts = []
    for i, line in enumerate(dat_text.splitlines(), start=1):
        parts = line.split("<>")
        if len(parts) < 4:
            continue
        raw_message = parts[3]
        # strip HTML tags
        text = BeautifulSoup(raw_message, "html.parser").get_text(separator=" ").strip()
        # skip empty, very short, or anchor-only posts
        if len(text) < 5 or re.fullmatch(r"[>>0-9\s]+", text):
            continue
        posts.append(Post(
            thread_url=thread_url,
            thread_title=thread_title,
            post_number=i,
            comment=text,
        ))
    return posts


def scrape_thread(thread: dict) -> list[Post]:
    dat_url = thread_url_to_dat_url(thread["url"])
    if not dat_url:
        print(f"  Could not convert URL to DAT: {thread['url']}")
        return []

    resp = requests.get(dat_url, headers=HEADERS, timeout=10, verify=False)
    if resp.status_code != 200:
        print(f"  DAT fetch failed ({resp.status_code}): {dat_url}")
        return []

    resp.encoding = "shift_jis"
    return parse_dat(resp.text, thread["url"], thread["title"])


def run(keyword: str, max_threads: int, output: str):
    print(f"Searching 5ch for: {keyword}")
    threads = search_threads(keyword, max_threads)
    print(f"Found {len(threads)} threads")

    all_posts: list[Post] = []

    for i, thread in enumerate(threads, 1):
        print(f"  [{i}/{len(threads)}] {thread['title'][:60]}")
        posts = scrape_thread(thread)
        print(f"    → {len(posts)} posts")
        all_posts.extend(posts)
        time.sleep(1.0)

    print(f"\nTotal posts collected: {len(all_posts)}")

    with open(output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in fields(Post)])
        writer.writeheader()
        for p in all_posts:
            writer.writerow({
                "thread_url": p.thread_url,
                "thread_title": p.thread_title,
                "post_number": p.post_number,
                "comment": p.comment,
            })

    print(f"Saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", required=True)
    parser.add_argument("--max_threads", type=int, default=5)
    parser.add_argument("--output", default="data/scraped_5ch.csv")
    args = parser.parse_args()
    run(args.keyword, args.max_threads, args.output)
