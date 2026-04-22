"""
Fetches recent tweets mentioning a keyword using the Twitter/X API v2.

Requires a Bearer Token from the Twitter Developer Portal.
Pass it via --bearer_token or set the TWITTER_BEARER_TOKEN environment variable.

Usage:
    python src/scraper_twitter.py --keyword "任天堂" --max_tweets 100 --output data/scraped_twitter.csv
"""

import argparse
import csv
import os
import time
from dataclasses import dataclass, fields

import tweepy


@dataclass
class Tweet:
    tweet_url: str
    author: str
    comment: str


def run(keyword: str, bearer_token: str, max_tweets: int, output: str) -> list[Tweet]:
    client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)

    # Exclude retweets to avoid duplicate content; filter to Japanese
    query = f"{keyword} lang:ja -is:retweet"
    print(f"Searching Twitter/X for: {query}")

    tweets: list[Tweet] = []
    # API max per request: 100 (Basic+) or 10 (Free tier)
    per_page = min(max_tweets, 100)

    try:
        paginator = tweepy.Paginator(
            client.search_recent_tweets,
            query=query,
            tweet_fields=["author_id", "created_at", "text"],
            expansions=["author_id"],
            user_fields=["username"],
            max_results=per_page,
        )

        for page in paginator:
            if page.data is None:
                break

            # Build author_id → username map from the includes block
            user_map: dict[int, str] = {}
            if page.includes and "users" in page.includes:
                user_map = {u.id: u.username for u in page.includes["users"]}

            for tweet in page.data:
                author = user_map.get(tweet.author_id, str(tweet.author_id))
                tweets.append(Tweet(
                    tweet_url=f"https://twitter.com/i/web/status/{tweet.id}",
                    author=author,
                    comment=tweet.text,
                ))
                if len(tweets) >= max_tweets:
                    break

            if len(tweets) >= max_tweets:
                break

            time.sleep(1.0)

    except tweepy.errors.Unauthorized:
        print("Error: Invalid or expired Bearer Token.")
        return []
    except tweepy.errors.Forbidden as e:
        print(f"Error: Access forbidden — your API plan may not support this endpoint.\n{e}")
        return []
    except tweepy.errors.TweepyException as e:
        print(f"Twitter API error: {e}")
        return []

    print(f"Collected {len(tweets)} tweets")

    with open(output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in fields(Tweet)])
        writer.writeheader()
        for t in tweets:
            writer.writerow({"tweet_url": t.tweet_url, "author": t.author, "comment": t.comment})

    print(f"Saved to {output}")
    return tweets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", required=True, help="Search keyword e.g. 任天堂")
    parser.add_argument("--bearer_token", default=os.environ.get("TWITTER_BEARER_TOKEN"), help="Twitter API v2 Bearer Token")
    parser.add_argument("--max_tweets", type=int, default=100)
    parser.add_argument("--output", default="data/scraped_twitter.csv")
    args = parser.parse_args()

    if not args.bearer_token:
        parser.error("Provide --bearer_token or set the TWITTER_BEARER_TOKEN environment variable.")

    run(args.keyword, args.bearer_token, args.max_tweets, args.output)
