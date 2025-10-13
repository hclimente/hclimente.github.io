import sqlite3
from time import strptime

import feedparser

from .db import DB_PATH


def fetch_rss_sources():
    """
    Fetch a list of RSS feed sources.
    Returns:
        dict: A dictionary of RSS feed sources with their names as keys and URLs as values.
    """

    # read from an sqlite database
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT name, feed_url, last_checked FROM sources"
        ).fetchall()

    if rows:
        for name, feed_url, last_checked in rows:
            yield {"name": name, "url": feed_url, "cutoff": last_checked}


def fetch_rss_feed(
    url: str,
    cuttoff_date: str = "2025-10-05",
    max_items: int = 3,
):
    """
    Fetch the latest news articles from an RSS feed and provide concise summaries.
    Args:
        url (str): The URL of the RSS feed to fetch.
        cuttoff_date (str): The cutoff date for articles in ISO 8601 format (YYYY-MM-DD). Articles published after this date will be included. Defaults to "2025-10-12".
        max_items (int): The maximum number of items to return. Defaults to 3.
    Returns:
        list: A list of dictionaries, each containing 'title', 'link', 'summary', and 'date' of an article.
    """

    cuttoff_date = strptime(cuttoff_date, "%Y-%m-%d")

    feed = feedparser.parse(url)

    # Check if the feed and entries were parsed correctly.
    if "bozo" in feed and feed.bozo == 1:
        print(f"Warning: The feed at {url} may be malformed.")
        # You might still get data, but it's good to know.

    for item in feed.entries[:max_items]:
        item_date = strptime(item.updated, "%Y-%m-%d")
        if item_date < cuttoff_date:
            break

        yield {
            "title": item.title,
            "link": item.link,
            "summary": item.summary,
            "date": item.updated,
        }
