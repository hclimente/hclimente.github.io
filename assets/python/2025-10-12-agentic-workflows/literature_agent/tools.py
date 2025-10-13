import requests
import sqlite3
from typing import Optional

from .utils import fetch_rss_sources, fetch_rss_feed


def dump_articles():
    """
    Fetch articles from RSS feeds and store them in a database.
    """

    conn = sqlite3.connect("rss_sources.db")

    # append to a text file
    for source in fetch_rss_sources():
        print(f"Fetching articles from {source['name']} ({source['url']})")
        for article in fetch_rss_feed(
            url=source["url"],
            cuttoff_date=source["cutoff"],
            max_items=3,
        ):
            conn.execute(
                "INSERT INTO articles (journal, title, link, summary, date) VALUES (?, ?, ?, ?, ?)",
                (
                    source["name"],
                    article["title"],
                    article["link"],
                    article["summary"],
                    article["date"],
                ),
            )
            conn.commit()

        conn.execute(
            "UPDATE sources SET last_checked = CURRENT_DATE WHERE journal = ?",
            source["name"],
        )
        conn.commit()

    conn.close()


def fetch_url_content(url: str) -> Optional[str]:
    """
    Fetches the text content of a given URL.

    This function is designed to be a tool for an ADK agent,
    allowing it to read the content of a web page. It returns
    None if the request fails (e.g., bad URL, timeout, or 4xx/5xx status).

    Args:
        url: The full URL (e.g., 'https://www.example.com') to fetch.

    Returns:
        The text content of the web page as a string, or None on failure.
    """
    if not url.startswith(("http://", "https://")):
        # Skeptical check: Don't try to fetch a clearly invalid URL.
        print(f"Error: URL must start with 'http://' or 'https://'. Received: {url}")
        return None

    try:
        # A reasonable timeout is crucial for reliable agents
        response = requests.get(url, timeout=10)

        # Check for successful HTTP status codes (200-299)
        response.raise_for_status()

        return response.text

    except requests.exceptions.RequestException as e:
        # Tell it like it is: Something broke.
        print(f"Failed to fetch content from {url}. Error: {e}")
        return None
