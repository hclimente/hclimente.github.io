import feedparser


def fetch_rss_feed(
    url: str = "http://feeds.bbci.co.uk/news/rss.xml",
    cuttoff_date: str = "2025-08-08T00:00:00Z",
):
    # Let feedparser do the heavy lifting of fetching and parsing.
    feed = feedparser.parse(url)

    # The 'entries' attribute is a list of all items, usually newest first.
    # We simply slice the first 20 items from this list.
    last_20_items = feed.entries[:20]

    # Check if the feed and entries were parsed correctly.
    if "bozo" in feed and feed.bozo == 1:
        print(f"Warning: The feed at {url} may be malformed.")
        # You might still get data, but it's good to know.

    if not last_20_items:
        print("Could not retrieve any items from the feed.")
    else:
        # Print the title and link for each of the 20 items.
        print(
            f"--- Displaying the latest {len(last_20_items)} items from '{feed.feed.title}' ---\n"
        )
        for index, item in enumerate(last_20_items, 1):
            # The item object acts like a dictionary. Common keys are 'title', 'link', 'summary', 'published'.
            print(f"{index}. Title: {item.title}")
            print(f"   Link: {item.link}\n")
