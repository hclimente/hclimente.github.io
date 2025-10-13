import sqlite3

DB_PATH = "rss_sources.db"


def create_journal_table():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sources (
                name TEXT PRIMARY KEY,
                feed_url TEXT NOT NULL,
                last_checked DATE
            )
        """)

        # add entries
        sources = [
            ("Nature", "https://www.nature.com/nature.rss", "2025-10-05"),
        ]
        conn.executemany(
            """
            INSERT INTO sources (name, feed_url, last_checked)
            VALUES (?, ?, ?)
        """,
            sources,
        )


def create_articles_table():
    with sqlite3.connect(DB_PATH) as conn:
        # create table to store articles
        conn.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY,
                journal_name TEXT FOREIGN KEY REFERENCES sources(name),
                title TEXT NOT NULL,
                link TEXT NOT NULL,
                summary TEXT,
                date DATE NOT NULL,
                reviewed BOOLEAN DEFAULT 0,
                priority INTEGER DEFAULT 0
            )
        """)
