from google.adk.agents.llm_agent import Agent

from .tools import (
    fetch_rss_sources,
    fetch_rss_feed,
)

root_agent = Agent(
    model="gemini-2.5-flash",
    name="root_agent",
    description="A helpful assistant for getting scientific articles.",
    instruction="Fetch the latest scientific articles from an RSS feed and provide concise summaries.",
    tools=[fetch_rss_sources, fetch_rss_feed],
)
