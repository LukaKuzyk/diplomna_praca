#!/usr/bin/env python3
"""
Automatically download Google Trends data using pytrends.
Saves search_data.csv and news_data.csv into src/data/.
"""
import argparse
import logging
import os
import time
from datetime import datetime, timedelta

import pandas as pd
from pytrends.request import TrendReq


SEARCH_KEYWORDS = ['iPhone', 'ai', 'election', 'trump', 'stock']
NEWS_KEYWORDS = ['war', 'unemployment', 'tariffs', 'earnings', 'ai']

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def download_trends(keywords: list, timeframe: str, category: int = 0,
                    geo: str = '', gprop: str = '') -> pd.DataFrame:
    """Download Google Trends interest-over-time for a list of keywords.

    pytrends allows max 5 keywords per request.
    """
    pytrend = TrendReq(hl='en-US', tz=360)
    pytrend.build_payload(keywords, cat=category, timeframe=timeframe, geo=geo, gprop=gprop)

    df = pytrend.interest_over_time()
    if 'isPartial' in df.columns:
        df = df.drop(columns=['isPartial'])
    return df


def download_with_retry(keywords, timeframe, category=0, geo='', gprop='',
                        max_retries=3, delay=60):
    """Wrapper that retries on 429 (rate-limit) errors."""
    for attempt in range(1, max_retries + 1):
        try:
            return download_trends(keywords, timeframe, category, geo, gprop)
        except Exception as e:
            if '429' in str(e) and attempt < max_retries:
                logging.warning(f"Rate limited. Waiting {delay}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(delay)
            else:
                raise


def build_timeframe(years: int) -> str:
    """Build pytrends timeframe string for the last N years."""
    end = datetime.now()
    start = end - timedelta(days=years * 365)
    return f"{start.strftime('%Y-%m-%d')} {end.strftime('%Y-%m-%d')}"


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description='Download Google Trends data')
    parser.add_argument('--years', type=int, default=5, help='Number of years of history (default: 5)')
    parser.add_argument('--force', action='store_true', help='Force re-download even if recent files exist')
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    search_path = os.path.join(DATA_DIR, 'search_data.csv')
    news_path = os.path.join(DATA_DIR, 'news_data.csv')

    # Skip if files are fresh (< 1 day old) and --force not set
    if not args.force:
        all_fresh = True
        for path in [search_path, news_path]:
            if not os.path.exists(path):
                all_fresh = False
                break
            age_hours = (time.time() - os.path.getmtime(path)) / 3600
            if age_hours > 24:
                all_fresh = False
                break
        if all_fresh:
            logging.info("Trends data is fresh (< 24h old). Use --force to re-download.")
            return

    timeframe = build_timeframe(args.years)
    logging.info(f"Downloading Google Trends data for timeframe: {timeframe}")

    # Download search trends (Web Search)
    logging.info(f"Downloading search trends: {SEARCH_KEYWORDS}")
    search_df = download_with_retry(SEARCH_KEYWORDS, timeframe)
    search_df.columns = ['iphone_search', 'ai_search', 'election_search', 'trump_search', 'stock_search']
    search_df.index.name = 'month'
    search_df.to_csv(search_path)
    logging.info(f"Saved search trends to {search_path} ({len(search_df)} rows)")

    # Small delay between requests to avoid rate-limiting
    time.sleep(5)

    # Download news trends (Google News, gprop='news')
    logging.info(f"Downloading news trends: {NEWS_KEYWORDS}")
    news_df = download_with_retry(NEWS_KEYWORDS, timeframe, gprop='news')
    news_df.columns = ['war_news', 'unemployment_news', 'tariffs_news', 'earnings_news', 'ai_news']
    news_df.index.name = 'month'
    news_df.to_csv(news_path)
    logging.info(f"Saved news trends to {news_path} ({len(news_df)} rows)")

    logging.info("Google Trends download completed successfully!")


if __name__ == "__main__":
    main()
