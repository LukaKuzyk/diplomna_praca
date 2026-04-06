#!/usr/bin/env python3
"""
Automatically download Google Trends data using pytrends.
Saves search_data.csv and news_data.csv into src/data/.
"""
import argparse
import logging
import os
import time
import random
from datetime import datetime, timedelta

import pandas as pd
from pytrends.request import TrendReq
from config_tickers import TICKER_CONFIG

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
    parser.add_argument('--ticker', type=str, default='AAPL', help='Ticker to download trends for. Use "ALL" for all of them.')
    parser.add_argument('--years', type=int, default=5, help='Number of years of history (default: 5)')
    parser.add_argument('--force', action='store_true', help='Force re-download even if recent files exist')
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    if args.ticker.upper() == 'ALL':
        tickers_to_process = list(TICKER_CONFIG.keys())
    else:
        if args.ticker.upper() not in TICKER_CONFIG:
            logging.error(f"Ticker {args.ticker} not found in config_tickers.py!")
            return
        tickers_to_process = [args.ticker.upper()]

    timeframe = build_timeframe(args.years)

    for current_ticker in tickers_to_process:
        search_path = os.path.join(DATA_DIR, f'{current_ticker}_search_data.csv')
        news_path = os.path.join(DATA_DIR, f'{current_ticker}_news_data.csv')

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
                logging.info(f"[{current_ticker}] Trends data is fresh (< 24h old). Use --force to re-download.")
                continue

        logging.info(f"[{current_ticker}] Downloading Google Trends data for timeframe: {timeframe}")

        config = TICKER_CONFIG[current_ticker]
        search_keywords = config["search_kw"]
        news_keywords = config["news_kw"]
        
        # Download search trends (Web Search)
        logging.info(f"[{current_ticker}] Downloading search trends: {search_keywords}")
        try:
            search_df = download_with_retry(search_keywords, timeframe)
            # Map dynamic columns to generic kw_1_search, kw_2_search, etc so features.py can process them without hardcoded names
            search_df.columns = [f'kw_{i+1}_search' for i in range(len(search_keywords))]
            search_df.index.name = 'month'
            search_df.to_csv(search_path)
            logging.info(f"[{current_ticker}] Saved search trends to {search_path} ({len(search_df)} rows)")
        except Exception as e:
            logging.error(f"[{current_ticker}] Failed to download search trends: {e}")

        # Small random delay between requests to avoid rate-limiting
        sleep_time = random.randint(15, 25)
        logging.info(f"[{current_ticker}] Sleeping {sleep_time}s to avoid Google rate limit...")
        time.sleep(sleep_time)

        # Download news trends (Google News, gprop='news')
        logging.info(f"[{current_ticker}] Downloading news trends: {news_keywords}")
        try:
            news_df = download_with_retry(news_keywords, timeframe, gprop='news')
            news_df.columns = [f'kw_{i+1}_news' for i in range(len(news_keywords))]
            news_df.index.name = 'month'
            news_df.to_csv(news_path)
            logging.info(f"[{current_ticker}] Saved news trends to {news_path} ({len(news_df)} rows)")
            
            sleep_time_end = random.randint(20, 35)
            logging.info(f"[{current_ticker}] Done with {current_ticker}. Sleeping {sleep_time_end}s before next ticker...")
            time.sleep(sleep_time_end)
        except Exception as e:
            logging.error(f"[{current_ticker}] Failed to download news trends: {e}")

    logging.info("Google Trends download completed successfully!")


if __name__ == "__main__":
    main()
