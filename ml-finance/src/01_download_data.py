#!/usr/bin/env python3
"""
Download AAPL stock data and prepare features for ML analysis
"""
import argparse
import logging
import os
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from utils import set_seed, date_utc_index, setup_logging
from config import DEFAULT_YEARS
import numpy as np


def is_file_fresh(filepath: str, max_age_days: int = 1, use_static_data: bool = True) -> bool:
    """Check if file exists and is fresher than max_age_days. If use_static_data=True, always return True if file exists."""
    if not os.path.exists(filepath):
        return False
    
    if use_static_data:
        return True  # Always use existing file for static data
    
    file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
    cutoff_time = datetime.now() - timedelta(days=max_age_days)
    
    return file_time > cutoff_time


def download_stock_data(ticker: str, years: int) -> pd.DataFrame:
    """Download stock data from yfinance"""
    logging.info(f"Downloading {ticker} data for {years} years...")

    # Download data
    stock = yf.Ticker(ticker)
    df = stock.history(period=f"{years}y", interval="1d")

    if df.empty:
        raise ValueError(f"No data found for ticker {ticker}")

    # Remove duplicate dates
    initial_len = len(df)
    df = df[~df.index.duplicated(keep='first')]
    if len(df) < initial_len:
        logging.info(f"Removed {initial_len - len(df)} duplicate dates")

    logging.info(f"Downloaded {len(df)} data points for {ticker}")
    return df


def download_earnings_dates(ticker: str) -> pd.DataFrame:
    """Download historical earnings dates for the ticker"""
    try:
        stock = yf.Ticker(ticker)
        earnings = stock.get_earnings_dates(limit=40)
        if earnings is not None and not earnings.empty:
            earnings_dates = earnings.index.normalize().tz_localize(None)
            earnings_df = pd.DataFrame({'earnings_date': 1}, index=earnings_dates)
            earnings_df = earnings_df[~earnings_df.index.duplicated(keep='first')]
            logging.info(f"Downloaded {len(earnings_df)} earnings dates for {ticker}")
            return earnings_df
    except Exception as e:
        logging.warning(f"Could not download earnings dates: {e}")
    return pd.DataFrame()


def create_features(df: pd.DataFrame, earnings_df: pd.DataFrame = None) -> pd.DataFrame:
    """Create target features: close, log_ret, rv_5, vix, qqq, earnings"""
    logging.info("Creating features...")

    # Ensure we have the required columns
    required_cols = ['Close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in data")

    # Create features
    df_features = df.copy()

    # Close price (already exists)
    df_features['close'] = df_features['Close']

    # Log returns: log(Close/Close.shift(1))
    df_features['log_ret'] = np.log(df_features['Close'] / df_features['Close'].shift(1))

    # Realized volatility: sqrt( (log_ret.rolling(5).var()) * 252 )
    df_features['rv_5'] = np.sqrt(df_features['log_ret'].rolling(5).var() * 252)

    # SNP500 change (if available)
    if 'snp500' in df_features.columns:
        df_features['snp500_change'] = df_features['snp500'].pct_change(fill_method=None)
        df_features['snp500_change'] = df_features['snp500_change'].fillna(0)

    # VIX (if available)
    if 'vix_close' in df_features.columns:
        df_features['vix_change'] = df_features['vix_close'].pct_change(fill_method=None).fillna(0)
    else:
        logging.warning("vix_close column not found, setting to 0")
        df_features['vix_close'] = 0.0
        df_features['vix_change'] = 0.0

    # QQQ change (if available)
    if 'qqq_close' in df_features.columns:
        df_features['qqq_change'] = df_features['qqq_close'].pct_change(fill_method=None).fillna(0)
    else:
        logging.warning("qqq_close column not found, setting to 0")
        df_features['qqq_change'] = 0.0

    # Earnings week binary feature
    df_features['earnings_week'] = 0
    if earnings_df is not None and not earnings_df.empty:
        for idx in df_features.index:
            idx_naive = idx.tz_localize(None) if idx.tzinfo else idx
            days_to_earnings = (earnings_df.index - idx_naive).days
            future_days = days_to_earnings[(days_to_earnings >= 0) & (days_to_earnings <= 7)]
            if len(future_days) > 0:
                df_features.loc[idx, 'earnings_week'] = 1
        logging.info(f"Marked {df_features['earnings_week'].sum():.0f} days as earnings_week=1")

    # Remove NA values (after feature calculation), drop only if essential features are NaN
    initial_rows = len(df_features)
    df_features = df_features.dropna(subset=['close', 'log_ret', 'rv_5'])

    logging.info(f"Removed {initial_rows - len(df_features)} rows with NA values")
    logging.info(f"Final dataset: {len(df_features)} rows")

    return df_features


def main():
    """Main function"""
    # Set seed for reproducibility
    set_seed(42)

    # Setup logging
    setup_logging()

    # Parse arguments
    parser = argparse.ArgumentParser(description='Download stock data and create features')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker (default: AAPL)')
    parser.add_argument('--years', type=int, default=DEFAULT_YEARS, help=f'Number of years of data (default: {DEFAULT_YEARS})')
    args = parser.parse_args()

    logging.info(f"Starting data download for {args.ticker} ({args.years} years)")

    # File paths
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    raw_data_path = os.path.join(data_dir, f'{args.ticker.lower()}.csv')
    features_path = os.path.join(data_dir, f'{args.ticker.lower()}_features.csv')

    # Check if raw data is fresh and has all required columns
    force_redownload = False
    if os.path.exists(raw_data_path):
        temp_df = pd.read_csv(raw_data_path, nrows=1)
        required_market_cols = ['snp500', 'vix_close', 'qqq_close']
        if any(col not in temp_df.columns for col in required_market_cols):
            force_redownload = True
            logging.info("Missing market data columns (VIX/QQQ), forcing re-download...")

    if is_file_fresh(raw_data_path, use_static_data=True) and not force_redownload:
        logging.info(f"Using static data file {raw_data_path}...")
        df = pd.read_csv(raw_data_path, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        logging.info(f"Downloading fresh data for {args.ticker}...")
        # Download stock data
        df_stock = download_stock_data(args.ticker, args.years)

        # Download SNP500 data (^GSPC is S&P 500 index)
        df_snp500 = download_stock_data('^GSPC', args.years)
        df_snp500 = df_snp500.rename(columns={'Close': 'snp500'})

        # Download VIX (^VIX — fear index)
        df_vix = download_stock_data('^VIX', args.years)
        df_vix = df_vix.rename(columns={'Close': 'vix_close'})

        # Download QQQ (Nasdaq-100 ETF — tech sector proxy)
        df_qqq = download_stock_data('QQQ', args.years)
        df_qqq = df_qqq.rename(columns={'Close': 'qqq_close'})

        # Merge on date index, keep all stock dates
        df = df_stock.copy()
        for market_df, col in [(df_snp500, 'snp500'), (df_vix, 'vix_close'), (df_qqq, 'qqq_close')]:
            df = pd.merge(df, market_df[[col]], left_index=True, right_index=True, how='left')
            df[col] = df[col].ffill().bfill()

        # Save raw data
        os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
        df.to_csv(raw_data_path)
        logging.info(f"Raw data saved to {raw_data_path}")

    # Convert to UTC datetime index
    df = date_utc_index(df, col='Date')

    # Download earnings dates
    earnings_df = download_earnings_dates(args.ticker)
    if not earnings_df.empty:
        earnings_path = os.path.join(data_dir, f'{args.ticker.lower()}_earnings.csv')
        earnings_df.to_csv(earnings_path)
        logging.info(f"Earnings dates saved to {earnings_path}")

    # Create features
    df_features = create_features(df, earnings_df)

    # Save features
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    df_features.to_csv(features_path)
    logging.info(f"Features saved to {features_path}")

    # Display summary
    logging.info("Data summary:")
    logging.info(f"Date range: {df_features.index.min()} to {df_features.index.max()}")
    logging.info(f"Total observations: {len(df_features)}")
    logging.info(f"Columns: {list(df_features.columns)}")

    # Basic statistics
    logging.info("Basic statistics for features:")
    for col in ['close', 'log_ret', 'rv_5']:
        if col in df_features.columns:
            desc = df_features[col].describe()
            logging.info(f"{col}: mean={desc['mean']:.6f}, std={desc['std']:.6f}, min={desc['min']:.6f}, max={desc['max']:.6f}")


if __name__ == "__main__":
    main()