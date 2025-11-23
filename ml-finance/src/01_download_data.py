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


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create target features: close, log_ret, rv_5, snp500_change"""
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
        df_features['snp500_change'] = df_features['snp500_change'].fillna(0)  # Fill first NaN with 0
    else:
        logging.warning("snp500 column not found, setting snp500_change to 0")
        df_features['snp500_change'] = 0.0

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
    parser = argparse.ArgumentParser(description='Download AAPL data and create features')
    parser.add_argument('--ticker', type=str, default='MSFT', help='Stock ticker (default: AAPL)')
    parser.add_argument('--years', type=int, default=5, help='Number of years of data (default: 5)')
    args = parser.parse_args()

    logging.info(f"Starting data download for {args.ticker} ({args.years} years)")

    # File paths
    raw_data_path = f'data/{args.ticker.lower()}.csv'
    features_path = f'data/{args.ticker.lower()}_features.csv'

    # Check if raw data is fresh and has SNP500 data
    force_redownload = False
    if os.path.exists(raw_data_path):
        # Check if file has old US10Y data or missing SNP500
        temp_df = pd.read_csv(raw_data_path, nrows=1)
        if 'us10y' in temp_df.columns or 'snp500' not in temp_df.columns:
            force_redownload = True
            logging.info("Old data format detected, forcing re-download...")

    if is_file_fresh(raw_data_path, use_static_data=True) and not force_redownload:
        logging.info(f"Using static data file {raw_data_path}...")
        df = pd.read_csv(raw_data_path, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        logging.info(f"Downloading fresh data for {args.ticker}...")
        # Download AAPL data
        df_aapl = download_stock_data(args.ticker, args.years)

        # Download SNP500 data (^GSPC is S&P 500 index)
        df_snp500 = download_stock_data('^GSPC', args.years)
        df_snp500 = df_snp500.rename(columns={'Close': 'snp500'})

        # Merge on date index, keep all stock dates
        df = pd.merge(df_aapl, df_snp500[['snp500']], left_index=True, right_index=True, how='left')

        # Forward fill SNP500 values for non-trading days
        df['snp500'] = df['snp500'].fillna(method='ffill').fillna(method='bfill')

        # Save raw data
        os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
        df.to_csv(raw_data_path)
        logging.info(f"Raw data saved to {raw_data_path}")

    # Convert to UTC datetime index
    df = date_utc_index(df, col='Date')

    # Create features
    df_features = create_features(df)

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