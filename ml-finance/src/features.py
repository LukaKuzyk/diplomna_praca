"""
Shared feature engineering for the ML Finance pipeline.
"""
import logging
import os
import numpy as np
import pandas as pd


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    # Avoid division by zero
    rs = gain / loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_atr(high, low, close, period=14):
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr


def calculate_cci(high, low, close, period=20):
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(period).mean()
    mad_tp = (tp - sma_tp).abs().rolling(period).mean()
    cci = (tp - sma_tp) / (0.015 * mad_tp)
    return cci


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create ML features from the dataset"""
    logging.info("Creating ML features...")

    features_df = df.copy()

    # Ensure index is DatetimeIndex for calendar features
    if not isinstance(features_df.index, pd.DatetimeIndex):
        logging.warning("Converting index to DatetimeIndex for calendar features")
        try:
            # Try with utc=True first for timezone-aware data
            features_df.index = pd.to_datetime(features_df.index, utc=True)
        except (ValueError, TypeError):
            # Fallback to regular conversion
            features_df.index = pd.to_datetime(features_df.index)

    # Load search data (prefer auto-generated, fallback to manual)
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    search_path = os.path.join(data_dir, 'search_data.csv')
    if not os.path.exists(search_path):
        search_path = os.path.join(data_dir, 'search_data_01.csv')
        logging.info("Using manually downloaded search data (search_data_01.csv)")

    if os.path.exists(search_path) and search_path.endswith('_01.csv'):
        # Manual CSV has 2 metadata rows before the header
        search_df = pd.read_csv(search_path, skiprows=2, header=0)
        search_df.columns = ['month', 'iphone_search', 'ai_search', 'election_search', 'trump_search', 'stock_search']
    else:
        search_df = pd.read_csv(search_path, index_col=0, parse_dates=True)
        search_df.index.name = 'month'
        search_df = search_df.reset_index()

    search_df['month'] = pd.to_datetime(search_df['month'], utc=True)
    search_df = search_df.set_index('month')
    search_daily = search_df.reindex(features_df.index, method='ffill')
    features_df = features_df.join(search_daily)

    # Load news data (prefer auto-generated, fallback to manual)
    news_path = os.path.join(data_dir, 'news_data.csv')
    if not os.path.exists(news_path):
        news_path = os.path.join(data_dir, 'news_data_01.csv')
        logging.info("Using manually downloaded news data (news_data_01.csv)")

    if os.path.exists(news_path) and news_path.endswith('_01.csv'):
        news_df = pd.read_csv(news_path, skiprows=2, header=0)
        news_df.columns = ['month', 'war_news', 'unemployment_news', 'tariffs_news', 'earnings_news', 'ai_news']
        for col in ['war_news', 'unemployment_news', 'tariffs_news', 'earnings_news', 'ai_news']:
            news_df[col] = news_df[col].replace('<1', 0.5).astype(float)
    else:
        news_df = pd.read_csv(news_path, index_col=0, parse_dates=True)
        news_df.index.name = 'month'
        news_df = news_df.reset_index()

    news_df['month'] = pd.to_datetime(news_df['month'], utc=True)
    news_df = news_df.set_index('month')
    news_daily = news_df.reindex(features_df.index, method='ffill')
    features_df = features_df.join(news_daily)

    # Lag features for search and news
    search_news_cols = ['iphone_search', 'ai_search', 'election_search', 'trump_search', 'stock_search',
                        'war_news', 'unemployment_news', 'tariffs_news', 'earnings_news', 'ai_news']
    for col in search_news_cols:
        for lag in [1, 2, 3]:
            features_df[f'{col}_lag_{lag}'] = features_df[col].shift(lag)

    # Lag features for log_ret
    for lag in [1, 2, 3, 5, 7, 10, 14, 15, 20, 21, 30]:
        features_df[f'log_ret_lag_{lag}'] = features_df['log_ret'].shift(lag)

    # Volume features
    features_df['volume'] = df['Volume']
    for lag in [1, 2, 5]:
        features_df[f'volume_lag_{lag}'] = features_df['volume'].shift(lag)

    # Rolling statistics
    features_df['rolling_skew_20'] = features_df['log_ret'].rolling(20).skew()
    features_df['rolling_kurt_20'] = features_df['log_ret'].rolling(20).kurt()

    # SNP500 change (assuming it's in df, or placeholder)
    # If not available, this will be NaN
    if 'snp500' in df.columns:
        features_df['snp500_change'] = df['snp500'].pct_change()
    else:
        # Placeholder, set to 0 or some value
        features_df['snp500_change'] = 0.0

    # Technical indicators
    # SMA(5) and SMA(20)
    features_df['sma_5'] = features_df['close'].rolling(5).mean()
    features_df['sma_20'] = features_df['close'].rolling(20).mean()

    # RSI(14)
    features_df['rsi_14'] = calculate_rsi(features_df['close'], 14)

    # MACD
    exp1 = features_df['close'].ewm(span=12, adjust=False).mean()
    exp2 = features_df['close'].ewm(span=26, adjust=False).mean()
    features_df['macd'] = exp1 - exp2
    features_df['macd_signal'] = features_df['macd'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    sma20 = features_df['close'].rolling(20).mean()
    std20 = features_df['close'].rolling(20).std()
    features_df['bb_upper'] = sma20 + 2 * std20
    features_df['bb_lower'] = sma20 - 2 * std20
    features_df['bb_middle'] = sma20

    # Stochastic Oscillator
    low14 = features_df['close'].rolling(14).min()
    high14 = features_df['close'].rolling(14).max()
    features_df['stoch_k'] = 100 * (features_df['close'] - low14) / (high14 - low14)
    features_df['stoch_d'] = features_df['stoch_k'].rolling(3).mean()

    # ATR
    features_df['atr_14'] = calculate_atr(features_df['High'], features_df['Low'], features_df['close'], 14)

    # CCI
    features_df['cci_20'] = calculate_cci(features_df['High'], features_df['Low'], features_df['close'], 20)

    # Momentum
    features_df['momentum_5'] = (features_df['close'] - features_df['close'].shift(5)) / features_df['close'].shift(5)
    features_df['momentum_10'] = (features_df['close'] - features_df['close'].shift(10)) / features_df['close'].shift(10)

    # Volume MA
    features_df['volume_ma_5'] = features_df['volume'].rolling(5).mean()
    features_df['volume_ma_20'] = features_df['volume'].rolling(20).mean()

    # Volatility feature (already exists as rv_5)
    features_df['volatility'] = features_df['rv_5']

    # Calendar features
    features_df['day_of_week'] = features_df.index.dayofweek  # 0=Monday, 4=Friday
    features_df['month'] = features_df.index.month  # 1-12

    # Fill NaN in features with 0
    features_df = features_df.fillna(0)

    # Remove rows with NaN values in essential columns
    initial_rows = len(features_df)
    features_df = features_df.dropna(subset=['close', 'log_ret', 'rv_5'])
    logging.info(f"Removed {initial_rows - len(features_df)} rows due to NaN values")

    return features_df
