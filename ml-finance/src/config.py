"""
Centralized configuration for the ML Finance pipeline.
"""

DEFAULT_SEED = 42

SIGNAL_THRESHOLD = 0.002

FEATURE_COLS = [
    'log_ret_lag_1', 'log_ret_lag_2', 'log_ret_lag_3', 'log_ret_lag_5',
    'log_ret_lag_7', 'log_ret_lag_10', 'log_ret_lag_14', 'log_ret_lag_15',
    'log_ret_lag_20', 'log_ret_lag_21', 'log_ret_lag_30',
    'volume', 'volume_lag_1', 'volume_lag_2', 'volume_lag_5',
    'rolling_skew_20', 'rolling_kurt_20',
    'snp500_change',
    'sma_5', 'sma_20', 'rsi_14', 'macd', 'macd_signal',
    'bb_upper', 'bb_lower', 'bb_middle', 'stoch_k', 'stoch_d', 'volatility',
    'atr_14', 'cci_20', 'momentum_5', 'momentum_10', 'volume_ma_5', 'volume_ma_20',
    'day_of_week', 'month',
    'iphone_search', 'ai_search', 'election_search', 'trump_search', 'stock_search',
    'war_news', 'unemployment_news', 'tariffs_news', 'earnings_news', 'ai_news',
    'iphone_search_lag_1', 'iphone_search_lag_2', 'iphone_search_lag_3',
    'ai_search_lag_1', 'ai_search_lag_2', 'ai_search_lag_3',
    'election_search_lag_1', 'election_search_lag_2', 'election_search_lag_3',
    'trump_search_lag_1', 'trump_search_lag_2', 'trump_search_lag_3',
    'stock_search_lag_1', 'stock_search_lag_2', 'stock_search_lag_3',
    'war_news_lag_1', 'war_news_lag_2', 'war_news_lag_3',
    'unemployment_news_lag_1', 'unemployment_news_lag_2', 'unemployment_news_lag_3',
    'tariffs_news_lag_1', 'tariffs_news_lag_2', 'tariffs_news_lag_3',
    'earnings_news_lag_1', 'earnings_news_lag_2', 'earnings_news_lag_3',
    'ai_news_lag_1', 'ai_news_lag_2', 'ai_news_lag_3'
]
