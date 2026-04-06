"""
Centralized configuration for the ML Finance pipeline.
"""

DEFAULT_SEED = 42

DEFAULT_YEARS = 5

SIGNAL_THRESHOLD = 0.002

FEATURE_COLS = [
    'log_ret_lag_1', 'log_ret_lag_2', 'log_ret_lag_3', 'log_ret_lag_5',
    'log_ret_lag_7', 'log_ret_lag_10', 'log_ret_lag_14', 'log_ret_lag_15',
    'log_ret_lag_20', 'log_ret_lag_21', 'log_ret_lag_30',
    'volume', 'volume_lag_1', 'volume_lag_2', 'volume_lag_5',
    'rolling_skew_20', 'rolling_kurt_20',
    'sma_5', 'sma_20', 'rsi_14', 'macd', 'macd_signal',
    'bb_upper', 'bb_lower', 'bb_middle', 'stoch_k', 'stoch_d', 'volatility',
    'atr_14', 'cci_20', 'momentum_5', 'momentum_10', 'volume_ma_5', 'volume_ma_20',
    'day_of_week', 'month',
    'vix_close', 'vix_change', 'vix_change_lag_1', 'vix_change_lag_2', 'vix_change_lag_3',
    'qqq_change', 'qqq_change_lag_1', 'qqq_change_lag_2', 'qqq_change_lag_3',
    'snp500_change', 'snp500_change_lag_1', 'snp500_change_lag_2', 'snp500_change_lag_3',
    'earnings_week',
    'kw_1_search', 'kw_2_search', 'kw_3_search',
    'kw_1_news', 'kw_2_news', 'kw_3_news',
    'kw_1_search_lag_1', 'kw_1_search_lag_2', 'kw_1_search_lag_3',
    'kw_2_search_lag_1', 'kw_2_search_lag_2', 'kw_2_search_lag_3',
    'kw_3_search_lag_1', 'kw_3_search_lag_2', 'kw_3_search_lag_3',
    'kw_1_news_lag_1', 'kw_1_news_lag_2', 'kw_1_news_lag_3',
    'kw_2_news_lag_1', 'kw_2_news_lag_2', 'kw_2_news_lag_3',
    'kw_3_news_lag_1', 'kw_3_news_lag_2', 'kw_3_news_lag_3'
]
