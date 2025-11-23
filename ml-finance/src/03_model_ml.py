#!/usr/bin/env python3
import argparse
import logging
import os
import warnings
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from utils import (
    set_seed, setup_logging, train_test_splits,
    evaluate_regression, directional_accuracy, save_predictions_csv
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Try to import XGBoost, fallback to RandomForest
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available, will use RandomForest as fallback")

# Try to import LightGBM
try:
    from lightgbm import LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

# Try to import CatBoost
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


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

    # RSI(14) - Relative Strength Index
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        # Avoid division by zero
        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        return rsi

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
    def calculate_atr(high, low, close, period=14):
        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr
    features_df['atr_14'] = calculate_atr(features_df['High'], features_df['Low'], features_df['close'], 14)

    # CCI
    def calculate_cci(high, low, close, period=20):
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(period).mean()
        mad_tp = (tp - sma_tp).abs().rolling(period).mean()
        cci = (tp - sma_tp) / (0.015 * mad_tp)
        return cci
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


def get_ml_models(random_state: int = 42) -> Dict[str, tuple]:
    """Get dictionary of ML models to compare"""
    models = {}

    # Linear Regression (baseline)
    models['linear'] = (LinearRegression(), StandardScaler())

    # Random Forest
    models['rf'] = (RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state
    ), StandardScaler())

    # XGBoost (if available)
    if XGBOOST_AVAILABLE:
        models['xgb'] = (XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state
        ), StandardScaler())
        # logging.info("XGBoost available for comparison")
    else:
        logging.warning("XGBoost not available, skipping XGBoost model")

    # Gradient Boosting Regressor
    models['gbr'] = (GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=random_state), StandardScaler())

    # LightGBM (if available)
    if LGBM_AVAILABLE:
        models['lgbm'] = (LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=random_state), StandardScaler())

    # CatBoost (if available)
    if CATBOOST_AVAILABLE:
        models['cat'] = (CatBoostRegressor(iterations=100, depth=5, learning_rate=0.05, random_state=random_state, verbose=False), StandardScaler())

    return models


def run_ml_walk_forward(train_window: int, test_window: int, step: int, ticker: str = 'AAPL') -> pd.DataFrame:
    """Run walk-forward validation for ML model"""
    logging.info("Starting ML walk-forward validation")

    # Load data
    data_path = os.path.join(os.path.dirname(__file__), 'data', f'{ticker.lower()}_features.csv')
    df = pd.read_csv(data_path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)

    # Create features
    df_features = create_features(df)

    # Define feature columns (exclude target and other non-feature columns)
    feature_cols = [
        'log_ret_lag_1',
        'log_ret_lag_2',
        'log_ret_lag_3', 'log_ret_lag_5',
        'log_ret_lag_7',
        'log_ret_lag_10',
        'log_ret_lag_14',
        'log_ret_lag_15', 'log_ret_lag_20',
        'log_ret_lag_21', 'log_ret_lag_30',
        'volume', 'volume_lag_1', 'volume_lag_2', 'volume_lag_5',
        'rolling_skew_20', 'rolling_kurt_20',
        'snp500_change',
        'sma_5', 'sma_20', 'rsi_14', 'macd', 'macd_signal',
        'bb_upper', 'bb_lower', 'bb_middle', 'stoch_k', 'stoch_d', 'volatility',
        'atr_14', 'cci_20', 'momentum_5', 'momentum_10', 'volume_ma_5', 'volume_ma_20',
        'day_of_week', 'month'
    ]

    # Check if all features exist
    missing_features = [col for col in feature_cols if col not in df_features.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    # Storage for all predictions
    all_predictions = []

    # Walk-forward validation
    for train_split, test_split, window_id in train_test_splits(
        df_features['log_ret'], train_window, test_window, step
    ):
        logging.info(f"Window {window_id}: train={len(train_split)}, test={len(test_split)}")

        # Get features for train/test
        train_features = df_features.loc[train_split.index, feature_cols]
        test_features = df_features.loc[test_split.index, feature_cols]

        # Skip if not enough data
        if len(train_features) < 50 or len(test_features) == 0:
            logging.warning(f"Skipping window {window_id} due to insufficient data")
            continue

        # Fit and predict with multiple models
        models = get_ml_models()
        predictions = {'date': test_split.index, 'y_true': test_split.values, 'window_id': window_id, 'target': 'log_ret'}

        for model_name, (model, scaler) in models.items():
            # Fit model
            X_scaled = scaler.fit_transform(train_features)
            model.fit(X_scaled, train_split)

            # Predict
            X_test_scaled = scaler.transform(test_features)
            y_pred = model.predict(X_test_scaled)

            predictions[f'y_pred_{model_name}'] = y_pred

        # Store predictions
        window_results = pd.DataFrame(predictions)
        all_predictions.append(window_results)

    # Combine all predictions
    if not all_predictions:
        raise ValueError("No valid predictions generated")

    results_df = pd.concat(all_predictions, ignore_index=True)

    # Calculate metrics for each model
    models = get_ml_models()
    for model_name in models.keys():
        pred_col = f'y_pred_{model_name}'
        mask = results_df['y_true'].notna() & results_df[pred_col].notna()
        if mask.sum() > 0:
            ml_metrics = evaluate_regression(
                results_df.loc[mask, 'y_true'],
                results_df.loc[mask, pred_col]
            )

            ml_da = directional_accuracy(
                results_df.loc[mask, 'y_true'],
                results_df.loc[mask, pred_col],
                threshold=0.0003
            )

            ml_metrics['Directional_Accuracy'] = ml_da

            logging.info(f"{model_name.upper()} Model Metrics: {ml_metrics}")

    return results_df


def main():
    """Main function"""
    set_seed(42)
    setup_logging()

    parser = argparse.ArgumentParser(description='Run ML models for log-return forecasting')
    parser.add_argument('--ticker', type=str, default='O', help='Stock ticker (default: AAPL)')
    parser.add_argument('--train_window', type=int, default=252,
                       help='Training window size (default: 252 ~1 year)')
    parser.add_argument('--test_window', type=int, default=30,
                       help='Test window size (default: 30 ~1 month)')
    parser.add_argument('--step', type=int, default=30,
                       help='Step size for walk-forward (default: 30 ~1 month)')

    args = parser.parse_args()

    logging.info("Running ML models for log-return forecasting")
    models = get_ml_models()
    logging.info(f"Comparing models: {list(models.keys())}")

    # Run walk-forward validation

    results_df = run_ml_walk_forward(args.train_window, args.test_window, args.step, args.ticker)

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), 'models', f'{args.ticker.lower()}_ml_predictions.csv')
    save_predictions_csv(output_path, results_df)

    logging.info("ML modeling completed")
    logging.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()