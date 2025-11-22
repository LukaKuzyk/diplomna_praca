#!/usr/bin/env python3
"""
ML models for AAPL log-return forecasting: XGBoost/RandomForest
"""
import argparse
import logging
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

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


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
    for lag in [1, 2, 5, 10]:
        features_df[f'log_ret_lag_{lag}'] = features_df['log_ret'].shift(lag)

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

    # Volatility feature (already exists as rv_5)
    features_df['volatility'] = features_df['rv_5']

    # Calendar features
    features_df['day_of_week'] = features_df.index.dayofweek  # 0=Monday, 4=Friday
    features_df['month'] = features_df.index.month  # 1-12

    # Remove rows with NaN values (due to lagging and rolling windows)
    initial_rows = len(features_df)
    features_df = features_df.dropna()
    logging.info(f"Removed {initial_rows - len(features_df)} rows due to NaN values")

    return features_df


class MLModel:
    """ML model wrapper with XGBoost/RandomForest fallback"""

    def __init__(self, use_xgboost: bool = True, random_state: int = 42):
        self.use_xgboost = use_xgboost and XGBOOST_AVAILABLE
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()

        if self.use_xgboost:
            logging.info("Using XGBoost model")
            self.model = XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=random_state
            )
        else:
            logging.info("Using RandomForest model (XGBoost not available)")
            self.model = RandomForestRegressor(
                n_estimators=400,
                max_depth=6,
                random_state=random_state
            )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Fit model
        self.model.fit(X_scaled, y)
        logging.info("Model fitted successfully")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


def run_ml_walk_forward(train_window: int, test_window: int, step: int) -> pd.DataFrame:
    """Run walk-forward validation for ML model"""
    logging.info("Starting ML walk-forward validation")

    # Load data
    data_path = 'data/aapl_features.csv'
    df = pd.read_csv(data_path, index_col=0)
    df.index = pd.to_datetime(df.index)

    # Create features
    df_features = create_features(df)

    # Define feature columns (exclude target and other non-feature columns)
    feature_cols = [
        'log_ret_lag_1', 'log_ret_lag_2', 'log_ret_lag_5', 'log_ret_lag_10',
        'sma_5', 'sma_20', 'rsi_14', 'volatility',
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

        # Fit model
        ml_model = MLModel(use_xgboost=XGBOOST_AVAILABLE)
        ml_model.fit(train_features, train_split)

        # Make predictions
        y_pred_ml = ml_model.predict(test_features)

        # Store predictions
        window_results = pd.DataFrame({
            'date': test_split.index,
            'y_true': test_split.values,
            'y_pred_ml': y_pred_ml,
            'window_id': window_id,
            'target': 'log_ret'
        })
        all_predictions.append(window_results)

    # Combine all predictions
    if not all_predictions:
        raise ValueError("No valid predictions generated")

    results_df = pd.concat(all_predictions, ignore_index=True)

    # Calculate metrics
    mask = results_df['y_true'].notna() & results_df['y_pred_ml'].notna()
    if mask.sum() > 0:
        ml_metrics = evaluate_regression(
            results_df.loc[mask, 'y_true'],
            results_df.loc[mask, 'y_pred_ml']
        )

        ml_da = directional_accuracy(
            results_df.loc[mask, 'y_true'],
            results_df.loc[mask, 'y_pred_ml']
        )

        ml_metrics['Directional_Accuracy'] = ml_da

        logging.info(f"ML Model Metrics: {ml_metrics}")

    return results_df


def main():
    """Main function"""
    set_seed(42)
    setup_logging()

    parser = argparse.ArgumentParser(description='Run ML models for log-return forecasting')
    parser.add_argument('--train_window', type=int, default=1260,
                       help='Training window size (default: 1260 ~5 years)')
    parser.add_argument('--test_window', type=int, default=252,
                       help='Test window size (default: 252 ~1 year)')
    parser.add_argument('--step', type=int, default=126,
                       help='Step size for walk-forward (default: 126 ~6 months)')

    args = parser.parse_args()

    logging.info("Running ML model for log-return forecasting")
    logging.info(f"XGBoost available: {XGBOOST_AVAILABLE}")

    # Run walk-forward validation

    results_df = run_ml_walk_forward(args.train_window, args.test_window, args.step)

    # Save results
    output_path = 'models/ml_predictions.csv'
    save_predictions_csv(output_path, results_df)

    logging.info("ML modeling completed")
    logging.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()