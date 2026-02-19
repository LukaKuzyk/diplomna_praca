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
    evaluate_regression, directional_accuracy, buy_and_hold_accuracy,
    save_predictions_csv
)
from config import SIGNAL_THRESHOLD, FEATURE_COLS
from features import create_features
from models import get_ml_models

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler


def run_ml_walk_forward(train_window: int, test_window: int, step: int, ticker: str = 'AAPL') -> pd.DataFrame:
    """Run walk-forward validation for ML model"""
    logging.info("Starting ML walk-forward validation")

    # Load data
    data_path = os.path.join(os.path.dirname(__file__), 'data', f'{ticker.lower()}_features.csv')
    df = pd.read_csv(data_path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)

    # Create features
    df_features = create_features(df)

    # Check if all features exist
    missing_features = [col for col in FEATURE_COLS if col not in df_features.columns]
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
        train_features = df_features.loc[train_split.index, FEATURE_COLS]
        test_features = df_features.loc[test_split.index, FEATURE_COLS]

        # Skip if not enough data
        if len(train_features) < 50 or len(test_features) == 0:
            logging.warning(f"Skipping window {window_id} due to insufficient data")
            continue

        # Fit and predict with multiple models
        models = get_ml_models()
        predictions = {'date': test_split.index, 'y_true': test_split.values, 'window_id': window_id, 'target': 'log_ret'}

        for model_name, (model, scaler) in models.items():
            logging.info(f"  Training {model_name.upper()}...")
            X_scaled = scaler.fit_transform(train_features)
            model.fit(X_scaled, train_split)

            X_test_scaled = scaler.transform(test_features)
            y_pred = model.predict(X_test_scaled)
            logging.info(f"  {model_name.upper()} done (mean pred: {y_pred.mean():.6f})")

            predictions[f'y_pred_{model_name}'] = y_pred

        # Store predictions
        window_results = pd.DataFrame(predictions)
        all_predictions.append(window_results)
        last_trained_models = models

    # Combine all predictions
    if not all_predictions:
        raise ValueError("No valid predictions generated")

    results_df = pd.concat(all_predictions, ignore_index=True)

    # Save feature importances from tree-based models (last window)
    importance_data = {}
    for model_name, (model, scaler) in last_trained_models.items():
        if hasattr(model, 'feature_importances_'):
            importance_data[model_name.upper()] = model.feature_importances_

    if importance_data:
        importance_df = pd.DataFrame(importance_data, index=FEATURE_COLS)
        importance_path = os.path.join(os.path.dirname(__file__), 'reports', f'{ticker.lower()}_feature_importance.csv')
        os.makedirs(os.path.dirname(importance_path), exist_ok=True)
        importance_df.to_csv(importance_path)
        logging.info(f"Feature importances saved to {importance_path}")

    # Calculate metrics for each model
    models = get_ml_models()
    bh_acc = buy_and_hold_accuracy(results_df['y_true'])
    logging.info(f"Baseline (Buy & Hold) accuracy: {bh_acc:.1%}")

    for model_name in models.keys():
        pred_col = f'y_pred_{model_name}'
        mask = results_df['y_true'].notna() & results_df[pred_col].notna()
        if mask.sum() > 0:
            ml_metrics = evaluate_regression(
                results_df.loc[mask, 'y_true'],
                results_df.loc[mask, pred_col]
            )

            da = directional_accuracy(
                results_df.loc[mask, 'y_true'],
                results_df.loc[mask, pred_col],
                threshold=SIGNAL_THRESHOLD
            )

            ml_metrics['Raw_DA'] = da['raw_da']
            ml_metrics['Confident_DA'] = da['confident_da']
            ml_metrics['Coverage'] = da['coverage']

            logging.info(
                f"{model_name.upper()}: Raw DA={da['raw_da']:.1%}, "
                f"Confident DA={da['confident_da']:.1%} (coverage={da['coverage']:.1%}), "
                f"B&H baseline={bh_acc:.1%}"
            )

    return results_df


def main():
    """Main function"""
    set_seed(42)
    setup_logging()

    parser = argparse.ArgumentParser(description='Run ML models for log-return forecasting')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker (default: AAPL)')
    parser.add_argument('--train_window', type=int, default=1008,
                       help='Training window size (default: 252 ~1 year)')
    parser.add_argument('--test_window', type=int, default=63,
                       help='Test window size (default: 30 ~1 month)')
    parser.add_argument('--step', type=int, default=63,
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