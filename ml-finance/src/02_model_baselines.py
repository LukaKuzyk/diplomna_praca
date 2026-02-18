#!/usr/bin/env python3
"""
Baseline models for AAPL price/return forecasting: Naive, ARIMA, GARCH
"""
import argparse
import logging
import os
import warnings
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import (
    set_seed, setup_logging, train_test_splits,
    evaluate_regression, directional_accuracy, save_predictions_csv
)
from models import BaselineModels

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def run_walk_forward(target: str, train_window: int, test_window: int, step: int) -> pd.DataFrame:
    """Run walk-forward validation for baseline models"""
    logging.info(f"Starting walk-forward validation for {target}")

    # Load data
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'aapl_features.csv')
    df = pd.read_csv(data_path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)

    # Get target series
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in data columns: {df.columns.tolist()}")

    series = df[target].dropna()

    # Initialize models
    baseline_models = BaselineModels()

    # Storage for all predictions
    all_predictions = []

    # Walk-forward validation
    for train_split, test_split, window_id in train_test_splits(series, train_window, test_window, step):
        logging.info(f"Window {window_id}: train={len(train_split)}, test={len(test_split)}")

        # Naive forecast
        y_pred_naive = baseline_models.naive_forecast(train_split, test_split, target)

        # ARIMA forecast
        try:
            arima_model = baseline_models.fit_arima(train_split, target)
            y_pred_arima, y_lower, y_upper = baseline_models.forecast_arima(arima_model, len(test_split))
        except Exception as e:
            logging.warning(f"ARIMA failed for window {window_id}: {e}")
            y_pred_arima = y_pred_naive.copy()
            y_lower = np.zeros(len(test_split))
            y_upper = np.zeros(len(test_split))

        # GARCH forecast (only for log_ret target)
        if target == 'log_ret':
            try:
                garch_model = baseline_models.fit_garch(train_split)
                y_pred_garch_mean, _, garch_vol = baseline_models.forecast_garch(garch_model, len(test_split))
            except Exception as e:
                logging.warning(f"GARCH failed for window {window_id}: {e}")
                garch_vol = np.zeros(len(test_split))
                y_pred_garch_mean = np.zeros(len(test_split))
        else:
            garch_vol = np.zeros(len(test_split))
            y_pred_garch_mean = np.full(len(test_split), np.nan)  # Not applicable for price forecasting

        # Store predictions
        window_results = pd.DataFrame({
            'date': test_split.index,
            'y_true': test_split.values,
            'y_pred_arima': y_pred_arima,
            'y_pred_garch_mean': y_pred_garch_mean,
            'y_lower': y_lower,
            'y_upper': y_upper,
            'window_id': window_id,
            'target': target
        })
        all_predictions.append(window_results)

    # Combine all predictions
    results_df = pd.concat(all_predictions, ignore_index=True)

    # Calculate metrics for ARIMA
    mask = results_df['y_true'].notna() & results_df['y_pred_arima'].notna()
    if mask.sum() > 0:
        arima_metrics = evaluate_regression(
            results_df.loc[mask, 'y_true'],
            results_df.loc[mask, 'y_pred_arima']
        )

        if target == 'log_ret':
            arima_da = directional_accuracy(
                results_df.loc[mask, 'y_true'],
                results_df.loc[mask, 'y_pred_arima']
            )
            arima_metrics['Directional_Accuracy'] = arima_da

        logging.info(f"ARIMA Metrics for {target}: {arima_metrics}")

    return results_df


def main():
    """Main function"""
    set_seed(42)
    setup_logging()

    parser = argparse.ArgumentParser(description='Run baseline models (ARIMA, GARCH)')
    parser.add_argument('--target', type=str, choices=['close', 'log_ret'], default='log_ret',
                       help='Target variable to forecast (default: log_ret)')
    parser.add_argument('--train_window', type=int, default=504,
                       help='Training window size (default: 1260 ~5 years)')
    parser.add_argument('--test_window', type=int, default=242,
                       help='Test window size (default: 252 ~1 year)')
    parser.add_argument('--step', type=int, default=126,
                       help='Step size for walk-forward (default: 126 ~6 months)')

    args = parser.parse_args()

    logging.info(f"Running baseline models for target: {args.target}")

    # Run walk-forward validation
    results_df = run_walk_forward(args.target, args.train_window, args.test_window, args.step)

    # Save results
    output_path = f'models/baseline_{args.target}_predictions.csv'
    save_predictions_csv(output_path, results_df)

    logging.info(f"Baseline modeling completed for {args.target}")
    logging.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()