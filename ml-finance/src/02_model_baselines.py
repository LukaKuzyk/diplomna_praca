#!/usr/bin/env python3
"""
Baseline models for AAPL price/return forecasting: Naive, ARIMA, GARCH
"""
import argparse
import logging
import warnings
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import matplotlib.pyplot as plt

from utils import (
    set_seed, setup_logging, train_test_splits,
    evaluate_regression, directional_accuracy, save_predictions_csv
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class BaselineModels:
    """Baseline models for time series forecasting"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def naive_forecast(self, train: pd.Series, test: pd.Series, target: str) -> np.ndarray:
        """Naive (Random Walk) forecast"""
        if target == 'close':
            # For price: last known price
            return np.full(len(test), train.iloc[-1])
        elif target == 'log_ret':
            # For returns: predict zero (no change)
            return np.zeros(len(test))
        else:
            raise ValueError(f"Unknown target: {target}")

    def fit_arima(self, train: pd.Series, target: str) -> ARIMA:
        """Fit ARIMA model"""
        if target == 'log_ret':
            # ARIMA(1,0,1) for log returns
            order = (1, 0, 1)
        elif target == 'close':
            # ARIMA(1,1,1) for price (needs differencing)
            order = (1, 1, 1)
        else:
            raise ValueError(f"Unknown target: {target}")

        try:
            model = ARIMA(train, order=order)
            model_fit = model.fit()
            return model_fit
        except Exception as e:
            logging.warning(f"ARIMA fitting failed: {e}")
            # Fallback to simpler model
            if target == 'log_ret':
                model = ARIMA(train, order=(1, 0, 0))
            else:
                model = ARIMA(train, order=(0, 1, 0))
            return model.fit()

    def forecast_arima(self, model_fit: ARIMA, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get ARIMA forecast with confidence intervals"""
        try:
            forecast = model_fit.get_forecast(steps=steps)
            forecast_mean = forecast.predicted_mean
            forecast_ci = forecast.conf_int(alpha=0.05)  # 95% CI

            return forecast_mean, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1]
        except Exception as e:
            logging.warning(f"ARIMA forecasting failed: {e}")
            # Return naive forecast if ARIMA fails
            return np.zeros(steps), np.zeros(steps), np.zeros(steps)

    def fit_garch(self, returns: pd.Series) -> arch_model:
        """Fit GARCH(1,1) model"""
        try:
            model = arch_model(returns, mean='AR', vol='GARCH', p=1, q=1, dist='StudentsT')
            return model.fit(disp='off')
        except Exception as e:
            logging.warning(f"GARCH fitting failed: {e}")
            # Fallback to simpler GARCH
            try:
                model = arch_model(returns, mean='Constant', vol='GARCH', p=1, q=1)
                return model.fit(disp='off')
            except Exception as e2:
                logging.error(f"GARCH fallback also failed: {e2}")
                raise

    def forecast_garch(self, model_fit: arch_model, steps: int = 1) -> Tuple[float, float, float]:
        """Get GARCH volatility forecast"""
        try:
            forecast = model_fit.forecast(horizon=steps)
            # Get conditional volatility for next period
            vol_forecast = np.sqrt(forecast.variance.iloc[-1, 0])
            return 0.0, 0.0, vol_forecast  # mean=0, lower=0, upper=volatility
        except Exception as e:
            logging.warning(f"GARCH forecasting failed: {e}")
            return 0.0, 0.0, 0.0


def run_walk_forward(target: str, train_window: int, test_window: int, step: int) -> pd.DataFrame:
    """Run walk-forward validation for baseline models"""
    logging.info(f"Starting walk-forward validation for {target}")

    # Load data
    data_path = 'data/aapl_features.csv'
    df = pd.read_csv(data_path, index_col=0)
    df.index = pd.to_datetime(df.index)

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
                _, _, garch_vol = baseline_models.forecast_garch(garch_model, len(test_split))
                y_pred_garch_mean = np.zeros(len(test_split))  # Mean forecast = 0
            except Exception as e:
                logging.warning(f"GARCH failed for window {window_id}: {e}")
                garch_vol = np.zeros(len(test_split))
                y_pred_garch_mean = np.zeros(len(test_split))
        else:
            garch_vol = np.zeros(len(test_split))
            y_pred_garch_mean = np.zeros(len(test_split))

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
    parser.add_argument('--target', type=str, choices=['close', 'log_ret'], required=True,
                       help='Target variable to forecast')
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