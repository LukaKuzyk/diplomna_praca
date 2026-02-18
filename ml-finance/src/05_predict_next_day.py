#!/usr/bin/env python3
"""
Predict next day AAPL log-return and provide trading recommendation
"""
import argparse
import logging
import os
import warnings
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from utils import (
    set_seed, setup_logging, evaluate_regression,
    directional_accuracy, buy_and_hold_accuracy, ensure_dirs
)
from config import SIGNAL_THRESHOLD, FEATURE_COLS
from features import create_features
from models import get_ml_models, MLModelPredictor

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def load_model_metrics(ticker: str) -> Dict[str, any]:
    """Load directional accuracy metrics for each model"""
    metrics_path = Path(os.path.join(os.path.dirname(__file__), 'reports', f'{ticker.lower()}_ml_metrics_summary.txt'))

    if not metrics_path.exists():
        metrics_path = Path(os.path.join(os.path.dirname(__file__), 'reports', f'{ticker.lower()}_metrics_summary.txt'))
        if not metrics_path.exists():
            logging.warning(f"Metrics file not found: {metrics_path}")
            return {}

    da_metrics = {}
    try:
        with open(metrics_path, 'r') as f:
            content = f.read()

        lines = content.split('\n')
        current_model = None
        current_dict = {}

        for line in lines:
            line = line.strip()
            if line.endswith('_Returns:'):
                current_model = line.replace('_Returns:', '').replace('ML_', '')
                current_dict = {}
            elif line.startswith('Raw_DA:') and current_model:
                current_dict['raw_da'] = float(line.split(':')[1].strip())
            elif line.startswith('Confident_DA:') and current_model:
                current_dict['confident_da'] = float(line.split(':')[1].strip())
            elif line.startswith('Coverage:') and current_model:
                current_dict['coverage'] = float(line.split(':')[1].strip())
                da_metrics[current_model.lower()] = current_dict
                current_model = None
            elif line.startswith('Directional_Accuracy:') and current_model:
                # Legacy format fallback
                da_value = float(line.split(':')[1].strip())
                da_metrics[current_model.lower()] = da_value
                current_model = None
            elif line.startswith('Buy_and_Hold_DA:'):
                da_metrics['_bh_accuracy'] = float(line.split(':')[1].strip())

    except Exception as e:
        logging.warning(f"Error loading metrics: {e}")

    return da_metrics


def predict_next_day(ticker: str = 'AAPL') -> Dict[str, any]:
    """Predict next day log-return and provide recommendation using all models"""
    logging.info(f"Starting next day prediction for {ticker} with all models...")

    # Load data
    data_path = Path(os.path.join(os.path.dirname(__file__), 'data', f'{ticker.lower()}_features.csv'))
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)

    # Create features
    df_features = create_features(df)

    # Check if all features exist
    feature_cols = [col for col in FEATURE_COLS if col in df_features.columns]
    missing_features = [col for col in FEATURE_COLS if col not in df_features.columns]
    if missing_features:
        logging.warning(f"Missing features: {missing_features}")

    # Use all available data for training
    train_features = df_features[feature_cols]
    train_target = df_features['log_ret']
    train_close = df_features['close']

    if len(train_features) < 50:
        raise ValueError("Not enough data for training")

    # Initialize ML models
    ml_predictor = MLModelPredictor()

    # Fit models
    ml_predictor.fit(train_features, train_target)

    # Prepare features for next day prediction
    last_row = df_features.iloc[-1]
    next_day_features = pd.DataFrame([last_row[feature_cols].values], columns=feature_cols)

    # Get predictions from all ML models
    ml_predictions = ml_predictor.predict_all(next_day_features)

    # Load directional accuracy metrics
    da_metrics = load_model_metrics(ticker)
    bh_acc = da_metrics.pop('_bh_accuracy', buy_and_hold_accuracy(df_features['log_ret'].dropna()))

    # Pick primary model as the one with highest Raw DA from backtesting
    best_da = 0
    primary_model = 'xgb' if 'xgb' in ml_predictions else list(ml_predictions.keys())[0]
    for model_key in ml_predictions:
        da_entry = da_metrics.get(model_key, {})
        raw_da = da_entry.get('raw_da', 0) if isinstance(da_entry, dict) else da_entry
        if raw_da > best_da:
            best_da = raw_da
            primary_model = model_key

    predicted_log_ret_ml = ml_predictions[primary_model]

    # Get last known close price
    last_close = df_features['close'].iloc[-1]

    # Calculate predicted close prices for all ML models
    predictions = {}
    for model_name, pred_ret in ml_predictions.items():
        pred_close = last_close * np.exp(pred_ret)
        predictions[f'ML_{model_name.upper()}'] = {'log_ret': pred_ret, 'close': pred_close}

    # Add main ML prediction
    predictions['ML'] = {'log_ret': predicted_log_ret_ml, 'close': last_close * np.exp(predicted_log_ret_ml)}

    # Overall recommendation based on threshold logic (only-long strategy)
    threshold = SIGNAL_THRESHOLD
    if predicted_log_ret_ml > threshold:
        recommendation = "BUY"
        reason = f"ML prediction positive: {predicted_log_ret_ml:.6f} > {threshold}"
    else:
        recommendation = "HOLD/CASH"
        reason = f"ML prediction weak: {predicted_log_ret_ml:.6f} ≤ {threshold}"

    expected_return_pct = predicted_log_ret_ml * 100

    result = {
        'ticker': ticker.upper(),
        'last_date': df_features.index[-1].strftime('%Y-%m-%d'),
        'last_close': last_close,
        'predictions': predictions,
        'da_metrics': da_metrics,
        'bh_accuracy': bh_acc,
        'recommendation': recommendation,
        'reason': reason,
        'expected_return_pct': expected_return_pct,
        'threshold': threshold,
        'primary_model': primary_model.upper(),
        'historical_data': df_features[['close']].tail(30)
    }

    logging.info(f"Prediction completed for {ticker}: {result['recommendation']}")
    return result


def create_prediction_plot(result: Dict[str, any], output_dir: str = 'reports/figures') -> None:
    """Create plot showing monthly price chart with next day ML predictions"""
    ensure_dirs(output_dir)
    logging.info(f"Creating prediction plot in {output_dir}...")

    ticker = result['ticker']
    # Load historical predictions
    ml_pred_path = Path(os.path.join(os.path.dirname(__file__), 'models', f'{ticker.lower()}_ml_predictions.csv'))

    # Get last 30 days of historical data (1 month)
    historical_data = result['historical_data'].tail(30)  # Last 30 days
    predictions = result['predictions']
    last_date = pd.to_datetime(result['last_date'])
    next_date = last_date + pd.Timedelta(days=1)

    # Set up the plotting area - single plot focused on price
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Plot historical price for the last month
    ax.plot(historical_data.index, historical_data['close'],
            label='Historical Price', color='blue', linewidth=2)

    # Add current price point
    ax.scatter(last_date, result['last_close'], color='blue', s=50, zorder=5,
               label=f'Today: ${result["last_close"]:.2f}')

    # Plot next day predictions as prominent dots
    ml_predictions = {k: v for k, v in predictions.items() if k.startswith('ML_')}

    colors = ['red', 'orange', 'green', 'purple', 'brown', 'cyan', 'magenta']
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']

    for i, (model_name, pred) in enumerate(ml_predictions.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        # Plot prediction point
        ax.scatter(next_date, pred['close'], color=color, marker=marker, s=150, zorder=6,
                  label=f'{model_name}: ${pred["close"]:.2f}')

        # Add arrow from current price to prediction
        ax.annotate('', xy=(next_date, pred['close']), xytext=(last_date, result['last_close']),
                   arrowprops=dict(arrowstyle='->', color=color, alpha=0.7, linewidth=2))

    # Add recommendation text
    recommendation = result['recommendation']
    if 'BUY' in recommendation:
        rec_color = 'green'
        rec_symbol = '↗️'
    else:
        rec_color = 'orange'
        rec_symbol = '➡️'

    ax.text(0.02, 0.98, f'{rec_symbol} {recommendation}',
            transform=ax.transAxes, fontsize=14, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3',
            facecolor=rec_color, alpha=0.1))

    # Add expected return
    expected_return = result['expected_return_pct']
    return_color = 'green' if expected_return > 0 else 'red'
    ax.text(0.02, 0.90, f'Expected Return: {expected_return:.2f}%',
            transform=ax.transAxes, fontsize=12, color=return_color,
            verticalalignment='top')

    ax.axvline(x=last_date, color='black', linestyle='--', alpha=0.7, label='Today')
    ax.set_title(f'{ticker}: Monthly Price Chart with Next Day Predictions', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Close Price (USD)', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    plt.tight_layout()

    # Save plot
    plt.savefig(f'{output_dir}/next_day_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved next_day_predictions.png")


def main():
    """Main function"""
    set_seed(42)
    setup_logging()

    parser = argparse.ArgumentParser(description='Predict next day stock return and provide recommendation')
    parser.add_argument('--ticker', type=str, default='MSFT', help='Stock ticker (default: AAPL)')
    args = parser.parse_args()

    logging.info(f"Predicting next day {args.ticker} movement...")

    try:
        result = predict_next_day(args.ticker)

        # Print results
        print("\n" + "="*60)
        print(f"NEXT DAY PREDICTION FOR {result['ticker']}")
        print("="*60)
        print(f"Last Date: {result['last_date']}")
        print(f"Last Close Price: ${result['last_close']:.2f}")
        print(f"Primary Model: {result['primary_model']}")
        print(f"Signal Threshold: {result['threshold']}")
        print("\nML Model Predictions:")
        da_metrics = result.get('da_metrics', {})
        bh_acc = result.get('bh_accuracy', 0)
        for model, pred in result['predictions'].items():
            if model.startswith('ML_'):
                model_key = model.replace('ML_', '').lower()
                da_value = da_metrics.get(model_key, {})
                if isinstance(da_value, dict):
                    da_str = f"Raw DA: {da_value.get('raw_da', 0):.1%}, Confident DA: {da_value.get('confident_da', 0):.1%} ({da_value.get('coverage', 0):.0%} coverage)"
                elif isinstance(da_value, (int, float)):
                    da_str = f"DA: {da_value:.1%}"
                else:
                    da_str = "DA: N/A"
                print(f"  {model}: Price ${pred['close']:.2f}, Return {pred['log_ret']:.6f}, {da_str}")
        print(f"\nBaseline (Buy & Hold DA): {bh_acc:.1%}")
        print(f"Expected Return (ML): {result['expected_return_pct']:.2f}%")
        print(f"RECOMMENDATION: {result['recommendation']}")
        print(f"Reason: {result['reason']}")
        print("="*60)

        # Save key results to file for report generation
        output_file = os.path.join(os.path.dirname(__file__), 'reports', f'{args.ticker.lower()}_next_day_prediction.txt')
        ensure_dirs(os.path.dirname(output_file))

        with open(output_file, 'w') as f:
            f.write(f"Best Model: {result['primary_model']}\n")
            f.write(f"Predicted Return: {result['expected_return_pct']:.6f}\n")
            primary_da = da_metrics.get(result['primary_model'].lower(), {})
            raw_da = primary_da.get('raw_da', 0) if isinstance(primary_da, dict) else primary_da
            f.write(f"Raw_DA: {raw_da:.6f}\n")
            f.write(f"Buy_Hold_DA: {result['bh_accuracy']:.6f}\n")
            f.write(f"Recommendation: {result['recommendation']}\n")
            f.write(f"Last Close: {result['last_close']:.2f}\n")
            f.write(f"Last Date: {result['last_date']}\n")
            f.write(f"Signal Threshold: {result['threshold']}\n")

        logging.info(f"Next day prediction results saved to {output_file}")

        # Create plot
        output_dir = os.path.join(os.path.dirname(__file__), 'reports', f'{args.ticker.lower()}_figures')
        create_prediction_plot(result, output_dir)

        logging.info("Next day prediction and plot completed successfully!")

    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        raise


if __name__ == "__main__":
    main()