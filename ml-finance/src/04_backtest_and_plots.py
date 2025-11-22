#!/usr/bin/env python3
"""
Backtesting and visualization for AAPL forecasting models
"""
import argparse
import logging
import warnings
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

from utils import (
    set_seed, setup_logging, evaluate_regression,
    directional_accuracy, calculate_mape, ensure_dirs
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')


def load_all_predictions() -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Load all prediction files and combine them"""
    logging.info("Loading all prediction files...")

    # Load base data
    data_path = Path('data/aapl_features.csv')
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df_base = pd.read_csv(data_path, index_col=0)
    df_base.index = pd.to_datetime(df_base.index)

    # Load prediction files
    models_dir = Path('models')
    prediction_files = {
        'baseline_close': models_dir / 'baseline_close_predictions.csv',
        'baseline_log_ret': models_dir / 'baseline_log_ret_predictions.csv',
        'ml': models_dir / 'ml_predictions.csv'
    }

    predictions = {}

    for name, filepath in prediction_files.items():
        if filepath.exists():
            df = pd.read_csv(filepath)
            # Set date column as index properly
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            else:
                # Fallback to first column if no date column
                df.index = pd.to_datetime(df.index)
            predictions[name] = df
            logging.info(f"Loaded {name}: {len(df)} predictions")
        else:
            logging.warning(f"Prediction file not found: {filepath}")

    return df_base, predictions


def combine_predictions(df_base: pd.DataFrame, predictions: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine all predictions into a single timeline"""
    logging.info("Combining predictions...")

    # Start with base data
    combined_df = df_base.copy()

    # Add predictions from each model
    for name, pred_df in predictions.items():
        if len(pred_df) == 0:
            continue

        # Merge predictions based on date index
        for col in pred_df.columns:
            if col not in ['window_id', 'target']:
                # Create a temporary DataFrame with just the column we want to merge
                temp_df = pred_df[col].to_frame()
                temp_df.columns = [f"{name}_{col}"]

                # Merge with combined_df (both have datetime index)
                combined_df = combined_df.merge(
                    temp_df,
                    left_index=True,
                    right_index=True,
                    how='left'
                )

    # Remove duplicates and sort by date
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
    combined_df = combined_df.sort_index()

    logging.info(f"Combined dataset: {len(combined_df)} rows")
    return combined_df


def calculate_strategy_performance(returns: pd.Series, ml_predictions: pd.Series,
                                transaction_cost: float = 0.0005) -> Dict[str, float]:
    """Calculate directional strategy performance"""
    logging.info("Calculating strategy performance...")

    # Align data
    data = pd.DataFrame({
        'returns': returns,
        'ml_pred': ml_predictions
    }).dropna()

    if len(data) == 0:
        return {'error': 'No aligned data for strategy'}

    # Generate positions based on ML predictions
    positions = np.sign(data['ml_pred'])

    # Calculate strategy returns (without leverage)
    strategy_returns = positions.shift(1) * data['returns']  # Shift because we use previous day's signal

    # Apply transaction costs (only when position changes)
    position_changes = positions != positions.shift(1)
    tc_per_trade = position_changes * transaction_cost
    strategy_returns = strategy_returns - tc_per_trade

    # Remove first NaN
    strategy_returns = strategy_returns.dropna()

    if len(strategy_returns) == 0:
        return {'error': 'No valid strategy returns'}

    # Calculate performance metrics
    total_return = (1 + strategy_returns).prod() - 1
    annual_return = strategy_returns.mean() * 252  # Annualized
    annual_vol = strategy_returns.std() * np.sqrt(252)

    # Sharpe ratio (assuming 0% risk-free rate)
    sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0

    # Maximum drawdown
    cumulative = (1 + strategy_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'num_trades': len(strategy_returns)
    }


def create_plots(combined_df: pd.DataFrame, output_dir: str = 'reports/figures') -> None:
    """Create all required plots"""
    ensure_dirs(output_dir)
    logging.info(f"Creating plots in {output_dir}...")

    # Set up the plotting area
    fig_size = (12, 8)

    # 1. Price vs ARIMA predictions
    plt.figure(figsize=fig_size)

    # Filter data for plotting (last 500 points for clarity)
    plot_data = combined_df.tail(500).copy()

    # Plot actual price
    plt.plot(plot_data.index, plot_data['close'], label='Actual Price', alpha=0.7)

    # Plot ARIMA predictions for close
    arima_close_col = None
    for col in plot_data.columns:
        if 'baseline_close_y_pred_arima' in col:
            arima_close_col = col
            break

    if arima_close_col and plot_data[arima_close_col].notna().any():
        plt.plot(plot_data.index, plot_data[arima_close_col],
                label='ARIMA Forecast', alpha=0.8, linestyle='--')

        # Add confidence intervals if available
        ci_lower_col = arima_close_col.replace('y_pred_arima', 'y_lower')
        ci_upper_col = arima_close_col.replace('y_pred_arima', 'y_upper')

        if ci_lower_col in plot_data.columns and ci_upper_col in plot_data.columns:
            plt.fill_between(plot_data.index,
                           plot_data[ci_lower_col],
                           plot_data[ci_upper_col],
                           alpha=0.2, label='95% Confidence Interval')

    plt.title('AAPL Price vs ARIMA Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/price_vs_arima.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved price_vs_arima.png")

    # 2. Returns predictions
    plt.figure(figsize=fig_size)

    # Plot actual returns
    plt.plot(plot_data.index, plot_data['log_ret'], label='Actual Returns',
             alpha=0.7, color='blue')

    # Plot ML predictions
    ml_pred_col = None
    for col in plot_data.columns:
        if 'ml_y_pred_ml' in col:
            ml_pred_col = col
            break

    if ml_pred_col and plot_data[ml_pred_col].notna().any():
        plt.plot(plot_data.index, plot_data[ml_pred_col],
                label='ML Forecast', alpha=0.8, color='red', linestyle='--')

    # Plot ARIMA predictions for returns
    arima_ret_col = None
    for col in plot_data.columns:
        if 'baseline_log_ret_y_pred_arima' in col:
            arima_ret_col = col
            break

    if arima_ret_col and plot_data[arima_ret_col].notna().any():
        plt.plot(plot_data.index, plot_data[arima_ret_col],
                label='ARIMA Returns Forecast', alpha=0.8, color='green', linestyle='--')

    # Add zero line
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)

    plt.title('AAPL Log Returns: Actual vs Forecasts')
    plt.xlabel('Date')
    plt.ylabel('Log Returns')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/returns_pred.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved returns_pred.png")

    # 3. Volatility forecast
    plt.figure(figsize=fig_size)

    # Plot realized volatility
    plt.plot(plot_data.index, plot_data['rv_5'], label='Realized Volatility (5-day)',
             alpha=0.7, color='blue')

    # Plot GARCH volatility forecast (if available)
    garch_vol_col = None
    for col in plot_data.columns:
        if 'y_upper' in col and 'baseline_log_ret' in col:
            garch_vol_col = col
            break

    if garch_vol_col and plot_data[garch_vol_col].notna().any():
        plt.plot(plot_data.index, plot_data[garch_vol_col],
                label='GARCH Volatility Forecast', alpha=0.8, color='red', linestyle='--')

    plt.title('AAPL Volatility: Realized vs GARCH Forecast')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/vol_forecast.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved vol_forecast.png")

    # 4. Strategy equity curve
    if ml_pred_col and plot_data[ml_pred_col].notna().any():
        plt.figure(figsize=fig_size)

        # Calculate strategy performance
        strategy_perf = calculate_strategy_performance(
            plot_data['log_ret'],
            plot_data[ml_pred_col]
        )

        if 'error' not in strategy_perf:
            # Calculate cumulative returns
            data_strategy = pd.DataFrame({
                'returns': plot_data['log_ret'],
                'ml_pred': plot_data[ml_pred_col]
            }).dropna()

            positions = np.sign(data_strategy['ml_pred'])
            strategy_returns = positions.shift(1) * data_strategy['returns']
            strategy_returns = strategy_returns.dropna()

            # Apply transaction costs
            position_changes = positions != positions.shift(1)
            tc_per_trade = position_changes * 0.0005  # 5bp
            strategy_returns = strategy_returns - tc_per_trade
            strategy_returns = strategy_returns.dropna()

            # Calculate equity curve
            equity_curve = (1 + strategy_returns).cumprod()

            # Plot equity curve
            plt.plot(equity_curve.index, equity_curve.values, label='Strategy Equity', color='green')

            # Add buy-and-hold for comparison
            bh_returns = plot_data['log_ret'].dropna()
            bh_equity = (1 + bh_returns).cumprod()
            plt.plot(bh_equity.index, bh_equity.values, label='Buy & Hold', alpha=0.7, color='blue')

            plt.title('AAPL Directional Strategy vs Buy & Hold')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Returns')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/strategy_equity.png', dpi=300, bbox_inches='tight')
            plt.close()
            logging.info("Saved strategy_equity.png")

            # Store strategy performance for summary
            return strategy_perf

    return {}


def calculate_final_metrics(combined_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Calculate final aggregated metrics"""
    logging.info("Calculating final metrics...")

    metrics = {}

    # ARIMA Close metrics
    arima_close_cols = [col for col in combined_df.columns if 'baseline_close_y_pred_arima' in col]
    if arima_close_cols:
        col = arima_close_cols[0]
        mask = combined_df['close'].notna() & combined_df[col].notna()
        if mask.sum() > 0:
            arima_metrics = evaluate_regression(
                combined_df.loc[mask, 'close'],
                combined_df.loc[mask, col]
            )
            arima_metrics['MAPE'] = calculate_mape(
                combined_df.loc[mask, 'close'],
                combined_df.loc[mask, col]
            )
            metrics['ARIMA_Close'] = arima_metrics

    # ARIMA Returns metrics
    arima_ret_cols = [col for col in combined_df.columns if 'baseline_log_ret_y_pred_arima' in col]
    if arima_ret_cols:
        col = arima_ret_cols[0]
        mask = combined_df['log_ret'].notna() & combined_df[col].notna()
        if mask.sum() > 0:
            arima_ret_metrics = evaluate_regression(
                combined_df.loc[mask, 'log_ret'],
                combined_df.loc[mask, col]
            )
            arima_ret_metrics['Directional_Accuracy'] = directional_accuracy(
                combined_df.loc[mask, 'log_ret'],
                combined_df.loc[mask, col]
            )
            metrics['ARIMA_Returns'] = arima_ret_metrics

    # ML Returns metrics
    ml_cols = [col for col in combined_df.columns if 'ml_y_pred_ml' in col]
    if ml_cols:
        col = ml_cols[0]
        mask = combined_df['log_ret'].notna() & combined_df[col].notna()
        if mask.sum() > 0:
            ml_metrics = evaluate_regression(
                combined_df.loc[mask, 'log_ret'],
                combined_df.loc[mask, col]
            )
            ml_metrics['Directional_Accuracy'] = directional_accuracy(
                combined_df.loc[mask, 'log_ret'],
                combined_df.loc[mask, col]
            )
            metrics['ML_Returns'] = ml_metrics

    return metrics


def save_metrics_summary(metrics: Dict[str, Dict[str, float]],
                        strategy_perf: Dict[str, float],
                        output_path: str = 'reports/metrics_summary.txt') -> None:
    """Save metrics summary to file"""
    ensure_dirs(output_path)

    with open(output_path, 'w') as f:
        f.write("AAPL Forecasting Models - Final Metrics Summary\n")
        f.write("=" * 50 + "\n\n")

        for model_name, model_metrics in metrics.items():
            f.write(f"{model_name}:\n")
            for metric_name, value in model_metrics.items():
                f.write(f"  {metric_name}: {value:.6f}\n")
            f.write("\n")

        if strategy_perf and 'error' not in strategy_perf:
            f.write("Directional Strategy Performance:\n")
            f.write(f"  Total Return: {strategy_perf.get('total_return', 0):.6f}\n")
            f.write(f"  Annual Return: {strategy_perf.get('annual_return', 0):.6f}\n")
            f.write(f"  Annual Volatility: {strategy_perf.get('annual_volatility', 0):.6f}\n")
            f.write(f"  Sharpe Ratio: {strategy_perf.get('sharpe_ratio', 0):.6f}\n")
            f.write(f"  Max Drawdown: {strategy_perf.get('max_drawdown', 0):.6f}\n")
            f.write(f"  Number of Trades: {strategy_perf.get('num_trades', 0)}\n")

    logging.info(f"Metrics summary saved to {output_path}")


def main():
    """Main function"""
    set_seed(42)
    setup_logging()

    parser = argparse.ArgumentParser(description='Create backtesting plots and metrics')
    args = parser.parse_args()

    logging.info("Starting backtesting and visualization...")

    try:
        # Load and combine all data
        df_base, predictions = load_all_predictions()
        combined_df = combine_predictions(df_base, predictions)

        # Create plots and get strategy performance
        strategy_perf = create_plots(combined_df)

        # Calculate final metrics
        final_metrics = calculate_final_metrics(combined_df)

        # Save metrics summary
        save_metrics_summary(final_metrics, strategy_perf)

        logging.info("Backtesting and visualization completed successfully!")

    except Exception as e:
        logging.error(f"Error in backtesting: {e}")
        raise


if __name__ == "__main__":
    main()