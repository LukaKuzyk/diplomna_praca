#!/usr/bin/env python3
"""
Backtesting and visualization for AAPL forecasting models
"""
import argparse
import logging
import os
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


def load_all_predictions(ticker: str = 'AAPL') -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Load all prediction files and combine them"""
    logging.info("Loading all prediction files...")

    # Load base data
    import os
    data_path = Path(os.path.join(os.path.dirname(__file__), 'data', f'{ticker.lower()}_features.csv'))
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df_base = pd.read_csv(data_path, index_col=0)
    df_base.index = pd.to_datetime(df_base.index, utc=True)

    # Load prediction files
    models_dir = Path(os.path.join(os.path.dirname(__file__), 'models'))
    prediction_files = {
        'baseline_close': models_dir / 'baseline_close_predictions.csv',
        'baseline_log_ret': models_dir / 'baseline_log_ret_predictions.csv',
        'ml': models_dir / f'{ticker.lower()}_ml_predictions.csv'
    }

    predictions = {}

    for name, filepath in prediction_files.items():
        if filepath.exists():
            df = pd.read_csv(filepath)
            # Set date column as index properly
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], utc=True)
                df.set_index('date', inplace=True)
            else:
                # Fallback to first column if no date column
                df.index = pd.to_datetime(df.index, utc=True)
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

    # Collect all prediction columns
    all_preds = pd.DataFrame(index=combined_df.index)

    for name, pred_df in predictions.items():
        if len(pred_df) == 0:
            continue

        # Remove duplicate indices, keeping the last prediction for each date
        pred_df = pred_df[~pred_df.index.duplicated(keep='last')]

        for col in pred_df.columns:
            if col not in ['window_id', 'target']:
                all_preds[f"{name}_{col}"] = pred_df[col]

    # Concatenate all predictions at once
    combined_df = pd.concat([combined_df, all_preds], axis=1)

    # Remove duplicates and sort by date
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
    combined_df = combined_df.sort_index()

    logging.info(f"Combined dataset: {len(combined_df)} rows")
    return combined_df


def generate_signals_with_threshold(predictions: pd.Series, threshold: float = 0.0003) -> pd.Series:
    """
    Generate trading signals from ML predictions with threshold.

    Args:
        predictions: ML predicted log returns
        threshold: Minimum absolute prediction value to generate signal

    Returns:
        Series of signals: 1 (long), -1 (short), 0 (neutral)
    """
    signals = pd.Series(0, index=predictions.index)
    signals[predictions > threshold] = 1
    signals[predictions < -threshold] = -1
    return signals


def generate_only_long_signals(predictions: pd.Series, threshold: float = 0.0003) -> pd.Series:
    """
    Generate only-long trading signals from ML predictions with threshold.

    Args:
        predictions: ML predicted log returns
        threshold: Minimum prediction value to generate long signal

    Returns:
        Series of signals: 1 (long), 0 (neutral/cash)
    """
    signals = pd.Series(0, index=predictions.index)
    signals[predictions > threshold] = 1
    return signals


def calculate_strategy_performance(returns: pd.Series, ml_predictions: pd.Series,
                                   transaction_cost: float = 0.0, threshold: float = 0.0003,
                                   strategy_type: str = 'directional') -> Dict[str, float]:
    """
    Calculate strategy performance with different signal generation modes.

    Args:
        returns: Log returns series
        ml_predictions: ML predicted log returns
        transaction_cost: Cost per trade (default 0.0)
        threshold: Signal threshold (default 0.0003)
        strategy_type: 'directional' (long/short) or 'only_long' (long only)

    Returns:
        Dictionary with performance metrics
    """
    logging.info(f"Calculating {strategy_type} strategy performance...")

    # Align data - convert log returns to simple returns for financial calculations
    data = pd.DataFrame({
        'returns': np.exp(returns) - 1,  # Convert log returns to simple returns
        'ml_pred': ml_predictions
    }).dropna()

    if len(data) == 0:
        return {'error': 'No aligned data for strategy'}

    # Generate signals based on strategy type
    if strategy_type == 'only_long':
        signals = generate_only_long_signals(data['ml_pred'], threshold)
    else:  # directional
        signals = generate_signals_with_threshold(data['ml_pred'], threshold)

    positions = signals

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


def create_plots(combined_df: pd.DataFrame, output_dir: str = None, ticker: str = 'MSFT') -> None:
    if output_dir is None:
        output_dir = 'reports/figures'
    """Create all required plots"""
    import os
    os.makedirs(output_dir, exist_ok=True)
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

    plt.title(f'{ticker.upper()} Price vs ARIMA Forecast')
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

    # Plot ML predictions (all models)
    ml_pred_cols = [col for col in plot_data.columns if col.startswith('ml_y_pred_')]
    colors = ['red', 'orange', 'purple', 'brown']
    for i, ml_pred_col in enumerate(ml_pred_cols):
        model_name = ml_pred_col.replace('ml_y_pred_', '').upper()
        if plot_data[ml_pred_col].notna().any():
            plt.plot(plot_data.index, plot_data[ml_pred_col],
                     label=f'{model_name} Forecast', alpha=0.8, color=colors[i % len(colors)], linestyle='--')

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

    plt.title(f'{ticker.upper()} Log Returns: Actual vs Forecasts')
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

    plt.title(f'{ticker.upper()} Volatility: Realized vs GARCH Forecast')
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
    # 4. Strategy equity curve
    ml_pred_cols = [col for col in plot_data.columns if col.startswith('ml_y_pred_')]
    if ml_pred_cols:
        plt.figure(figsize=fig_size)

        # --- Buy & Hold на всій історії ---
        # лог-ретурни -> прості -> кумулятивна дохідність
        simple_returns = np.exp(plot_data['log_ret'].fillna(0)) - 1
        bh_equity = (1 + simple_returns).cumprod()
        plt.plot(bh_equity.index, bh_equity.values,
                 label='Buy & Hold', alpha=0.7, color='blue', linewidth=2)

        colors = ['green', 'red', 'orange', 'purple', 'cyan', 'magenta']
        strategy_types = ['directional', 'only_long']
        best_perf = {}
        best_model = None
        best_strategy_type = None

        strategy_metrics = {}

        for i, ml_pred_col in enumerate(ml_pred_cols):
            model_name = ml_pred_col.replace('ml_y_pred_', '').upper()
            pred_series = plot_data[ml_pred_col]

            # пропускаємо модель, якщо взагалі немає прогнозів
            if not pred_series.notna().any():
                continue

            for strat_type in strategy_types:
                # ---- 1. Метрики стратегії (як у тебе було) ----
                strategy_perf = calculate_strategy_performance(
                    plot_data['log_ret'],
                    pred_series,
                    transaction_cost=0.0,
                    threshold=0.0003,
                    strategy_type=strat_type
                )

                if 'error' in strategy_perf:
                    continue

                key = f"{model_name}_{strat_type}"
                strategy_metrics[key] = strategy_perf

                # ---- 2. Побудова equity-кривої БЕЗ dropna ----
                #   a) сигнали на ВСЬОМУ індексі plot_data
                if strat_type == 'only_long':
                    raw_signals = generate_only_long_signals(pred_series, 0.0003)
                else:
                    raw_signals = generate_signals_with_threshold(pred_series, 0.0003)

                # вирівнюємо по індексу, NaN -> 0 (немає позиції)
                signals = raw_signals.reindex(plot_data.index).fillna(0)

                #   b) позиція з лагом на 1 день (щоб не було заглядання в майбутнє)
                positions = signals.shift(1).fillna(0)

                #   c) прості ретурни (для грошей)
                simple_returns = np.exp(plot_data['log_ret'].fillna(0)) - 1

                #   d) прибуток стратегії
                strategy_returns = positions * simple_returns

                #   e) equity-крива
                equity_curve = (1 + strategy_returns).cumprod()

                # ---- 3. Плотинг ----
                linestyle = '--' if strat_type == 'only_long' else '-'
                label_suffix = ' (Long Only)' if strat_type == 'only_long' else ''
                plt.plot(
                    equity_curve.index,
                    equity_curve.values,
                    label=f'{model_name}{label_suffix}',
                    color=colors[i % len(colors)],
                    linewidth=1.5,
                    linestyle=linestyle
                )

                # ---- 4. Вибір найкращої стратегії по Sharpe ----
                sharpe = strategy_perf.get('sharpe_ratio', -999)
                if not best_perf or sharpe > best_perf.get('sharpe_ratio', -999):
                    best_perf = strategy_perf
                    best_model = model_name
                    best_strategy_type = strat_type

        if best_perf:
            strategy_desc = "Long Only" if best_strategy_type == 'only_long' else "Directional"
            plt.title(
                f'{ticker.upper()} ML Strategies vs Buy & Hold\n'
                f'Best: {best_model} ({strategy_desc}, Sharpe: {best_perf.get("sharpe_ratio", 0):.2f})'
            )
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

            # Повертаємо метрики для текстового summary
            return strategy_metrics

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
                combined_df.loc[mask, col],
                threshold=0.0003
            )
            metrics['ARIMA_Returns'] = arima_ret_metrics

    # ML Returns metrics for each model
    ml_cols = [col for col in combined_df.columns if col.startswith('ml_y_pred_')]
    for col in ml_cols:
        model_name = col.replace('ml_y_pred_', '').upper()
        mask = combined_df['log_ret'].notna() & combined_df[col].notna()
        if mask.sum() > 0:
            ml_metrics = evaluate_regression(
                combined_df.loc[mask, 'log_ret'],
                combined_df.loc[mask, col]
            )
            ml_metrics['Directional_Accuracy'] = directional_accuracy(
                combined_df.loc[mask, 'log_ret'],
                combined_df.loc[mask, col],
                threshold=0.0003
            )
            metrics[f'ML_{model_name}_Returns'] = ml_metrics

    return metrics


def save_metrics_summary(metrics: Dict[str, Dict[str, float]],
                         strategy_metrics: Dict[str, Dict[str, float]],
                         output_path: str = None) -> None:
    """
    Save comprehensive metrics summary including all strategy types.

    Args:
        metrics: Model prediction metrics
        strategy_metrics: Strategy performance metrics for each model/strategy combination
        output_path: Path to save the summary
    """
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), 'reports', 'metrics_summary.txt')

    ensure_dirs(output_path)

    with open(output_path, 'w') as f:
        f.write("Forecasting Models - Final Metrics Summary\n")
        f.write("=" * 50 + "\n\n")

        # Model prediction metrics
        for model_name, model_metrics in metrics.items():
            f.write(f"{model_name}:\n")
            for metric_name, value in model_metrics.items():
                f.write(f"  {metric_name}: {value:.6f}\n")
            f.write("\n")

        # Strategy performance metrics
        if strategy_metrics:
            f.write("Strategy Performance Metrics:\n")
            f.write("-" * 30 + "\n\n")

            # Group by model
            models = set()
            for key in strategy_metrics.keys():
                if '_' in key:
                    model = key.split('_')[0]
                    models.add(model)

            for model in sorted(models):
                f.write(f"{model} Strategies:\n")

                # Directional strategy
                dir_key = f"{model}_directional"
                if dir_key in strategy_metrics:
                    perf = strategy_metrics[dir_key]
                    f.write("  Directional (Long/Short):\n")
                    f.write(f"    Total Return: {perf.get('total_return', 0):.6f}\n")
                    f.write(f"    Annual Return: {perf.get('annual_return', 0):.6f}\n")
                    f.write(f"    Annual Volatility: {perf.get('annual_volatility', 0):.6f}\n")
                    f.write(f"    Sharpe Ratio: {perf.get('sharpe_ratio', 0):.6f}\n")
                    f.write(f"    Max Drawdown: {perf.get('max_drawdown', 0):.6f}\n")
                    f.write(f"    Number of Trades: {perf.get('num_trades', 0)}\n")

                # Only-long strategy
                long_key = f"{model}_only_long"
                if long_key in strategy_metrics:
                    perf = strategy_metrics[long_key]
                    f.write("  Only-Long:\n")
                    f.write(f"    Total Return: {perf.get('total_return', 0):.6f}\n")
                    f.write(f"    Annual Return: {perf.get('annual_return', 0):.6f}\n")
                    f.write(f"    Annual Volatility: {perf.get('annual_volatility', 0):.6f}\n")
                    f.write(f"    Sharpe Ratio: {perf.get('sharpe_ratio', 0):.6f}\n")
                    f.write(f"    Max Drawdown: {perf.get('max_drawdown', 0):.6f}\n")
                    f.write(f"    Number of Trades: {perf.get('num_trades', 0)}\n")

                f.write("\n")

    logging.info(f"Metrics summary saved to {output_path}")


def main():
    """Main function"""
    set_seed(42)
    setup_logging()

    parser = argparse.ArgumentParser(description='Create backtesting plots and metrics')
    parser.add_argument('--ticker', type=str, default='MSFT', help='Stock ticker (default: AAPL)')
    args = parser.parse_args()

    logging.info("Starting backtesting and visualization...")

    try:
        # Load and combine all data
        df_base, predictions = load_all_predictions(args.ticker)
        combined_df = combine_predictions(df_base, predictions)

        # Create plots and get strategy performance
        output_dir = os.path.join(os.path.dirname(__file__), 'reports', f'{args.ticker.lower()}_figures')
        strategy_perf = create_plots(combined_df, output_dir, args.ticker)

        # Calculate final metrics
        final_metrics = calculate_final_metrics(combined_df)

        # Save metrics summary
        output_path = os.path.join(os.path.dirname(__file__), 'reports', f'{args.ticker.lower()}_metrics_summary.txt')
        save_metrics_summary(final_metrics, strategy_perf, output_path)

        logging.info("Backtesting and visualization completed successfully!")

    except Exception as e:
        logging.error(f"Error in backtesting: {e}")
        raise


if __name__ == "__main__":
    main()
