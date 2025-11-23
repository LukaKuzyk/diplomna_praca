#!/usr/bin/env python3
"""
ML Model Backtesting and Advanced Visualization for Stock Forecasting
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
import seaborn as sns
from pathlib import Path
from scipy import stats

from utils import (
    set_seed, setup_logging, evaluate_regression,
    directional_accuracy, ensure_dirs
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Signal threshold for trading decisions
SIGNAL_THRESHOLD = 0.002


def load_ml_predictions(ticker: str = 'AAPL') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load ML predictions and base data"""
    logging.info("Loading ML predictions and base data...")

    # Load base data
    data_path = Path(os.path.join(os.path.dirname(__file__), 'data', f'{ticker.lower()}_features.csv'))
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df_base = pd.read_csv(data_path, index_col=0)
    df_base.index = pd.to_datetime(df_base.index, utc=True)

    # Load ML predictions
    ml_path = Path(os.path.join(os.path.dirname(__file__), 'models', f'{ticker.lower()}_ml_predictions.csv'))
    if not ml_path.exists():
        raise FileNotFoundError(f"ML predictions file not found: {ml_path}")

    df_ml = pd.read_csv(ml_path)
    df_ml['date'] = pd.to_datetime(df_ml['date'], utc=True)
    df_ml.set_index('date', inplace=True)

    logging.info(f"Loaded base data: {len(df_base)} rows")
    logging.info(f"Loaded ML predictions: {len(df_ml)} rows")

    return df_base, df_ml


def combine_ml_data(df_base: pd.DataFrame, df_ml: pd.DataFrame) -> pd.DataFrame:
    """Combine base data with ML predictions"""
    logging.info("Combining base data with ML predictions...")

    # Start with base data
    combined_df = df_base.copy()

    # Remove duplicate indices in ML predictions
    df_ml = df_ml[~df_ml.index.duplicated(keep='last')]

    # Add ML predictions
    for col in df_ml.columns:
        if col not in ['window_id', 'target', 'y_true']:
            combined_df[f"ml_{col}"] = df_ml[col]

    # Remove duplicates and sort
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
    combined_df = combined_df.sort_index()

    logging.info(f"Combined dataset: {len(combined_df)} rows")
    return combined_df


def create_model_comparison_plot(combined_df: pd.DataFrame, output_dir: str, ticker: str) -> None:
    """Create comprehensive model comparison plot"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{ticker.upper()} ML Models: Comprehensive Analysis', fontsize=16, fontweight='bold')

    # Get model columns
    model_cols = [col for col in combined_df.columns if col.startswith('ml_y_pred_')]
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan']

    # 1. Predictions vs Actual (scatter plot)
    ax1 = axes[0, 0]
    actual_returns = combined_df['log_ret']
    for i, col in enumerate(model_cols):
        pred_returns = combined_df[col]
        mask = actual_returns.notna() & pred_returns.notna()
        if mask.sum() > 0:
            model_name = col.replace('ml_y_pred_', '').upper()
            ax1.scatter(actual_returns[mask], pred_returns[mask], alpha=0.6, color=colors[i % len(colors)],
                       label=f'{model_name}', s=20)

    # Perfect prediction line
    min_val = min(actual_returns.min(), combined_df[model_cols].min().min())
    max_val = max(actual_returns.max(), combined_df[model_cols].max().max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, label='Perfect Prediction')
    ax1.set_xlabel('Actual Returns')
    ax1.set_ylabel('Predicted Returns')
    ax1.set_title('Model Predictions vs Actual Returns')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Prediction Error Distribution
    ax2 = axes[0, 1]
    errors_data = []
    labels = []
    for col in model_cols:
        pred_returns = combined_df[col]
        mask = actual_returns.notna() & pred_returns.notna()
        if mask.sum() > 0:
            errors = pred_returns[mask] - actual_returns[mask]
            errors_data.append(errors)
            labels.append(col.replace('ml_y_pred_', '').upper())

    if errors_data:
        ax2.hist(errors_data, bins=30, alpha=0.7, label=labels, density=True)
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Prediction Error')
        ax2.set_ylabel('Density')
        ax2.set_title('Prediction Error Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # 3. Rolling Directional Accuracy
    ax3 = axes[1, 0]
    window_size = 50  # Rolling window for accuracy calculation
    for i, col in enumerate(model_cols):
        pred_returns = combined_df[col]
        mask = actual_returns.notna() & pred_returns.notna()
        if mask.sum() > window_size:
            # Calculate rolling directional accuracy
            actual_sign = np.sign(actual_returns[mask])
            pred_sign = np.sign(pred_returns[mask])
            accuracy = (actual_sign == pred_sign).rolling(window=window_size).mean()
            ax3.plot(accuracy.index, accuracy.values, color=colors[i % len(colors)],
                    label=col.replace('ml_y_pred_', '').upper(), linewidth=2)

    ax3.axhline(y=0.5, color='black', linestyle='--', alpha=0.7, label='Random (50%)')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Directional Accuracy (Rolling)')
    ax3.set_title(f'Rolling Directional Accuracy (Window={window_size})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # 4. Model Correlation Heatmap
    ax4 = axes[1, 1]
    pred_data = combined_df[model_cols].dropna()
    if len(pred_data) > 0:
        corr_matrix = pred_data.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax4, cbar_kws={'shrink': 0.8})
        ax4.set_title('Model Prediction Correlations')
        ax4.set_xticklabels([col.replace('ml_y_pred_', '').upper() for col in model_cols],
                           rotation=45, ha='right')
        ax4.set_yticklabels([col.replace('ml_y_pred_', '').upper() for col in model_cols],
                           rotation=0)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved model_comparison.png")


def create_strategy_performance_plot(combined_df: pd.DataFrame, output_dir: str, ticker: str) -> None:
    """Create strategy performance comparison plot"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{ticker.upper()} ML Strategy Performance Analysis', fontsize=16, fontweight='bold')

    model_cols = [col for col in combined_df.columns if col.startswith('ml_y_pred_')]
    first_pred_mask = combined_df[model_cols].notna().any(axis=1)
    first_pred_date = combined_df[[col for col in combined_df.columns if col.startswith('ml_y_pred_')]].notna().any(
        axis=1).idxmax()
    combined_df = combined_df.loc[first_pred_date:]

    # combined_cut = combined_df.loc[first_pred_date:].copy()
    # combined_df = combined_cut
    logging.info(f"First prediction date: {first_pred_date}")
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan']

    # Calculate strategy metrics for each model
    strategy_results = {}

    for col in model_cols:
        model_name = col.replace('ml_y_pred_', '').upper()
        pred_returns = combined_df[col]
        actual_returns = combined_df['log_ret']

        # Only-long strategy
        signals = pd.Series(0, index=pred_returns.index)
        signals[pred_returns > SIGNAL_THRESHOLD] = 1

        # Calculate returns
        data = pd.DataFrame({
            'returns': np.exp(actual_returns) - 1,
            'signals': signals
        }).dropna()

        if len(data) > 0:
            strategy_returns = data['signals'].shift(1) * data['returns']
            strategy_returns = strategy_returns.dropna()

            if len(strategy_returns) > 0:
                cumulative = (1 + strategy_returns).cumprod()
                strategy_results[model_name] = {
                    'cumulative': cumulative,
                    'total_return': cumulative.iloc[-1] - 1,
                    'sharpe': strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
                }

    # 1. Equity Curves
    ax1 = axes[0, 0]
    # Buy & Hold
    bh_returns = np.exp(combined_df['log_ret'].dropna()) - 1
    bh_cumulative = (1 + bh_returns).cumprod()
    ax1.plot(bh_cumulative.index, bh_cumulative.values, 'k-', linewidth=2, label='Buy & Hold')

    for i, (model_name, results) in enumerate(strategy_results.items()):
        ax1.plot(results['cumulative'].index, results['cumulative'].values,
                color=colors[i % len(colors)], linewidth=2, label=f'{model_name} Strategy')

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Returns')
    ax1.set_title('Strategy Equity Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # 2. Total Returns Bar Chart
    ax2 = axes[0, 1]
    models = list(strategy_results.keys())
    returns = [results['total_return'] for results in strategy_results.values()]
    ax2.bar(models, returns, color=colors[:len(models)], alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.7)
    ax2.set_ylabel('Total Return')
    ax2.set_title('Total Strategy Returns')
    ax2.tick_params(axis='x', rotation=45)

    # 3. Sharpe Ratios
    ax3 = axes[1, 0]
    sharpes = [results['sharpe'] for results in strategy_results.values()]
    ax3.bar(models, sharpes, color=colors[:len(models)], alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.7)
    ax3.set_ylabel('Sharpe Ratio')
    ax3.set_title('Strategy Sharpe Ratios')
    ax3.tick_params(axis='x', rotation=45)

    # 4. Monthly Returns Heatmap
    ax4 = axes[1, 1]
    if len(combined_df) > 60 and strategy_results:  # Need at least 2-3 months and strategy data
        # Calculate monthly returns for each strategy
        monthly_data = {}
        for model_name, results in strategy_results.items():
            # Resample to monthly and calculate monthly returns
            monthly_cumulative = results['cumulative'].resample('M').last()
            monthly_ret = monthly_cumulative.pct_change().dropna()  # Remove first NaN
            monthly_data[model_name] = monthly_ret

        monthly_df = pd.DataFrame(monthly_data)
        if len(monthly_df) > 0 and len(monthly_df.columns) > 0:
            # Create clean month labels
            month_labels = [d.strftime('%Y-%m') for d in monthly_df.index]

            sns.heatmap(monthly_df.T, cmap='RdYlGn', center=0, ax=ax4,
                       cbar_kws={'label': 'Monthly Return'},
                       xticklabels=month_labels, yticklabels=monthly_df.columns)
            ax4.set_title('Monthly Strategy Returns')
            ax4.set_xlabel('Month')
            # Rotate x-axis labels for better readability
            plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        else:
            ax4.text(0.5, 0.5, 'Insufficient data for\nmonthly analysis',
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Monthly Returns (N/A)')
    else:
        ax4.text(0.5, 0.5, 'Insufficient data for\nmonthly analysis',
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Monthly Returns (N/A)')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/strategy_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved strategy_performance.png")


def create_prediction_stability_plot(combined_df: pd.DataFrame, output_dir: str, ticker: str) -> None:
    """Create prediction stability and confidence analysis plot"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{ticker.upper()} ML Prediction Stability Analysis', fontsize=16, fontweight='bold')

    model_cols = [col for col in combined_df.columns if col.startswith('ml_y_pred_')]
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan']

    # 1. Prediction Variance Over Time
    ax1 = axes[0, 0]
    for i, col in enumerate(model_cols):
        pred_returns = combined_df[col]
        rolling_std = pred_returns.rolling(window=30).std()
        ax1.plot(rolling_std.index, rolling_std.values, color=colors[i % len(colors)],
                label=col.replace('ml_y_pred_', '').upper(), linewidth=2)

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Prediction Volatility (Rolling Std)')
    ax1.set_title('Prediction Stability Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # 2. Model Agreement Matrix
    ax2 = axes[0, 1]
    if len(model_cols) > 1:
        # Calculate agreement between models (same directional prediction)
        agreement_matrix = pd.DataFrame(index=[col.replace('ml_y_pred_', '').upper() for col in model_cols],
                                       columns=[col.replace('ml_y_pred_', '').upper() for col in model_cols])

        for i, col1 in enumerate(model_cols):
            for j, col2 in enumerate(model_cols):
                pred1 = combined_df[col1]
                pred2 = combined_df[col2]
                mask = pred1.notna() & pred2.notna()
                if mask.sum() > 0:
                    agreement = ((np.sign(pred1[mask]) == np.sign(pred2[mask])) & (np.abs(pred1[mask]) > SIGNAL_THRESHOLD) & (np.abs(pred2[mask]) > SIGNAL_THRESHOLD)).mean()
                    agreement_matrix.iloc[i, j] = agreement

        # Convert to float and handle NaN
        agreement_matrix = agreement_matrix.astype(float)
        sns.heatmap(agreement_matrix, annot=True, cmap='Blues', vmin=0, vmax=1, ax=ax2,
                   cbar_kws={'label': 'Agreement Rate'})
        ax2.set_title('Model Directional Agreement')
    else:
        ax2.text(0.5, 0.5, 'Need multiple models\nfor agreement analysis',
                ha='center', va='center', transform=ax2.transAxes)

    # 3. Prediction Magnitude Distribution
    ax3 = axes[1, 0]
    for i, col in enumerate(model_cols):
        pred_returns = combined_df[col].dropna()
        if len(pred_returns) > 0:
            ax3.hist(np.abs(pred_returns), bins=30, alpha=0.7, color=colors[i % len(colors)],
                    label=col.replace('ml_y_pred_', '').upper(), density=True)

    ax3.axvline(x=SIGNAL_THRESHOLD, color='black', linestyle='--', alpha=0.7, label='Signal Threshold')
    ax3.set_xlabel('Prediction Magnitude')
    ax3.set_ylabel('Density')
    ax3.set_title('Prediction Confidence Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Hit Rate by Prediction Magnitude
    ax4 = axes[1, 1]
    actual_returns = combined_df['log_ret']

    magnitude_bins = [(0, 0.001), (0.001, 0.002), (0.002, 0.005), (0.005, 0.01), (0.01, 0.1)]
    bin_labels = ['0-0.1%', '0.1-0.2%', '0.2-0.5%', '0.5-1%', '1%+']

    for i, col in enumerate(model_cols[:3]):  # Show only first 3 models to avoid clutter
        pred_returns = combined_df[col]
        mask = actual_returns.notna() & pred_returns.notna()
        if mask.sum() > 0:
            pred_magnitude = np.abs(pred_returns[mask])
            actual_sign = np.sign(actual_returns[mask])
            pred_sign = np.sign(pred_returns[mask])

            hit_rates = []
            for bin_start, bin_end in magnitude_bins:
                bin_mask = (pred_magnitude >= bin_start) & (pred_magnitude < bin_end)
                if bin_mask.sum() > 0:
                    hit_rate = (actual_sign[bin_mask] == pred_sign[bin_mask]).mean()
                    hit_rates.append(hit_rate)
                else:
                    hit_rates.append(np.nan)

            ax4.plot(bin_labels, hit_rates, 'o-', color=colors[i % len(colors)],
                    label=col.replace('ml_y_pred_', '').upper(), linewidth=2, markersize=8)

    ax4.axhline(y=0.5, color='black', linestyle='--', alpha=0.7, label='Random')
    ax4.set_xlabel('Prediction Magnitude')
    ax4.set_ylabel('Directional Accuracy')
    ax4.set_title('Accuracy by Prediction Confidence')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/prediction_stability.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved prediction_stability.png")


def create_feature_importance_plot(combined_df: pd.DataFrame, output_dir: str, ticker: str) -> None:
    """Create feature analysis plot (correlation and importance)"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{ticker.upper()} Feature Analysis', fontsize=16, fontweight='bold')

    # Get feature columns (exclude predictions and target)
    exclude_cols = ['log_ret', 'close'] + [col for col in combined_df.columns if col.startswith('ml_')]
    feature_cols = [col for col in combined_df.columns if col not in exclude_cols and not col.startswith('ml_')]

    if len(feature_cols) > 10:
        feature_cols = feature_cols[:10]  # Limit to top 10 features

    # 1. Feature Correlation with Target
    ax1 = axes[0, 0]
    correlations = []
    for col in feature_cols:
        corr = combined_df[col].corr(combined_df['log_ret'])
        if not np.isnan(corr):
            correlations.append((col, corr))

    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    features = [x[0] for x in correlations[:10]]
    corrs = [x[1] for x in correlations[:10]]

    bars = ax1.barh(features, corrs, color=['red' if x < 0 else 'green' for x in corrs], alpha=0.7)
    ax1.set_xlabel('Correlation with Returns')
    ax1.set_title('Feature Correlation with Target')
    ax1.grid(True, alpha=0.3)

    # 2. Feature Volatility
    ax2 = axes[0, 1]
    volatilities = []
    for col in feature_cols:
        if combined_df[col].std() > 0:
            volatilities.append((col, combined_df[col].std()))

    volatilities.sort(key=lambda x: x[1], reverse=True)
    features_vol = [x[0] for x in volatilities[:10]]
    vols = [x[1] for x in volatilities[:10]]

    ax2.barh(features_vol, vols, color='skyblue', alpha=0.7)
    ax2.set_xlabel('Standard Deviation')
    ax2.set_title('Feature Volatility')
    ax2.grid(True, alpha=0.3)

    # 3. Feature Distribution Comparison
    ax3 = axes[1, 0]
    # Show distribution of top 3 correlated features
    top_features = [x[0] for x in correlations[:3]]
    for i, feature in enumerate(top_features):
        data = combined_df[feature].dropna()
        if len(data) > 0:
            ax3.hist(data, bins=30, alpha=0.7, label=feature, density=True)

    ax3.set_xlabel('Feature Value')
    ax3.set_ylabel('Density')
    ax3.set_title('Top Feature Distributions')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Autocorrelation Analysis
    ax4 = axes[1, 1]
    lags = range(1, 11)
    autocorr = [combined_df['log_ret'].autocorr(lag=lag) for lag in lags]

    ax4.bar(lags, autocorr, color='lightcoral', alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.7)
    ax4.set_xlabel('Lag (Days)')
    ax4.set_ylabel('Autocorrelation')
    ax4.set_title('Returns Autocorrelation')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved feature_analysis.png")


def create_plots(combined_df: pd.DataFrame, output_dir: str, ticker: str) -> None:
    """Create all ML-focused plots"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    ensure_dirs(output_dir)
    logging.info(f"Creating ML analysis plots in {output_dir}...")

    # Create comprehensive model comparison plot
    create_model_comparison_plot(combined_df, output_dir, ticker)

    # Create strategy performance analysis
    create_strategy_performance_plot(combined_df, output_dir, ticker)

    # Create prediction stability analysis
    create_prediction_stability_plot(combined_df, output_dir, ticker)

    # Create feature analysis plot
    create_feature_importance_plot(combined_df, output_dir, ticker)

    logging.info("All ML analysis plots created successfully")


def calculate_ml_metrics(combined_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Calculate ML model performance metrics"""
    logging.info("Calculating ML model metrics...")

    metrics = {}

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
                threshold=SIGNAL_THRESHOLD
            )
            metrics[f'ML_{model_name}_Returns'] = ml_metrics

    return metrics


def save_ml_metrics_summary(metrics: Dict[str, Dict[str, float]], output_path: str) -> None:
    """Save ML model metrics summary"""
    ensure_dirs(output_path)

    with open(output_path, 'w') as f:
        f.write("ML Models Performance Summary\n")
        f.write("=" * 40 + "\n\n")

        for model_name, model_metrics in metrics.items():
            f.write(f"{model_name}:\n")
            for metric_name, value in model_metrics.items():
                f.write(f"  {metric_name}: {value:.6f}\n")
            f.write("\n")

    logging.info(f"ML metrics summary saved to {output_path}")


def main():
    """Main function"""
    set_seed(42)
    setup_logging()

    parser = argparse.ArgumentParser(description='Create ML model analysis plots and metrics')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker (default: AAPL)')
    args = parser.parse_args()

    logging.info(f"Starting ML analysis for {args.ticker}...")

    try:
        # Load and combine data
        df_base, df_ml = load_ml_predictions(args.ticker)
        combined_df = combine_ml_data(df_base, df_ml)

        # Create comprehensive plots
        output_dir = os.path.join(os.path.dirname(__file__), 'reports', f'{args.ticker.lower()}_figures')
        create_plots(combined_df, output_dir, args.ticker)

        # Calculate and save metrics
        metrics = calculate_ml_metrics(combined_df)
        output_path = os.path.join(os.path.dirname(__file__), 'reports', f'{args.ticker.lower()}_ml_metrics_summary.txt')
        save_ml_metrics_summary(metrics, output_path)

        logging.info("ML analysis completed successfully!")

    except Exception as e:
        logging.error(f"Error in ML analysis: {e}")
        raise


if __name__ == "__main__":
    main()
