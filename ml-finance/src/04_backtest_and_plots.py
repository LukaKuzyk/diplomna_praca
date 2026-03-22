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

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from utils import (
    set_seed, setup_logging, evaluate_regression,
    directional_accuracy, buy_and_hold_accuracy, ensure_dirs, save_predictions_csv
)
from config import SIGNAL_THRESHOLD

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


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

    # Add ML predictions and y_true (next-day return target)
    for col in df_ml.columns:
        if col not in ['window_id', 'target']:
            combined_df[f"ml_{col}"] = df_ml[col]

    # Remove duplicates and sort
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
    combined_df = combined_df.sort_index()

    logging.info(f"Combined dataset: {len(combined_df)} rows")
    return combined_df


def create_model_comparison_plot(combined_df: pd.DataFrame, output_dir: str, ticker: str) -> None:
    """Create individual model comparison plots"""
    # Get model columns
    model_cols = [col for col in combined_df.columns if col.startswith('ml_y_pred_') and 'LINEAR' not in col]
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan']
    actual_returns = combined_df['ml_y_true']

    # 1. Predictions vs Actual (scatter plot)
    plt.figure(figsize=(10, 6))
    for i, col in enumerate(model_cols):
        pred_returns = combined_df[col]
        mask = actual_returns.notna() & pred_returns.notna()
        if mask.sum() > 0:
            model_name = col.replace('ml_y_pred_', '').upper()
            plt.scatter(actual_returns[mask], pred_returns[mask], alpha=0.6, color=colors[i % len(colors)],
                       label=f'{model_name}', s=20)

    # Perfect prediction line
    min_val = min(actual_returns.min(), combined_df[model_cols].min().min())
    max_val = max(actual_returns.max(), combined_df[model_cols].max().max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, label='Perfect Prediction')
    plt.xlabel('Actual Returns')
    plt.ylabel('Predicted Returns')
    plt.title(f'{ticker.upper()} Model Predictions vs Actual Returns')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comp_pred_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Prediction Error Distribution
    plt.figure(figsize=(10, 6))
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
        plt.hist(errors_data, bins=30, alpha=0.7, label=labels, density=True)
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        plt.xlabel('Prediction Error')
        plt.ylabel('Density')
        plt.title(f'{ticker.upper()} Prediction Error Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comp_error_dist.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Rolling Raw Directional Accuracy (honest, no threshold)
    plt.figure(figsize=(10, 6))
    window_size = 50
    all_model_cols = [col for col in combined_df.columns if col.startswith('ml_y_pred_') or col.startswith('ml_cl_')]
    for i, col in enumerate(all_model_cols):
        pred_returns = combined_df[col]
        mask = actual_returns.notna() & pred_returns.notna()
        if mask.sum() > window_size:
            actual_sign = np.sign(actual_returns[mask])
            if col.startswith('ml_cl_'):
                # Classifier probability: centered around 0.5
                pred_sign = np.sign(pred_returns[mask] - 0.5)
                label_name = col.replace('ml_cl_', 'CL_').upper()
            else:
                pred_sign = np.sign(pred_returns[mask])
                label_name = col.replace('ml_y_pred_', 'REG_').upper()
                
            accuracy = (actual_sign == pred_sign).rolling(window=window_size).mean()
            plt.plot(accuracy.index, accuracy.values, color=colors[i % len(colors)],
                    label=label_name, linewidth=2)

    # Buy & Hold baseline
    bh_mask = actual_returns.notna()
    if bh_mask.sum() > window_size:
        bh_accuracy = (actual_returns[bh_mask] > 0).rolling(window=window_size).mean()
        plt.plot(bh_accuracy.index, bh_accuracy.values, color='grey', linestyle=':',
                linewidth=2, label='Buy & Hold baseline')

    plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.7, label='Random (50%)')
    plt.xlabel('Date')
    plt.ylabel('Raw Directional Accuracy (Rolling)')
    plt.title(f'{ticker.upper()} Rolling Raw DA — No Threshold (Window={window_size})')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comp_rolling_da.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Model Correlation Heatmap (using signals to allow comparing reg vs clf)
    plt.figure(figsize=(10, 8))
    pred_data = combined_df[all_model_cols].dropna().copy()
    if len(pred_data) > 0:
        # Convert all to signals for fair correlation (-1, 0, 1)
        for c in all_model_cols:
            if c.startswith('ml_cl_'):
                pred_data[c] = np.sign(pred_data[c] - 0.5)
            else:
                pred_data[c] = np.sign(pred_data[c])
        
        corr_matrix = pred_data.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, cbar_kws={'shrink': 0.8})
        plt.title(f'{ticker.upper()} Model Signal Correlations')
        
        labels = [c.replace('ml_cl_', 'CL_').replace('ml_y_pred_', 'REG_').upper() for c in all_model_cols]
        plt.xticks(np.arange(len(labels))+0.5, labels, rotation=45, ha='right')
        plt.yticks(np.arange(len(labels))+0.5, labels, rotation=0)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comp_signal_corr.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved individual model comparison plots")


def create_strategy_performance_plot(combined_df: pd.DataFrame, output_dir: str, ticker: str) -> None:
    """Create individual strategy performance comparison plots"""
    model_cols = [col for col in combined_df.columns if col.startswith('ml_y_pred_')]
    first_pred_date = combined_df[[col for col in combined_df.columns if col.startswith('ml_y_pred_')]].notna().any(
        axis=1).idxmax()
    combined_df = combined_df.loc[first_pred_date:]
    logging.info(f"First prediction date: {first_pred_date}")
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan']

    # Calculate strategy metrics for each model
    strategy_results = {}
    all_model_cols = [col for col in combined_df.columns if col.startswith('ml_y_pred_') or col.startswith('ml_cl_')]
    for col in all_model_cols:
        if col.startswith('ml_cl_'):
            model_name = col.replace('ml_cl_', 'CL_').upper()
        else:
            model_name = col.replace('ml_y_pred_', 'REG_').upper()
            
        pred_returns = combined_df[col]
        actual_returns = combined_df['ml_y_true']

        # Only-long strategy
        signals = pd.Series(0, index=pred_returns.index)
        if col.startswith('ml_cl_'):
            signals[pred_returns > 0.5] = 1
        else:
            signals[pred_returns > SIGNAL_THRESHOLD] = 1

        # Calculate returns
        data = pd.DataFrame({
            'returns': np.exp(actual_returns) - 1,
            'signals': signals
        }).dropna()

        if len(data) > 0:
            strategy_returns = data['signals'] * data['returns']
            strategy_returns = strategy_returns.dropna()

            if len(strategy_returns) > 0:
                cumulative = (1 + strategy_returns).cumprod()
                strategy_results[model_name] = {
                    'cumulative': cumulative,
                    'total_return': cumulative.iloc[-1] - 1,
                    'sharpe': strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
                }

    # 1. Equity Curves
    plt.figure(figsize=(10, 6))
    bh_returns = np.exp(combined_df['log_ret'].dropna()) - 1
    bh_cumulative = (1 + bh_returns).cumprod()
    plt.plot(bh_cumulative.index, bh_cumulative.values, 'k-', linewidth=2, label='Buy & Hold')

    for i, (model_name, results) in enumerate(strategy_results.items()):
        plt.plot(results['cumulative'].index, results['cumulative'].values,
                color=colors[i % len(colors)], linewidth=2, label=f'{model_name} Strategy')

    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.title(f'{ticker.upper()} Strategy Equity Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.tight_layout()
    plt.savefig(f'{output_dir}/strat_perf_equity_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Total Returns Bar Chart
    plt.figure(figsize=(10, 6))
    models = list(strategy_results.keys())
    returns = [results['total_return'] for results in strategy_results.values()]
    plt.bar(models, returns, color=colors[:len(models)], alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.7)
    plt.ylabel('Total Return')
    plt.title(f'{ticker.upper()} Total Strategy Returns')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/strat_perf_total_returns.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Monthly Returns Heatmap
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    if len(combined_df) > 60 and strategy_results:
        monthly_data = {}
        for model_name, results in strategy_results.items():
            monthly_cumulative = results['cumulative'].resample('ME').last() if pd.__version__ >= '2.2.0' else results['cumulative'].resample('M').last()
            monthly_ret = monthly_cumulative.pct_change().dropna()
            monthly_data[model_name] = monthly_ret

        monthly_df = pd.DataFrame(monthly_data)
        if len(monthly_df) > 0 and len(monthly_df.columns) > 0:
            month_labels = [d.strftime('%Y-%m') for d in monthly_df.index]
            sns.heatmap(monthly_df.T, cmap='RdYlGn', center=0, ax=ax,
                       cbar_kws={'label': 'Monthly Return'},
                       xticklabels=month_labels, yticklabels=monthly_df.columns)
            ax.set_title(f'{ticker.upper()} Monthly Strategy Returns')
            ax.set_xlabel('Month')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        else:
            ax.text(0.5, 0.5, 'Insufficient data for\nmonthly analysis', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Monthly Returns (N/A)')
    else:
        ax.text(0.5, 0.5, 'Insufficient data for\nmonthly analysis', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Monthly Returns (N/A)')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/strat_perf_monthly_returns.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved individual strategy performance plots")


def create_feature_importance_plot(combined_df: pd.DataFrame, output_dir: str, ticker: str) -> None:
    """Create individual feature importance plots"""
    importance_path = os.path.join(os.path.dirname(__file__), 'reports', f'{ticker.lower()}_feature_importance.csv')
    if not os.path.exists(importance_path):
        logging.warning(f"Feature importance file not found: {importance_path}, skipping plot")
        return

    importance_df = pd.read_csv(importance_path, index_col=0)

    # 1. Top-20 Average Importance
    plt.figure(figsize=(10, 8))
    avg_importance = importance_df.mean(axis=1).sort_values(ascending=True)
    top20 = avg_importance.tail(20)
    colors_bar = plt.cm.viridis(np.linspace(0.3, 0.9, len(top20)))
    plt.barh(range(len(top20)), top20.values, color=colors_bar)
    plt.yticks(range(len(top20)), top20.index, fontsize=8)
    plt.xlabel('Average Importance')
    plt.title(f'{ticker.upper()} Top 20 Features (avg across tree models)')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feat_imp_top20_avg.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Per-model Top-10 comparison
    plt.figure(figsize=(10, 8))
    top10_features = avg_importance.tail(10).index.tolist()
    top10_data = importance_df.loc[top10_features]
    x = np.arange(len(top10_features))
    n_models = len(top10_data.columns)
    bar_width = 0.8 / n_models
    model_colors = plt.cm.Set2(np.linspace(0, 1, n_models))

    for i, model_name in enumerate(top10_data.columns):
        offset = (i - n_models / 2 + 0.5) * bar_width
        plt.barh(x + offset, top10_data[model_name].values, height=bar_width,
                label=model_name, color=model_colors[i], alpha=0.85)

    plt.yticks(x, top10_features, fontsize=8)
    plt.xlabel('Importance')
    plt.title(f'{ticker.upper()} Top 10 Features — per model')
    plt.legend(fontsize=7, loc='lower right')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feat_imp_top10_models.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Feature Correlation with Target
    plt.figure(figsize=(10, 8))
    features_path = os.path.join(os.path.dirname(__file__), 'data', f'{ticker.lower()}_features.csv')
    correlations = {}
    if os.path.exists(features_path):
        from features import create_features
        raw_data = pd.read_csv(features_path, index_col=0)
        
        # We need raw data as it was given, but let's be careful about dates if missing
        try:
            raw_data.index = pd.to_datetime(raw_data.index, utc=True)
        except:
            pass
            
        feature_data = create_features(raw_data)
        next_day_ret = feature_data['log_ret'].shift(-1)
        for col in importance_df.index:
            if col in feature_data.columns:
                corr = feature_data[col].corr(next_day_ret)
                if not np.isnan(corr):
                    correlations[col] = corr

    if correlations:
        corr_series = pd.Series(correlations)
        top_corr = corr_series.reindex(corr_series.abs().sort_values(ascending=True).tail(20).index)
        bar_colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in top_corr.values]
        plt.barh(range(len(top_corr)), top_corr.values, color=bar_colors, alpha=0.8)
        plt.yticks(range(len(top_corr)), top_corr.index, fontsize=8)
        plt.xlabel('Pearson Correlation with next-day log_ret')
        plt.title(f'{ticker.upper()} Top 20 Features — Correlation with Target')
        plt.axvline(x=0, color='black', linewidth=0.8)
        plt.grid(True, alpha=0.3, axis='x')
    else:
        plt.text(0.5, 0.5, 'Feature data not available\nfor correlation analysis', ha='center', va='center')
        
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feat_imp_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Feature Category Breakdown
    plt.figure(figsize=(10, 8))
    categories = {
        'Technical\n(SMA, RSI, MACD, BB, etc.)': ['sma_5', 'sma_20', 'rsi_14', 'macd', 'macd_signal',
                                                    'bb_upper', 'bb_lower', 'bb_middle', 'stoch_k', 'stoch_d',
                                                    'atr_14', 'cci_20', 'momentum_5', 'momentum_10'],
        'Return Lags\n(log_ret_lag_*)': [c for c in importance_df.index if c.startswith('log_ret_lag')],
        'Volume\n(volume, volume_lag, MA)': [c for c in importance_df.index if 'volume' in c],
        'Market\n(VIX, QQQ)': [c for c in importance_df.index if 'vix' in c or 'qqq' in c],
        'Search Trends\n(Google Trends)': [c for c in importance_df.index if 'search' in c],
        'News Trends\n(Google News)': [c for c in importance_df.index if 'news' in c and 'earnings' not in c],
        'Statistical\n(skew, kurt, vol)': ['rolling_skew_20', 'rolling_kurt_20', 'volatility'],
        'Calendar\n(day, month)': ['day_of_week', 'month'],
        'Earnings\n(earnings_week)': ['earnings_week'],
    }

    cat_importance = {}
    for cat_name, cols in categories.items():
        valid_cols = [c for c in cols if c in importance_df.index]
        if valid_cols:
            cat_importance[cat_name] = importance_df.loc[valid_cols].mean(axis=1).sum()

    if cat_importance:
        cat_series = pd.Series(cat_importance).sort_values(ascending=True)
        custom_colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']
        pie_colors = custom_colors[:len(cat_series)]
        plt.barh(range(len(cat_series)), cat_series.values, color=pie_colors)
        plt.yticks(range(len(cat_series)), cat_series.index, fontsize=8)
        plt.xlabel('Total Importance (sum)')
        plt.title(f'{ticker.upper()} Importance by Feature Category')
        plt.grid(True, alpha=0.3, axis='x')
    else:
        plt.text(0.5, 0.5, 'No matching categories found', ha='center', va='center')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/feat_imp_categories.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved individual feature importance plots")


def create_shap_analysis_plot(output_dir: str, ticker: str) -> None:
    """Create individual SHAP analysis plots using RandomForest model on the full dataset."""
    if not SHAP_AVAILABLE:
        logging.warning("SHAP not available, skipping SHAP analysis plot")
        return

    features_path = os.path.join(os.path.dirname(__file__), 'data', f'{ticker.lower()}_features.csv')
    if not os.path.exists(features_path):
        logging.warning(f"Features file not found: {features_path}, skipping SHAP plot")
        return

    from features import create_features, select_features_lasso
    from sklearn.preprocessing import StandardScaler

    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError:
        logging.warning("RandomForest not available, skipping SHAP analysis")
        return

    raw_data = pd.read_csv(features_path, index_col=0)
    raw_data.index = pd.to_datetime(raw_data.index, utc=True)
    feature_data = create_features(raw_data)
    target = feature_data['log_ret'].shift(-1).dropna()
    feature_data = feature_data.loc[target.index]
    selected_features = select_features_lasso(feature_data, target)

    X = feature_data[selected_features]
    y = target

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)

    model = RandomForestRegressor(n_estimators=100, max_depth=5, 
                                  max_features='sqrt', random_state=42, n_jobs=-1)
    model.fit(X_scaled, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_scaled)

    # 1. Summary bee swarm plot
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title(f'{ticker.upper()} SHAP Summary (Bee Swarm)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shap_beeswarm.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Mean absolute SHAP values (bar)
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.title(f'{ticker.upper()} Mean |SHAP| Value (Global Importance)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shap_bar.png', dpi=300, bbox_inches='tight')
    plt.close()

    logging.info("Saved individual SHAP analysis plots")


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

    # Create feature analysis plot
    create_feature_importance_plot(combined_df, output_dir, ticker)

    # Create SHAP analysis plot
    create_shap_analysis_plot(output_dir, ticker)

    logging.info("All ML analysis plots created successfully")


def calculate_ml_metrics(combined_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Calculate ML model performance metrics"""
    logging.info("Calculating ML model metrics...")

    metrics = {}

    # Buy & Hold baseline
    bh_acc = buy_and_hold_accuracy(combined_df['log_ret'].dropna())
    metrics['Baseline'] = {'Buy_and_Hold_DA': bh_acc}

    # ML Returns metrics for each model
    ml_cols = [col for col in combined_df.columns if col.startswith('ml_y_pred_')]
    for col in ml_cols:
        model_name = col.replace('ml_y_pred_', '').upper()
        mask = combined_df['ml_y_true'].notna() & combined_df[col].notna()
        if mask.sum() > 0:
            ml_metrics = evaluate_regression(
                combined_df.loc[mask, 'ml_y_true'],
                combined_df.loc[mask, col]
            )
            da = directional_accuracy(
                combined_df.loc[mask, 'ml_y_true'],
                combined_df.loc[mask, col],
                threshold=SIGNAL_THRESHOLD
            )
            ml_metrics['Raw_DA'] = da['raw_da']
            ml_metrics['Confident_DA'] = da['confident_da']
            ml_metrics['Coverage'] = da['coverage']
            ml_metrics['Total_Test_Days'] = int(mask.sum())
            metrics[f'ML_REG_{model_name}_Returns'] = ml_metrics

    # ML Classification metrics for each model
    cl_cols = [col for col in combined_df.columns if col.startswith('ml_cl_')]
    for col in cl_cols:
        model_name = col.replace('ml_cl_', '').upper()
        mask = combined_df['ml_y_true'].notna() & combined_df[col].notna()
        if mask.sum() > 0:
            mapped_preds = combined_df.loc[mask, col] - 0.5
            da = directional_accuracy(
                combined_df.loc[mask, 'ml_y_true'],
                mapped_preds,
                threshold=0.05
            )
            metrics[f'ML_CL_{model_name}_Probability'] = {
                'Raw_DA': da['raw_da'],
                'Confident_DA': da['confident_da'],
                'Coverage': da['coverage'],
                'Mean_Probability': float(combined_df.loc[mask, col].mean()),
                'Total_Test_Days': int(mask.sum())
            }

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
