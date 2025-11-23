#!/usr/bin/env python3
"""
Predict next day AAPL log-return and provide trading recommendation
"""
import argparse
import logging
import warnings
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

from utils import (
    set_seed, setup_logging, evaluate_regression,
    directional_accuracy, ensure_dirs
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Try to import all ML libraries
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

SIGNAL_THRESHOLD = 0.002


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create ML features from the dataset (updated to match 03_model_ml.py)"""
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
    if 'snp500' in df.columns:
        features_df['snp500_change'] = df['snp500'].pct_change()
    else:
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
    """Get dictionary of ML models to compare (same as in 03_model_ml.py)"""
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

    # Gradient Boosting Regressor
    models['gbr'] = (GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=random_state), StandardScaler())

    # LightGBM (if available)
    if LGBM_AVAILABLE:
        models['lgbm'] = (LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=random_state), StandardScaler())

    # CatBoost (if available)
    if CATBOOST_AVAILABLE:
        models['cat'] = (CatBoostRegressor(iterations=100, depth=5, learning_rate=0.05, random_state=random_state, verbose=False), StandardScaler())

    return models


class MLModelPredictor:
    """ML model predictor with multiple models"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}

        # Initialize all available models
        all_models = get_ml_models(random_state)
        for model_name, (model, scaler) in all_models.items():
            self.models[model_name] = model
            self.scalers[model_name] = scaler
            logging.info(f"Initialized {model_name.upper()} model")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit all models"""
        for model_name, model in self.models.items():
            # Scale features
            X_scaled = self.scalers[model_name].fit_transform(X)

            # Fit model
            model.fit(X_scaled, y)
            logging.info(f"{model_name.upper()} model fitted successfully")

    def predict(self, X: pd.DataFrame, model_name: str = 'xgb') -> float:
        """Make prediction with specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")

        X_scaled = self.scalers[model_name].transform(X)
        prediction = self.models[model_name].predict(X_scaled)[0]
        return prediction

    def predict_all(self, X: pd.DataFrame) -> Dict[str, float]:
        """Make predictions with all models"""
        predictions = {}
        for model_name in self.models.keys():
            predictions[model_name] = self.predict(X, model_name)
        return predictions


def load_model_metrics(ticker: str) -> Dict[str, float]:
    """Load directional accuracy metrics for each model"""
    import os
    # Try ML metrics file first
    metrics_path = Path(os.path.join(os.path.dirname(__file__), 'reports', f'{ticker.lower()}_ml_metrics_summary.txt'))

    if not metrics_path.exists():
        # Fallback to regular metrics file
        metrics_path = Path(os.path.join(os.path.dirname(__file__), 'reports', f'{ticker.lower()}_metrics_summary.txt'))
        if not metrics_path.exists():
            logging.warning(f"Metrics file not found: {metrics_path}")
            return {}

    da_metrics = {}
    try:
        with open(metrics_path, 'r') as f:
            content = f.read()

        # Parse DA values for each model
        lines = content.split('\n')
        current_model = None

        for line in lines:
            line = line.strip()
            if line.endswith('_Returns:'):
                current_model = line.replace('_Returns:', '').replace('ML_', '')
            elif line.startswith('Directional_Accuracy:') and current_model:
                da_value = float(line.split(':')[1].strip())
                da_metrics[current_model.lower()] = da_value
                current_model = None

    except Exception as e:
        logging.warning(f"Error loading metrics: {e}")

    return da_metrics


class BaselineModels:
    """Baseline models for time series forecasting (simplified from 02_model_baselines.py)"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def naive_forecast(self, train: pd.Series, target: str) -> float:
        """Naive forecast"""
        if target == 'close':
            return train.iloc[-1]
        elif target == 'log_ret':
            return 0.0
        else:
            raise ValueError(f"Unknown target: {target}")

    def fit_arima(self, train: pd.Series, target: str) -> ARIMA:
        """Fit ARIMA model"""
        if target == 'log_ret':
            order = (1, 0, 1)
        elif target == 'close':
            order = (1, 1, 1)
        else:
            raise ValueError(f"Unknown target: {target}")

        try:
            model = ARIMA(train, order=order)
            return model.fit()
        except Exception as e:
            logging.warning(f"ARIMA fitting failed: {e}")
            # Fallback
            if target == 'log_ret':
                model = ARIMA(train, order=(1, 0, 0))
            else:
                model = ARIMA(train, order=(0, 1, 0))
            return model.fit()

    def forecast_arima(self, model_fit: ARIMA) -> float:
        """Get ARIMA forecast for next day"""
        try:
            forecast = model_fit.forecast(steps=1)
            return forecast.iloc[0]
        except Exception as e:
            logging.warning(f"ARIMA forecasting failed: {e}")
            return 0.0

    def fit_garch(self, returns: pd.Series) -> arch_model:
        """Fit GARCH model"""
        try:
            model = arch_model(returns, mean='AR', vol='GARCH', p=1, q=1, dist='StudentsT')
            return model.fit(disp='off')
        except Exception as e:
            logging.warning(f"GARCH fitting failed: {e}")
            try:
                model = arch_model(returns, mean='Constant', vol='GARCH', p=1, q=1)
                return model.fit(disp='off')
            except Exception as e2:
                logging.error(f"GARCH fallback failed: {e2}")
                raise

    def forecast_garch(self, model_fit: arch_model) -> float:
        """Get GARCH volatility forecast (for log_ret mean)"""
        try:
            forecast = model_fit.forecast(horizon=1)
            return 0.0  # Mean forecast = 0 for simplicity
        except Exception as e:
            logging.warning(f"GARCH forecasting failed: {e}")
            return 0.0


def predict_next_day(ticker: str = 'AAPL') -> Dict[str, any]:
    """Predict next day log-return and provide recommendation using all models"""
    logging.info(f"Starting next day prediction for {ticker} with all models...")

    # Load data
    import os
    data_path = Path(os.path.join(os.path.dirname(__file__), 'data', f'{ticker.lower()}_features.csv'))
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)

    # Create features
    df_features = create_features(df)

    # Define feature columns for ML (same as in 03_model_ml.py)
    feature_cols = [
        'log_ret_lag_1', 'log_ret_lag_2', 'log_ret_lag_3', 'log_ret_lag_5', 'log_ret_lag_7', 'log_ret_lag_10', 'log_ret_lag_14', 'log_ret_lag_15', 'log_ret_lag_20', 'log_ret_lag_21', 'log_ret_lag_30',
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
        logging.warning(f"Missing features: {missing_features}")
        # Remove missing features from the list
        feature_cols = [col for col in feature_cols if col in df_features.columns]

    # Use all available data for training
    train_features = df_features[feature_cols]
    train_target = df_features['log_ret']
    train_close = df_features['close']

    if len(train_features) < 50:
        raise ValueError("Not enough data for training")

    # Initialize models
    baseline_models = BaselineModels()
    ml_predictor = MLModelPredictor()

    # Fit models
    ml_predictor.fit(train_features, train_target)

    # Prepare features for next day prediction
    last_row = df_features.iloc[-1]
    next_day_features = pd.DataFrame([last_row[feature_cols].values], columns=feature_cols)

    # Get predictions from all ML models
    ml_predictions = ml_predictor.predict_all(next_day_features)

    # Get the best model prediction (using the one with highest directional accuracy from backtesting)
    # For simplicity, use XGB if available, otherwise the first available
    primary_model = 'xgb' if 'xgb' in ml_predictions else list(ml_predictions.keys())[0]
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

    # Load directional accuracy metrics
    da_metrics = load_model_metrics(ticker)

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
        'recommendation': recommendation,
        'reason': reason,
        'expected_return_pct': expected_return_pct,
        'threshold': threshold,
        'primary_model': primary_model.upper(),
        'historical_data': df_features[['close']].tail(30)  # Last 30 days (1 month) for plotting
    }

    logging.info(f"Prediction completed for {ticker}: {result['recommendation']}")
    return result


def create_prediction_plot(result: Dict[str, any], output_dir: str = 'reports/figures') -> None:
    """Create plot showing monthly price chart with next day ML predictions"""
    import os
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
    import os
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
        for model, pred in result['predictions'].items():
            if model.startswith('ML_'):
                model_key = model.replace('ML_', '').lower()
                da_value = da_metrics.get(model_key, 'N/A')
                if da_value != 'N/A':
                    da_str = f"DA: {da_value:.1%}"
                else:
                    da_str = "DA: N/A"
                print(f"  {model}: Price ${pred['close']:.2f}, Return {pred['log_ret']:.6f}, {da_str}")
        print(f"\nExpected Return (ML): {result['expected_return_pct']:.2f}%")
        print(f"RECOMMENDATION: {result['recommendation']}")
        print(f"Reason: {result['reason']}")
        print("="*60)

        # Save key results to file for report generation
        output_file = os.path.join(os.path.dirname(__file__), 'reports', f'{args.ticker.lower()}_next_day_prediction.txt')
        ensure_dirs(os.path.dirname(output_file))

        with open(output_file, 'w') as f:
            f.write(f"Best Model: {result['primary_model']}\n")
            f.write(f"Predicted Return: {result['expected_return_pct']:.6f}\n")
            f.write(f"Confidence: {da_metrics.get(result['primary_model'].lower(), 0):.6f}\n")
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