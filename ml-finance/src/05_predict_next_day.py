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
    """Create ML features from the dataset (same as in 03_model_ml.py)"""
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
    """ML model wrapper with XGBoost/RandomForest fallback (same as in 03_model_ml.py)"""

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


def predict_next_day() -> Dict[str, any]:
    """Predict next day log-return and provide recommendation using all models"""
    logging.info("Starting next day prediction with all models...")

    # Load data
    data_path = Path('data/aapl_features.csv')
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)

    # Create features
    df_features = create_features(df)

    # Define feature columns for ML
    feature_cols = [
        'log_ret_lag_1', 'log_ret_lag_2', 'log_ret_lag_5', 'log_ret_lag_10',
        'sma_5', 'sma_20', 'rsi_14', 'volatility',
        'day_of_week', 'month'
    ]

    # Check if all features exist
    missing_features = [col for col in feature_cols if col not in df_features.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    # Use all available data for training
    train_features = df_features[feature_cols]
    train_target = df_features['log_ret']
    train_close = df_features['close']

    if len(train_features) < 50:
        raise ValueError("Not enough data for training")

    # Initialize models
    baseline_models = BaselineModels()
    ml_model = MLModel(use_xgboost=XGBOOST_AVAILABLE)

    # Fit models
    ml_model.fit(train_features, train_target)
    arima_model = baseline_models.fit_arima(train_target, 'log_ret')
    garch_model = baseline_models.fit_garch(train_target)

    # Prepare features for next day prediction
    last_row = df_features.iloc[-1]
    next_day_features = pd.DataFrame([last_row[feature_cols].values], columns=feature_cols)

    # Predictions
    predicted_log_ret_ml = ml_model.predict(next_day_features)[0]
    predicted_log_ret_arima = baseline_models.forecast_arima(arima_model)
    predicted_log_ret_naive = baseline_models.naive_forecast(train_target, 'log_ret')
    predicted_log_ret_garch = baseline_models.forecast_garch(garch_model)

    # Get last known close price
    last_close = df_features['close'].iloc[-1]

    # Calculate predicted close prices
    predicted_close_ml = last_close * np.exp(predicted_log_ret_ml)
    predicted_close_arima = last_close * np.exp(predicted_log_ret_arima)
    predicted_close_naive = last_close * np.exp(predicted_log_ret_naive)
    predicted_close_garch = last_close * np.exp(predicted_log_ret_garch)

    # Collect predictions
    predictions = {
        'ML': {'log_ret': predicted_log_ret_ml, 'close': predicted_close_ml},
        'ARIMA': {'log_ret': predicted_log_ret_arima, 'close': predicted_close_arima},
        'Naive': {'log_ret': predicted_log_ret_naive, 'close': predicted_close_naive},
        'GARCH': {'log_ret': predicted_log_ret_garch, 'close': predicted_close_garch}
    }

    # Overall recommendation based on ML (primary model)
    if predicted_log_ret_ml > 0.001:
        recommendation = "КУПУВАТИ (Buy)"
        reason = f"Прогноз ML позитивний: {predicted_log_ret_ml:.6f}"
    elif predicted_log_ret_ml < -0.001:
        recommendation = "ПРОДАВАТИ (Sell)"
        reason = f"Прогноз ML негативний: {predicted_log_ret_ml:.6f}"
    else:
        recommendation = "ТРИМАТИ (Hold)"
        reason = f"Прогноз ML нейтральний: {predicted_log_ret_ml:.6f}"

    expected_return_pct = predicted_log_ret_ml * 100

    result = {
        'last_date': df_features.index[-1].strftime('%Y-%m-%d'),
        'last_close': last_close,
        'predictions': predictions,
        'recommendation': recommendation,
        'reason': reason,
        'expected_return_pct': expected_return_pct,
        'historical_data': df_features[['close']].tail(60)  # Last 60 days for plotting
    }

    logging.info(f"Prediction completed: {result}")
    return result


def create_prediction_plot(result: Dict[str, any], output_dir: str = 'reports/figures') -> None:
    """Create plot showing historical price and next day predictions from all models"""
    ensure_dirs(output_dir)
    logging.info(f"Creating prediction plot in {output_dir}...")

    historical_data = result['historical_data']
    predictions = result['predictions']
    last_date = pd.to_datetime(result['last_date'])
    next_date = last_date + pd.Timedelta(days=1)

    # Set up the plotting area
    plt.figure(figsize=(14, 8))

    # Plot historical price
    plt.plot(historical_data.index, historical_data['close'], label='Історична ціна (Historical Price)', color='blue', linewidth=2)

    # Plot predictions as points
    colors = {'ML': 'red', 'ARIMA': 'green', 'Naive': 'orange', 'GARCH': 'purple'}
    markers = {'ML': 'o', 'ARIMA': 's', 'Naive': '^', 'GARCH': 'D'}

    for model, pred in predictions.items():
        plt.scatter(next_date, pred['close'], color=colors[model], marker=markers[model], s=100,
                   label=f'{model} прогноз: ${pred["close"]:.2f} ({pred["log_ret"]:.4f})', zorder=5)

    # Add vertical line for today
    plt.axvline(x=last_date, color='black', linestyle='--', alpha=0.7, label='Сьогодні (Today)')

    # Formatting
    plt.title('AAPL: Історична ціна та прогнози на наступний день\n(Historical Price and Next Day Predictions)', fontsize=16)
    plt.xlabel('Дата (Date)', fontsize=12)
    plt.ylabel('Ціна закриття (Close Price, USD)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
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

    parser = argparse.ArgumentParser(description='Predict next day AAPL return and provide recommendation')
    args = parser.parse_args()

    logging.info("Predicting next day AAPL movement...")

    try:
        result = predict_next_day()

        # Print results
        print("\n" + "="*60)
        print("ПРОГНОЗ НА НАСТУПНИЙ ДЕНЬ AAPL")
        print("="*60)
        print(f"Остання дата: {result['last_date']}")
        print(f"Остання ціна закриття: ${result['last_close']:.2f}")
        print("\nПрогнози моделей:")
        for model, pred in result['predictions'].items():
            print(f"  {model}: Ціна ${pred['close']:.2f}, Доход {pred['log_ret']:.6f}")
        print(f"\nОчікуваний дохід (ML): {result['expected_return_pct']:.2f}%")
        print(f"РЕКОМЕНДАЦІЯ: {result['recommendation']}")
        print(f"Причина: {result['reason']}")
        print("="*60)

        # Create plot
        create_prediction_plot(result)

        logging.info("Next day prediction and plot completed successfully!")

    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        raise


if __name__ == "__main__":
    main()