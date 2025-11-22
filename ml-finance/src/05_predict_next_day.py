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
    for lag in [1, 2, 5, 10, 15, 20]:
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
                n_estimators=500,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state
            )
        else:
            logging.info("Using RandomForest model (XGBoost not available)")
            self.model = RandomForestRegressor(
                n_estimators=500,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
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
    import os
    data_path = Path(os.path.join(os.path.dirname(__file__), '..', 'data', 'aapl_features.csv'))
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)

    # Create features
    df_features = create_features(df)

    # Define feature columns for ML
    feature_cols = [
        'log_ret_lag_1', 'log_ret_lag_2', 'log_ret_lag_5', 'log_ret_lag_10', 'log_ret_lag_15', 'log_ret_lag_20',
        'sma_5', 'sma_20', 'rsi_14', 'macd', 'macd_signal',
        'bb_upper', 'bb_lower', 'bb_middle', 'stoch_k', 'stoch_d', 'volatility',
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
    """Create plot showing historical predictions vs actual and next day predictions"""
    ensure_dirs(output_dir)
    logging.info(f"Creating prediction plot in {output_dir}...")

    # Load historical predictions
    ml_pred_path = Path('models/ml_predictions.csv')
    baseline_pred_path = Path('models/baseline_log_ret_predictions.csv')

    historical_data = result['historical_data']
    predictions = result['predictions']
    last_date = pd.to_datetime(result['last_date'])
    next_date = last_date + pd.Timedelta(days=1)

    # Set up the plotting area
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

    # Plot 1: Historical price
    ax1.plot(historical_data.index, historical_data['close'], label='Історична ціна (Historical Price)', color='blue', linewidth=2)

    # Load and plot historical ML predictions (last 30 days)
    if ml_pred_path.exists():
        ml_preds = pd.read_csv(ml_pred_path)
        ml_preds['date'] = pd.to_datetime(ml_preds['date'], utc=True)
        ml_preds.set_index('date', inplace=True)

        # Get last 30 days of predictions
        recent_preds = ml_preds.tail(30)

        # Plot predicted vs actual prices for historical data
        for idx, row in recent_preds.iterrows():
            if idx in historical_data.index:
                actual_price = historical_data.loc[idx, 'close']
                # Calculate predicted price from log_ret
                prev_price = historical_data.loc[:idx].iloc[-2]['close'] if len(historical_data.loc[:idx]) > 1 else actual_price
                for col in recent_preds.columns:
                    if col.startswith('y_pred_'):
                        pred_ret = row[col]
                        pred_price = prev_price * np.exp(pred_ret)
                        model_name = col.replace('y_pred_', '').upper()
                        color = {'LINEAR': 'cyan', 'RF': 'magenta', 'XGB': 'red'}.get(model_name, 'gray')
                        ax1.scatter(idx, pred_price, color=color, alpha=0.6, s=20)
                        ax1.scatter(idx, actual_price, color='blue', alpha=0.6, s=20)

    # Plot next day predictions as points
    colors = {'ML': 'red', 'ARIMA': 'green', 'Naive': 'orange', 'GARCH': 'purple'}
    markers = {'ML': 'o', 'ARIMA': 's', 'Naive': '^', 'GARCH': 'D'}

    for model, pred in predictions.items():
        ax1.scatter(next_date, pred['close'], color=colors[model], marker=markers[model], s=100,
                   label=f'{model} прогноз: ${pred["close"]:.2f}', zorder=5)

    ax1.axvline(x=last_date, color='black', linestyle='--', alpha=0.7, label='Сьогодні (Today)')
    ax1.set_title('AAPL: Історична ціна та прогнози (Historical Price and Predictions)', fontsize=14)
    ax1.set_ylabel('Ціна закриття (Close Price, USD)', fontsize=12)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Returns predictions vs actual
    if ml_pred_path.exists() and baseline_pred_path.exists():
        ml_preds = pd.read_csv(ml_pred_path)
        ml_preds['date'] = pd.to_datetime(ml_preds['date'], utc=True)
        ml_preds.set_index('date', inplace=True)

        baseline_preds = pd.read_csv(baseline_pred_path)
        baseline_preds['date'] = pd.to_datetime(baseline_preds['date'], utc=True)
        baseline_preds.set_index('date', inplace=True)

        # Get last 30 days
        recent_ml = ml_preds.tail(30)
        recent_baseline = baseline_preds.tail(30)

        # Plot actual returns
        actual_returns = historical_data['close'].pct_change().tail(30)
        ax2.plot(actual_returns.index, actual_returns.values, label='Фактичні дохідності (Actual Returns)', color='blue', linewidth=2)

        # Plot predicted returns
        for idx, row in recent_ml.iterrows():
            for col in recent_ml.columns:
                if col.startswith('y_pred_'):
                    pred_ret = row[col]
                    model_name = col.replace('y_pred_', '').upper()
                    color = {'LINEAR': 'cyan', 'RF': 'magenta', 'XGB': 'red'}.get(model_name, 'gray')
                    ax2.scatter(idx, pred_ret, color=color, alpha=0.6, s=20, label=f'{model_name} Pred' if idx == recent_ml.index[0] else "")

        # Plot ARIMA predictions
        if 'y_pred_arima' in recent_baseline.columns:
            arima_preds = recent_baseline['y_pred_arima'].tail(30)
            ax2.scatter(arima_preds.index, arima_preds.values, color='green', alpha=0.6, s=20, label='ARIMA Pred')

        # Plot GARCH predictions
        if 'y_pred_garch_mean' in recent_baseline.columns:
            garch_preds = recent_baseline['y_pred_garch_mean'].tail(30)
            ax2.scatter(garch_preds.index, garch_preds.values, color='purple', alpha=0.6, s=20, label='GARCH Pred')

    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.axvline(x=last_date, color='black', linestyle='--', alpha=0.7)
    ax2.set_title('Доходності: Фактичні vs Прогнози (Returns: Actual vs Predictions)', fontsize=14)
    ax2.set_xlabel('Дата (Date)', fontsize=12)
    ax2.set_ylabel('Доходність (Return)', fontsize=12)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
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