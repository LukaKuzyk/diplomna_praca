"""
Shared model definitions for the ML Finance pipeline.
"""
import logging
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

from config import DEFAULT_SEED

# Conditional ML imports
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

try:
    from ngboost import NGBRegressor
    NGBOOST_AVAILABLE = True
except ImportError:
    NGBOOST_AVAILABLE = False

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, SGDRegressor
from sklearn.preprocessing import StandardScaler


class BaselineModels:
    """Baseline models for time series forecasting"""

    def __init__(self, random_state: int = DEFAULT_SEED):
        self.random_state = random_state

    def naive_forecast(self, train: pd.Series, test: pd.Series, target: str) -> np.ndarray:
        """Naive (Random Walk) forecast"""
        if target == 'close':
            return np.full(len(test), train.iloc[-1])
        elif target == 'log_ret':
            return np.zeros(len(test))
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
            model_fit = model.fit()
            return model_fit
        except Exception as e:
            logging.warning(f"ARIMA fitting failed: {e}")
            if target == 'log_ret':
                model = ARIMA(train, order=(1, 0, 0))
            else:
                model = ARIMA(train, order=(0, 1, 0))
            return model.fit()

    def forecast_arima(self, model_fit: ARIMA, steps: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get ARIMA forecast with confidence intervals"""
        try:
            forecast = model_fit.get_forecast(steps=steps)
            forecast_mean = forecast.predicted_mean
            forecast_ci = forecast.conf_int(alpha=0.05)  # 95% CI

            return forecast_mean, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1]
        except Exception as e:
            logging.warning(f"ARIMA forecasting failed: {e}")
            return np.zeros(steps), np.zeros(steps), np.zeros(steps)

    def fit_garch(self, returns: pd.Series) -> arch_model:
        """Fit GARCH(1,1) model"""
        try:
            model = arch_model(returns, mean='AR', vol='GARCH', p=1, q=1, dist='StudentsT')
            return model.fit(disp='off')
        except Exception as e:
            logging.warning(f"GARCH fitting failed: {e}")
            try:
                model = arch_model(returns, mean='Constant', vol='GARCH', p=1, q=1)
                return model.fit(disp='off')
            except Exception as e2:
                logging.error(f"GARCH fallback also failed: {e2}")
                raise

    def forecast_garch(self, model_fit: arch_model, steps: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get GARCH mean and volatility forecast"""
        try:
            forecast = model_fit.forecast(horizon=steps)
            mean_forecast = forecast.mean.iloc[-1, :].values
            vol_forecast = np.sqrt(forecast.variance.iloc[-1, :].values)
            lower = mean_forecast - vol_forecast
            upper = mean_forecast + vol_forecast
            return mean_forecast, lower, upper
        except Exception as e:
            logging.warning(f"GARCH forecasting failed: {e}")
            return np.zeros(steps), np.zeros(steps), np.zeros(steps)


def get_ml_models(random_state: int = DEFAULT_SEED) -> Dict[str, tuple]:
    """Get dictionary of ML models to compare"""
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
    else:
        logging.warning("XGBoost not available, skipping XGBoost model")

    # Gradient Boosting Regressor
    models['gbr'] = (GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=random_state), StandardScaler())

    # LightGBM (if available)
    if LGBM_AVAILABLE:
        models['lgbm'] = (LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=random_state), StandardScaler())

    # CatBoost (if available)
    if CATBOOST_AVAILABLE:
        models['cat'] = (CatBoostRegressor(iterations=100, depth=5, learning_rate=0.05, random_state=random_state, verbose=False), StandardScaler())

    # ElasticNet
    models['elasticnet'] = (ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=random_state), StandardScaler())

    # ExtraTrees
    models['extratrees'] = (ExtraTreesRegressor(n_estimators=100, max_depth=10, min_samples_split=5, random_state=random_state), StandardScaler())

    # SGD
    models['sgd'] = (SGDRegressor(max_iter=1000, tol=1e-3, alpha=0.01, random_state=random_state), StandardScaler())

    # NGBoost (if available)
    if NGBOOST_AVAILABLE:
        models['ngb'] = (NGBRegressor(n_estimators=100, learning_rate=0.1, random_state=random_state), StandardScaler())

    return models


class MLModelPredictor:
    """ML model predictor with multiple models"""

    def __init__(self, random_state: int = DEFAULT_SEED):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}

        all_models = get_ml_models(random_state)
        for model_name, (model, scaler) in all_models.items():
            self.models[model_name] = model
            self.scalers[model_name] = scaler
            logging.info(f"Initialized {model_name.upper()} model")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit all models"""
        for model_name, model in self.models.items():
            X_scaled = self.scalers[model_name].fit_transform(X)
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
