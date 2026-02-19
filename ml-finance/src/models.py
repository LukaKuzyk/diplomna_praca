"""
Shared model definitions for the ML Finance pipeline.
"""
import logging
from typing import Dict, Tuple
import numpy as np
import pandas as pd

from config import DEFAULT_SEED

# Conditional ML imports
try:
    from xgboost import XGBRegressor, XGBClassifier
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

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, ElasticNet, SGDRegressor, LogisticRegression
from sklearn.preprocessing import StandardScaler


class BaselineModels:
    """Baseline models for time series forecasting

    NOTE: requires 'statsmodels' and 'arch' packages.
    Not used in the main ML pipeline — kept for reference.
    """

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

    def fit_arima(self, train: pd.Series, target: str):
        """Fit ARIMA model"""
        from statsmodels.tsa.arima.model import ARIMA

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

    def forecast_arima(self, model_fit, steps: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get ARIMA forecast with confidence intervals"""
        try:
            forecast = model_fit.get_forecast(steps=steps)
            forecast_mean = forecast.predicted_mean
            forecast_ci = forecast.conf_int(alpha=0.05)  # 95% CI

            return forecast_mean, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1]
        except Exception as e:
            logging.warning(f"ARIMA forecasting failed: {e}")
            return np.zeros(steps), np.zeros(steps), np.zeros(steps)

    def fit_garch(self, returns: pd.Series):
        """Fit GARCH(1,1) model"""
        from arch import arch_model

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

    def forecast_garch(self, model_fit, steps: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def get_tuned_ml_models(random_state: int = DEFAULT_SEED) -> Dict[str, tuple]:
    """Get ML models with GridSearchCV tuning for RF and XGB"""
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

    models = {}
    tscv = TimeSeriesSplit(n_splits=3)

    # Linear Regression (baseline, no tuning needed)
    models['linear'] = (LinearRegression(), StandardScaler())

    # Random Forest with GridSearchCV
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5],
    }
    rf_base = RandomForestRegressor(min_samples_leaf=2, random_state=random_state)
    models['rf'] = (
        GridSearchCV(rf_base, rf_params, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1, refit=True),
        StandardScaler()
    )

    # XGBoost with GridSearchCV
    if XGBOOST_AVAILABLE:
        xgb_params = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
        }
        xgb_base = XGBRegressor(subsample=0.8, colsample_bytree=0.8, random_state=random_state)
        models['xgb'] = (
            GridSearchCV(xgb_base, xgb_params, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1, refit=True),
            StandardScaler()
        )

    # Other models keep fixed hyperparameters
    models['gbr'] = (GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=random_state), StandardScaler())

    if LGBM_AVAILABLE:
        models['lgbm'] = (LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=random_state), StandardScaler())

    if CATBOOST_AVAILABLE:
        models['cat'] = (CatBoostRegressor(iterations=100, depth=5, learning_rate=0.05, random_state=random_state, verbose=False), StandardScaler())

    models['elasticnet'] = (ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=random_state), StandardScaler())
    models['extratrees'] = (ExtraTreesRegressor(n_estimators=100, max_depth=10, min_samples_split=5, random_state=random_state), StandardScaler())
    models['sgd'] = (SGDRegressor(max_iter=1000, tol=1e-3, alpha=0.01, random_state=random_state), StandardScaler())

    if NGBOOST_AVAILABLE:
        models['ngb'] = (NGBRegressor(n_estimators=100, learning_rate=0.1, random_state=random_state), StandardScaler())

    return models


def get_classification_models(random_state: int = DEFAULT_SEED) -> Dict[str, tuple]:
    """Get dictionary of ML classification models"""
    models = {}

    # Logistic Regression (baseline classifier)
    models['cl_logreg'] = (LogisticRegression(
        random_state=random_state, max_iter=1000, class_weight='balanced'
    ), StandardScaler())

    # Random Forest Classifier
    models['cl_rf'] = (RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=random_state
    ), StandardScaler())

    # XGBoost Classifier
    if XGBOOST_AVAILABLE:
        models['cl_xgb'] = (XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            random_state=random_state
        ), StandardScaler())
    else:
        logging.warning("XGBoost not available, skipping XGBoost classifier")

    return models


def get_tuned_classification_models(random_state: int = DEFAULT_SEED) -> Dict[str, tuple]:
    """Get classification models with GridSearchCV tuning for RF and XGB"""
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

    models = get_classification_models(random_state)
    tuned_models = {}
    cv = TimeSeriesSplit(n_splits=3)

    for name, (model, scaler) in models.items():
        if name == 'cl_rf':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5]
            }
            grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
            tuned_models[name] = (grid_search, scaler)

        elif name == 'cl_xgb':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1]
            }
            grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
            tuned_models[name] = (grid_search, scaler)

        else:
            # Keep other models without tuning
            tuned_models[name] = (model, scaler)

    return tuned_models


class MLModelPredictor:
    """Wrapper class to handle training and prediction with multiple ML models"""

    def __init__(self, random_state: int = DEFAULT_SEED, model_type: str = 'regression'):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.is_fitted = False

        if model_type == 'regression':
            all_models = get_ml_models(random_state)
        elif model_type == 'classification':
            all_models = get_classification_models(random_state)
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Choose 'regression' or 'classification'.")

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
        self.is_fitted = True

    def predict(self, X: pd.DataFrame, model_name: str) -> float:
        """Make prediction with specific model"""
        if not self.is_fitted:
            raise ValueError("Models not fitted yet")
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")

        model = self.models[model_name]
        scaler = self.scalers[model_name]

        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
        return prediction

    def predict_all(self, X: pd.DataFrame) -> Dict[str, float]:
        """Make predictions with all models"""
        if not self.is_fitted:
            raise ValueError("Models not fitted yet")

        predictions = {}
        for model_name, model in self.models.items():
            scaler = self.scalers[model_name]
            X_scaled = scaler.transform(X)
            y_pred = model.predict(X_scaled)[0] # Assuming single prediction for single X
            predictions[model_name] = y_pred

        return predictions

    def predict_proba_all(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Make probability predictions (for class 1: UP) with all trained classification models"""
        if not self.is_fitted:
            raise ValueError("Models not fitted yet")

        predictions = {}
        for model_name, model in self.models.items():
            if not hasattr(model, 'predict_proba'):
                continue
            scaler = self.scalers[model_name]
            X_scaled = scaler.transform(X)
            # Probability of target=1 (UP)
            y_pred_proba = model.predict_proba(X_scaled)[:, 1]
            predictions[model_name] = y_pred_proba

        return predictions
