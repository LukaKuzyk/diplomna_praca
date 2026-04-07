"""
Shared model definitions for the ML Finance pipeline.
"""
import logging
from typing import Dict, Tuple
import numpy as np
import pandas as pd

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

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

from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor,
    RandomForestClassifier, StackingRegressor, StackingClassifier
)
from sklearn.linear_model import LinearRegression, ElasticNet, SGDRegressor, LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score


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
        models['lgbm'] = (LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=random_state, verbose=-1), StandardScaler())

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


def _optuna_tune_regressor(X, y, model_name, random_state, n_trials=30):
    """Run Optuna hyperparameter search for a single regression model.
    Returns the best-fitted estimator."""
    tscv = TimeSeriesSplit(n_splits=3)

    def objective(trial):
        if model_name == 'rf':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 25),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            }
            model = RandomForestRegressor(**params, random_state=random_state)

        elif model_name == 'xgb' and XGBOOST_AVAILABLE:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            }
            model = XGBRegressor(**params, random_state=random_state)

        elif model_name == 'gbr':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            }
            model = GradientBoostingRegressor(**params, random_state=random_state)

        elif model_name == 'lgbm' and LGBM_AVAILABLE:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 15, 127),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            }
            model = LGBMRegressor(**params, random_state=random_state, verbose=-1)

        elif model_name == 'cat' and CATBOOST_AVAILABLE:
            params = {
                'iterations': trial.suggest_int('iterations', 50, 300),
                'depth': trial.suggest_int('depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            }
            model = CatBoostRegressor(**params, random_state=random_state, verbose=False)
        else:
            return float('inf')

        scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
        return -scores.mean()

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    logging.info(f"Optuna {model_name.upper()} best params: {best}")
    return best


def get_tuned_ml_models(random_state: int = DEFAULT_SEED) -> Dict[str, tuple]:
    """Get ML models with Optuna tuning for tree-based models.
    Call .fit() on the returned models — they are NOT pre-fitted."""
    if not OPTUNA_AVAILABLE:
        logging.warning("Optuna not available, falling back to default hyperparameters")
        return get_ml_models(random_state)

    # Tuning requires data — defer to OptunaTunedWrapper
    models = get_ml_models(random_state)

    tunable = ['rf', 'xgb', 'gbr']
    if LGBM_AVAILABLE:
        tunable.append('lgbm')
    if CATBOOST_AVAILABLE:
        tunable.append('cat')

    for name in tunable:
        if name in models:
            models[name] = (OptunaRegressorWrapper(name, random_state), StandardScaler())

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
    """Get classification models with Optuna tuning for tree-based classifiers."""
    if not OPTUNA_AVAILABLE:
        logging.warning("Optuna not available, falling back to default hyperparameters")
        return get_classification_models(random_state)

    models = get_classification_models(random_state)

    for name in ['cl_rf', 'cl_xgb']:
        if name in models:
            models[name] = (OptunaClassifierWrapper(name, random_state), StandardScaler())

    return models


class OptunaRegressorWrapper:
    """Sklearn-compatible wrapper that runs Optuna tuning inside .fit()."""

    def __init__(self, model_name: str, random_state: int = DEFAULT_SEED, n_trials: int = 30):
        self.model_name = model_name
        self.random_state = random_state
        self.n_trials = n_trials
        self.best_estimator_ = None
        self.best_params_ = None

    def fit(self, X, y):
        best_params = _optuna_tune_regressor(X, y, self.model_name, self.random_state, self.n_trials)
        self.best_params_ = best_params
        self.best_estimator_ = self._build_model(best_params)
        self.best_estimator_.fit(X, y)
        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def _build_model(self, params):
        if self.model_name == 'rf':
            return RandomForestRegressor(**params, random_state=self.random_state)
        elif self.model_name == 'xgb':
            return XGBRegressor(**params, random_state=self.random_state)
        elif self.model_name == 'gbr':
            return GradientBoostingRegressor(**params, random_state=self.random_state)
        elif self.model_name == 'lgbm':
            return LGBMRegressor(**params, random_state=self.random_state, verbose=-1)
        elif self.model_name == 'cat':
            return CatBoostRegressor(**params, random_state=self.random_state, verbose=False)

    @property
    def feature_importances_(self):
        if self.best_estimator_ and hasattr(self.best_estimator_, 'feature_importances_'):
            return self.best_estimator_.feature_importances_
        raise AttributeError("No feature_importances_ available")


class OptunaClassifierWrapper:
    """Sklearn-compatible wrapper that runs Optuna tuning inside .fit() for classifiers."""

    def __init__(self, model_name: str, random_state: int = DEFAULT_SEED, n_trials: int = 30):
        self.model_name = model_name
        self.random_state = random_state
        self.n_trials = n_trials
        self.best_estimator_ = None
        self.best_params_ = None

    def fit(self, X, y):
        tscv = TimeSeriesSplit(n_splits=3)

        def objective(trial):
            if self.model_name == 'cl_rf':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 5, 25),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                }
                model = RandomForestClassifier(**params, class_weight='balanced', random_state=self.random_state)
            elif self.model_name == 'cl_xgb' and XGBOOST_AVAILABLE:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                }
                model = XGBClassifier(**params, eval_metric='logloss', random_state=self.random_state)
            else:
                return 0.0

            scores = cross_val_score(model, X, y, cv=tscv, scoring='roc_auc', n_jobs=-1)
            return scores.mean()

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        self.best_params_ = study.best_params
        logging.info(f"Optuna {self.model_name.upper()} best params: {self.best_params_}")
        self.best_estimator_ = self._build_model(self.best_params_)
        self.best_estimator_.fit(X, y)
        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)

    def _build_model(self, params):
        if self.model_name == 'cl_rf':
            return RandomForestClassifier(**params, class_weight='balanced', random_state=self.random_state)
        elif self.model_name == 'cl_xgb':
            return XGBClassifier(**params, eval_metric='logloss', random_state=self.random_state)

    @property
    def feature_importances_(self):
        if self.best_estimator_ and hasattr(self.best_estimator_, 'feature_importances_'):
            return self.best_estimator_.feature_importances_
        raise AttributeError("No feature_importances_ available")


def get_stacking_regressor(random_state: int = DEFAULT_SEED):
    """Build a StackingRegressor: RF + XGB + GBR → Ridge meta-learner."""
    estimators = [
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=5,
                                     min_samples_leaf=2, random_state=random_state)),
        ('gbr', GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.05,
                                          random_state=random_state)),
    ]
    if XGBOOST_AVAILABLE:
        estimators.append(
            ('xgb', XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8, random_state=random_state))
        )

    return StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=3,
        n_jobs=-1
    )


def get_stacking_classifier(random_state: int = DEFAULT_SEED):
    """Build a StackingClassifier: RF + XGB + LogReg → LogReg meta-learner."""
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=5,
                                      min_samples_leaf=2, class_weight='balanced',
                                      random_state=random_state)),
        ('logreg', LogisticRegression(random_state=random_state, max_iter=1000, class_weight='balanced')),
    ]
    if XGBOOST_AVAILABLE:
        estimators.append(
            ('xgb', XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
                                 random_state=random_state))
        )

    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(random_state=random_state, max_iter=1000),
        cv=3,
        n_jobs=-1
    )


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
