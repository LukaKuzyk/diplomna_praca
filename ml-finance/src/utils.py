"""
Utility functions for ML Finance AAPL Analysis
"""
import os
import logging
import random
from datetime import datetime, timezone
from typing import Tuple, Generator, Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all libraries"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def date_utc_index(df: pd.DataFrame, col: str = "Date") -> pd.DataFrame:
    """Convert date column to UTC datetime index"""
    df = df.copy()
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], utc=True)
        df[col] = df[col].dt.tz_localize('UTC')
        df.set_index(col, inplace=True)
    return df


def train_test_splits(series: pd.Series, train_window: int, test_window: int, step: int):
    """Generate train/test splits for walk-forward validation,
    including the last (possibly shorter) test window."""

    start_idx = 0
    window_id = 0
    n = len(series)

    while True:
        train_end = start_idx + train_window

        # Якщо train_end виходить за межі – стоп
        if train_end >= n:
            break

        # Останній test_end — не далі, ніж кінець серії
        test_end = min(train_end + test_window, n)

        train_split = series.iloc[start_idx:train_end]
        test_split = series.iloc[train_end:test_end]

        # Видаємо навіть якщо test_split короткий, але не порожній
        if len(test_split) > 0:
            yield train_split, test_split, window_id

        # Якщо ми дійшли до кінця — завершити цикл
        if test_end == n:
            break

        start_idx += step
        window_id += 1


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate regression metrics: RMSE, MAE"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    return {
        'RMSE': rmse,
        'MAE': mae
    }


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.0) -> float:
    """Calculate directional accuracy (sign hit-rate) with optional threshold"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Apply threshold to predictions
    if threshold > 0:
        pred_direction = np.where(y_pred > threshold, 1,
                                  np.where(y_pred < -threshold, -1, 0))
    else:
        pred_direction = np.sign(y_pred)

    # Calculate true directions
    true_direction = np.sign(y_true)

    # Calculate accuracy (ignore zeros in predictions and true where pred is 0)
    # For pred_direction == 0, we consider it as no signal, so skip those days
    mask = (pred_direction != 0) & (true_direction != 0)
    if np.sum(mask) == 0:
        return 0.0

    correct = np.sum((true_direction == pred_direction) & mask)
    total = np.sum(mask)

    return correct / total if total > 0 else 0.0


def ensure_dirs(path: str) -> None:
    """Create directories if they don't exist"""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def save_predictions_csv(path: str, df: pd.DataFrame) -> None:
    """Save predictions DataFrame to CSV with directory creation"""
    ensure_dirs(path)
    df.to_csv(path, index=True)
    logging.info(f"Predictions saved to {path}")


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Avoid division by zero
    mask = y_true != 0
    if np.sum(mask) == 0:
        return 0.0

    return mean_absolute_percentage_error(y_true[mask], y_pred[mask])


def setup_logging(level: str = "INFO") -> None:
    """Setup colored logging (rich → colorlog → plain fallback)"""
    log_level = getattr(logging, level.upper())
    try:
        from rich.logging import RichHandler
        logging.basicConfig(
            level=log_level, format='%(message)s', datefmt='[%X]',
            handlers=[RichHandler(rich_tracebacks=True, markup=True, show_path=False)]
        )
    except ImportError:
        try:
            import colorlog
            handler = colorlog.StreamHandler()
            handler.setFormatter(colorlog.ColoredFormatter(
                '%(log_color)s%(asctime)s - %(levelname)s - %(message)s',
                log_colors={'DEBUG': 'cyan', 'INFO': 'green', 'WARNING': 'yellow',
                            'ERROR': 'red', 'CRITICAL': 'bold_red'}
            ))
            logging.basicConfig(level=log_level, handlers=[handler])
        except ImportError:
            logging.basicConfig(level=log_level,
                                format='%(asctime)s - %(levelname)s - %(message)s',
                                handlers=[logging.StreamHandler()])