#!/usr/bin/env python3
import argparse
import logging
import os
import warnings
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from utils import (
    set_seed, setup_logging, train_test_splits,
    evaluate_regression, directional_accuracy, buy_and_hold_accuracy,
    save_predictions_csv
)
from config import SIGNAL_THRESHOLD, FEATURE_COLS
from features import create_features, select_features_lasso
from models import get_ml_models, get_tuned_ml_models, get_classification_models, get_tuned_classification_models

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler


def run_ml_walk_forward(train_window: int, test_window: int, step: int, ticker: str = 'AAPL', tune: bool = False) -> pd.DataFrame:
    """Run walk-forward validation for ML model"""
    logging.info("Starting ML walk-forward validation")

    # Load data
    data_path = os.path.join(os.path.dirname(__file__), 'data', f'{ticker.lower()}_features.csv')
    df = pd.read_csv(data_path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)

    # Create features
    df_features = create_features(df)

    # Check if all features exist
    missing_features = [col for col in FEATURE_COLS if col not in df_features.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    # Target: next day's log return (prevents data leakage)
    target = df_features['log_ret'].shift(-1)
    target = target.dropna()
    df_features = df_features.loc[target.index]
    
    # Classification target: 1 if return > 0, else 0
    target_class = (target > 0).astype(int)

    # Storage for all predictions
    all_predictions = []

    # Walk-forward validation
    for train_split, test_split, window_id in train_test_splits(
        target, train_window, test_window, step
    ):
        logging.info(f"Window {window_id}: train={len(train_split)}, test={len(test_split)}")

        # Get features for train/test
        train_features = df_features.loc[train_split.index, FEATURE_COLS]
        test_features = df_features.loc[test_split.index, FEATURE_COLS]

        # Skip if not enough data
        if len(train_features) < 50 or len(test_features) == 0:
            logging.warning(f"Skipping window {window_id} due to insufficient data")
            continue

        # Feature selection with Lasso
        selected_features = select_features_lasso(train_features, train_split)
        if not selected_features:
            logging.warning("Lasso selected 0 features, falling back to all features")
            selected_features = FEATURE_COLS
            
        train_features = train_features[selected_features]
        test_features = test_features[selected_features]
        last_selected_features = selected_features

        # Fit and predict with multiple models
        reg_models = get_tuned_ml_models() if tune else get_ml_models()
        clf_models = get_tuned_classification_models() if tune else get_classification_models()
        
        predictions = {'date': test_split.index, 'y_true': test_split.values, 'window_id': window_id, 'target': 'log_ret_next'}

        # Train regressors
        for model_name, (model, scaler) in reg_models.items():
            logging.info(f"  Training {model_name.upper()} (Regressor)...")
            X_scaled = scaler.fit_transform(train_features)
            model.fit(X_scaled, train_split)

            # Log best params for GridSearchCV-wrapped models
            if hasattr(model, 'best_params_'):
                logging.info(f"  {model_name.upper()} best params: {model.best_params_}")

            X_test_scaled = scaler.transform(test_features)
            y_pred = model.predict(X_test_scaled)
            logging.info(f"  {model_name.upper()} done (mean pred: {y_pred.mean():.6f})")

            predictions[f'y_pred_{model_name}'] = y_pred

        # Train classifiers
        train_split_class = target_class.loc[train_split.index]
        for model_name, (model, scaler) in clf_models.items():
            logging.info(f"  Training {model_name.upper()} (Classifier)...")
            X_scaled = scaler.fit_transform(train_features)
            model.fit(X_scaled, train_split_class)

            # Log best params for GridSearchCV-wrapped models
            if hasattr(model, 'best_params_'):
                logging.info(f"  {model_name.upper()} best params: {model.best_params_}")

            if hasattr(model, 'predict_proba'):
                X_test_scaled = scaler.transform(test_features)
                # Predict probability of class 1 (UP)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                logging.info(f"  {model_name.upper()} done (mean prob: {y_pred_proba.mean():.6f})")
                predictions[f'{model_name}'] = y_pred_proba

        # Store predictions
        window_results = pd.DataFrame(predictions)
        all_predictions.append(window_results)
        # Store both sets of models for feature importances
        last_trained_models = {**reg_models, **clf_models}

    # Combine all predictions
    if not all_predictions:
        raise ValueError("No valid predictions generated")

    results_df = pd.concat(all_predictions, ignore_index=True)

    # Save feature importances from tree-based models (last window)
    importance_data = {}
    for model_name, (model, scaler) in last_trained_models.items():
        # Handle GridSearchCV-wrapped models
        estimator = model.best_estimator_ if hasattr(model, 'best_estimator_') else model
        if hasattr(estimator, 'feature_importances_'):
            importance_data[model_name.upper()] = estimator.feature_importances_

    if importance_data:
        importance_df = pd.DataFrame(importance_data, index=last_selected_features)
        importance_path = os.path.join(os.path.dirname(__file__), 'reports', f'{ticker.lower()}_feature_importance.csv')
        os.makedirs(os.path.dirname(importance_path), exist_ok=True)
        importance_df.to_csv(importance_path)
        logging.info(f"Feature importances saved to {importance_path}")

    # Calculate metrics for each model
    reg_models = get_ml_models()
    clf_models = get_classification_models()
    bh_acc = buy_and_hold_accuracy(results_df['y_true'])
    logging.info(f"Baseline (Buy & Hold) accuracy: {bh_acc:.1%}")

    # Regressors
    for model_name in reg_models.keys():
        pred_col = f'y_pred_{model_name}'
        mask = results_df['y_true'].notna() & results_df[pred_col].notna()
        if mask.sum() > 0:
            ml_metrics = evaluate_regression(
                results_df.loc[mask, 'y_true'],
                results_df.loc[mask, pred_col]
            )

            da = directional_accuracy(
                results_df.loc[mask, 'y_true'],
                results_df.loc[mask, pred_col],
                threshold=SIGNAL_THRESHOLD
            )

            ml_metrics['Raw_DA'] = da['raw_da']
            ml_metrics['Confident_DA'] = da['confident_da']
            ml_metrics['Coverage'] = da['coverage']

            logging.info(
                f"{model_name.upper()}: Raw DA={da['raw_da']:.1%}, "
                f"Confident DA={da['confident_da']:.1%} (coverage={da['coverage']:.1%}), "
                f"B&H baseline={bh_acc:.1%}"
            )

    # Classifiers
    for model_name in clf_models.keys():
        pred_col = f'{model_name}'
        if pred_col not in results_df.columns: continue
        mask = results_df['y_true'].notna() & results_df[pred_col].notna()
        if mask.sum() > 0:
            # For classifiers, probability > 0.5 means UP (+1), else DOWN (-1)
            # We map this to a pseudo continuous array where predictions center around 0
            # so standard directional_accuracy works. prob = 0.6 -> 0.1, prob = 0.4 -> -0.1
            mapped_preds = results_df.loc[mask, pred_col] - 0.5

            da = directional_accuracy(
                results_df.loc[mask, 'y_true'],
                mapped_preds,
                # Threshold for confidence: > 0.55 or < 0.45. Since we subtracted 0.5, threshold is > 0.05
                threshold=0.05
            )

            logging.info(
                f"{model_name.upper()}: Raw DA={da['raw_da']:.1%}, "
                f"Confident DA (P>0.55)={da['confident_da']:.1%} (coverage={da['coverage']:.1%}), "
                f"B&H baseline={bh_acc:.1%}"
            )

    return results_df


def main():
    """Main function"""
    set_seed(42)
    setup_logging()

    parser = argparse.ArgumentParser(description='Run ML models for log-return forecasting')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker (default: AAPL)')
    parser.add_argument('--train_window', type=int, default=1008,
                       help='Training window size (default: 252 ~1 year)')
    parser.add_argument('--test_window', type=int, default=63,
                       help='Test window size (default: 30 ~1 month)')
    parser.add_argument('--step', type=int, default=63,
                       help='Step size for walk-forward (default: 30 ~1 month)')

    parser.add_argument('--tune', action='store_true',
                       help='Enable GridSearchCV hyperparameter tuning for RF and XGB')

    args = parser.parse_args()

    logging.info("Running ML models for log-return forecasting")
    if args.tune:
        logging.info("Hyperparameter tuning ENABLED (GridSearchCV)")
    models = get_ml_models()
    logging.info(f"Comparing models: {list(models.keys())}")

    # Run walk-forward validation

    results_df = run_ml_walk_forward(args.train_window, args.test_window, args.step, args.ticker, tune=args.tune)

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), 'models', f'{args.ticker.lower()}_ml_predictions.csv')
    save_predictions_csv(output_path, results_df)

    logging.info("ML modeling completed")
    logging.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()