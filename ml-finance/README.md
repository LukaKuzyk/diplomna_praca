# ML Finance AAPL Analysis

A comprehensive machine learning project for forecasting AAPL stock prices, returns, and volatility using multiple modeling approaches including ARIMA, GARCH, and XGBoost/RandomForest.

## Project Structure

```
ml-finance/
├── data/                    # Raw and processed data
│   ├── aapl.csv            # Raw downloaded AAPL data
│   └── aapl_features.csv   # Processed features and targets
├── models/                  # Model predictions and outputs
│   ├── baseline_close_predictions.csv
│   ├── baseline_log_ret_predictions.csv
│   └── ml_predictions.csv
├── reports/                 # Analysis reports and figures
│   └── figures/
│       ├── price_vs_arima.png
│       ├── returns_pred.png
│       ├── vol_forecast.png
│       └── strategy_equity.png
├── src/                     # Source code
│   ├── __init__.py
│   ├── utils.py            # Utility functions
│   ├── 01_download_data.py # Data downloading and feature engineering
│   ├── 02_model_baselines.py # ARIMA and GARCH models
│   ├── 03_model_ml.py      # ML models (XGBoost/RandomForest)
│   └── 04_backtest_and_plots.py # Backtesting and visualization
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

1. **Create virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the scripts in order to download data, train models, and generate analysis:

### Step 1: Download Data and Create Features
```bash
python src/01_download_data.py --ticker AAPL --years 10
```
- Downloads AAPL stock data for the last 10 years
- Creates target features: `close`, `log_ret`, `rv_5`
- Caches raw data in `data/aapl.csv` (refreshes if older than 1 day)
- Saves processed features in `data/aapl_features.csv`

### Step 2: Train Baseline Models
```bash
# For price forecasting
python src/02_model_baselines.py --target close

# For log-returns forecasting
python src/02_model_baselines.py --target log_ret
```
- Implements Naive (Random Walk) model
- Trains ARIMA models with walk-forward validation
- Fits GARCH(1,1) models for volatility forecasting
- Uses 5-year training window, 1-year test window, 6-month step
- Saves predictions in `models/`

### Step 3: Train ML Models
```bash
python src/03_model_ml.py
```
- Creates ML features: lagged returns, technical indicators (SMA, RSI), calendar features
- Trains XGBoost model (falls back to RandomForest if XGBoost unavailable)
- Uses same walk-forward validation as baseline models
- Saves ML predictions in `models/ml_predictions.csv`

### Step 4: Generate Analysis and Plots
```bash
python src/04_backtest_and_plots.py
```
- Combines all model predictions
- Creates comprehensive visualizations:
  - `price_vs_arima.png`: AAPL price vs ARIMA forecasts with confidence intervals
  - `returns_pred.png`: Log returns vs ML/ARIMA predictions
  - `vol_forecast.png`: Realized volatility vs GARCH forecasts
  - `strategy_equity.png`: Directional strategy performance vs buy-and-hold
- Calculates final performance metrics
- Implements directional trading strategy with 5bp transaction costs
- Saves metrics summary in `reports/metrics_summary.txt`

## Models and Features

### Target Variables
- **`close`**: Daily closing price
- **`log_ret`**: Log returns = `log(Close/Close.shift(1))`
- **`rv_5`**: Realized volatility = `sqrt(log_ret.rolling(5).var() * 252)`

### Baseline Models
- **Naive (Random Walk)**: `close_{t+h} = close_t` or `log_ret_{t+h} = 0`
- **ARIMA**: `(1,0,1)` for returns, `(1,1,1)` for prices with automatic order selection
- **GARCH(1,1)**: For volatility forecasting with Student's t-distribution

### ML Models
- **XGBoost** (primary): `n_estimators=300, max_depth=4, learning_rate=0.05`
- **RandomForest** (fallback): `n_estimators=400, max_depth=6`

### Features for ML
- **Lag features**: `log_ret` at lags 1, 2, 5, 10
- **Technical indicators**: SMA(5), SMA(20), RSI(14)
- **Volatility**: `rv_5` (realized volatility)
- **Calendar features**: Day of week (0-4), Month (1-12)

## Walk-Forward Validation

- **Training window**: 1260 trading days (~5 years)
- **Test window**: 252 trading days (~1 year)
- **Step size**: 126 trading days (roll forward by 6 months)
- **Validation approach**: Train on past data, predict next year, repeat

## Performance Metrics

### Regression Metrics (RMSE, MAE, MAPE)
- For price and return forecasting accuracy

### Directional Accuracy
- Sign hit-rate for return predictions
- Measures ability to predict return direction

### Strategy Performance
- **Sharpe ratio**: Risk-adjusted returns (annualized)
- **Maximum drawdown**: Largest peak-to-trough decline
- **Total return**: Cumulative strategy performance
- **Transaction costs**: 5 basis points per rebalance

## Output Files

### Data Files
- `data/aapl.csv`: Raw downloaded price data
- `data/aapl_features.csv`: Processed features and targets

### Model Predictions
- `models/baseline_close_predictions.csv`: ARIMA/Naive price predictions
- `models/baseline_log_ret_predictions.csv`: ARIMA/GARCH return predictions
- `models/ml_predictions.csv`: ML model return predictions

### Reports and Figures
- `reports/figures/price_vs_arima.png`: Price forecasting visualization
- `reports/figures/returns_pred.png`: Return predictions comparison
- `reports/figures/vol_forecast.png`: Volatility forecasting
- `reports/figures/strategy_equity.png`: Strategy performance
- `reports/metrics_summary.txt`: Final performance metrics

## Requirements

- **Python**: 3.12+
- **Key packages**:
  - `pandas>=2.0`, `numpy>=1.24`
  - `yfinance>=0.2` (data download)
  - `scikit-learn>=1.3` (ML models)
  - `statsmodels>=0.14` (ARIMA)
  - `arch>=6.3` (GARCH)
  - `xgboost>=2.0` (primary ML model)
  - `matplotlib>=3.7` (visualization)

## Notes

- All timestamps use UTC timezone
- Random seed fixed at 42 for reproducibility
- Data cached for 1 day to avoid excessive API calls
- XGBoost fallback to RandomForest if unavailable
- All models use consistent walk-forward validation framework

## Results Summary

After running all scripts, you should find:
- ✅ At least 4 PNG plots in `reports/figures/`
- ✅ Model prediction CSV files in `models/`
- ✅ Performance metrics in `reports/metrics_summary.txt`
- ✅ Complete analysis of AAPL forecasting performance across multiple models