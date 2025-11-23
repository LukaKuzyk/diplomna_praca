#!/usr/bin/env python3
"""
Complete ML Pipeline Runner for Stock Forecasting
Runs the full pipeline: ML training -> Backtesting -> Next Day Prediction
"""
import subprocess
import sys
import os
import argparse
import logging

def setup_logging():
    """Setup basic logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def run_command(command, description):
    """Run a command and handle errors"""
    logging.info(f"Starting: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logging.info(f"Completed: {description}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed: {description}")
        logging.error(f"Error output: {e.stderr}")
        return False

def main():
    """Main function to run the complete pipeline"""
    setup_logging()

    parser = argparse.ArgumentParser(description='Run complete ML pipeline for stock forecasting')
    parser.add_argument('--ticker', type=str, default='AAPL',
                       help='Stock ticker to analyze (default: AAPL)')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip ML model training (use existing predictions)')
    parser.add_argument('--skip-backtest', action='store_true',
                       help='Skip backtesting and plotting')
    parser.add_argument('--skip-prediction', action='store_true',
                       help='Skip next day prediction')

    args = parser.parse_args()

    ticker = args.ticker.upper()
    logging.info(f"Starting complete ML pipeline for {ticker}")

    success_count = 0
    total_steps = 3

    # Step 1: ML Model Training
    if not args.skip_training:
        cmd = f"python src/03_model_ml.py --ticker {ticker}"
        if run_command(cmd, f"ML Model Training for {ticker}"):
            success_count += 1
        else:
            logging.error("ML training failed, stopping pipeline")
            sys.exit(1)
    else:
        logging.info("Skipping ML model training")
        success_count += 1

    # Step 2: Backtesting and Visualization
    if not args.skip_backtest:
        cmd = f"python src/04_backtest_and_plots.py --ticker {ticker}"
        if run_command(cmd, f"Backtesting and Visualization for {ticker}"):
            success_count += 1
        else:
            logging.error("Backtesting failed")
    else:
        logging.info("Skipping backtesting and visualization")
        success_count += 1

    # Step 3: Next Day Prediction
    if not args.skip_prediction:
        cmd = f"python src/05_predict_next_day.py --ticker {ticker}"
        if run_command(cmd, f"Next Day Prediction for {ticker}"):
            success_count += 1
        else:
            logging.error("Next day prediction failed")
    else:
        logging.info("Skipping next day prediction")
        success_count += 1

    # Step 4: Generate Report
    if success_count >= 3:  # Only generate report if at least training, backtest, and prediction succeeded
        cmd = f"python src/06_generate_report.py --ticker {ticker}"
        if run_command(cmd, f"Report Generation for {ticker}"):
            success_count += 1
        else:
            logging.warning("Report generation failed, but analysis is still valid")
    else:
        logging.info("Skipping report generation due to incomplete pipeline")

    # Summary
    total_steps = 4  # Updated to include report generation
    logging.info("=" * 60)
    logging.info("PIPELINE EXECUTION SUMMARY")
    logging.info("=" * 60)
    logging.info(f"Stock Ticker: {ticker}")
    logging.info(f"Steps Completed: {success_count}/{total_steps}")

    if success_count == total_steps:
        logging.info("✅ COMPLETE SUCCESS: All pipeline steps completed successfully!")
        logging.info("📊 Check the 'src/reports/' directory for results, plots, and reports")
        logging.info("📈 Check the 'src/models/' directory for model predictions")
        logging.info("📄 Check for PDF/HTML reports in 'reports/' directory")
    elif success_count >= 3:
        logging.info("⚠️  MOSTLY SUCCESSFUL: Core analysis completed, report generation may have failed")
        logging.info("📊 Check the 'src/reports/' directory for results and plots")
    else:
        logging.warning(f"⚠️  PARTIAL SUCCESS: {success_count}/{total_steps} steps completed")
        sys.exit(1)

if __name__ == "__main__":
    main()