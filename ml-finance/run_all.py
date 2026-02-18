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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def setup_logging():
    """Setup colored logging (rich → colorlog → plain fallback)"""
    try:
        from rich.logging import RichHandler
        logging.basicConfig(
            level=logging.INFO, format='%(message)s', datefmt='[%X]',
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
            logging.basicConfig(level=logging.INFO, handlers=[handler])
        except ImportError:
            logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s - %(levelname)s - %(message)s',
                                handlers=[logging.StreamHandler(sys.stdout)])

def run_command(command, description):
    """Run a command and stream output in real-time"""
    logging.info(f"Starting: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True,
                                stdout=sys.stdout, stderr=sys.stderr)
        logging.info(f"Completed: {description}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed: {description}")
        return False

def main():
    """Main function to run the complete pipeline"""
    setup_logging()

    parser = argparse.ArgumentParser(description='Run complete ML pipeline for stock forecasting')
    parser.add_argument('--ticker', type=str, default='AAPL',
                       help='Stock ticker to analyze (default: AAPL)')
    parser.add_argument('--skip-trends', action='store_true',
                       help='Skip Google Trends download')
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
    total_steps = 5

    # Step 0: Download Google Trends data
    if not args.skip_trends:
        cmd = f"python {os.path.join(BASE_DIR, 'src', 'download_trends.py')}"
        if run_command(cmd, "Google Trends Download"):
            success_count += 1
        else:
            logging.warning("Trends download failed — will use existing data")
            success_count += 1  # Non-fatal
    else:
        logging.info("Skipping Google Trends download")
        success_count += 1

    # Step 1: Download stock data and create base features
    cmd = f"python {os.path.join(BASE_DIR, 'src', '01_download_data.py')} --ticker {ticker}"
    if run_command(cmd, f"Data Download for {ticker}"):
        success_count += 1
    else:
        logging.error("Data download failed, stopping pipeline")
        sys.exit(1)

    # Step 2: ML Model Training
    if not args.skip_training:
        cmd = f"python {os.path.join(BASE_DIR, 'src', '03_model_ml.py')} --ticker {ticker}"
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
        cmd = f"python {os.path.join(BASE_DIR, 'src', '04_backtest_and_plots.py')} --ticker {ticker}"
        if run_command(cmd, f"Backtesting and Visualization for {ticker}"):
            success_count += 1
        else:
            logging.error("Backtesting failed")
    else:
        logging.info("Skipping backtesting and visualization")
        success_count += 1

    # Step 3: Next Day Prediction
    if not args.skip_prediction:
        cmd = f"python {os.path.join(BASE_DIR, 'src', '05_predict_next_day.py')} --ticker {ticker}"
        if run_command(cmd, f"Next Day Prediction for {ticker}"):
            success_count += 1
        else:
            logging.error("Next day prediction failed")
    else:
        logging.info("Skipping next day prediction")
        success_count += 1

    # Step 4: Generate Report
    if success_count >= 5:
        cmd = f"python {os.path.join(BASE_DIR, 'src', '06_generate_report.py')} --ticker {ticker}"
        if run_command(cmd, f"Report Generation for {ticker}"):
            success_count += 1
        else:
            logging.warning("Report generation failed, but analysis is still valid")
    else:
        logging.info("Skipping report generation due to incomplete pipeline")

    # Summary
    total_steps = 6
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
    elif success_count >= 5:
        logging.info("⚠️  MOSTLY SUCCESSFUL: Core analysis completed, report generation may have failed")
        logging.info("📊 Check the 'src/reports/' directory for results and plots")
    else:
        logging.warning(f"⚠️  PARTIAL SUCCESS: {success_count}/{total_steps} steps completed")
        sys.exit(1)

if __name__ == "__main__":
    main()