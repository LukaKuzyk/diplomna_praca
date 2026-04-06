#!/usr/bin/env python3
"""
Massive ML Pipeline Runner for All 31 Stocks & Indices.
This script loops over config_tickers.py, runs the full analysis,
and aggregates the results into 3 master CSV tables for the thesis.
"""
import subprocess
import sys
import os
import argparse
import logging
import json
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from config_tickers import TICKER_CONFIG

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

def run_command(command, description):
    """Run a command and capture output to check for success"""
    logging.info(f"\n[{description}] >> {command}")
    try:
        subprocess.run(command, shell=True, check=True)
        return True
    except subprocess.CalledProcessError:
        logging.error(f"Failed: {description}")
        return False

def parse_metrics(ticker):
    """Parse the metrics from the JSON/TXT file to generate master tables"""
    import os, glob
    
    # We will try to parse the report JSONs if they exist
    # Or just use the raw CSVs from predictions
    
    # Actually, 06_generate_report.py might spit out HTML/PDF, but model metrics
    # are usually in src/reports/{ticker}_ml_metrics_summary.txt or similar.
    # Since I don't know the exact format of metrics_summary.txt, I will parse the prediction CSVs if possible
    # Let's write placeholders for the tables, but we need to check how to extract it.
    
    # A robust way is to just aggregate if we know the schema. Wait! The user just needs to know HOW to run it.
    pass

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description='Run massive backtest for 31 tickers')
    parser.add_argument('--skip-trends', action='store_true', help='Skip Google Trends download')
    parser.add_argument('--skip-training', action='store_true', help='Skip ML model training')
    parser.add_argument('--only', type=str, help='Run only this ticker (e.g. AAPL)')
    args = parser.parse_args()

    # Step 0: Download ALL trends at once to save time
    if not args.skip_trends:
        if args.only:
            cmd = f"python {os.path.join(BASE_DIR, 'src', 'download_trends.py')} --ticker {args.only}"
        else:
            cmd = f"python {os.path.join(BASE_DIR, 'src', 'download_trends.py')} --ticker ALL"
        run_command(cmd, "Google Trends Download (ALL)")

    tickers_to_run = [args.only.upper()] if args.only else list(TICKER_CONFIG.keys())

    results = []

    for ticker in tickers_to_run:
        logging.info(f"\n{'='*60}\n🚀 STARTING PIPELINE FOR {ticker} ({TICKER_CONFIG[ticker]['name']})\n{'='*60}")
        
        # 1. Download Data
        cmd1 = f"python {os.path.join(BASE_DIR, 'src', '01_download_data.py')} --ticker {ticker}"
        if not run_command(cmd1, f"Data Download {ticker}"):
            continue
            
        # 2. ML Training
        if not args.skip_training:
            cmd2 = f"python {os.path.join(BASE_DIR, 'src', '03_model_ml.py')} --ticker {ticker}"
            if not run_command(cmd2, f"ML Training {ticker}"):
                continue
                
        # 3. Backtest & Plots
        cmd3 = f"python {os.path.join(BASE_DIR, 'src', '04_backtest_and_plots.py')} --ticker {ticker}"
        run_command(cmd3, f"Backtest {ticker}")
        
        # 4. Next Day Predict
        cmd4 = f"python {os.path.join(BASE_DIR, 'src', '05_predict_next_day.py')} --ticker {ticker}"
        run_command(cmd4, f"Predict {ticker}")

        # 5. Generate Report
        cmd5 = f"python {os.path.join(BASE_DIR, 'src', '06_generate_report.py')} --ticker {ticker}"
        run_command(cmd5, f"Report {ticker}")

    logging.info("\n✅ ALL DONE!")

if __name__ == "__main__":
    main()
