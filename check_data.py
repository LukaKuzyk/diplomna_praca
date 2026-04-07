import os
import pandas as pd
import glob
import sys

# Get tickers from config if possible, otherwise from directory
sys.path.insert(0, '/Users/mac-pro/PycharmProjects/diplomna_praca/ml-finance/src')
try:
    from config_tickers import TICKER_CONFIG
    tickers = list(TICKER_CONFIG.keys())
except ImportError:
    print("Could not import TICKER_CONFIG")
    tickers = []

data_dir = "/Users/mac-pro/PycharmProjects/diplomna_praca/ml-finance/src/data"

missing_search = []
missing_news = []
tz_info = {}

print(f"Checking {len(tickers)} tickers...\n")

for t in tickers:
    search_file = os.path.join(data_dir, f"{t}_search_data.csv")
    news_file = os.path.join(data_dir, f"{t}_news_data.csv")
    
    if not os.path.exists(search_file):
        missing_search.append(t)
    else:
        try:
            df = pd.read_csv(search_file, index_col=0)
            if not df.empty:
                idx = pd.to_datetime(df.index)
                tz = getattr(idx.dtype, 'tz', None)
                if tz is None and hasattr(idx, 'tzinfo'):
                    tz = idx.tzinfo
                tz_info[f"{t}_search"] = str(tz)
        except Exception as e:
            tz_info[f"{t}_search"] = f"Error reading"

    if not os.path.exists(news_file):
        missing_news.append(t)
    else:
        try:
            df = pd.read_csv(news_file, index_col=0)
            if not df.empty:
                idx = pd.to_datetime(df.index)
                tz = getattr(idx.dtype, 'tz', None)
                if tz is None and hasattr(idx, 'tzinfo'):
                    tz = idx.tzinfo
                tz_info[f"{t}_news"] = str(tz)
        except Exception as e:
            tz_info[f"{t}_news"] = f"Error reading"

print(f"⚠️ Missing Search Data ({len(missing_search)}): {', '.join(missing_search)}")
print(f"⚠️ Missing News Data ({len(missing_news)}): {', '.join(missing_news)}")

print("\n🕒 Timezone Summary:")
tz_counts = {}
for k, v in tz_info.items():
    tz_counts[v] = tz_counts.get(v, 0) + 1
for tz, count in tz_counts.items():
    print(f"  - {tz}: {count} files")

