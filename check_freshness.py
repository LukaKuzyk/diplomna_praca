import os
import sys
import time

sys.path.insert(0, '/Users/mac-pro/PycharmProjects/diplomna_praca/ml-finance/src')
try:
    from config_tickers import TICKER_CONFIG
    tickers = list(TICKER_CONFIG.keys())
except:
    tickers = []

data = "/Users/mac-pro/PycharmProjects/diplomna_praca/ml-finance/src/data"
now = time.time()
ms = []; mn = []; stale = []; fresh = 0

for t in tickers:
    sf = os.path.join(data, f"{t}_search_data.csv")
    nf = os.path.join(data, f"{t}_news_data.csv")
    
    if not os.path.exists(sf): ms.append(t)
    elif now - os.path.getmtime(sf) > 86400: stale.append(f"{t}_s")
    else: fresh += 1
        
    if not os.path.exists(nf): mn.append(t)
    elif now - os.path.getmtime(nf) > 86400: stale.append(f"{t}_n")
    else: fresh += 1

print(f"Missing Search: {ms}")
print(f"Missing News: {mn}")
print(f"Fresh files (<24h): {fresh}")
print(f"Stale files (>24h): {len(stale)}")
