import pandas as pd
import numpy as np

df = pd.read_csv('/Users/mac-pro/PycharmProjects/diplomna_praca/ml-finance/src/models/jnj_ml_predictions.csv', index_col=0, parse_dates=True)
p_cols = [c for c in df.columns if 'y_pred_' in c or 'cl_' in c]
first_pred_date = df[p_cols].notna().any(axis=1).idxmax()
df_slice = df.loc[first_pred_date:].copy()

# Print start and end dates
print("Test range:", df_slice.index[0], "to", df_slice.index[-1])

print('BnH logic 1:', (np.exp(df_slice["y_true"].fillna(0).cumsum()).iloc[-1] - 1) * 100)

bh_returns = np.exp(df_slice['y_true'].dropna()) - 1
bh_cumulative = (1 + bh_returns).cumprod()
print('BnH logic 2:', (bh_cumulative.iloc[-1] - 1) * 100)

for c in p_cols:
    p = df_slice[c]
    sig = np.where(p > 0.5, 1, 0) if c.startswith('cl_') else np.where(p > 0.002, 1, 0)
    data = pd.DataFrame({'returns': np.ex    data = pd.DataFrame({'returns': np.ex    data = pd.DataFrame({'returns': n'] * data['returns']
    cum = (1+strat).cumprod()
    if len(cum) > 0:
        val = (cum.iloc[-1]-1)*100
        if c == 'cl_rf':
            print('cl_rf exact match:', val)
        if c == 'y_pred_elasticnet':
            print('elasticnet exact match:', val)
        if c == 'y_pred_xgb':
            print('xgb match:', val)
