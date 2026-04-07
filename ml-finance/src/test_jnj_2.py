import pandas as pd
import numpy as np

df = pd.read_csv('/Users/mac-pro/PycharmProjects/diplomna_praca/ml-finance/src/models/jnj_ml_predictions.csv', index_col=0, parse_dates=True)
p_cols = [c for c in df.columns if 'y_pred_' in c or 'cl_' in c]
first_pred_date = df[p_cols].notna().any(axis=1).idxmax()
df_slice = df.loc[first_pred_date:].copy()

print("Original slice length:", len(df_slice))

ret1 = (np.exp(df_slice["y_true"].fillna(0).cumsum()).iloc[-1] - 1) * 100
ret2 = ( (1 + (np.exp(df_slice['y_true'].dropna()) - 1)).cumprod().iloc[-1] - 1 ) * 100

print('BnH logic 1:', ret1)
print('BnH logic 2:', ret2)

sig = np.where(df_slice['cl_rf'] > 0.5, 1, 0)
d = pd.DataFrame({'returns': np.exp(df_slice['y_true'])-1, 'signals': sig}).dropna()
print('CL_RF match:', ( (1 + (d['returns']*d['signals'])).cumprod().iloc[-1]-1 )*100)

