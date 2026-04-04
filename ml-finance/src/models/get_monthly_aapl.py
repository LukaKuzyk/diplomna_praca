import os
import numpy as np
import pandas as pd

base = '/Users/mac-pro/PycharmProjects/diplomna_praca/ml-finance/src'
df_base = pd.read_csv(os.path.join(base, 'data', 'aapl_features.csv'), index_col=0)
df_base.index = pd.to_datetime(df_base.index, utc=True)
df_ml = pd.read_csv(os.path.join(base, 'models', 'aapl_ml_predictions.csv'))
df_ml['date'] = pd.to_datetime(df_ml['date'], utc=True)
df_ml.set_index('date', inplace=True)
df_ml = df_ml[~df_ml.index.duplicated(keep='last')]
combined = df_base.copy()
for col in df_ml.columns:
    if col not in ['window_id', 'target']:
        combined[f'ml_{col}'] = df_ml[col]
combined = combined[~combined.index.duplicated(keep='first')].sort_index()

pred_cols = [c for c in combined.columns if c.startswith('ml_y_pred_')]
first_pred_date = combined[pred_cols].notna().any(axis=1).idxmax()
combined = combined.loc[first_pred_date:]

all_model_cols = [c for c in combined.columns if c.startswith('ml_y_pred_') or c.startswith('ml_cl_')]

monthly_data = {}
for col in all_model_cols:
    model_name = col.replace('ml_cl_', 'CL_').replace('ml_y_pred_', 'REG_').upper()
    pred = combined[col]
    actual = combined['ml_y_true']
    signals = pd.Series(0, index=pred.index)
    if col.startswith('ml_cl_'):
        signals[pred > 0.5] = 1
    else:
        signals[pred > 0.002] = 1

    data = pd.DataFrame({'returns': np.exp(actual) - 1, 'signals': signals}).dropna()
    strat_rets = (data['signals'] * data['returns']).dropna()
    
    if len(strat_rets) > 0:
        cum = (1 + strat_rets).cumprod()
        monthly_cum = cum.resample('M').last() if pd.__version__ < '2.2.0' else cum.resample('ME').last()
        monthly_ret = monthly_cum.pct_change().dropna()
        monthly_data[model_name] = monthly_ret

monthly_df = pd.DataFrame(monthly_data)
monthly_df.index = monthly_df.index.strftime('%Y-%m')
print(monthly_df.to_string(float_format=lambda x: f'{x*100:.2f}%'))