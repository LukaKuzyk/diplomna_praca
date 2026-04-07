import subprocess
try:
    code = """
import os, glob, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from config_tickers import TICKER_CONFIG

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
MODELS_DIR, REPORTS_DIR = os.path.join(BASE_DIR, 'models'), os.path.join(BASE_DIR, 'reports')
MASTER_DIR, FIGURES_DIR = os.path.join(REPORTS_DIR, 'master_tables'), os.path.join(REPORTS_DIR, 'master_figures')
os.makedirs(MASTER_DIR, exist_ok=True); os.makedirs(FIGURES_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")

def get_cat(f):
    f = f.lower()
    if 'search' in f: return 'Search'
    if 'news' in f: return 'News'
    if any(t in f for t in ['sma','ema','macd','rsi','bollinger','stoch','cci','atr','volat','kurt','skew']): return 'Technical'
    if any(m in f for m in ['snp500','vix','qqq','rv_']): return 'Macro'
       any(c in f for c in ['month','day','quarter','earn']): return 'Calendar'
    if any(p in f for p in ['close','open','high','low','ret','volume']): return 'Price'
    return 'Other'

def analyze(df):
    if 'y_true' not in df.columns: return {}
    y, res = df['y_true'], {}
    bh_wealth = np.exp(y.cumsum())
    bh_max_dd = ((bh_wealth - bh_wealth.cummax()) / bh_wealth.cummax()).min() if not bh_wealth.isna().all() else 0
    bh_ret = bh_wealth.iloc[-1] - 1 if not bh_wealth.empty else 0
    for c in [x for x in df.columns if x.startswith('y_pred_') or x.startswith('cl_')]:
        p = df[c]
        sig = np.where(p > 0.5, 1, 0) if c.startswith('cl_') else np.where(p > 0, 1, 0)
        s_ret = y * sig
        w = np.exp(s_ret.cumsum()); ret = w.iloc[-1] - 1 if not w.empty else 0
        mdd = ((w - w.cummax()) / w.cummax()).min() if not w.isna().all() else 0
        act = np.sum(sig != 0); win = np.sum(s_ret > 0) / act if act > 0 else 0
        psig = np.where(p > 0.5, 1, -1) if c.startswith('cl_') else np.sign(p)
        val = np.sign(y) != 0; da = np.mean(psig[val] == np.sign(y)[val]) if val.sum() > 0 else 0
        m_name = c.replace('y_pred_','').replace('cl_','CL_').upper()
        res[c] = {'Name': m_name, 'ML_Ret': ret*100, 'BH_Ret': bh_ret*100, 'Alpha': (ret-bh_ret)*100,
                  'ML_DD': mdd*100, 'BH_DD': bh_max_dd*100, 'Win': win*100, 'Trades': int(np.abs(np.diff(sig)).sum()) if len(sig)>1 else 0, 'DA': da*100}
    return res

print("Spustam agregaciu...")
t1, t2, t3, das, bms, td, feats = [], [], [], [], [], [], {}
for t in TICKER_CONFIG:
    sec = TICKER_CONFIG[t].get('sector', 'Unknown')
    pfile = os.path.join(MODELS_DIR, f"{t}_ml_predictions.csv")
    if os.path.exists(pfile):
        df = pd.read_csv(pfile, index_col=0, parse_dates=True)
        res = analyze(df)
        if res:
            b = res[max(res.keys(), key=lambda k: res[k]['Alpha'])]
            t1.append({'Ticker': t, 'Sektor': sec, 'Najlepsi ML Model': b['Name'], 'Buy & Hold Vynos (%)': b['BH_Ret'], 'ML Strategia Vynos (%)': b['ML_Ret'], 'Nadvynos / Alpha (%)': b['Alpha'], 'Smerova presnost - DA (%)': b['DA']})
            t2.append({'Ticker': t, 'Buy & Hold Max Drawdown (%)': b['BH_DD'], 'ML Strategia Max Drawdown (%)': b['ML_DD'], 'Pocet ML Obchodov': b['Trades'], 'Win Rate (%)': b['Win']})
            bms.append(b['Name']); td.append({'Ticker': t, 'DA': b['DA']})
            for k, v in res.items(): das.append({'Model': v['Name'], 'DA': v['DA']})
    
    ffile = os.path.join(REPORTS_DIR, f"{t}_feature_importance.csv")
    if os.path.exists(ffile):
        sf = pd.read_csv(ffile, index_col=0).mean(axis=1).sort_values(ascending=False)
        top = sf.head(3).index.tolist(); top += ["N/A"]*(3-len(top))
        cscores = {}
        for f, imp in sf.items(): cscores[get_cat(f)] = cscores.get(get_cat(f), 0) + imp
        topc = max(cscores.keys(), key=lambda k: cscores[k]) if cscores else "N/A"
        t3.append({'Ticker': t, 'Top 1 Kategoria': topc, 'Top 1 Konkretny atribut': top[0], 'Top 2 Konkretny atribut': top[1], 'Top 3 Konkretny atribut': top[2]})
        for f, imp in sf.items(): feats[f] = feats.get(f, []) + [imp]

pd.DataFrame(t1).round(2).to_csv(os.path.join(MASTER_DIR, 'Tabulka_1_Vykonnost_a_Ziskovost.csv'), index=False)
pd.DataFrame(t2).round(2).to_csv(os.path.join(MASTER_DIR, 'Tabulka_2_Riziko_a_Stabilita.csv'), index=False)
pd.DataFrame(t3).to_csv(os.path.join(MASTER_DIR, 'Tabulka_3_Co_riadi_trh.csv'), index=False)

if das:
    d = pd.DataFrame(das).groupby('Model')['DA'].mean().reset_index().sort_values('DA', ascending=False)
    plt.figure(figsize=(10,6)); sns.barplot(data=d, x='DA', y='Model', palette='viridis'); plt.axvline(50, color='r', ls='--'); plt.title('Obr 1: Priemerná smerová presnosť (DA) modelov'); plt.xlabel('DA (%)'); plt.savefig(os.path.join(FIGURES_DIR, 'Obr_1_Priemerna_presnost_DA.png'), bbox_inches='tight', dpi=300); plt.close()
if bms:
    from collections import Counter
    c = Counter(bms)
    plt.figure(figsize=(8,8)); plt.pie(c.values(), labels=c.keys(), autopct='%1.1f%%', colors=sns.color_palette('pastel')); plt.title('Obr 2: Najčastejšie víťazné modely'); plt.savefig(os.path.join(FIGURES_DIR, 'Obr_2_Vyhra_najlepsich_modelov.png'), bbox_inches='tight', dpi=300); plt.close()
if feats:
    gdf = pd.DataFrame([(f, np.mean(v)) for f,v in feats.items()], columns=['Atribút','Dôležitosť'])
    gdf['Kategória'] = gdf['Atribút'].apply(get_cat); gdf = gdf.sort_values('Dôležitosť', ascending=False).head(15)
    plt.figure(figsize=(10,8)); sns.barplot(data=gdf, x='Dôležitosť', y='Atribút', hue='Kategória', dodge=False); plt.title('Obr 3: Top 15 globálnych indikátorov'); plt.savefig(os.path.join(FIGURES_DIR, 'Obr_3_Top_15_globalnych_indikatorov.png'), bbox_inches='tight', dpi=300); plt.close()
if td:
    tf = pd.DataFrame(td).sort_values('DA', ascending=False)
    plt.figure(figsize=(14,6)); sns.barplot(data=tf, x='Ticker', y='DA', palette='coolwarm'); plt.axhline(50, color='r', ls='--'); plt.title('Obr 4: Smerová presnosť podľa Tickerov'); plt.xticks(rotation=45); plt.savefig(os.path.join(FIGURES_DIR, 'Obr_4_Distribucia_presnosti_podla_tickerov.png'), bbox_inches='tight', dpi=300); plt.close()
print("Grafy ulozene.")
"""
    with open('/Users/mac-pro/PycharmProjects/diplomna_praca/ml-finance/src/07_aggregate_results.py', 'w') as f:
        f.write(code)
except Exception as e:
    pass
