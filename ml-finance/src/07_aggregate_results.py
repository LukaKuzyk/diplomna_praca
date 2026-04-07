import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from collections import Counter

# Add parent directory to path so config_tickers can be imported natively
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_tickers import TICKER_CONFIG

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
MASTER_TABLES_DIR = os.path.join(REPORTS_DIR, 'master_tables')
MASTER_FIGURES_DIR = os.path.join(REPORTS_DIR, 'master_figures')

os.makedirs(MASTER_TABLES_DIR, exist_ok=True)
os.makedirs(MASTER_FIGURES_DIR, exist_ok=True)

# Nastavenie pre lepšie grafy
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})

# Pomocná funkcia pre zaradenie indikátorov do kategórií
def get_feature_category(feature_name):
    name = feature_name.lower()
    if 'search' in name: return 'Search Trend'
    if 'news' in name: return 'News Sentiment'
    if any(tech in name for tech in ['sma', 'ema', 'macd', 'rsi', 'bollinger', 'stoch', 'cci', 'atr', 'volat', 'kurt', 'skew']):
        return 'Technical'
    if any(macro in name for macro in ['snp500', 'vix', 'qqq', 'rv_']):
        return 'Macro'
    if any(cal in name for cal in ['month', 'day', 'quarter', 'earn']):
        return 'Calendar'
    if any(price in name for price in ['close', 'open', 'high', 'low', 'ret', 'volume']):
        return 'Price Action'
    return 'Základné dáta'

# Tabulky
tabulka1_data = [] # Výkonnosť a ziskovosť
tabulka2_data = [] # Riziko a stabilita
tabulka3_data = [] # Čo ovláda trh (Dôležitosť príznakov)

# Agregačné vektory pre globálne grafy
all_model_das = []
best_models = []
all_feature_importances = {}
ticker_da_distribution = []

print("Získavam reporty a agregujem výsledky...")

for ticker in TICKER_CONFIG:
    sector = TICKER_CONFIG[ticker].get('sector', 'Neznámy')
    
    # 1. Vyhodnotenie modelov (Z models folderu pre vypocet equity a MaxDD)
    pred_file = os.path.join(MODELS_DIR, f"{ticker}_ml_predictions.csv")
    if os.path.exists(pred_file):
        df_preds = pd.read_csv(pred_file, index_col=0, parse_dates=True)
        if 'y_true' in df_preds.columns:
            # Aby tabuľka plne korešpondovala s HTML reportom, začíname od prvého dňa s predikciou
            model_cols = [c for c in df_preds.columns if c.startswith('y_pred_') or c.startswith('cl_')]
            if model_cols:
                first_pred_date = df_preds[model_cols].notna().any(axis=1).idxmax()
                df_preds = df_preds.loc[first_pred_date:].copy()
            
            y_true = df_preds['y_true']
            
            # Buy and Hold (baseline ako v HTML reportoch)
            bh_returns = np.exp(y_true.dropna()) - 1
            bnh_wealth = (1 + bh_returns).cumprod()
            bnh_total_ret_pct = (bnh_wealth.iloc[-1] - 1) * 100 if not bnh_wealth.empty else 0
            bnh_drawdown = ((bnh_wealth - bnh_wealth.cummax()) / bnh_wealth.cummax()).min() * 100 if not bnh_wealth.isna().all() else 0

            best_model_name = ""
            best_model_return = -np.inf
            best_model_alpha = -np.inf
            best_model_maxdd = 0
            best_model_winrate = 0
            best_model_trades = 0
            best_model_da = 0

            for col in model_cols:
                y_pred = df_preds[col]
                model_name = col.replace('y_pred_', '').replace('cl_', 'CL_').upper()
                
                # Signál s použitím SIGNAL_THRESHOLD = 0.002
                if col.startswith('cl_'):
                    signal = np.where(y_pred > 0.5, 1, 0)
                    pred_sign = np.where(y_pred > 0.5, 1, -1)
                else:
                    signal = np.where(y_pred > 0.002, 1, 0)
                    pred_sign = np.sign(y_pred)
                
                # Výpočet wealth krivky pre ML stratégiu
                data_eval = pd.DataFrame({'returns': np.exp(y_true) - 1, 'signals': signal}).dropna()
                
                if len(data_eval) > 0:
                    strategy_returns = data_eval['signals'] * data_eval['returns']
                    strat_wealth = (1 + strategy_returns).cumprod()
                    strat_total_ret_pct = (strat_wealth.iloc[-1] - 1) * 100 if not strat_wealth.empty else 0
                    max_dd = ((strat_wealth - strat_wealth.cummax()) / strat_wealth.cummax()).min() * 100 if not strat_wealth.isna().all() else 0
                    winning_trades = np.sum(strategy_returns > 0)
                else:
                    strat_total_ret_pct = 0
                    max_dd = 0
                    winning_trades = 0
                
                alpha = strat_total_ret_pct - bnh_total_ret_pct
                
                # Trading stats
                trades = int(np.abs(np.diff(signal[~np.isnan(y_pred)])).sum()) if len(signal[~np.isnan(y_pred)]) > 1 else 0
                total_taken_trades = np.sum(signal[~np.isnan(y_pred)] != 0)
                win_rate = (winning_trades / total_taken_trades * 100) if total_taken_trades > 0 else 0
                
                # DA calculation
                valid_mask = (~np.isnan(y_pred)) & (~np.isnan(y_true))
                valid_y = y_true[valid_mask]
                valid_pred_sign = pred_sign[valid_mask]
                valid_nonzero = np.sign(valid_y) != 0
                
                da = np.mean(valid_pred_sign[valid_nonzero] == np.sign(valid_y)[valid_nonzero]) * 100 if valid_nonzero.sum() > 0 else 0
                
                all_model_das.append({'Model': model_name, 'DA': da})
                
                if alpha > best_model_alpha:
                    best_model_alpha = alpha
                    best_model_name = model_name
                    best_model_return = strat_total_ret_pct
                    best_model_maxdd = max_dd
                    best_model_winrate = win_rate
                    best_model_trades = trades
                    best_model_da = da
            
            if best_model_name:
                tabulka1_data.append({
                    'Ticker': ticker,
                    'Sektor': sector,
                    'Najlepší Model': best_model_name,
                    'B&H Výnos (%)': round(bnh_total_ret_pct, 2),
                    'ML Výnos (%)': round(best_model_return, 2),
                    'Alpha voči B&H (%)': round(best_model_alpha, 2),
                    'DA (%)': round(best_model_da, 2)
                })
                tabulka2_data.append({
                    'Ticker': ticker,
                    'B&H MaxDD (%)': round(bnh_drawdown, 2),
                    'ML MaxDD (%)': round(best_model_maxdd, 2),
                    'Počet Obchodov': best_model_trades,
                    'Win Rate (%)': round(best_model_winrate, 2)
                })
                best_models.append(best_model_name)
                ticker_da_distribution.append({'Ticker': ticker, 'Best Model DA (%)': best_model_da})

    # 2. Dôležitosť príznakov (Z reports foldra CSVs)
    feat_file = os.path.join(REPORTS_DIR, f"{ticker}_feature_importance.csv")
    if os.path.exists(feat_file):
        df_feats = pd.read_csv(feat_file, index_col=0)
        # Priemer pre kazdy feature napr. z troch modelov (RF, XGB, LGBM)
        mean_feat_imp = df_feats.mean(axis=1).sort_values(ascending=False)
        
        # Obohatíme globálny zoznam pre Obr. 3
        for feat, imp in mean_feat_imp.items():
            if feat not in all_feature_importances:
                all_feature_importances[feat] = []
            all_feature_importances[feat].append(imp)
            
        top_features = mean_feat_imp.head(3).index.tolist()
        
        # Zistíme dominantnú kategóriu
        category_score = {}
        for feat, imp in list(mean_feat_imp.items())[:10]: # Berieme top 10
            cat = get_feature_category(feat)
            category_score[cat] = category_score.get(cat, 0) + imp
        
        top_category = max(category_score, key=category_score.get) if category_score else "N/A"
        
        while len(top_features) < 3: top_features.append("N/A")
        
        tabulka3_data.append({
            'Ticker': ticker,
            'Dominantná kategória': top_category,
            'Top 1 indikátor': top_features[0],
            'Top 2 indikátor': top_features[1],
            'Top 3 indikátor': top_features[2],
        })


# --- Generovanie Outputov ---

# Ulozenie Tabuliek
df_tab1 = pd.DataFrame(tabulka1_data)
df_tab2 = pd.DataFrame(tabulka2_data)
df_tab3 = pd.DataFrame(tabulka3_data)

if not df_tab1.empty:
    df_tab1.to_csv(os.path.join(MASTER_TABLES_DIR, 'Tabulka_1_Vykonnost_a_Ziskovost.csv'), index=False)
    df_tab2.to_csv(os.path.join(MASTER_TABLES_DIR, 'Tabulka_2_Riziko_a_Stabilita.csv'), index=False)
    df_tab3.to_csv(os.path.join(MASTER_TABLES_DIR, 'Tabulka_3_Co_ovlada_trh.csv'), index=False)
    print(f"✔️ Zlúčené tabuľky boli uložené do priečinka: {MASTER_TABLES_DIR}")

# Obr 1: Priemerna DA podla modelu (Barplot)
if all_model_das:
    df_model_das = pd.DataFrame(all_model_das)
    mean_das = df_model_das.groupby('Model')['DA'].mean().reset_index().sort_values('DA', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=mean_das, x='DA', y='Model', palette='viridis')
    plt.axvline(50, color='red', linestyle='--', label='50% Hranica (Náhoda)')
    plt.title('Obr. 1: Priemerná smerová presnosť modelov naprieč všetkými 31 tickers')
    plt.xlabel('Smerová presnosť - DA (%)')
    plt.ylabel('Moel ML')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MASTER_FIGURES_DIR, 'Obr_1_Priemerna_presnost_DA.png'), dpi=300)
    plt.close()

# Obr 2: Ktorý model vyhral najčastejšie? (Pie Chart Alpha)
if best_models:
    counter = Counter(best_models)
    labels = list(counter.keys())
    sizes = list(counter.values())
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel', len(labels)))
    plt.title('Obr. 2: Percentuálne zastúpenie najlepších modelov\n(Podľa dosiahnutej Alphy voči B&H)')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(MASTER_FIGURES_DIR, 'Obr_2_Najlepsie_modely_Alpha.png'), dpi=300)
    plt.close()

# Obr 3: Top Globálne indikátory naprieč 31 trhmi
if all_feature_importances:
    global_importances = []
    for feat, imps in all_feature_importances.items():
        global_importances.append({
            'Indikátor': feat,
            'Priemerná Dôležitosť': np.mean(imps),
            'Kategória': get_feature_category(feat)
        })
    df_global_imp = pd.DataFrame(global_importances).sort_values('Priemerná Dôležitosť', ascending=False).head(15)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df_global_imp, x='Priemerná Dôležitosť', y='Indikátor', hue='Kategória', dodge=False, palette='Set2')
    plt.title('Obr. 3: Top 15 najdôležitejších spoločných indikátorových premenných\n(Priemer za 31 Tickerov)')
    plt.xlabel('Priemerný Feature Importance (MDI)')
    plt.ylabel('')
    plt.legend(title='Typ dát', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(MASTER_FIGURES_DIR, 'Obr_3_Top_15_Globalne_Indikatory.png'), dpi=300)
    plt.close()

# Obr 4: Distribucia Smerovej presnosti (DA) najlepseho modelu (Heatmap / Barplot)
if ticker_da_distribution:
    df_ticker_da = pd.DataFrame(ticker_da_distribution).sort_values('Best Model DA (%)', ascending=False)
    
    plt.figure(figsize=(14, 6))
    ax = sns.barplot(data=df_ticker_da, x='Ticker', y='Best Model DA (%)', palette='coolwarm')
    plt.axhline(50, color='red', linestyle='--', linewidth=2)
    plt.title('Obr. 4: Distribúcia Smerovej Presnosti najúčinnejšieho modelu pre dané aktívum')
    plt.ylabel('Smerová Predpovedná Úspešnosť - DA (%)')
    plt.xlabel('Spoločnosti (Tickre)')
    plt.xticks(rotation=45)
    
    # Pridanie hodnot na vrcholy baru
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.1f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points',
                   fontsize=9)
                   
    plt.tight_layout()
    plt.savefig(os.path.join(MASTER_FIGURES_DIR, 'Obr_4_Distribucia_presnosti_podla_tickerov.png'), dpi=300)
    plt.close()

print(f"✔️ Generovanie finálnych 4 grafov dokončené. Nájdete ich v priečinku: {MASTER_FIGURES_DIR}")
