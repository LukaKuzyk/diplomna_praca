# Framework strojového učenia pre predikciu finančných trhov

Komplexný algoritmický framework pre predikciu výnosov akcií, cien a smerových pohybov s využitím pokročilých klasifikačných a regresných modelov strojového učenia založených na rozhodovacích stromoch. Framework je navrhnutý tak, aby fungoval pre akýkoľvek akciový symbol (predvolene AAPL) a ponúka dynamické načítavanie dát, rozsiahlu tvorbu príznakov (feature engineering), robustnú validáciu pomocou kĺzavého okna (walk-forward), najmodernejšie stacking modely a detailnú vizuálnu analýzu backtestingu.

Tento repozitár slúži ako praktická, kódová implementácia diplomovej práce, ktorá skúma efektívnosť a ekonomickú životaschopnosť modelov strojového učenia v moderných kvantitatívnych financiách.

---

## 1. Prehľad projektu a výskumné ciele

Hlavným cieľom tohto projektu je kvantitatívne vyhodnotiť, či dokážu moderné modely strojového učenia identifikovať nelineárne vzorce v historických trhových dátach a tak presnejšie predikovať budúci smer vývoja ceny než pasívny benchmark (stratégia Buy & Hold). Framework skúma:

- **Predikčnú silu**: Dokážu algoritmy presne predpovedať smer logaritmických výnosov na nasledujúci deň?
- **Ekonomickú životaschopnosť**: Je obchodovanie na základe týchto signálov ziskové aj po započítaní reálnych trhových obmedzení, akými sú transakčné náklady (predvolene 5 bázických bodov)?
- **Dôležitosť príznakov**: Ktoré premenné (hybnosť, objem, technické indikátory alebo širšie trhové indexy ako S&P 500/VIX) majú najväčšiu predikčnú váhu?
- **Stabilitu modelov**: Ponúkajú pokročilé ansámblové metódy, ako Stacking Regresory/Klasifikátory, lepšie výnosy očistené o riziko (Sharpe Ratio) a nižšie prepady (drawdowns) v porovnaní so štandardnými modelmi?

---

## 2. Štruktúra projektu

```
ml-finance/
├── data/                    # Surové trhové dáta a zmapované odvodené príznaky
│   ├── [ticker].csv         # Surové dáta zvoleného inštrumentu (OHLCV) spojené s trhovým kontextom (VIX, QQQ)
│   ├── [ticker]_earnings.csv# Historické dátumy zverejňovania hospodárskych výsledkov
│   └── [ticker]_features.csv# Finálny predspracovaný tabuľkový dataset pripravený pre ML
├── models/                  # Vypočítané predikcie vygenerované počas walk-forward validácie
│   └── [ticker]_ml_predictions.csv
├── reports/
│   ├── [ticker]_figures/    # Vygenerované vizualizačné grafy rozdelené podľa typu analýzy
│   ├── [ticker]_feature_importance.csv  # Vypočítané váhy dôležitosti jednotlivých príznakov
│   └── [ticker]_ml_metrics_summary.txt  # Finálne záznamy výkonnostných metrík (RMSE, DA, Sharpe)
├── src/                     # Adresár so zdrojovým kódom
│   ├── 01_download_data.py  # Sťahovanie dát, spájanie trhového kontextu a feature engineering
│   ├── 03_model_ml.py       # ML Pipeline, definície modelov, trénovanie a walk-forward validácia
│   ├── 04_backtest_and_plots.py  # Výpočet výkonnosti, backtesting portfólia a tvorba grafov
│   ├── config.py            # Globálne konštanty (napr. signálne prahy, zoznamy príznakov)
│   ├── features.py          # Logika technických indikátorov a výber príznakov (Lasso)
│   ├── models.py            # Inicializácia modelov XGBoost, Random Forest a Stacking Ensembles
│   └── utils.py             # Vyhodnocovacie metriky a objekty na rozdelenie dát (cross-validation)
└── README.md
```

---

## 3. Dátová pipeline a tvorba príznakov (`01_download_data.py`)

### Aké dáta sa používajú?

Dáta sa dynamicky sťahujú pomocou API knižnice `yfinance`. Zadaním parametra `--ticker [SYMBOL]` skript stiahne:

1. **Dáta primárneho aktíva**: Historické denné ceny Open, High, Low, Close a Volume pre cieľový inštrument (napr. AAPL).
2. **Makroekonomický a trhový kontext**: Dáta pre index S&P 500 (`^GSPC`), index volatility VIX (`^VIX`) a technologický proxy index Nasdaq 100 (`QQQ`).
3. **Korporátne udalosti**: Historické dátumy zverejňovania výsledkov pre vytvorenie binárnych kalendárnych príznakov.

### Ako sú dáta spracované (Feature Engineering)?

Surové časové rady sú už z podstaty zašumené a nestacionárne. Pipeline vytvára robustný tabuľkový dataset, ktorý obsahuje:

- **Cieľové premenné**: Budúci spojitý logaritmický výnos (`log_ret.shift(-1)`) a binárne smerové triedy.
- **Historické lagy**: Spätné spojité výnosy (`log_ret_lag_1`, `log_ret_lag_5` atď.) a oneskorenia objemu.
- **Technické indikátory**:
  - Trendové: Jednoduché kĺzavé priemery (`sma_5`, `sma_20`), MACD, Hybnosť (Momentum).
  - Oscilátory: Index relatívnej sily (`rsi_14`), Stochastický oscilátor (`stoch_k`, `stoch_d`), Commodity Channel Index (`cci_20`).
  - Volatilita: Bollingerove pásma, Priemerný skutočný rozsah (`atr_14`), Realizovaná 5-dňová variancia (`rv_5`).
- **Štatistické príznaky**: Kĺzavá šikmosť (skewness) a špicatosť (kurtosis).
- **Kalendár a zverejňovanie výsledkov (Earnings)**: Mesiac, deň v týždni a príznak `earnings_week` pre zachytenie sezónnosti a zhlukovania volatility.

### Ukladanie dát

Dáta sú lokálne pre vybraný čas udržiavané v adresári `data/`, aby sa predišlo opakovaným a nadbytočným API volaniam. Kompletne pripravený dataset sa ukladá ako `data/[ticker]_features.csv`.

---

## 4. Modely strojového učenia a validácia (`03_model_ml.py`)

### Prístup k modelovaniu

Pipeline trénuje dva typy modelov pre rozdielne paradigmy predikcie:

1. **Regresory (Spojitý výstup)**: XGBoost, Random Forest a Meta-Stacking Regressor, ktoré kombinujú ich predikcie. Výstupom je exaktná hodnota očakávaného logaritmického výnosu.
2. **Klasifikátory (Pravdepodobnostný výstup)**: XGBoost, Random Forest a Meta-Stacking Classifier. Výstupom je pravdepodobnosť, že výnos v nasledujúci deň bude prísne kladný (> 0.5 pravdepodobnosť).

### Walk-Forward Validácia (Time-Series Cross-Validation)

Aby sa definitívne zabránilo _únikom informácií (data leakage)_ a posunu (tzv. _look-ahead bias_), štandardné náhodné rozdeľovanie na trénovacie a testovacie sady sa tu striktne nepoužíva. Namiesto toho model beží na roztvorenom posúvajúcom sa okne:

- **Trénovacie okno**: 1008 obchodných dní (~4 roky histórie).
- **Testovacie okno**: 63 obchodných dní (~1 kvartál budúcich predpovedí).
- **Krok (Step)**: 63 dní.
  Model sa iteratívne trénuje na historických dátach, urobí predikcie pre nasledujúcich 63 dní, okno sa posunie dopredu o 63 dní a model sa nanovo pretrénuje.

### Výber príznakov (Feature Selection)

Pred fázou samotného trénovania sa v rámci každého okna využíva **Lasso Regresia (L1 regularizácia)** na dynamický výber len tých najviac relevantných prediktívnych premenných, čím sa dáta očisťujú od šumu a vysoko kolineárnych indikátorov.

---

## 5. Backtesting, analýza a vizualizácie (`04_backtest_and_plots.py`)

Záverečným krokom je spojenie všetkých predikčných okien s jej _reálnymi_ historickými výsledkami a nasimulovanie obchodnej stratégie. Ak regresor predikuje výnos presahujúci aktivačný prah `SIGNAL_THRESHOLD` (alebo je pravdepodobnosť klasifikátora > 0.55), stratégia simuluje nákup a držanie počas jedného dňa.

### Vygenerované grafy (`reports/[ticker]_figures/`)

Výsledná vizuálna diagnostika pokrýva štyri hlavné oblasti hodnotenia:

#### A. Porovnanie modelov a analýza chybovosti (Model Comparison)

- `model_comp_pred_vs_actual.png`: Bodový graf (Scatter plot) mapujúci predikované verzus reálne výnosy.
- `model_comp_error_dist.png`: Graf rozdelenia hustoty predikčných chýb pre regresory.
- `model_comp_rolling_da.png`: Kĺzavý priemer (50-dňový) pre čistú smerovú presnosť v porovnaní s pasívnym benchmarkom (Buy & Hold).
- `model_comp_signal_corr.png`: Teplotná mapa (Heatmap) naznačujúca, ako veľmi sa rôzne modely zhodujú vo svojich predikciách.

#### B. Výkonnosť stratégie a ekonomický backtest (Strategy Performance)

- `strat_perf_equity_curves.png`: Porovnanie kumulatívneho rastu ML portfólia (nákupné signály) proti pasívnej stratégii Buy & Hold.
- `strat_perf_total_returns.png` a `strat_perf_sharpe_ratios.png`: Stĺpcové porovnania celkovej ziskovosti a efektívnosti výnosov očistených o riziko.
- `strat_perf_monthly_returns.png`: Matica riadenia rizík, ktorá ukazuje percentuálne výnosy mesiac po mesiaci.

#### C. Stabilita predikcií a istota modelov (Prediction Stability)

- `pred_stab_volatility.png`: Znázorňuje mieru volatility, s akou sa menia predikcie jednotlivých modelov v čase.
- `pred_stab_hit_rate.png`: Analyzuje presnosť v závislosti na sile signálu. Zodpovedá otázku: _„Prináša vyšší očakávaný výnos naozaj vyššiu úspešnosť zásahu?“_
- `pred_stab_magnitude_dist.png`: Porovnáva základnú distribúciu spojitých výstupov (regresií) a prediktívnej istoty klasifikátorov (Pravdepodobnosti).

#### D. Dôležitosť príznakov a interpretovateľnosť SHAP

- `feat_imp_top20_avg.png` a `feat_imp_top10_models.png`: Štandardné váhy dôležitosti Gini/Tree, indikujúce najvplyvnejšie premenné.
- `feat_imp_categories.png`: Zoskupenie pre určenie, či sú kľúčovým determinantom Trendy, Trh, Objem alebo oneskorenia výnosov.
- **SHAP analýza** (`shap_beeswarm.png`, `shap_dep_1.png`): V prípade inštalovanej knižnice `shap` analytika ponúka modely založené na interpretovateľnej AI (XAI) definujúce, _ako presne_ vyššie alebo nižšie hodnoty indikátorov ovplyvnili logaritmický výnos.

### Vývoj a zber metrík

Záverečné hodnoty sú zaznamenávané v textovom reporte `reports/[ticker]_ml_metrics_summary.txt`:

- **Raw DA (Directional Accuracy)**: Celkové percento správne odhadnutého smeru.
- **Confident DA**: Presnosť iba u obchodov, ktoré prekonajú signálny prah istoty.
- **Coverage (Pokrytie)**: Percento dní, kedy model skutočne vykonal transakciu (identifikoval silný signál).

---

## 6. Spustenie a inštalácia

1. **Vytvorenie virtuálneho prostredia:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Na OS Windows: .venv\Scripts\activate
   ```

2. **Inštalácia závislostí:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Spustenie kompletnej pipeline:**

   ```bash
   # Stiahnutie dát a tvorba príznakov (napr. u MSFT za 5 rokov)
   python src/01_download_data.py --ticker MSFT --years 5

   # Trénovanie modelov za pomoci Walk-Forward validácie (pre GridSearch zapnite --tune)
   python src/03_model_ml.py --ticker MSFT --train_window 1008 --test_window 63

   # Simulácia backtestingu a generovanie vizualizácií
   python src/04_backtest_and_plots.py --ticker MSFT
   ```
