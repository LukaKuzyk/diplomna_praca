import pandas as pd
import matplotlib.pyplot as plt

# Завантаження
df = pd.read_csv("models/baseline_log_ret_predictions.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# Вибрати останні 30 днів
df_plot = df.tail(360)

# Побудова графіка
plt.figure(figsize=(12,6))
plt.plot(df_plot["date"], df_plot["y_true"], label="Фактична ціна", color='blue')
plt.plot(df_plot["date"], df_plot["y_pred_arima"], label="ARIMA прогноз", color='orange')
plt.plot(df_plot["date"], df_plot["y_pred_garch_mean"], label="GARCH прогноз", color='green')

# Довірчий інтервал (необов'язково)
plt.fill_between(df_plot["date"], df_plot["y_lower"], df_plot["y_upper"], color='orange', alpha=0.2)

# Прогноз на останній день
last_row = df_plot.iloc[-1]
plt.scatter(last_row["date"], last_row["y_pred_arima"], color='red', label=f"Прогноз на {last_row['date'].date()}")

# Стилі
plt.title("Прогнози моделей vs. Фактична ціна (останні 30 днів)")
plt.xlabel("Дата")
plt.ylabel("Ціна")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("reports/figures/arima_garch_preds.png")
plt.show()
