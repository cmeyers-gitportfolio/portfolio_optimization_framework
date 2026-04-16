import psycopg2
import pandas as pd
import numpy as np
import os

WINDOW = 36
CASH_ASSET = "RF_USD"

DB_CONFIG = {
    "host": "localhost",
    "database": "portfolio_engine",
    "user": "postgres",
    "password": "MY_PASSWORD"
}

# ==========================
# 1. Cargar datos
# ==========================

conn = psycopg2.connect(**DB_CONFIG)

query = """
SELECT fecha, ticker, adj_close
FROM precios
ORDER BY fecha;
"""

df = pd.read_sql(query, conn)
conn.close()

prices = df.pivot(index="fecha", columns="ticker", values="adj_close")
prices.index = pd.to_datetime(prices.index)
prices = prices.sort_index()

if CASH_ASSET not in prices.columns:
    raise ValueError("RF_USD no está en base.")

rf_series = prices[CASH_ASSET]
risky_prices = prices.drop(columns=[CASH_ASSET])

risky_prices = risky_prices.dropna()
returns = risky_prices.pct_change().dropna()

rf_aligned = rf_series.reindex(returns.index).ffill()

# ==========================
# 2. Exportar base limpia
# ==========================

output_path = "output_modulo3"
os.makedirs(output_path, exist_ok=True)

returns.to_csv(f"{output_path}/returns_full.csv")
rf_aligned.to_csv(f"{output_path}/rf_full.csv")

print("✅ Módulo 3 listo.")
print("Datos base exportados para módulo 4.")