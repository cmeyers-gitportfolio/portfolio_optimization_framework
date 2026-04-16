import yfinance as yf
import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime

# =============================
# 1. Configuración
# =============================

start = "2015-01-01"
end = datetime.today().strftime("%Y-%m-%d")

tickers = [
    "CHILE.SN",
    "BSAC",
    "FALABELLA.SN",
    "SQM-B.SN",
    "CMPC.SN",
    "VOO",
    "ACWX",
    "AGG",
    "CFIETFCC.SN",
    "IAU",
    "UUP",          # NUEVO ETF FX
    "^IRX",
    "CLPUSD=X"
]

# =============================
# 2. Descargar datos
# =============================

data = yf.download(
    tickers,
    start=start,
    end=end,
    interval="1mo",
    auto_adjust=True,
    progress=False
)["Close"]

data = data.resample("MS").last()

# =============================
# 3. Separar tipo de cambio
# =============================

fx = data["CLPUSD=X"]

clp_assets = [
    "CHILE.SN",
    "FALABELLA.SN",
    "SQM-B.SN",
    "CMPC.SN",
    "CFIETFCC.SN"
]

# =============================
# 4. Convertir CLP → USD
# =============================

for ticker in clp_assets:
    data[ticker] = data[ticker] / fx

# =============================
# 5. Procesar RF (^IRX)
# =============================

rf_annual = data["^IRX"] / 100
rf_monthly = (1 + rf_annual)**(1/12) - 1
data["RF_USD"] = rf_monthly

# =============================
# 6. Eliminar columnas auxiliares
# =============================

data = data.drop(columns=["^IRX", "CLPUSD=X"])
data = data.dropna(how="all")

# =============================
# 7. Formato largo
# =============================

data = data.reset_index()
data_long = data.melt(
    id_vars="Date",
    var_name="ticker",
    value_name="adj_close"
)

data_long = data_long.dropna()

# =============================
# 8. Conexión PostgreSQL
# =============================

conn = psycopg2.connect(
    host="localhost",
    database="portfolio_engine",
    user="postgres",
    password="MY_PASSWORD"
)

cur = conn.cursor()

insert_query = """
INSERT INTO precios (fecha, ticker, adj_close)
VALUES (%s, %s, %s)
ON CONFLICT (fecha, ticker)
DO UPDATE SET adj_close = EXCLUDED.adj_close;
"""

for _, row in data_long.iterrows():
    cur.execute(insert_query, (
        row["Date"],
        row["ticker"],
        float(row["adj_close"])
    ))

conn.commit()
cur.close()
conn.close()

print("Datos descargados, convertidos a USD y actualizados en PostgreSQL correctamente.")