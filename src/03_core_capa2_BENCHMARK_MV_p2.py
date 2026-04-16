import numpy as np
import pandas as pd
import psycopg2
from scipy.optimize import minimize

# =====================================================
# 1. CONFIGURACIÓN
# =====================================================

WINDOW = 36

DB_CONFIG = {
    "host": "localhost",
    "database": "portfolio_engine",
    "user": "postgres",
    "password": "MY_PASSWORD"
}

ASSET_CLASSES = {
    "RV": ['VOO','ACWX','BSAC','CHILE.SN','CMPC.SN','FALABELLA.SN','SQM-B.SN'],
    "RF": ['AGG','CFIETFCC.SN'],
    "Commodities": ['IAU'],
    "FX": ['UUP']
}

CASH_ASSET = "RF_USD"

ALL_RISKY_ASSETS = sum(ASSET_CLASSES.values(), [])

# =====================================================
# 2. CARGAR DATOS
# =====================================================

conn = psycopg2.connect(**DB_CONFIG)

query = """
SELECT fecha, ticker, adj_close
FROM precios
ORDER BY fecha;
"""

df = pd.read_sql(query, conn)
conn.close()

prices = df.pivot(index="fecha", columns="ticker", values="adj_close")
prices = prices.sort_index()

# --------------------------
# Separar RF antes de todo
# --------------------------

if CASH_ASSET not in prices.columns:
    raise ValueError("RF_USD no está en la base.")

rf_series = prices[CASH_ASSET]
risky_prices = prices[ALL_RISKY_ASSETS]

# Ventana común SOLO riesgosos
risky_prices = risky_prices.dropna()

# Retornos SOLO riesgosos
returns = risky_prices.pct_change().dropna()

# Alinear RF
rf_aligned = rf_series.reindex(returns.index)

if rf_aligned.isna().sum() > 0:
    rf_aligned = rf_aligned.ffill()

# =====================================================
# 3. OPTIMIZADOR MV
# =====================================================

def optimize_mv(cov_matrix):

    n = cov_matrix.shape[0]
    init_guess = np.ones(n) / n
    bounds = [(0.02,0.15)] * n

    constraints = [{'type':'eq', 'fun':lambda w: np.sum(w)-1}]

    for cls, assets in ASSET_CLASSES.items():
        idx = [ALL_RISKY_ASSETS.index(a) for a in assets if a in ALL_RISKY_ASSETS]

        if cls == "RV":
            constraints.append({'type':'ineq','fun':lambda w,idx=idx: np.sum(w[idx]) - 0.30})
            constraints.append({'type':'ineq','fun':lambda w,idx=idx: 0.70 - np.sum(w[idx])})

        elif cls == "RF":
            constraints.append({'type':'ineq','fun':lambda w,idx=idx: np.sum(w[idx]) - 0.10})
            constraints.append({'type':'ineq','fun':lambda w,idx=idx: 0.50 - np.sum(w[idx])})

        elif cls == "Commodities":
            constraints.append({'type':'ineq','fun':lambda w,idx=idx: 0.20 - np.sum(w[idx])})

        elif cls == "FX":
            constraints.append({'type':'ineq','fun':lambda w,idx=idx: 0.20 - np.sum(w[idx])})

    def portfolio_variance(w):
        return w.T @ cov_matrix @ w

    result = minimize(portfolio_variance,
                      init_guess,
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints)

    return result.x

# =====================================================
# 4. WALK-FORWARD OOS
# =====================================================

weights_history = []
portfolio_returns = []
portfolio_rf = []

dates = returns.index

for t in range(WINDOW, len(returns)-1):

    window_data = returns.iloc[t-WINDOW:t]
    cov_matrix = window_data.cov().values

    weights = optimize_mv(cov_matrix)
    weights_history.append(weights)

    next_ret = returns.iloc[t+1].values
    port_ret = np.dot(weights, next_ret)
    portfolio_returns.append(port_ret)

    # RF rolling promedio 36m
    rf_window = rf_aligned.iloc[t-WINDOW:t]
    portfolio_rf.append(rf_window.mean())

weights_df = pd.DataFrame(weights_history,
                          index=dates[WINDOW:len(returns)-1],
                          columns=ALL_RISKY_ASSETS)

portfolio_returns = pd.Series(portfolio_returns,
                              index=dates[WINDOW:len(returns)-1])

rf_oos = pd.Series(portfolio_rf,
                   index=dates[WINDOW:len(returns)-1])

# =====================================================
# 5. MÉTRICAS
# =====================================================

cumulative = (1 + portfolio_returns).cumprod()
total_return = cumulative.iloc[-1] - 1
annual_return = (1 + total_return)**(12/len(portfolio_returns)) - 1
annual_vol = portfolio_returns.std() * np.sqrt(12)

annual_rf = rf_oos.mean() * 12

sharpe = (annual_return - annual_rf) / annual_vol

running_max = cumulative.cummax()
drawdown = cumulative / running_max - 1
max_dd = drawdown.min()

var_95 = np.percentile(portfolio_returns,5)
cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()

calmar = annual_return / abs(max_dd)
turnover = weights_df.diff().abs().sum(axis=1).mean()

# =====================================================
# 6. RESULTADOS
# =====================================================

print("\n========== BENCHMARK MV RESULTADOS ==========\n")
print(f"Periodo OOS: {portfolio_returns.index[0]} → {portfolio_returns.index[-1]}")
print(f"Annual Return: {annual_return:.4f}")
print(f"Annual Volatility: {annual_vol:.4f}")
print(f"Sharpe (RF dinámico): {sharpe:.4f}")
print(f"Max Drawdown: {max_dd:.4f}")
print(f"CVaR 95%: {cvar_95:.4f}")
print(f"Calmar Ratio: {calmar:.4f}")
print(f"Turnover Promedio: {turnover:.4f}")
print("\n=============================================\n")

# =====================================================
# 7. EXPORTAR CSV (CARPETA ORDENADA)
# =====================================================

import os

output_path = "output_benchmark"
os.makedirs(output_path, exist_ok=True)

weights_df.to_csv(f"{output_path}/benchmark_weights.csv")
portfolio_returns.to_csv(f"{output_path}/benchmark_returns.csv")
cumulative.to_csv(f"{output_path}/benchmark_equity_curve.csv")
rf_oos.to_csv(f"{output_path}/benchmark_rf_oos.csv")

# =====================================================
# 8. GUARDAR EN SQL
# =====================================================

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# Crear tabla con PRIMARY KEY (clave para ON CONFLICT)
cur.execute("""
CREATE TABLE IF NOT EXISTS benchmark_results (
    fecha DATE PRIMARY KEY,
    retorno FLOAT
);
""")

for date, ret in portfolio_returns.items():
    cur.execute("""
        INSERT INTO benchmark_results (fecha, retorno)
        VALUES (%s,%s)
        ON CONFLICT (fecha)
        DO UPDATE SET retorno = EXCLUDED.retorno;
    """, (date, float(ret)))

conn.commit()
cur.close()
conn.close()

print("Resultados guardados en SQL y en carpeta output_benchmark correctamente.")

print("\n=============================================\n")

print(weights_df.mean())

print("\n=============================================\n")

print(rf_series.describe())
print(rf_series.tail())

print("\n=============================================\n")