import numpy as np
import pandas as pd
import psycopg2
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize
import os

WINDOW = 36
alpha = 0.95

DB_CONFIG = {
    "host": "localhost",
    "database": "portfolio_engine",
    "user": "postgres",
    "password": "MY_PASSWORD"
}

# ==========================
# 1. Cargar datos base
# ==========================

returns = pd.read_csv("output_modulo3/returns_full.csv", index_col=0, parse_dates=True)
rf_series = pd.read_csv("output_modulo3/rf_full.csv", index_col=0, parse_dates=True).squeeze()

tickers = returns.columns.tolist()
n = len(tickers)

# ==========================
# 2. Clasificación activos
# ==========================

RV = ["VOO","ACWX","BSAC","CHILE.SN","CMPC.SN","FALABELLA.SN","SQM-B.SN"]
RF = ["AGG","CFIETFCC.SN"]
COM = ["IAU"]
FX = ["UUP"]

RV = [t for t in RV if t in tickers]
RF = [t for t in RF if t in tickers]
COM = [t for t in COM if t in tickers]
FX = [t for t in FX if t in tickers]

def sum_class(w, asset_list):
    if not asset_list:
        return 0
    return np.sum([w[tickers.index(t)] for t in asset_list])

bounds = [(0.02,0.15)]*n

constraints = [
    {"type":"eq","fun":lambda w: np.sum(w)-1},
    {"type":"ineq","fun":lambda w: sum_class(w,RV)-0.30},
    {"type":"ineq","fun":lambda w: 0.70-sum_class(w,RV)},
    {"type":"ineq","fun":lambda w: sum_class(w,RF)-0.10},
    {"type":"ineq","fun":lambda w: 0.50-sum_class(w,RF)},
    {"type":"ineq","fun":lambda w: 0.20-sum_class(w,COM)},
    {"type":"ineq","fun":lambda w: 0.20-sum_class(w,FX)}
]

# ==========================
# 3. Walk Forward Rolling
# ==========================

weights_history = []
portfolio_returns = []
portfolio_rf = []

dates = returns.index

for t in range(WINDOW, len(returns)-1):

    window_data = returns.iloc[t-WINDOW:t]

    # μ shrinkage
    mu_hist = window_data.mean()
    mu_global = mu_hist.mean()
    mu_shrink = 0.5*mu_hist + 0.5*mu_global
    mu_annual = mu_shrink*12

    # Σ shrinkage
    lw = LedoitWolf()
    lw.fit(window_data)
    Sigma = lw.covariance_*12

    def portfolio_vol(w):
        return np.sqrt(w @ Sigma @ w)

    def min_var(w):
        return w @ Sigma @ w

    def neg_sharpe(w):
        ret = w @ mu_annual.values
        vol = portfolio_vol(w)
        rf_roll = rf_series.iloc[t-WINDOW:t].mean()*12
        return -(ret - rf_roll)/vol

    def cvar_obj(w):
        port = window_data.values @ w
        var = np.percentile(port,(1-alpha)*100)
        cvar = port[port<=var].mean()
        return -cvar

    w0 = np.ones(n)/n

    w_min = minimize(min_var,w0,bounds=bounds,constraints=constraints).x
    w_sh = minimize(neg_sharpe,w0,bounds=bounds,constraints=constraints).x
    w_cv = minimize(cvar_obj,w0,bounds=bounds,constraints=constraints).x

    w_rob = 0.4*w_min + 0.3*w_sh + 0.3*w_cv
    w_rob = w_rob/np.sum(w_rob)

    weights_history.append(w_rob)

    next_ret = returns.iloc[t+1].values
    portfolio_returns.append(np.dot(w_rob,next_ret))

    portfolio_rf.append(rf_series.iloc[t-WINDOW:t].mean())

# ==========================
# 4. Construcción OOS
# ==========================

weights_df = pd.DataFrame(weights_history,
                          index=dates[WINDOW:len(returns)-1],
                          columns=tickers)

portfolio_returns = pd.Series(portfolio_returns,
                              index=weights_df.index)

rf_oos = pd.Series(portfolio_rf,
                   index=weights_df.index)

cumulative = (1+portfolio_returns).cumprod()

# ---- MÉTRICAS CORREGIDAS ----

total_return = cumulative.iloc[-1] - 1

annual_return = (1 + total_return)**(12/len(portfolio_returns)) - 1
annual_vol = portfolio_returns.std() * np.sqrt(12)
annual_rf = rf_oos.mean() * 12

if annual_vol != 0:
    sharpe = (annual_return - annual_rf) / annual_vol
else:
    sharpe = np.nan

running_max = cumulative.cummax()
drawdown = cumulative / running_max - 1
max_dd = drawdown.min()

# ==========================
# 5. Exportar
# ==========================

output_path="output_modulo4"
os.makedirs(output_path,exist_ok=True)

weights_df.to_csv(f"{output_path}/robusto_weights.csv")
portfolio_returns.to_csv(f"{output_path}/robusto_returns_oos.csv")
cumulative.to_csv(f"{output_path}/robusto_equity_curve.csv")
rf_oos.to_csv(f"{output_path}/robusto_rf_oos.csv")

metrics = pd.Series({
    "Annual Return":annual_return,
    "Annual Vol":annual_vol,
    "Sharpe":sharpe,
    "MaxDD":max_dd
})

metrics.to_csv(f"{output_path}/robusto_metrics.csv")

# ==========================
# 6. Guardar en SQL
# ==========================

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS robusto_results (
    fecha DATE PRIMARY KEY,
    retorno FLOAT
);
""")

for date, ret in portfolio_returns.items():
    cur.execute("""
    INSERT INTO robusto_results (fecha, retorno)
    VALUES (%s,%s)
    ON CONFLICT (fecha)
    DO UPDATE SET retorno = EXCLUDED.retorno;
    """,(date,float(ret)))

conn.commit()
cur.close()
conn.close()

# =====================================================
# 7. REPORTE COMPLETO EN TERMINAL
# =====================================================

print("\n========== ROLLING ROBUSTO RESULTADOS ==========\n")

print(f"Periodo OOS: {portfolio_returns.index[0]} → {portfolio_returns.index[-1]}")
print(f"Observaciones OOS: {len(portfolio_returns)}")

# Métricas principales
print("\n--- Métricas Anuales ---")
print(f"Annual Return: {annual_return:.4f}")
print(f"Annual Volatility: {annual_vol:.4f}")
print(f"Sharpe (RF dinámico): {sharpe:.4f}")

# Drawdown
print("\n--- Riesgo ---")
print(f"Max Drawdown: {max_dd:.4f}")

# CVaR
var_95 = np.percentile(portfolio_returns,5)
cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
print(f"CVaR 95%: {cvar_95:.4f}")

# Calmar
calmar = annual_return / abs(max_dd)
print(f"Calmar Ratio: {calmar:.4f}")

# Turnover
turnover = weights_df.diff().abs().sum(axis=1).mean()
print(f"Turnover Promedio: {turnover:.4f}")

# Distribución retornos
print("\n--- Distribución Retornos Mensuales ---")
print(portfolio_returns.describe())

# =====================================================
# Pesos promedio
# =====================================================

print("\n--- Pesos Promedio por Activo ---")
print(weights_df.mean().round(4))

print("\n--- Pesos Promedio por Clase ---")

avg_weights = weights_df.mean()

rv_weight = sum(avg_weights[t] for t in RV if t in avg_weights.index)
rf_weight = sum(avg_weights[t] for t in RF if t in avg_weights.index)
com_weight = sum(avg_weights[t] for t in COM if t in avg_weights.index)
fx_weight = sum(avg_weights[t] for t in FX if t in avg_weights.index)

print(f"RV promedio: {rv_weight:.4f}")
print(f"RF promedio: {rf_weight:.4f}")
print(f"Commodities promedio: {com_weight:.4f}")
print(f"FX promedio: {fx_weight:.4f}")

# =====================================================
# Diagnóstico estructural
# =====================================================

print("\n--- Diagnóstico Estructural ---")

if rv_weight < 0.40:
    print("Portafolio estructuralmente defensivo.")
elif rv_weight > 0.60:
    print("Portafolio estructuralmente agresivo.")
else:
    print("Portafolio balanceado.")

print("\n==============================================\n")


# =====================================================
# 8. MÉTRICAS AVANZADAS VS BENCHMARK
# =====================================================

print("\n========== MÉTRICAS AVANZADAS VS BENCHMARK ==========\n")

# -----------------------------------------------------
# Cargar Benchmark OOS
# -----------------------------------------------------

benchmark_returns = pd.read_csv(
    "output_benchmark/benchmark_returns.csv",
    index_col=0,
    parse_dates=True
).squeeze()

# Alinear fechas
aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
aligned.columns = ["model", "benchmark"]

excess = aligned["model"] - aligned["benchmark"]

# -----------------------------------------------------
# 1) CAGR EXACTO
# -----------------------------------------------------

start_date = portfolio_returns.index[0]
end_date = portfolio_returns.index[-1]

years = (end_date - start_date).days / 365.25

cagr = (cumulative.iloc[-1])**(1/years) - 1

print(f"CAGR exacto: {cagr:.4f}")

# -----------------------------------------------------
# 2) SORTINO
# -----------------------------------------------------

downside = portfolio_returns[portfolio_returns < 0]

if len(downside) > 0:
    downside_std = downside.std() * np.sqrt(12)
    sortino = (cagr - annual_rf) / downside_std if downside_std != 0 else np.nan
else:
    sortino = np.nan

print(f"Sortino Ratio: {sortino:.4f}")

# -----------------------------------------------------
# 3) TRACKING ERROR
# -----------------------------------------------------

tracking_error = excess.std() * np.sqrt(12)

print(f"Tracking Error: {tracking_error:.4f}")

# -----------------------------------------------------
# 4) INFORMATION RATIO
# -----------------------------------------------------

if tracking_error != 0:
    information_ratio = (excess.mean() * 12) / tracking_error
else:
    information_ratio = np.nan

print(f"Information Ratio: {information_ratio:.4f}")

# -----------------------------------------------------
# 5) STABILITY METRICS
# -----------------------------------------------------

# A) % Outperformance
outperf_ratio = (excess > 0).sum() / len(excess)

# B) Consistency Ratio
excess_mean = excess.mean()
excess_std = excess.std()
consistency_ratio = excess_mean / excess_std if excess_std != 0 else np.nan

# C) Autocorrelación del exceso
autocorr_1 = excess.autocorr(lag=1)

print("\n--- Stability Metrics ---")
print(f"% Outperformance: {outperf_ratio:.4f}")
print(f"Consistency Ratio: {consistency_ratio:.4f}")
print(f"Autocorr Exceso (lag1): {autocorr_1:.4f}")

# -----------------------------------------------------
# 6) Diagnóstico de Skill
# -----------------------------------------------------

print("\n--- Diagnóstico de Skill ---")

if information_ratio > 0.75:
    print("Skill institucional fuerte.")
elif information_ratio > 0.50:
    print("Skill aceptable y consistente.")
elif information_ratio > 0.25:
    print("Skill débil pero positivo.")
else:
    print("No hay evidencia robusta de alpha estructural.")

print("\n====================================================\n")

# =====================================================
# 9. SIGNIFICANCIA ESTADÍSTICA DEL ALPHA
# =====================================================

T = len(excess)

mean_excess = excess.mean()
std_excess = excess.std()

t_stat = mean_excess / (std_excess / np.sqrt(T))

print("\n========== TEST DE SIGNIFICANCIA ==========\n")
print(f"Observaciones: {T}")
print(f"Mean Excess Monthly: {mean_excess:.6f}")
print(f"Std Excess Monthly: {std_excess:.6f}")
print(f"T-stat Alpha: {t_stat:.4f}")

if abs(t_stat) > 2:
    print("Alpha estadísticamente significativo (5%).")
else:
    print("Alpha NO estadísticamente significativo.")

print("\n===========================================\n")

# =====================================================
# 10. ANÁLISIS DE RÉGIMEN: PERÍODO INFLACIÓN
# =====================================================

inflation_start = "2022-01-01"
inflation_end = "2023-12-31"

regime_data = aligned.loc[inflation_start:inflation_end]

if len(regime_data) > 0:

    regime_excess = regime_data["model"] - regime_data["benchmark"]

    regime_cum_model = (1 + regime_data["model"]).prod() - 1
    regime_cum_bench = (1 + regime_data["benchmark"]).prod() - 1

    regime_dd_model = (1 + regime_data["model"]).cumprod()
    regime_dd_model = regime_dd_model / regime_dd_model.cummax() - 1

    regime_dd_bench = (1 + regime_data["benchmark"]).cumprod()
    regime_dd_bench = regime_dd_bench / regime_dd_bench.cummax() - 1

    print("\n========== RÉGIMEN INFLACIÓN ==========\n")
    print(f"Retorno acumulado Modelo: {regime_cum_model:.4f}")
    print(f"Retorno acumulado Benchmark: {regime_cum_bench:.4f}")
    print(f"MaxDD Modelo: {regime_dd_model.min():.4f}")
    print(f"MaxDD Benchmark: {regime_dd_bench.min():.4f}")
    print(f"% Outperformance Régimen: {(regime_excess>0).mean():.4f}")
    print("\n========================================\n")

# =====================================================
# 11. EQUITY CURVE COMPARATIVA
# =====================================================

import matplotlib.pyplot as plt

benchmark_cum = (1 + aligned["benchmark"]).cumprod()
model_cum = (1 + aligned["model"]).cumprod()

plt.figure()
plt.plot(model_cum)
plt.plot(benchmark_cum)
plt.title("Equity Curve Comparativa")
plt.xlabel("Fecha")
plt.ylabel("Crecimiento del Capital")
plt.show()

# =====================================================
# 12. Drawdown Comparativo
# =====================================================

model_dd = model_cum / model_cum.cummax() - 1
bench_dd = benchmark_cum / benchmark_cum.cummax() - 1

plt.figure()
plt.plot(model_dd)
plt.plot(bench_dd)
plt.title("Drawdown Comparativo")
plt.xlabel("Fecha")
plt.ylabel("Drawdown")
plt.show()

# =====================================================
# 13. Exceso Acumulado
# =====================================================

excess_cum = (1 + excess).cumprod()

plt.figure()
plt.plot(excess_cum)
plt.title("Exceso Acumulado del Modelo vs Benchmark")
plt.xlabel("Fecha")
plt.ylabel("Crecimiento del Exceso")
plt.show()

print("========================================")

print("✅ Módulo 4 Rolling Robusto completado.")