# ==========================================================
# MODULO 5 – RISK ENGINE
# VaR / CVaR COMPLETO – VERSION INSTITUCIONAL
# Export CSV + PostgreSQL
# ==========================================================

import os
import uuid
import numpy as np
import pandas as pd
import psycopg2
from datetime import datetime
from scipy.stats import norm, skew, kurtosis, t
from scipy.optimize import minimize

# ==========================================================
# CONFIGURACIÓN GENERAL
# ==========================================================

ALPHAS = [0.95, 0.99]
HORIZONS = [1, 12]
OUTPUT_DIR = "output_modulo5"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# PostgreSQL Config (AJUSTAR A TU ENTORNO)
DB_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "dbname": "portfolio_engine",
    "user": "postgres",
    "password": "MY_PASSWORD"
}

# ==========================================================
# CARGA DE DATOS
# ==========================================================

model = pd.read_csv("output_modulo4/robusto_returns_oos.csv", index_col=0, parse_dates=True)
benchmark = pd.read_csv("output_benchmark/benchmark_returns.csv", index_col=0, parse_dates=True)

model_returns = model.squeeze()
benchmark_returns = benchmark.squeeze()
excess_returns = model_returns - benchmark_returns

# ==========================================================
# FUNCIONES VaR / CVaR
# ==========================================================

def historical_var(returns, alpha):
    return np.percentile(returns, (1-alpha)*100)

def historical_cvar(returns, alpha):
    var = historical_var(returns, alpha)
    return returns[returns <= var].mean()

def parametric_var(returns, alpha):
    mu = returns.mean()
    sigma = returns.std()
    z = norm.ppf(1-alpha)
    return mu + z*sigma

def cornish_fisher_var(returns, alpha):
    mu = returns.mean()
    sigma = returns.std()
    s = skew(returns)
    k = kurtosis(returns)
    z = norm.ppf(1-alpha)

    z_cf = (z +
            (1/6)*(z**2-1)*s +
            (1/24)*(z**3-3*z)*(k-3) -
            (1/36)*(2*z**3-5*z)*(s**2))

    return mu + z_cf*sigma

# ==========================================================
# Student-t MLE
# ==========================================================

def fit_t_distribution(returns):

    def neg_log_likelihood(params):
        df, loc, scale = params
        if df <= 2 or scale <= 0:
            return np.inf
        return -np.sum(t.logpdf(returns, df, loc, scale))

    initial = [5, returns.mean(), returns.std()]
    bounds = [(2.1, 50), (None, None), (1e-6, None)]
    result = minimize(neg_log_likelihood, initial, bounds=bounds)

    return result.x

def student_t_var(returns, alpha):
    df, loc, scale = fit_t_distribution(returns)
    q = t.ppf(1-alpha, df)
    return loc + q*scale

# ==========================================================
# HORIZON ADJUSTMENT
# ==========================================================

def compound_returns(returns, horizon):
    if horizon == 1:
        return returns
    return (1+returns).rolling(horizon).apply(np.prod, raw=True) - 1

# ==========================================================
# MOTOR PRINCIPAL
# ==========================================================

def compute_var_block(name, returns, run_id, results_list):

    print(f"\n========== {name} ==========")

    for horizon in HORIZONS:

        r = compound_returns(returns, horizon).dropna()
        print(f"\n--- Horizon: {horizon}M ---")

        for alpha in ALPHAS:

            var_hist = historical_var(r, alpha)
            cvar_hist = historical_cvar(r, alpha)
            var_norm = parametric_var(r, alpha)
            var_cf = cornish_fisher_var(r, alpha)
            var_t = student_t_var(r, alpha)

            print(f"\nAlpha: {alpha}")
            print(f"VaR Hist:        {var_hist:.4f}")
            print(f"CVaR Hist:       {cvar_hist:.4f}")
            print(f"VaR Normal:      {var_norm:.4f}")
            print(f"VaR CornishFish: {var_cf:.4f}")
            print(f"VaR Student-t:   {var_t:.4f}")

            results_list.append({
                "run_id": run_id,
                "timestamp": datetime.now(),
                "portfolio": name,
                "horizon_m": horizon,
                "alpha": alpha,
                "var_hist": var_hist,
                "cvar_hist": cvar_hist,
                "var_normal": var_norm,
                "var_cornish_fisher": var_cf,
                "var_student_t": var_t,
                "n_obs": len(r)
            })

# ==========================================================
# EXPORTACIÓN CSV
# ==========================================================

def export_csv(results_df, run_id):

    file_name = f"var_cvar_summary_{run_id}.csv"
    path = os.path.join(OUTPUT_DIR, file_name)
    results_df.to_csv(path, index=False)

    print(f"\nCSV exportado en: {path}")

# ==========================================================
# EXPORTACIÓN POSTGRESQL
# ==========================================================

def export_postgres(results_df):

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    create_table_query = """
    CREATE TABLE IF NOT EXISTS risk_var_cvar_results (
        run_id UUID,
        timestamp TIMESTAMP,
        portfolio VARCHAR(20),
        horizon_m INTEGER,
        alpha FLOAT,
        var_hist FLOAT,
        cvar_hist FLOAT,
        var_normal FLOAT,
        var_cornish_fisher FLOAT,
        var_student_t FLOAT,
        n_obs INTEGER
    );
    """

    cur.execute(create_table_query)
    conn.commit()

    insert_query = """
    INSERT INTO risk_var_cvar_results (
        run_id, timestamp, portfolio, horizon_m, alpha,
        var_hist, cvar_hist, var_normal,
        var_cornish_fisher, var_student_t, n_obs
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    for _, row in results_df.iterrows():
        cur.execute(insert_query, tuple(row))

    conn.commit()
    cur.close()
    conn.close()

    print("Resultados insertados en PostgreSQL.")

# ==========================================================
# EJECUCIÓN
# ==========================================================

if __name__ == "__main__":

    print("\n================================================")
    print("RISK ENGINE – VaR / CVaR BLOQUE INSTITUCIONAL")
    print("================================================")

    
    run_id = str(uuid.uuid4())
    results = []

    compute_var_block("MODELO", model_returns, run_id, results)
    compute_var_block("BENCHMARK", benchmark_returns, run_id, results)
    compute_var_block("EXCESO", excess_returns, run_id, results)

    results_df = pd.DataFrame(results)

    print("\n========== EXPORTANDO RESULTADOS ==========")

    export_csv(results_df, run_id)
    export_postgres(results_df)

    print("\n========== FIN BLOQUE VaR ==========")

    # ==========================================================
# HISTORICAL REPLAY INSTITUCIONAL
# ==========================================================

print("\n================================================")
print("HISTORICAL REPLAY – ARQUITECTURA ACTUAL")
print("================================================")

# ----------------------------------------------------------
# 1. Cargar returns full
# ----------------------------------------------------------

returns_full = pd.read_csv(
    "output_modulo3/returns_full.csv",
    index_col=0,
    parse_dates=True
)

# ----------------------------------------------------------
# 2. Cargar pesos promedio
# ----------------------------------------------------------

robusto_weights = pd.read_csv(
    "output_modulo4/robusto_weights.csv",
    index_col=0
)

benchmark_weights = pd.read_csv(
    "output_benchmark/benchmark_weights.csv",
    index_col=0
)

w_model = robusto_weights.mean().values
w_bench = benchmark_weights.mean().values

# Asegurar orden consistente
returns_full = returns_full[robusto_weights.columns]

# ----------------------------------------------------------
# 3. Construir Replay
# ----------------------------------------------------------

replay_model = returns_full.values @ w_model
replay_bench = returns_full.values @ w_bench

replay_model = pd.Series(replay_model, index=returns_full.index)
replay_bench = pd.Series(replay_bench, index=returns_full.index)

# ----------------------------------------------------------
# 4. Métricas Replay
# ----------------------------------------------------------

def replay_metrics(name, series):

    cumulative = (1 + series).cumprod()
    total_return = cumulative.iloc[-1] - 1

    years = (series.index[-1] - series.index[0]).days / 365.25
    cagr = cumulative.iloc[-1]**(1/years) - 1

    vol = series.std() * np.sqrt(12)

    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    max_dd = drawdown.min()

    var_95 = np.percentile(series,5)
    cvar_95 = series[series <= var_95].mean()

    print(f"\n--- {name} ---")
    print(f"CAGR Histórico: {cagr:.4f}")
    print(f"Vol Anual: {vol:.4f}")
    print(f"Max Drawdown: {max_dd:.4f}")
    print(f"VaR 95%: {var_95:.4f}")
    print(f"CVaR 95%: {cvar_95:.4f}")

replay_metrics("MODELO REPLAY", replay_model)
replay_metrics("BENCHMARK REPLAY", replay_bench)

# ----------------------------------------------------------
# 5. Comparación estructural
# ----------------------------------------------------------

excess_replay = replay_model - replay_bench
outperf_ratio = (excess_replay > 0).mean()

print("\n--- Comparación Replay ---")
print(f"% Outperformance Histórico: {outperf_ratio:.4f}")

print("\n================================================\n")

# ==========================================================
# STRESS TEST EXPLÍCITO – SHOCK SISTÉMICO
# ==========================================================

print("\n================================================")
print("STRESS TEST – SHOCK SISTÉMICO EXPLÍCITO")
print("================================================")

# ----------------------------------------------------------
# 1. Definir shocks por activo
# ----------------------------------------------------------

shock_dict = {
    # Renta Variable (-30%)
    "VOO": -0.30,
    "ACWX": -0.30,
    "CHILE.SN": -0.30,
    "BSAC": -0.30,
    "CMPC.SN": -0.30,
    "FALABELLA.SN": -0.30,
    "SQM-B.SN": -0.30,
    
    # Renta Fija (-10%)
    "AGG": -0.10,
    "CFIETFCC.SN": -0.10,
    
    # Commodities (-15%)
    "IAU": -0.15,
    
    # FX (+15%)
    "UUP": 0.15
}

# Orden consistente
tickers = robusto_weights.columns
shock_vector = np.array([shock_dict[t] for t in tickers])

# ----------------------------------------------------------
# 2. Aplicar a pesos promedio
# ----------------------------------------------------------

model_stress = np.dot(w_model, shock_vector)
bench_stress = np.dot(w_bench, shock_vector)

print("\n--- RESULTADO STRESS ---")
print(f"Modelo:     {model_stress:.4f}")
print(f"Benchmark:  {bench_stress:.4f}")

# ----------------------------------------------------------
# 3. Diferencia estructural
# ----------------------------------------------------------

stress_diff = model_stress - bench_stress

print("\n--- DIFERENCIA ---")
print(f"Modelo - Benchmark: {stress_diff:.4f}")

if model_stress > bench_stress:
    print("→ El modelo es MÁS resiliente bajo shock extremo.")
else:
    print("→ El benchmark resiste mejor este shock.")

print("\n================================================\n")