# ==============================================
# MONTE CARLO ENGINE
# Simulación Forward del Portafolio
# ==============================================

import numpy as np
import pandas as pd
import psycopg2
import uuid
from datetime import datetime
from scipy.stats import skew, kurtosis

# ==============================
# CONFIGURACIÓN
# ==============================

N_SIM = 10000
HORIZON = 12

CSV_OUTPUT = "output_modulo5"

POSTGRES_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "portfolio_engine",
    "user": "postgres",
    "password": "MY_PASSWORD"
}

# ==============================
# CARGA DE DATOS
# ==============================

returns_assets = pd.read_csv(
    "output_modulo3/returns_full.csv",
    index_col=0,
    parse_dates=True
)

returns_assets = returns_assets.loc["2022-02-01":]

weights = pd.read_csv(
    "output_modulo4/robusto_weights.csv",
    index_col=0
)

weights_vec = weights.mean().values

benchmark_returns = pd.read_csv(
    "output_benchmark/benchmark_returns.csv",
    index_col=0,
    parse_dates=True
).squeeze()

# ==============================
# PARÁMETROS ESTADÍSTICOS
# ==============================

mu = returns_assets.mean().values
cov = returns_assets.cov().values

# ==============================
# CHECK DIMENSIONAL
# ==============================

print("\n--- DIMENSION CHECK ---")
print("Assets:", returns_assets.shape[1])
print("Weights:", len(weights_vec))
print("mu:", mu.shape)
print("cov:", cov.shape)

# ==============================
# MONTE CARLO NORMAL
# ==============================

def montecarlo_normal():

    sims = np.random.multivariate_normal(
        mean=mu,
        cov=cov,
        size=N_SIM
    )

    portfolio_returns = sims @ weights_vec

    return portfolio_returns


# ==============================
# MONTE CARLO STUDENT-T
# ==============================

def montecarlo_student_t(df=5):

    z = np.random.multivariate_normal(
        mean=np.zeros(len(mu)),
        cov=cov,
        size=N_SIM
    )

    chi = np.random.chisquare(df, N_SIM)

    t_samples = z / np.sqrt(chi[:, None] / df)

    sims = mu + t_samples

    portfolio_returns = sims @ weights_vec

    return portfolio_returns


# ==============================
# REGIME MIXTURE
# ==============================

def montecarlo_regime():

    p_crisis = 0.1
    vol_multiplier = 2.5
    drift_shock = -0.02

    sims = []

    for _ in range(N_SIM):

        if np.random.rand() < p_crisis:

            sim = np.random.multivariate_normal(
                mean=mu + drift_shock,
                cov=cov * vol_multiplier
            )

        else:

            sim = np.random.multivariate_normal(
                mean=mu,
                cov=cov
            )

        sims.append(sim)

    sims = np.array(sims)

    portfolio_returns = sims @ weights_vec

    return portfolio_returns


# ==============================
# MÉTRICAS
# ==============================

def compute_metrics(sim_returns):

    mean = np.mean(sim_returns)
    vol = np.std(sim_returns)

    var95 = np.percentile(sim_returns, 5)
    cvar95 = sim_returns[sim_returns <= var95].mean()

    worst = sim_returns.min()

    skewness = skew(sim_returns)
    kurt = kurtosis(sim_returns)

    return {
        "mean": mean,
        "vol": vol,
        "var95": var95,
        "cvar95": cvar95,
        "worst": worst,
        "skew": skewness,
        "kurt": kurt
    }


# ==============================
# EXPORT CSV
# ==============================

def export_csv(df, run_id):

    path = f"{CSV_OUTPUT}/montecarlo_summary_{run_id}.csv"

    df.to_csv(path)

    print(f"\nCSV exportado en: {path}")


# ==============================
# EXPORT POSTGRES
# ==============================

def export_postgres(df):

    conn = psycopg2.connect(**POSTGRES_CONFIG)
    cur = conn.cursor()

    cur.execute("""

    CREATE TABLE IF NOT EXISTS risk_montecarlo_results (

        run_id TEXT,
        method TEXT,
        mean DOUBLE PRECISION,
        vol DOUBLE PRECISION,
        var95 DOUBLE PRECISION,
        cvar95 DOUBLE PRECISION,
        worst DOUBLE PRECISION,
        skew DOUBLE PRECISION,
        kurt DOUBLE PRECISION,
        created_at TIMESTAMP

    )

    """)

    insert_query = """

    INSERT INTO risk_montecarlo_results
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)

    """

    for _, row in df.iterrows():
        cur.execute(insert_query, tuple(row))

    conn.commit()
    cur.close()
    conn.close()

    print("Resultados insertados en PostgreSQL.")


# ==============================
# TERMINAL REPORT
# ==============================

def terminal_report(df):

    print("\n================================================")
    print("MONTE CARLO – FORWARD RISK REPORT")
    print("================================================")

    for _, row in df.iterrows():

        print(f"\nMétodo: {row['method']}")
        print(f"Expected Return: {row['mean']:.4f}")
        print(f"Volatility: {row['vol']:.4f}")
        print(f"VaR 95%: {row['var95']:.4f}")
        print(f"CVaR 95%: {row['cvar95']:.4f}")
        print(f"Worst Case: {row['worst']:.4f}")
        print(f"Skewness: {row['skew']:.4f}")
        print(f"Kurtosis: {row['kurt']:.4f}")


# ==============================
# EJECUCIÓN
# ==============================

if __name__ == "__main__":

    run_id = str(uuid.uuid4())
    timestamp = datetime.now()

    print("\n========================================")
    print("MONTE CARLO ENGINE – SIMULACIÓN FORWARD")
    print("========================================")

    normal = montecarlo_normal()
    student = montecarlo_student_t()
    regime = montecarlo_regime()

    results = []

    for name, sim in {

        "normal": normal,
        "student_t": student,
        "regime_mixture": regime

    }.items():

        m = compute_metrics(sim)

        results.append({

            "run_id": run_id,
            "method": name,
            "mean": m["mean"],
            "vol": m["vol"],
            "var95": m["var95"],
            "cvar95": m["cvar95"],
            "worst": m["worst"],
            "skew": m["skew"],
            "kurt": m["kurt"],
            "created_at": timestamp

        })

    df = pd.DataFrame(results)

    terminal_report(df)

    export_csv(df, run_id)

    export_postgres(df)

    np.save("output_modulo5/sim_normal.npy", normal)
    np.save("output_modulo5/sim_student.npy", student)
    np.save("output_modulo5/sim_regime.npy", regime)

    print("\n================================================")
    print("FIN MONTE CARLO ENGINE")
    print("================================================")