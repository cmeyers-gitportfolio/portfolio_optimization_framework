# ==============================================
# DISTRIBUTION ENGINE (MEJORADO + EXPLICACIÓN)
# Visualización de riesgo tipo paper académico
# ==============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# ==============================
# CONFIGURACIÓN
# ==============================

OUTPUT_PATH = "output_modulo5/distributions"
os.makedirs(OUTPUT_PATH, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10,6)

# ==============================
# CARGA DE SIMULACIONES
# ==============================

normal = np.load("output_modulo5/sim_normal.npy")
student = np.load("output_modulo5/sim_student.npy")
regime = np.load("output_modulo5/sim_regime.npy")

benchmark_returns = pd.read_csv(
    "output_benchmark/benchmark_returns.csv",
    index_col=0,
    parse_dates=True
).squeeze()

benchmark_sim = np.random.choice(
    benchmark_returns,
    size=len(normal),
    replace=True
)

# ==============================
# HISTOGRAMAS
# ==============================

def plot_histogram(sim, name):

    mean = np.mean(sim)
    vol = np.std(sim)
    var = np.percentile(sim,5)
    cvar = sim[sim <= var].mean()
    skew = stats.skew(sim)
    kurt = stats.kurtosis(sim)

    print("\n------------------------------------------------")
    print(f"HISTOGRAM – {name}")
    print("------------------------------------------------")

    print(f"Mean return: {mean:.4f}")
    print(f"Volatility: {vol:.4f}")
    print(f"VaR 95%: {var:.4f}")
    print(f"CVaR 95%: {cvar:.4f}")
    print(f"Skewness: {skew:.4f}")
    print(f"Kurtosis: {kurt:.4f}")

    print("\nInterpretation:")

    if skew < 0:
        print("Distribution has negative skew → downside shocks dominate.")
    else:
        print("Distribution has positive skew → upside surprises more likely.")

    if kurt > 0:
        print("Fat tails present → extreme events more probable than normal.")
    else:
        print("Tail risk close to normal distribution.")

    plt.figure()

    sns.histplot(sim, bins=60, kde=True)

    plt.axvline(mean, color="green", linestyle="--", label="Mean")
    plt.axvline(var, color="orange", linestyle="--", label="VaR 95")
    plt.axvline(cvar, color="red", linestyle="--", label="CVaR 95")

    plt.title(f"Monte Carlo Distribution – {name}")
    plt.legend()

    plt.savefig(f"{OUTPUT_PATH}/hist_{name}.png")
    plt.close()


# ==============================
# MODELO vs BENCHMARK
# ==============================

def plot_model_vs_benchmark(sim):

    prob_outperform = np.mean(sim > benchmark_sim)

    print("\n------------------------------------------------")
    print("MODEL vs BENCHMARK DISTRIBUTION")
    print("------------------------------------------------")

    print(f"Probability model outperforms benchmark: {prob_outperform:.3f}")

    if prob_outperform > 0.5:
        print("Model distribution dominates benchmark on average.")
    else:
        print("Benchmark distribution dominates model.")

    plt.figure()

    sns.kdeplot(sim, label="Model")
    sns.kdeplot(benchmark_sim, label="Benchmark")

    plt.title("Distribution Comparison – Model vs Benchmark")
    plt.legend()

    plt.savefig(f"{OUTPUT_PATH}/model_vs_benchmark.png")
    plt.close()


# ==============================
# CDF
# ==============================

def plot_cdf(sim):

    sorted_returns = np.sort(sim)
    cdf = np.arange(len(sorted_returns)) / len(sorted_returns)

    bench_sorted = np.sort(benchmark_sim)
    bench_cdf = np.arange(len(bench_sorted)) / len(bench_sorted)

    print("\n------------------------------------------------")
    print("CDF ANALYSIS")
    print("------------------------------------------------")

    print("CDF shows cumulative probability of returns.")
    print("Left shift → higher probability of losses.")
    print("Right shift → better return distribution.")

    plt.figure()

    plt.plot(sorted_returns, cdf, label="Model")
    plt.plot(bench_sorted, bench_cdf, label="Benchmark")

    plt.title("Cumulative Distribution Function")
    plt.legend()

    plt.savefig(f"{OUTPUT_PATH}/cdf_comparison.png")
    plt.close()


# ==============================
# QQ PLOT
# ==============================

def plot_qq(sim):

    skew = stats.skew(sim)
    kurt = stats.kurtosis(sim)

    print("\n------------------------------------------------")
    print("QQ PLOT – NORMALITY TEST")
    print("------------------------------------------------")

    print(f"Skewness: {skew:.4f}")
    print(f"Kurtosis: {kurt:.4f}")

    print("If points deviate strongly from diagonal → distribution is non-normal.")

    plt.figure()

    stats.probplot(sim, dist="norm", plot=plt)

    plt.title("QQ Plot – Normality Check")

    plt.savefig(f"{OUTPUT_PATH}/qq_plot.png")
    plt.close()


# ==============================
# TAIL RISK
# ==============================

def plot_tail(sim):

    tail = sim[sim < np.percentile(sim,10)]

    worst = np.min(sim)
    var = np.percentile(sim,5)

    print("\n------------------------------------------------")
    print("TAIL RISK ANALYSIS")
    print("------------------------------------------------")

    print(f"Worst simulated loss: {worst:.4f}")
    print(f"VaR 95% threshold: {var:.4f}")
    print("Tail distribution shows extreme downside scenarios.")

    plt.figure()

    sns.histplot(tail, bins=40)

    plt.title("Tail Risk Distribution")

    plt.savefig(f"{OUTPUT_PATH}/tail_risk.png")
    plt.close()


# ==============================
# PROBABILIDADES
# ==============================

def compute_probabilities(sim):

    prob_loss = np.mean(sim < 0)
    prob_outperform = np.mean(sim > benchmark_sim)

    print("\n========================================")
    print("PROBABILITY METRICS")
    print("========================================")

    print(f"Probability of Loss: {prob_loss:.3f}")
    print(f"Probability of Outperform Benchmark: {prob_outperform:.3f}")

    if prob_loss < 0.5:
        print("Loss probability below 50% → distribution tilted toward positive outcomes.")

    if prob_outperform > 0.5:
        print("Model expected to outperform benchmark more often than not.")


# ==============================
# EJECUCIÓN
# ==============================

if __name__ == "__main__":

    print("\n========================================")
    print("DISTRIBUTION ENGINE – VISUAL RISK REPORT")
    print("========================================")

    plot_histogram(normal, "normal")
    plot_histogram(student, "student_t")
    plot_histogram(regime, "regime_mixture")

    plot_model_vs_benchmark(regime)

    plot_cdf(regime)

    plot_qq(regime)

    plot_tail(regime)

    compute_probabilities(regime)

    print("\nGráficos exportados en:")
    print(OUTPUT_PATH)

    print("\n========================================")
    print("FIN DISTRIBUTION ENGINE")
    print("========================================")