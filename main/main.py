# ============================================
# MAIN PIPELINE
# Robust Strategic Asset Allocation Framework
# ============================================

import os
import subprocess
import sys

# ==========================
# CONFIG
# ==========================

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    # CAPA 1 — DATA
    "01_core_capa1_inputs_precios_p2.py",
    "02_core_capa1_limpiar_base_p2.py",

    # CAPA 2 — PORTFOLIOS
    "03_core_capa2_BENCHMARK_MV_p2.py",
    "04_core_capa2_portfolio_construction_p2.py",

    # CAPA 3 — RISK
    "05_core_capa3_VaR_CVaR_p2.py",
    "06_core_capa3_montecarlo_engine_p2.py",
    "07_core_capa3_distribution_engine_p2.py",
    "10_core_capa3_regime_stress_engine_p2.py",

    # CAPA 4 — ROBUSTEZ
    "08_core_capa4_robustness_engine_p2.py",
    "09_core_capa4_estimation_error_engine_p2.py",
]

# ==========================
# EXECUTION FUNCTION
# ==========================

def run_script(script_name):

    script_path = os.path.join(BASE_PATH, script_name)

    print("\n========================================")
    print(f"Running: {script_name}")
    print("========================================\n")

    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,
        text=True
    )

    if result.returncode != 0:
        print(f"\n❌ Error executing {script_name}")
        sys.exit(1)

    print(f"\n✅ Finished: {script_name}")


# ==========================
# MAIN
# ==========================

def main():

    print("\n========================================")
    print("ROBUST ASSET ALLOCATION PIPELINE")
    print("========================================\n")

    for script in SCRIPTS:
        run_script(script)

    print("\n========================================")
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("========================================\n")


if __name__ == "__main__":
    main()