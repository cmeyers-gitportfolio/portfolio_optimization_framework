# =========================================================
# 10_core_modulo6_regime_stress_engine_p2.py
# =========================================================

import numpy as np
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("\n==============================")
print("REGIME STRESS ENGINE")
print("==============================\n")

# =========================================================
# CONFIGURACIÓN
# =========================================================

DB_CONFIG = {
    "host":"localhost",
    "database":"portfolio_engine",
    "user":"postgres",
    "password":"MY_PASSWORD"
}

# =========================================================
# CARGA DE DATOS
# =========================================================

returns = pd.read_csv(
    "output_modulo3/returns_full.csv",
    index_col=0,
    parse_dates=True
)

weights_df = pd.read_csv(
    "output_modulo4/robusto_weights.csv",
    index_col=0,
    parse_dates=True
)

portfolio_returns = pd.read_csv(
    "output_modulo4/robusto_returns_oos.csv",
    index_col=0,
    parse_dates=True
).squeeze()

tickers = returns.columns.tolist()

# =========================================================
# CLASIFICACIÓN ACTIVOS
# =========================================================

RV = ["VOO","ACWX","BSAC","CHILE.SN","CMPC.SN","FALABELLA.SN","SQM-B.SN"]
RF = ["AGG","CFIETFCC.SN"]
COM = ["IAU"]
FX = ["UUP"]

# Filtrar existentes

RV=[t for t in RV if t in tickers]
RF=[t for t in RF if t in tickers]
COM=[t for t in COM if t in tickers]
FX=[t for t in FX if t in tickers]

# =========================================================
# PESOS PROMEDIO
# =========================================================

w_avg = weights_df.mean().values

# =========================================================
# VOLATILIDAD HISTÓRICA
# =========================================================

vol = portfolio_returns.std()*np.sqrt(12)

# =========================================================
# DEFINICIÓN REGÍMENES
# =========================================================

regimes = {

"equity_crash":{

"RV":-0.25,
"RF":0.03,
"COM":-0.10,
"FX":0.08

},

"inflation_shock":{

"RV":-0.10,
"RF":-0.15,
"COM":0.12,
"FX":0.05

},

"liquidity_crisis":{

"RV":-0.30,
"RF":-0.05,
"COM":-0.15,
"FX":0.12

},

"usd_spike":{

"RV":-0.08,
"RF":-0.03,
"COM":-0.12,
"FX":0.15

}

}

# =========================================================
# FUNCIÓN SHOCK VECTOR
# =========================================================

def build_shock(regime):

    shock=np.zeros(len(tickers))

    for i,t in enumerate(tickers):

        if t in RV:
            shock[i]=regime["RV"]

        elif t in RF:
            shock[i]=regime["RF"]

        elif t in COM:
            shock[i]=regime["COM"]

        elif t in FX:
            shock[i]=regime["FX"]

        else:
            shock[i]=0

    return shock

# =========================================================
# SIMULACIÓN REGÍMENES
# =========================================================

results=[]

for name,regime in regimes.items():

    shock_vector=build_shock(regime)

    port_return=w_avg @ shock_vector

    sharpe=port_return/vol if vol!=0 else np.nan

    # estimación drawdown proporcional

    maxdd=port_return*3

    results.append({

    "regime":name,
    "return":port_return,
    "vol":vol,
    "sharpe":sharpe,
    "maxdd":maxdd

    })

df=pd.DataFrame(results)

# =========================================================
# EXPORT CSV
# =========================================================

os.makedirs("output_robustness",exist_ok=True)

df.to_csv(

"output_robustness/regime_stress_results.csv",
index=False

)

# =========================================================
# EXPORT SQL
# =========================================================

conn=psycopg2.connect(**DB_CONFIG)
cur=conn.cursor()

cur.execute("""

CREATE TABLE IF NOT EXISTS regime_stress_results(

regime TEXT,
return FLOAT,
vol FLOAT,
sharpe FLOAT,
maxdd FLOAT

)

""")

for _,row in df.iterrows():

    cur.execute("""

    INSERT INTO regime_stress_results
    VALUES (%s,%s,%s,%s,%s)

    """,(

    row["regime"],
    float(row["return"]),
    float(row["vol"]),
    float(row["sharpe"]),
    float(row["maxdd"])

    ))

conn.commit()
cur.close()
conn.close()

# =========================================================
# RESULTADOS
# =========================================================

print("\nREGIME STRESS RESULTS\n")

print(df)

# =========================================================
# GRÁFICOS
# =========================================================

sns.set_style("whitegrid")

plt.figure(figsize=(8,6))
sns.barplot(data=df,x="regime",y="return")
plt.title("Portfolio Return Under Stress Regimes")
plt.savefig("output_robustness/regime_returns.png",dpi=300,bbox_inches="tight")
plt.show()

plt.figure(figsize=(8,6))
sns.barplot(data=df,x="regime",y="maxdd")
plt.title("Estimated Drawdown Under Stress Regimes")
plt.savefig("output_robustness/regime_drawdown.png",dpi=300,bbox_inches="tight")
plt.show()

# =========================================================
# INTERPRETACIÓN
# =========================================================

print("\n==============================")
print("INTERPRETACIÓN")
print("==============================\n")

worst=df.loc[df["return"].idxmin()]

print("El régimen más adverso para el portafolio es:\n")

print(worst["regime"])

print("\nRetorno estimado bajo este escenario:")

print(f"{worst['return']:.2%}")

print("\nEsto indica dónde se concentra el riesgo estructural del modelo.")

print("\nRegime Stress Engine finalizado\n")