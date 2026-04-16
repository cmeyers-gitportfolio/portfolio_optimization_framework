import numpy as np
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize
import os

print("\n==============================")
print("ROBUSTNESS ENGINE")
print("==============================\n")

# =========================================================
# 1 CONFIGURACIÓN
# =========================================================

WINDOWS = [12,24,36]
SHRINK_GRID = [0.3,0.5,0.7]
CVAR_GRID = [0.90,0.95,0.975]

DB_CONFIG = {
    "host":"localhost",
    "database":"portfolio_engine",
    "user":"postgres",
    "password":"MY_PASSWORD"
}

# =========================================================
# 2 CARGA DE DATOS
# =========================================================

returns = pd.read_csv(
    "output_modulo3/returns_full.csv",
    index_col=0,
    parse_dates=True
)

tickers = returns.columns.tolist()
n = len(tickers)

# =========================================================
# CLASIFICACIÓN ACTIVOS
# =========================================================

RV = ["VOO","ACWX","BSAC","CHILE.SN","CMPC.SN","FALABELLA.SN","SQM-B.SN"]
RF = ["AGG","CFIETFCC.SN"]
COM = ["IAU"]
FX = ["UUP"]

RV=[t for t in RV if t in tickers]
RF=[t for t in RF if t in tickers]
COM=[t for t in COM if t in tickers]
FX=[t for t in FX if t in tickers]

def sum_class(w,assets):
    return np.sum([w[tickers.index(a)] for a in assets])

bounds=[(0.02,0.15)]*n

constraints=[

{'type':'eq','fun':lambda w:np.sum(w)-1},

{'type':'ineq','fun':lambda w:sum_class(w,RV)-0.30},
{'type':'ineq','fun':lambda w:0.70-sum_class(w,RV)},

{'type':'ineq','fun':lambda w:sum_class(w,RF)-0.10},
{'type':'ineq','fun':lambda w:0.50-sum_class(w,RF)},

{'type':'ineq','fun':lambda w:0.20-sum_class(w,COM)},
{'type':'ineq','fun':lambda w:0.20-sum_class(w,FX)}

]

# =========================================================
# FUNCIÓN MODELO
# =========================================================

def run_model(window,shrink,cvar_alpha):

    port_returns=[]
    weights_hist=[]

    for t in range(window,len(returns)-1):

        data=returns.iloc[t-window:t]

        mu_hist=data.mean()
        mu_global=mu_hist.mean()
        mu=shrink*mu_hist+(1-shrink)*mu_global
        mu=mu*12

        lw=LedoitWolf()
        lw.fit(data)
        Sigma=lw.covariance_*12

        def min_var(w):
            return w@Sigma@w

        def neg_sharpe(w):

            ret=w@mu.values
            vol=np.sqrt(w@Sigma@w)

            return -ret/vol

        def cvar_obj(w):

            port=data.values@w

            var=np.percentile(port,(1-cvar_alpha)*100)
            cvar=port[port<=var].mean()

            return -cvar

        w0=np.ones(n)/n

        w_min=minimize(min_var,w0,bounds=bounds,constraints=constraints).x
        w_sh=minimize(neg_sharpe,w0,bounds=bounds,constraints=constraints).x
        w_cv=minimize(cvar_obj,w0,bounds=bounds,constraints=constraints).x

        w=0.4*w_min+0.3*w_sh+0.3*w_cv
        w=w/np.sum(w)

        weights_hist.append(w)

        next_ret=returns.iloc[t+1].values
        port_returns.append(np.dot(w,next_ret))

    r=pd.Series(port_returns)
    weights=pd.DataFrame(weights_hist,columns=tickers)

    return r,weights

# =========================================================
# 3-5 SENSIBILIDAD
# =========================================================

results=[]

for w in WINDOWS:
    for s in SHRINK_GRID:
        for c in CVAR_GRID:

            print(f"Running W={w}  SHR={s}  CVaR={c}")

            r,weights=run_model(w,s,c)

            ann_ret=(1+r.mean())**12-1
            ann_vol=r.std()*np.sqrt(12)
            sharpe=ann_ret/ann_vol

            cum=(1+r).cumprod()
            dd=cum/cum.cummax()-1
            maxdd=dd.min()

            turnover=weights.diff().abs().sum(axis=1).mean()
            weight_vol=weights.std().mean()

            persistence=weights.corr().mean().mean()

            results.append({
                "rolling_window":w,
                "shrink":s,
                "cvar":c,
                "return":ann_ret,
                "vol":ann_vol,
                "sharpe":sharpe,
                "maxdd":maxdd,
                "turnover":turnover,
                "weight_vol":weight_vol,
                "persistence":persistence
            })

# =========================================================
# 7 TABLA CONSOLIDADA
# =========================================================

df=pd.DataFrame(results)

df["robust_score"]=(
df["sharpe"]
-0.5*abs(df["maxdd"])
-0.2*df["turnover"]
)

df=df.sort_values("robust_score",ascending=False)

print("\nROBUSTNESS SUMMARY\n")
print(df)

# =========================================================
# 8 EXPORT CSV
# =========================================================

os.makedirs("output_robustness",exist_ok=True)
df.to_csv("output_robustness/robustness_results.csv",index=False)

# =========================================================
# 9 EXPORT SQL
# =========================================================

conn=psycopg2.connect(**DB_CONFIG)
cur=conn.cursor()

cur.execute("""

CREATE TABLE IF NOT EXISTS robustness_results(

rolling_window INT,
shrink FLOAT,
cvar FLOAT,
return FLOAT,
vol FLOAT,
sharpe FLOAT,
maxdd FLOAT,
turnover FLOAT,
weight_vol FLOAT,
persistence FLOAT,
score FLOAT,
PRIMARY KEY (rolling_window,shrink,cvar)

)

""")

for _,row in df.iterrows():

    cur.execute("""

    INSERT INTO robustness_results
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    ON CONFLICT DO NOTHING

    """,(
        int(row["rolling_window"]),
        float(row["shrink"]),
        float(row["cvar"]),
        float(row["return"]),
        float(row["vol"]),
        float(row["sharpe"]),
        float(row["maxdd"]),
        float(row["turnover"]),
        float(row["weight_vol"]),
        float(row["persistence"]),
        float(row["robust_score"])
    ))

conn.commit()
cur.close()
conn.close()

# =========================================================
# 10 VISUALIZACIÓN
# =========================================================

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"]=120

os.makedirs("output_robustness",exist_ok=True)

pivot_sharpe=df.pivot_table(values="sharpe",index="rolling_window",columns="shrink")
pivot_dd=df.pivot_table(values="maxdd",index="rolling_window",columns="shrink")

# ---------------------------------------------------------
# Sharpe Heatmap
# ---------------------------------------------------------

plt.figure(figsize=(8,6))
sns.heatmap(pivot_sharpe,annot=True,cmap="viridis")
plt.title("Sharpe Robustness")

plt.savefig("output_robustness/robustness_sharpe_heatmap.png",dpi=300,bbox_inches="tight")
plt.show()


# ---------------------------------------------------------
# Drawdown Heatmap
# ---------------------------------------------------------

plt.figure(figsize=(8,6))
sns.heatmap(pivot_dd,annot=True,cmap="magma")
plt.title("Drawdown Robustness")

plt.savefig("output_robustness/robustness_drawdown_heatmap.png",dpi=300,bbox_inches="tight")
plt.show()


# ---------------------------------------------------------
# Stability Map
# ---------------------------------------------------------

plt.figure(figsize=(8,6))
plt.scatter(df["turnover"],df["sharpe"])
plt.xlabel("Turnover")
plt.ylabel("Sharpe")
plt.title("Stability Map")

plt.savefig("output_robustness/robustness_stability_map.png",dpi=300,bbox_inches="tight")
plt.show()

# =========================================================
# 11 REPORTE TERMINAL
# =========================================================

print("\n==============================")
print("ROBUSTNESS REPORT")
print("==============================\n")

best = df.iloc[0]

print("Mejor configuración encontrada\n")

print(f"Rolling Window : {int(best['rolling_window'])} meses")
print(f"Shrinkage      : {best['shrink']}")
print(f"CVaR Level     : {best['cvar']}")

print("\nMétricas asociadas\n")

print(f"Return anualizado : {best['return']:.2%}")
print(f"Volatilidad       : {best['vol']:.2%}")
print(f"Sharpe Ratio      : {best['sharpe']:.2f}")
print(f"Max Drawdown      : {best['maxdd']:.2%}")
print(f"Turnover promedio : {best['turnover']:.3f}")

print("\n==============================")
print("Rangos observados en el análisis")
print("==============================\n")

print(f"Sharpe range     : {df['sharpe'].min():.2f}  →  {df['sharpe'].max():.2f}")
print(f"MaxDD range      : {df['maxdd'].min():.2%}  →  {df['maxdd'].max():.2%}")
print(f"Turnover range   : {df['turnover'].min():.3f} → {df['turnover'].max():.3f}")

print("\n==============================")
print("INTERPRETACIÓN")
print("==============================\n")

if df['sharpe'].std() < 0.2:
    print("El Sharpe presenta baja dispersión entre configuraciones.")
    print("→ El sistema muestra estabilidad frente a cambios de parámetros.")
else:
    print("El Sharpe presenta dispersión relevante.")
    print("→ El sistema podría depender de parámetros específicos.")

if abs(df['maxdd'].std()) < 0.03:
    print("\nEl Max Drawdown es estable entre configuraciones.")
    print("→ El control de downside parece estructural.")
else:
    print("\nEl Drawdown varía significativamente entre configuraciones.")

if df['turnover'].mean() < 0.35:
    print("\nEl turnover promedio es moderado.")
    print("→ El sistema no depende de rebalanceos excesivos.")
else:
    print("\nEl turnover es elevado.")
    print("→ Podría existir sensibilidad operativa.")

print("\n==============================")
print("LECTURA DE LOS GRÁFICOS")
print("==============================\n")

print("Sharpe Heatmap")
print("• Muestra sensibilidad del rendimiento ajustado por riesgo")
print("• Superficies planas → sistema robusto")
print("• Picos aislados → posible overfitting")

print("\nDrawdown Heatmap")
print("• Evalúa estabilidad del downside")
print("• Drawdown consistente → arquitectura defensiva sólida")

print("\nStability Map")
print("• Relación entre turnover y Sharpe")
print("• Ideal: Sharpe alto con turnover moderado")

print("\nConclusión preliminar:")
print("El análisis sugiere que el sistema mantiene propiedades de riesgo-retorno")
print("relativamente estables frente a cambios razonables en los parámetros.")

print("\nRobustness Engine finalizado\n")
