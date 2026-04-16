import numpy as np
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("\n==============================")
print("ESTIMATION ERROR ENGINE")
print("==============================\n")

# =========================================================
# CONFIGURACIÓN
# =========================================================

N_SIM = 300

RETURN_NOISE = 0.20
COV_NOISE = 0.15
CORR_NOISE = 0.20

VOL_FLOOR = 1e-4
RIDGE = 1e-4
WEIGHT_CLIP = 0.50

DB_CONFIG = {
    "host":"localhost",
    "database":"portfolio_engine",
    "user":"postgres",
    "password":"MY_PASSWORD"
}

# =========================================================
# MATRIZ POSITIVA DEFINIDA
# =========================================================

def nearest_positive_definite(A):

    B = (A + A.T) / 2

    eigvals, eigvecs = np.linalg.eigh(B)

    eigvals[eigvals < 1e-6] = 1e-6

    B_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return B_pd


# =========================================================
# CARGA DE DATOS
# =========================================================

returns = pd.read_csv(
    "output_modulo3/returns_full.csv",
    index_col=0,
    parse_dates=True
)

tickers = returns.columns.tolist()
n = len(tickers)

# =========================================================
# ESTIMACIONES BASE
# =========================================================

mu = returns.mean()*12
cov = returns.cov()*12

mu = mu.values
cov = cov.values

cov = nearest_positive_definite(cov)

# regularización ridge
cov = cov + np.eye(n)*RIDGE


# =========================================================
# OPTIMIZADOR MV ESTABILIZADO
# =========================================================

def mv_weights(mu,cov):

    inv = np.linalg.pinv(cov)

    ones = np.ones(len(mu))

    w = inv @ mu
    w = w/(ones@w)

    # -----------------------------------------------------
    # ESTABILIZACIÓN DE PESOS
    # -----------------------------------------------------

    w = np.clip(w,-WEIGHT_CLIP,WEIGHT_CLIP)

    w = w / np.sum(np.abs(w))

    return w


# =========================================================
# MÉTRICAS
# =========================================================

def portfolio_metrics(w,returns):

    r = pd.Series(returns.values @ w)

    ann_ret = (1+r.mean())**12 - 1

    vol = max(r.std()*np.sqrt(12), VOL_FLOOR)

    sharpe = ann_ret/vol

    cum=(1+r).cumprod()
    dd=cum/cum.cummax()-1
    maxdd=dd.min()

    return ann_ret,vol,sharpe,maxdd


# =========================================================
# SIMULACIÓN
# =========================================================

results=[]

for i in range(N_SIM):

    # -----------------------------------------------------
    # 1 RETURN ERROR
    # -----------------------------------------------------

    mu_p = mu*(1+np.random.normal(0,RETURN_NOISE,n))

    w = mv_weights(mu_p,cov)

    r,v,s,d = portfolio_metrics(w,returns)

    results.append({
        "type":"return_error",
        "return":r,
        "vol":v,
        "sharpe":s,
        "maxdd":d
    })


    # -----------------------------------------------------
    # 2 COVARIANCE ERROR
    # -----------------------------------------------------

    noise = np.random.normal(0,COV_NOISE,cov.shape)

    cov_p = cov*(1+noise)

    cov_p=(cov_p+cov_p.T)/2

    cov_p = nearest_positive_definite(cov_p)

    cov_p = cov_p + np.eye(n)*RIDGE

    w = mv_weights(mu,cov_p)

    r,v,s,d = portfolio_metrics(w,returns)

    results.append({
        "type":"cov_error",
        "return":r,
        "vol":v,
        "sharpe":s,
        "maxdd":d
    })


    # -----------------------------------------------------
    # 3 CORRELATION ERROR
    # -----------------------------------------------------

    std = np.sqrt(np.diag(cov))

    corr = cov/np.outer(std,std)

    corr_noise = corr*(1+np.random.normal(0,CORR_NOISE,corr.shape))

    corr_noise=(corr_noise+corr_noise.T)/2

    cov_corr = np.outer(std,std)*corr_noise

    cov_corr = nearest_positive_definite(cov_corr)

    cov_corr = cov_corr + np.eye(n)*RIDGE

    w = mv_weights(mu,cov_corr)

    r,v,s,d = portfolio_metrics(w,returns)

    results.append({
        "type":"corr_error",
        "return":r,
        "vol":v,
        "sharpe":s,
        "maxdd":d
    })


df = pd.DataFrame(results)

# =========================================================
# EXPORT CSV
# =========================================================

os.makedirs("output_robustness",exist_ok=True)

df.to_csv(
    "output_robustness/estimation_error_results.csv",
    index=False
)

# =========================================================
# EXPORT SQL
# =========================================================

conn=psycopg2.connect(**DB_CONFIG)
cur=conn.cursor()

cur.execute("""

CREATE TABLE IF NOT EXISTS estimation_error_results(

type TEXT,
return FLOAT,
vol FLOAT,
sharpe FLOAT,
maxdd FLOAT

)

""")

for _,row in df.iterrows():

    cur.execute("""

    INSERT INTO estimation_error_results
    VALUES (%s,%s,%s,%s,%s)

    """,(
        row["type"],
        float(row["return"]),
        float(row["vol"]),
        float(row["sharpe"]),
        float(row["maxdd"])
    ))

conn.commit()
cur.close()
conn.close()


# =========================================================
# SUMMARY
# =========================================================

summary=df.groupby("type").agg(
    mean_sharpe=("sharpe","mean"),
    std_sharpe=("sharpe","std"),
    mean_dd=("maxdd","mean"),
    std_dd=("maxdd","std")
)

print("\nESTIMATION ERROR SUMMARY\n")
print(summary)


# =========================================================
# GRÁFICOS
# =========================================================

sns.set_style("whitegrid")

plt.figure(figsize=(8,6))
sns.boxplot(data=df,x="type",y="sharpe")
plt.title("Sharpe Sensitivity")
plt.savefig("output_robustness/estimation_error_sharpe.png",dpi=300,bbox_inches="tight")
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(data=df,x="type",y="maxdd")
plt.title("Drawdown Sensitivity")
plt.savefig("output_robustness/estimation_error_drawdown.png",dpi=300,bbox_inches="tight")
plt.show()


# =========================================================
# INTERPRETACIÓN PROFESIONAL
# =========================================================

print("\n==============================")
print("INTERPRETACIÓN")
print("==============================\n")

print("Este módulo evalúa cuánto cambia el desempeño del portafolio")
print("cuando los parámetros del modelo están mal estimados.\n")

print("En la práctica, ningún modelo conoce los valores reales de:")
print("• retornos esperados")
print("• covarianza")
print("• correlaciones\n")

print("Por lo tanto simulamos errores en estas estimaciones.\n")

print("Tipos de error evaluados:\n")

print("Return Error")
print("• cambia la dirección de la apuesta del portafolio")
print("• afecta principalmente el alfa esperado\n")

print("Covariance Error")
print("• cambia la estimación del riesgo")
print("• puede alterar fuertemente los pesos óptimos\n")

print("Correlation Error")
print("• cambia la diversificación")
print("• puede concentrar o dispersar riesgo estructural\n")

print("Interpretación de resultados:\n")

worst = summary["std_sharpe"].idxmax()

print("La mayor sensibilidad del sistema aparece en:", worst)

if worst == "cov_error":

    print("\nEsto indica que el portafolio depende fuertemente")
    print("de la estimación de la matriz de covarianza.")
    print("Este resultado es consistente con la literatura cuant.")
    
elif worst == "corr_error":

    print("\nLa diversificación estructural es el principal riesgo.")
    
else:

    print("\nLa estimación de retornos es el factor dominante.")


print("\nLectura práctica:")

print("Si Sharpe cambia mucho entre simulaciones")
print("→ el modelo es sensible al error de estimación")

print("Si Sharpe es relativamente estable")
print("→ el portafolio es robusto")

print("\nEste análisis permite detectar si el modelo")
print("podría fallar fuera de muestra.\n")

print("Estimation Error Engine finalizado\n")