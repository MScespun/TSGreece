import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.regression.linear_model import yule_walker

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from scipy.stats import norm
import statsmodels.api as sm
from scipy import stats as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.stats import shapiro, skew, kurtosis, norm
from scipy.stats import gaussian_kde





tabella = pd.read_csv("tabella.csv")
tabella['TIME_PERIOD'] = pd.to_datetime(tabella['TIME_PERIOD'])
tabella.set_index('TIME_PERIOD', inplace=True)
tabella.sort_index(inplace=True)
tabella.index = tabella.index.to_period('M').to_timestamp()
tabella = tabella["OBS_VALUE"]

#holt-winters

amodel = ExponentialSmoothing(tabella, trend='add', seasonal='add', seasonal_periods=12)
hwa = amodel.fit()
alpha = hwa.model.params["smoothing_level"]
beta = hwa.model.params["smoothing_trend"]
gamma = hwa.model.params["smoothing_seasonal"]
print(f"additivo - alpha: {alpha:.4f}, beta: {beta:.4f}, gamma: {gamma:.4f}")

mmodel = ExponentialSmoothing(tabella, trend='add', seasonal='mul', seasonal_periods=12)
hwm = mmodel.fit()

alpha = hwm.model.params["smoothing_level"]
beta = hwm.model.params["smoothing_trend"]
gamma = hwm.model.params["smoothing_seasonal"]
print(f"molti - alpha: {alpha:.4f}, beta: {beta:.4f}, gamma: {gamma:.4f}")

nt = 15 #numero di test set
ft = 1 #unità di tempo a cui estendnere la previsione
n = tabella.shape[0]
idt = tabella.index[0]
fdt = tabella.index[-1]
pdt = 12

max = np.array([1e+10,0,0,0])

#cerco i parametri tramite grid-search
a =np.arange(1,10)/10
a = np.concatenate(([0.5239],a))
b =np.arange(1,10)/10
b = np.concatenate(([0.0423],b))
c =np.arange(1,10)/10
c = np.concatenate(([0.0453],c))

for i in a:
    print(i)
    for j in b:
        for k in c:
            err =0

            for l in np.arange(n-nt-ft,n-ft):
                index_1 = tabella.index[l-1]
                index_2= tabella.index[l]
                train = tabella[idt:index_1]
                index_3 = tabella.index[l+ft-1]
                test = tabella[index_2:index_3]
                model_hw = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12)
                train.hw = model_hw.fit(smoothing_level=i, smoothing_trend=j, smoothing_seasonal=k, optimized= False)
                predictions = train.hw.forecast(steps=ft)
                err += np.sum((test.values - predictions.values) ** 2)
            if (max[0]>err):
                    max[0]=err
                    max[1]=i
                    max[2]=j
                    max[3]=k

print ("errore add: ",max[0], "alpha: ", max[1], "beta", max[2], "gamma", max[3] )

#cerco i parametri tramite grid-search
a =np.arange(1,10)/10
a = np.concatenate(([0.6061],a))
b =np.arange(1,10)/10
b = np.concatenate(([0.0001],b))
c =np.arange(1,10)/10
c = np.concatenate(([0.0875],c))

for i in a:
    print(i)
    for j in b:
        for k in c:
            err =0

            for l in np.arange(n-nt-ft,n-ft):
                index_1 = tabella.index[l-1]
                index_2= tabella.index[l]
                train = tabella[idt:index_1]
                index_3 = tabella.index[l+ft-1]
                test = tabella[index_2:index_3]
                model_hw = ExponentialSmoothing(train, trend='add', seasonal='mul', seasonal_periods=12)
                train.hw = model_hw.fit(smoothing_level=i, smoothing_trend=j, smoothing_seasonal=k, optimized= False)
                predictions = train.hw.forecast(steps=ft)
                err += np.sum((test.values - predictions.values) ** 2)
            if (max[0]>err):
                    max[0]=err
                    max[1]=i
                    max[2]=j
                    max[3]=k

print ("errore add: ",max[0], "alpha: ", max[1], "beta", max[2], "gamma", max[3] )

model_hwa = ExponentialSmoothing(tabella, trend='add', seasonal='add', seasonal_periods=12)
hwa = model_hwa.fit(smoothing_level=0.1, smoothing_trend=0.0423, smoothing_seasonal=0.0453, optimized=False)
model_hwm = ExponentialSmoothing(tabella, trend='add', seasonal='mul', seasonal_periods=12)
hwm = model_hwm.fit(smoothing_level=0.1, smoothing_trend=0.0001, smoothing_seasonal=0.6, optimized=False)

fitted_hwa = hwa.fittedvalues
fitted_hwm = hwm.fittedvalues

plt.figure(figsize=(10, 6))
plt.plot(tabella, label="Serie originale", color="black", marker="o")
plt.plot(fitted_hwa, label="Modello additivo", color="blue", linestyle="--", marker="o")
plt.plot(fitted_hwm, label="Modello moltiplicativo", color="red", linestyle="--", marker="o")

# Legenda
plt.legend(loc="upper left", fontsize=10, frameon=True, fancybox=True, shadow=True)
plt.title("Modelli Holt-Winters: Additivo vs Moltiplicativo")
plt.xlabel("Data")
plt.ylabel("Valore")
plt.grid()
plt.show()

# Estrazione dei residui
hwa_r = tabella - hwa.fittedvalues
hwm_r = tabella - hwm.fittedvalues

# Proporzione di varianza non spiegata
var_ratio_hwa = hwa_r.var() / tabella.var()
var_ratio_hwm = hwm_r.var() / tabella.var()

print("Proporzione di varianza non spiegata (additivo):", var_ratio_hwa)
print("Proporzione di varianza non spiegata (moltiplicativo):", var_ratio_hwm)

# Indicatori di forma (skewness e kurtosis)
fm = pd.DataFrame({
    "skewness": [skew(hwa_r), skew(hwm_r)],
    "kurtosis": [kurtosis(hwa_r), kurtosis(hwm_r)]
}, index=["additivo", "moltiplicativo"])

print("\nIndicatori di forma:")
print(fm)

# Scatterplot
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].scatter(hwa_r.index, hwa_r, s=20)
axes[0, 0].set_title("Residui (additivo)")
axes[0, 1].scatter(hwa.fittedvalues, hwa_r, s=20)
axes[0, 1].set_title("Residui vs Valori Stimati (additivo)")

axes[1, 0].scatter(hwm_r.index, hwm_r, s=20)
axes[1, 0].set_title("Residui (moltiplicativo)")
axes[1, 1].scatter(hwm.fittedvalues, hwm_r, s=20)
axes[1, 1].set_title("Residui vs Valori Stimati (moltiplicativo)")

plt.tight_layout()
plt.show()

# Autocorrelazione
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

plot_acf(hwa_r, lags=28, ax=axes[0])
axes[0].set_title("ACF Residui (additivo)")

plot_acf(hwm_r, lags=28, ax=axes[1])
axes[1].set_title("ACF Residui (moltiplicativo)")

plt.tight_layout()
plt.show()

# Densità empirica
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Additivo
kde = gaussian_kde(hwa_r)

axes[0].hist(hwa_r, bins=20, density=True, color="gray")
axes[0].plot(np.sort(hwa_r), norm.pdf(np.sort(hwa_r), np.mean(hwa_r), np.std(hwa_r)), color="red")
axes[0].plot(np.sort(hwa_r), kde(np.sort(hwa_r)), color="blue")
axes[0].set_title("Densità Residui (additivo)")

# Moltiplicativo
kde = gaussian_kde(hwm_r)

axes[1].hist(hwm_r, bins=20, density=True, color="gray")
axes[1].plot(np.sort(hwm_r), norm.pdf(np.sort(hwm_r), np.mean(hwm_r), np.std(hwm_r)), color="red")
axes[1].plot(np.sort(hwm_r), kde(np.sort(hwa_r)), color="blue")
axes[1].set_title("Densità Residui (moltiplicativo)")

plt.tight_layout()
plt.show()

# Grafico quantile-quantile
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sm.qqplot(hwa_r, line="s", ax=axes[0], marker="o", alpha=0.6)
axes[0].set_title("QQ-plot Residui (additivo)")

sm.qqplot(hwm_r, line="s", ax=axes[1], marker="o", alpha=0.6)
axes[1].set_title("QQ-plot Residui (moltiplicativo)")

plt.tight_layout()
plt.show()

# Test di Gaussianità
shapiro_hwa = shapiro(hwa_r)
shapiro_hwm = shapiro(hwm_r)

print("\nTest di Gaussianità (Shapiro-Wilk):")
print(f"Residui Additivo: p-value = {shapiro_hwa.pvalue:.4f}")
print(f"Residui Moltiplicativo: p-value = {shapiro_hwm.pvalue:.4f}")


#autoregressivo diretto yw
fig, ax = plt.subplots()
plot_acf(tabella, lags=30, ax=ax)
ax.set_xticks(range(0, 31, 1))
ax.set_xlabel('Lag')
ax.grid(True)
plt.title('Autocorrelation Function (ACF)')
plt.show()

fig, ax = plt.subplots()
plot_pacf(tabella, lags=30, ax=ax, method="ols")
ax.set_xticks(range(0, 31, 1))
ax.set_xlabel('Lag')
ax.grid(True)
plt.title('Partial Autocorrelation Function (PACF), OLS')
plt.show()

fig, ax = plt.subplots()
plot_pacf(tabella, lags=30, ax=ax, method="yw")
ax.set_xticks(range(0, 31, 1))
ax.set_xlabel('Lag')
ax.grid(True)
plt.title('Partial Autocorrelation Function (PACF), YW')
plt.show()

L = len(tabella)
l = 24

lm =LinearRegression()

mnt = np.zeros((L - l, l + 1))
for i in range(1, l + 2):
    mnt[:, i - 1] = tabella[i - 1:L - l - 1 + i]
mnt_df = pd.DataFrame(mnt, columns=[f"X{i}" for i in range(1, l + 2)])
y = mnt_df["X25"]
features= mnt_df[mnt_df.columns.difference(["X25"])]
#features = sm.add_constant(features), togliere commento se si vuole ridurre OLS
modello = sm.OLS(y, features).fit()

while ((modello.pvalues>0.05).any()):
   colonna_drop = modello.pvalues.idxmax()
   features = features.drop(columns=colonna_drop)
   modello = sm.OLS(y, features).fit()

print(modello.summary())




# Parametri
nt = 15  # Numero di test set
ft = 1   # Passi futuri da prevedere
n = len(tabella)
pdt = 12  # Periodo della serie (stagionalità)
li = 24 # Numero massimo di lag per il modello autoregressivo

err_lmr = np.zeros(nt)
err_ols = np.zeros(nt)
err_hwa = np.zeros(nt)
err_hwm = np.zeros(nt)

# Ciclo di validazione
for l in range(n - nt - ft, n - ft):
    # Costruzione di train e test
    train = tabella.iloc[:l]
    test = tabella.iloc[l:l + ft]

    # Verifica della lunghezza minima del train set
    if len(train) <= li:
        continue

    # Modello autoregressivo ridotto
    L = len(train)
    mtrain = np.zeros((L - li, li + 1))

    for i in range(0, li + 1):
        start_idx = i
        end_idx = L - li + i
        mtrain[:, i ] = train.iloc[start_idx:end_idx].values

    # Creazione del DataFrame
    mtrain_df = pd.DataFrame(mtrain, columns=[f"X{i}" for i in range(1, li + 2)])
    y_train = mtrain_df["X25"]
    X_train = mtrain_df[["X1", "X7", "X12", "X13", "X19", "X24"]]

    # Regressione lineare senza costanti
    modello = sm.OLS(y_train, X_train).fit()

    # Previsioni con il modello lineare
    train_lmr_p = np.zeros(L + ft)
    train_lmr_p[:L] = train.values

    for i in range(0, ft):
        x_lag = np.array([train_lmr_p[L + i - 24], train_lmr_p[L + i - 18],
                          train_lmr_p[L + i - 13], train_lmr_p[L + i - 12],
                          train_lmr_p[L + i - 6], train_lmr_p[L + i - 1]])
        # Previsione
        train_lmr_p[L + i] = modello.predict(x_lag.reshape(1,-1))[0]

    # Calcolo dell'errore

    err_lmr[l - (n - nt - ft)] = test.values[-1] - train_lmr_p[L + ft-1]

    # Modello autoregressivo OLS
    if len(train) > 25:  # Verifica che il training set sia sufficiente
        model_hwa = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12)
        hwa = model_hwa.fit(smoothing_level=0.1, smoothing_trend=0.0423, smoothing_seasonal=0.0453, optimized=False)
        model_hwm = ExponentialSmoothing(train, trend='add', seasonal='mul', seasonal_periods=12)
        hwm = model_hwm.fit(smoothing_level=0.1, smoothing_trend=0.0001, smoothing_seasonal=0.6, optimized=False)
        model_ols = AutoReg(train, lags=[1,12,13,14,24], old_names=False).fit()

        pred_hwa = hwa.predict(start=len(train), end=len(train) + ft - 1)
        pred_hwm = hwm.predict(start=len(train), end=len(train) + ft - 1)

        pred_ols = model_ols.predict(start=len(train), end=len(train) + ft - 1)
        err_ols[l - (n - nt - ft)] = test.values[-1] - pred_ols.values[-1]
        err_hwa[l - (n - nt - ft)] = test.values[-1] - pred_hwa.values[-1]
        err_hwm[l - (n - nt - ft)] = test.values[-1] - pred_hwm.values[-1]

# Errori totali
print(f"Modello autoregressivo ridotto - errore: {np.sum(err_lmr ** 2):.4f}")
print(f"Modello autoregressivo OLS - errore: {np.sum(err_ols ** 2):.4f}")
print(f"HW additivo - errore: {np.sum(err_hwa ** 2):.4f}")
print(f"HW moltiplicativo- errore: {np.sum(err_hwm ** 2):.4f}")

y = mnt_df["X25"]

features= mnt_df[mnt_df.columns.difference(["X25"])]
features = features[["X1", "X7", "X12", "X13", "X19", "X24"]]
modello = sm.OLS(y, features).fit()
RES = modello.resid

features= mnt_df[mnt_df.columns.difference(["X25"])]
features = features[["X1", "X12", "X13", "X14", "X24"]]
features = sm.add_constant(features)
modello = sm.OLS(y, features).fit()
RES2 = modello.resid
plt.plot(RES,'.',c='y', label="Residui  YW")
plt.plot(RES2,'.',c='r', label="Residui OLS")
plt.title("Residui modello autoregressivo")
plt.legend()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

plot_acf(RES, lags=31, ax=axes[0])
axes[0].set_title("ACF Residui YW")

plot_acf(RES2, lags=31, ax=axes[1])
axes[1].set_title("ACF Residui OLS")

plt.tight_layout()
plt.show()

#previsioni
model_hwa = ExponentialSmoothing(tabella, trend='add', seasonal='add', seasonal_periods=12)
hwa = model_hwa.fit(smoothing_level=0.1, smoothing_trend=0.0423, smoothing_seasonal=0.0453, optimized=False)
predictions = hwa.forecast(steps=12)
err = hwa.resid

plt.figure()
plt.plot(tabella, c='k', label="Serie")
plt.plot(predictions,c='r',label="Previsione")
plt.plot(predictions+np.quantile(err,0.975),c='g',label="Bande di incertezza 95%")
plt.plot(predictions+np.quantile(err,0.025),c='g')
plt.title("Previsioni")
plt.legend()
plt.show()

plt.figure()
plt.plot(tabella["2021":], c='k', label="Serie")
plt.plot(predictions,c='r', label="Previsione")
plt.plot(predictions+np.quantile(err,0.975),c='g', label="Bande di incertezza 95%")
plt.plot(predictions+np.quantile(err,0.025),c='g')
plt.title("Previsioni, grafico più recente")
plt.legend()
plt.show()

print(np.quantile(err,0.975)-np.quantile(err,0.025))