import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from scipy.stats import norm
import statsmodels.api as sm
from scipy import stats as st
from statsmodels.tsa.seasonal import STL

tabella = pd.read_csv("tabella.csv")
tabella['TIME_PERIOD'] = pd.to_datetime(tabella['TIME_PERIOD'])
tabella.set_index('TIME_PERIOD', inplace=True)
tabella.sort_index(inplace=True)
tabella = tabella["OBS_VALUE"]
print(tabella.tail())

plt.plot(tabella,c='b')
plt.title("Grafico della serie")
plt.xlabel("Tempo")
plt.ylabel("Consumo (GWh)")
plt.grid(True)
plt.show()


stl = STL(tabella, seasonal=7)  # Crea l'oggetto STL
result = stl.fit()
seasonal_component = result.seasonal

#cogliamo la stagionalità

diff_tp = tabella.diff().dropna()

plt.figure(figsize=(10, 5))
plt.plot(diff_tp, c='b')
plt.title('Serie delle differenze')
plt.xlabel('Tempo')
plt.ylabel('Differenze')
plt.grid(True)
plt.show()


fig, ax = plt.subplots()
plot_acf(tabella, lags=30, ax=ax)
ax.set_xticks(range(0, 31, 1))
ax.set_xlabel('Lag')
ax.grid(True)
plt.title('Autocorrelation Function (ACF)')
plt.show()

fig, ax = plt.subplots()
plot_acf(diff_tp, lags=30, ax=ax)
ax.set_xticks(range(0, 31, 1))
ax.set_xlabel('Lag')
ax.grid(True)
plt.title('ACF delle differenze')
plt.show()

tabella_sim = pd.concat([tabella, pd.Series([np.nan, np.nan])], ignore_index=True)
tabella_sim = tabella_sim.values.reshape(-1,12)
tabella_sim = tabella_sim.transpose()
plt.style.use('dark_background')
colors = plt.cm.hot(np.linspace(0, 1, 17))
for i in range(tabella_sim.shape[1]):
    plt.plot(range(1, 13), tabella_sim[:, i]-(tabella_sim[:,i]).mean(), color=colors[i])
plt.title("Andamento mensile centrato")
plt.show()
for i in range(tabella_sim.shape[1]):
    plt.plot(range(1, 13), tabella_sim[:, i], color=colors[i])
medie = np.nanmean(tabella_sim, axis=1)
print(medie)
plt.plot(range(1, 13), medie, marker='o', linestyle='-', linewidth=5, color='white', label='Media annuale')
sd_tab = np.nanstd(tabella_sim, axis=1)
plt.plot(range(1, 13), medie + sd_tab, marker='o', linestyle='-', linewidth=5, color='gray', label='Media + Dev. St.')
plt.plot(range(1, 13), medie - sd_tab, marker='o', linestyle='-', linewidth=5, color='gray', label='Media - Dev. St.')
plt.xlabel("Mesi")
plt.ylabel("Valori")
plt.title("Andamenti mensili e media annuale")
plt.legend()
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.show()
plt.style.use('default')


#decidiamo se modello moltiplicativo o additivo

result_a = seasonal_decompose(tabella, model='additive', period=12)
fig = result_a.plot()
for ax in fig.axes:
    ax.set_title("")
fig.suptitle("Decomposizione additiva")
plt.show()

result_m = seasonal_decompose(tabella, model='multiplicative', period=12)
fig = result_m.plot()
for ax in fig.axes:
    ax.set_title("")
fig.suptitle("Decompodizione moltiplicativa")
plt.show()

stl = STL(tabella, seasonal=7)
result = stl.fit()
fig = result.plot()
for ax in fig.axes:
    ax.set_title("")
fig.suptitle("Decomposizione STL")
plt.show()

stl = STL(np.log(tabella), seasonal=7)
result = stl.fit()
fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

# Componente osservata
axes[0].plot(np.exp(result.observed), label="Observed")
axes[0].set_ylabel("Observed")
axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))  # Attiva notazione scientifica
axes[0].legend()

# Componente trend
axes[1].plot(np.exp(result.trend), label="Trend", color="orange")
axes[1].set_ylabel("Trend")
axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
axes[1].legend()

# Componente stagionale
axes[2].plot(np.exp(result.seasonal), label="Seasonal", color="green")
axes[2].set_ylabel("Seasonal")
axes[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
axes[2].legend()

# Componente residuo
axes[3].plot(np.exp(result.resid), label="Residuals", color="red")
axes[3].set_ylabel("Residuals")
axes[3].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
axes[3].legend()

# Label asse x
axes[3].set_xlabel("Time")

# Mostra il grafico
plt.tight_layout()
fig.suptitle("Decomposizione STL moltiplicativa")
plt.show()

plt.plot(result_a.seasonal, c="k", label="Additivo")
plt.plot(np.nanmean(result_m.trend)*(result_m.seasonal-1), c='r',label="Moltiplicativo")
plt.title("Confronto stagionalità")
plt.legend()
plt.show()

plt.plot(result_a.resid, c="k", label="Additivo")
plt.plot(np.nanmean(result_m.trend)*(result_m.resid-1), c='r', label="Moltiplicativo")
plt.title("Confronto residui")
plt.legend()
plt.show()

#analisi dei residui
valori = tabella.values[6:-6]
aresid = result_a.resid.values
aresid = aresid[~np.isnan(aresid)]
mresid = result_m.resid.values
mresid = mresid[~np.isnan(mresid)]
plt.xlabel("Unità temporali")
plt.plot(aresid,'.')
plt.title("Residui additivi")
plt.show()
fig, ax = plt.subplots()
plot_acf(aresid, lags=30, ax=ax)
ax.set_xticks(range(0, 31, 1))
ax.set_xlabel('Lag')
ax.grid(True)
plt.title('ACF residui additivi')
plt.show()

fig, ax = plt.subplots()
plot_acf(mresid, lags=30, ax=ax)
ax.set_xticks(range(0, 31, 1))
ax.set_xlabel('Lag')
ax.grid(True)
plt.title('ACF residui moltiplicativi')
plt.show()


plt.plot(mresid,'.')
plt.title("Residui moltiplicativi")
plt.xlabel("Unità temporali")
plt.show()

acf_values = acf(aresid, fft=True)
acf_std = np.std(acf_values)
print("Deviazione standard dell'ACF:", acf_std)
acf_values = acf(np.log(mresid), fft=True)
acf_std = np.std(acf_values)
print("Deviazione standard dell'ACF:", acf_std)
print("Var additiva: ", np.var(aresid)/np.var(valori))
print("Var molt: ", np.var(np.log(mresid))/np.var(np.log(valori)))

#valori simili: probabilmente c'è l'effetto che sembra additivo perché approssimiamo al primo ordine

#vediamo se i residui seguono una distribuzione gaussiana
mresidl = mresid
x=np.linspace(min(mresidl), max(mresidl),1000)
plt.hist(mresidl,density=True, bins=40, label="Residui moltiplicativi")
plt.plot(x, norm.pdf(x,loc=np.mean(mresidl),scale=np.std(mresidl)), label="Gaussiana")
plt.title("Confronto distribuzione moltiplicativa e gaussiana")
plt.legend()
plt.show()
sm.qqplot((mresidl-np.mean(mresidl))/(np.std(mresidl)), line='45')
plt.title("QQ Plot: moltiplicativo")
plt.show()
ska = st.skew((aresid-np.mean(aresid))/np.std(aresid))
skm = st.skew((mresid-np.mean(mresid))/np.std(mresid))
ka = st.kurtosis((aresid-np.mean(aresid))/np.std(aresid))
km = st.kurtosis((aresid-np.mean(aresid))/np.std(aresid))
print(st.shapiro(mresidl))
print("Skewness_add: ", ska)
print("Skewness_molt: ", skm)
print("Kurtosi_add: ", ka)
print("Kurtosi_molt: ", km)

x=np.linspace(min(aresid), max(aresid),1000)
plt.hist(aresid,density=True, bins=40, label="Residui additivi")
plt.plot(x, norm.pdf(x,loc=np.mean(aresid),scale=np.std(aresid)), label="Gaussiana")
plt.title("Confronto distribuzione additiva e gaussiana")
plt.legend()
plt.show()
sm.qqplot((aresid-np.mean(aresid))/(np.std(aresid)), line='45')
plt.title("QQ Plot: additivo")
plt.show()

print(st.shapiro(aresid))


stl = STL(tabella, seasonal=7)
result = stl.fit()
resid =result.resid.values

plt.xlabel("Unità temporali")
plt.plot(resid,'.')
plt.title("Residui STL")
plt.show()

fig, ax = plt.subplots()
plot_acf(resid, lags=30, ax=ax)
ax.set_xticks(range(0, 31, 1))
ax.set_xlabel('Lag')
ax.grid(True)
plt.title('ACF residui STL')
plt.show()

acf_values = acf(resid, fft=True)
acf_std = np.std(acf_values)
print("Deviazione standard dell'ACF:", acf_std)
print("Var STL: ", np.var(resid)/np.var(valori))

x=np.linspace(min(resid), max(resid),1000)
plt.hist(resid,density=True, bins=40, label="Residui STL")
plt.plot(x, norm.pdf(x,loc=np.mean(resid),scale=np.std(resid)), label="Gaussiana")
plt.title("Confronto distribuzione STL e gaussiana")
plt.legend()
plt.show()
sm.qqplot((resid-np.mean(resid))/(np.std(resid)), line='45')
plt.title("QQ Plot: STL")
plt.show()

sk = st.skew((resid-np.mean(resid))/np.std(resid))
k = st.kurtosis((resid-np.mean(resid))/np.std(resid))
print("Skewness_stl: ", sk)
print("Kurtosi_stl: ", k)
print("Shaprio stl: ", st.shapiro(resid))

stl = STL(np.log(tabella), seasonal=7)
result = stl.fit()
resid =result.resid.values

plt.xlabel("Unità temporali")
plt.plot(resid,'.')
plt.title("Residui STL moltiplicaivo")
plt.show()

fig, ax = plt.subplots()
plot_acf(resid, lags=30, ax=ax)
ax.set_xticks(range(0, 31, 1))
ax.set_xlabel('Lag')
ax.grid(True)
plt.title('ACF residui STL moltiplicativi')
plt.show()

acf_values = acf(resid, fft=True)
acf_std = np.std(acf_values)
print("Deviazione standard dell'ACF:", acf_std)
print("Var STL molt.: ", np.var(resid)/np.var(np.log(valori)))

x=np.linspace(min(resid), max(resid),1000)
plt.hist(resid,density=True, bins=40, label="Residui STL molt.")
plt.plot(x, norm.pdf(x,loc=np.mean(resid),scale=np.std(resid)), label="Gaussiana")
plt.title("Confronto distribuzione STL molt. e gaussiana")
plt.legend()
plt.show()
sm.qqplot((resid-np.mean(resid))/(np.std(resid)), line='45')
plt.title("QQ Plot: STL molt.")
plt.show()
sk = st.skew((resid-np.mean(resid))/np.std(resid))
k = st.kurtosis((resid-np.mean(resid))/np.std(resid))
print("Skewness_stl_molt: ", sk)
print("Kurtosi_stl_molt: ", k)
print("Shaprio stl molt: ", st.shapiro(resid))





