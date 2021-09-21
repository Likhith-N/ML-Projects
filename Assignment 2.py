import matplotlib.pyplot as plt
import math
import pandas as pd


def avg(gas):
    value = sum(gas)/len(gas)*100
    return value


df = pd.read_csv("cityDay.csv")

Bnglr = df.iloc[6122:6153, :-1]
date = [x for x in range(1, 32)]
Ggases = ["NO", "NO2", "NOx", "NH3", "CO", "SO2"]
values = []
for word in Ggases:
    values.append(avg(Bnglr[word]))
# plot 1
plt.subplot(2, 2, 1)
plt.plot(date, Bnglr["CO"], 'r', label="CO")
plt.plot(date, Bnglr["NOx"], 'g', label="NOx")
plt.xlabel("Dates")
plt.ylabel("mg / m3")
plt.title("1. Daily CO & NOx values")
plt.legend()

# plot 2
plt.subplot(2, 2, 2)
plt.plot(date, Bnglr["NO"], 'or', mec='k', label="NO", alpha=0.5)
plt.plot(date, Bnglr["NO2"], 'og', mec='k', label="NO2", alpha=0.5)
plt.plot(date, Bnglr["NOx"], 'ob', mec='k', label="NOx", alpha=0.5)
plt.plot(date, Bnglr["NH3"], 'oc', mec='k', label="Nh3", alpha=0.5)
plt.plot(date, Bnglr["CO"], 'om', mec='k', label="CO", alpha=0.5)
plt.plot(date, Bnglr["SO2"], 'oy', mec='k', label="SO2", alpha=0.5)
plt.xlabel("Dates")
plt.ylabel("mg / m3")
plt.title("2. All gases values in scattered points")
plt.legend()

# plot 3
plt.subplot(2, 2, 3)
plt.bar(date, Bnglr["NO"], label="NO", width=0.2, color='b')
plt.bar(date, Bnglr["SO2"], label="SO2", width=0.4, color='y', alpha=0.5)
plt.xlabel("Dates")
plt.ylabel("mg / m3")
plt.title("3. Daily NO & SO2 values")
plt.legend()

# plot 4
plt.subplot(2, 2, 4)
plt.pie(values, labels=Ggases, autopct="%1.1f%%", colors=[
        'c', 'r', 'y', 'g', 'b', 'w'], shadow=True, explode=[0, 0, 0.3, 0, 0, 0])
plt.title("4. Gas values")

plt.show()
