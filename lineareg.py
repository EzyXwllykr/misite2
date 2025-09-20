from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([3, 6, 9, 13, 15])

modelo = LinearRegression()
modelo.fit(x, y)

dias = 7
capitulos_previstos = modelo.predict([[dias]])
print(f"Depois de {dias} dias, a previsão é que a pessoa leia {capitulos_previstos[0]:.0f} capítulos.")

plt.scatter(x, y, color="blue")
plt.plot(x, modelo.predict(x), color="red")
plt.show()