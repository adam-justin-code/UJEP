import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression

# 1. Vygenerování dat
X, y = make_blobs(n_samples=200, centers=2, random_state=42, cluster_std=1.0)

# 2. Přeškálování na Výšku a Váhu
X[:, 0] = X[:, 0] * 0.08 + 1.75  # Výška
X[:, 1] = X[:, 1] * 10 + 80      # Váha

# 3. Model pro nalezení přímky
model = LogisticRegression()
model.fit(X, y)

w = model.coef_[0]
b = model.intercept_[0]
slope = -w[0] / w[1]
intercept = -b / w[1]

# Body pro čáru
x_line = np.linspace(1.4, 2.1, 100)
y_line = slope * x_line + intercept

# 4. Vykreslení
plt.figure(figsize=(8, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', marker='x', label='Skupina A')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', marker='x', label='Skupina B')
plt.plot(x_line, y_line, color='#1f77b4', linewidth=2, label='Lineární hranice')

plt.title("Rozdělení blobů přímkou")
plt.xlabel("h [m]")
plt.ylabel("m [kg]")
plt.grid(True)
plt.legend()

plt.show()