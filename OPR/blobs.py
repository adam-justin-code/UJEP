import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from scipy.optimize import linprog

# 1. Vygenerování dat
# Random state 42 občas prohodí barvy, proto musíme detekovat polohu dynamicky
X, y = make_blobs(n_samples=300, centers=2, random_state=42, cluster_std=1.2)

# Přeškálování (Výška v m, Váha v kg)
X[:, 0] = X[:, 0] * 0.08 + 1.75  
X[:, 1] = X[:, 1] * 10 + 130 # Posunul jsem váhu výš, aby to odpovídalo tvému obrázku

# 2. Automatická detekce: Kdo je nahoře a kdo dole?
# Spočítáme průměrnou váhu pro skupinu 0 a 1
mean_y_0 = np.mean(X[y == 0][:, 1])
mean_y_1 = np.mean(X[y == 1][:, 1])

# Podle vzorce: M0 musí být body POD čárou, M1 body NAD čárou (nebo naopak, podle definice phi)
# Aby seděl vzorec z obrázku: 
# phi >= y + delta (Line is ABOVE points -> M_bottom)
# phi <= y - delta (Line is BELOW points -> M_top)

if mean_y_0 < mean_y_1:
    M_bottom = X[y == 0] # Skupina dole
    M_top = X[y == 1]    # Skupina nahoře
    label_bottom = "Skupina A (Dole)"
    label_top = "Skupina B (Nahoře)"
else:
    M_bottom = X[y == 1] # Skupina dole (byla to '1', teď je to naše 'bottom')
    M_top = X[y == 0]    # Skupina nahoře
    label_bottom = "Skupina B (Dole)"
    label_top = "Skupina A (Nahoře)"

# 3. Příprava LP
# Proměnné: [beta0, beta1, delta]
# Cíl: Maximize delta => Minimize (-delta)
c = [0, 0, -1] 

A_ub = [] 
b_ub = []

# Nerovnice pro SPODNÍ skupinu (M_bottom):
# Čára musí být NAD nimi: beta0 + beta1*x - delta >= y 
# Přepis: -beta0 - beta1*x + delta <= -y
for point in M_bottom:
    A_ub.append([-1, -point[0], 1])
    b_ub.append(-point[1])

# Nerovnice pro HORNÍ skupinu (M_top):
# Čára musí být POD nimi: beta0 + beta1*x + delta <= y
# Přepis: beta0 + beta1*x + delta <= y
for point in M_top:
    A_ub.append([1, point[0], 1])
    b_ub.append(point[1])

# Meze: delta musí být kladná
bounds = [(None, None), (None, None), (0, None)]

# Výpočet
res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

# 4. Vykreslení
plt.figure(figsize=(10, 7))

# Vykreslení bodů (použijeme naše seřazené M_bottom a M_top)
plt.scatter(M_bottom[:, 0], M_bottom[:, 1], color='blue', marker='x', label=label_bottom)
plt.scatter(M_top[:, 0], M_top[:, 1], color='red', marker='x', label=label_top)

if res.success:
    beta0, beta1, delta = res.x
    print(f"Úspěch! Delta (margin) = {delta:.4f}")
    
    # Body pro čáru
    x_range = np.linspace(min(X[:,0]), max(X[:,0]), 100)
    y_line = beta0 + beta1 * x_range
    
    # Vykreslení
    plt.plot(x_range, y_line, color='green', linewidth=3, label='Optimální dělící přímka')
    
    # Vykreslení koridoru (marginu)
    plt.plot(x_range, y_line - delta, '--', color='gray', alpha=0.7, label='Hranice marginu')
    plt.plot(x_range, y_line + delta, '--', color='gray', alpha=0.7)
    plt.fill_between(x_range, y_line - delta, y_line + delta, color='gray', alpha=0.1)

else:
    print("Řešení nenalezeno! Body se pravděpodobně prolínají.")

plt.title(f"Přesné rozdělení podle vzorce (Delta = {res.x[2]:.2f})")
plt.xlabel("Výška [m]")
plt.ylabel("Váha [kg]")
plt.legend()
plt.grid(True)
plt.show()