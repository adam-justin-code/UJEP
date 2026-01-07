import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from scipy.optimize import linprog

# 1. Vygenerování hustších dat (více blobů/bodů na střed)
# Zvýšíme n_samples a zvětšíme cluster_std, aby se body k sobě přiblížily
X, y = make_blobs(n_samples=300, centers=2, random_state=42, cluster_std=1.3)

# Přeškálování na reálné jednotky (Výška v m, Váha v kg)
X[:, 0] = X[:, 0] * 0.08 + 1.75  # Výška
X[:, 1] = X[:, 1] * 10 + 80      # Váha

# Rozdělení na skupiny pro LP formulaci
# Podle tvého obrázku: M0 jsou body "dole" (červené), M1 jsou body "nahoře" (modré)
# V našem generátoru je y==1 horní shluk (Modrá), y==0 spodní (Červená)
M0_x, M0_y = X[y == 0][:, 0], X[y == 0][:, 1]  # Skupina dole
M1_x, M1_y = X[y == 1][:, 0], X[y == 1][:, 1]  # Skupina nahoře

# 2. Příprava Lineárního programování
# Hledáme proměnné: [beta0, beta1, delta]
# Funkce přímky: phi = beta0 + beta1 * x
# Cíl: Maximalizovat delta => Minimalizovat (-delta)
c = [0, 0, -1] 

# Omezení (Constraints) podle tvého vzorce:
# Pro M0 (dole): beta0 + beta1*x - delta >= y  --> -beta0 - beta1*x + delta <= -y
# Pro M1 (nahoře): beta0 + beta1*x + delta <= y -->  beta0 + beta1*x + delta <= y

A_ub = [] # Matice koeficientů nerovnic
b_ub = [] # Pravá strana nerovnic

# Naplnění nerovnic pro M0 (červené body)
for x_val, y_val in zip(M0_x, M0_y):
    A_ub.append([-1, -x_val, 1])
    b_ub.append(-y_val)

# Naplnění nerovnic pro M1 (modré body)
for x_val, y_val in zip(M1_x, M1_y):
    A_ub.append([1, x_val, 1])
    b_ub.append(y_val)

# Meze proměnných: beta0, beta1 jsou libovolná čísla, delta musí být >= 0
bounds = [(None, None), (None, None), (0, None)]

# 3. Výpočet pomocí Simplexové metody
res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

if res.success:
    beta0, beta1, delta = res.x
    print(f"Nalezeno optimální řešení: delta = {delta:.4f}")
    
    # Vytvoření bodů pro čáru
    x_line = np.linspace(min(X[:,0]), max(X[:,0]), 100)
    y_line = beta0 + beta1 * x_line
    
    # Hranice koridoru (pro vizualizaci delty)
    y_lower = y_line - delta # Hranice pro horní skupinu
    y_upper = y_line + delta # Hranice pro spodní skupinu
else:
    print("Řešení nenalezeno (body se pravděpodobně prolínají příliš moc).")
    beta0, beta1, delta = 0, 0, 0
    x_line, y_line = [], []

# 4. Vykreslení
plt.figure(figsize=(10, 7))

# Body
plt.scatter(M0_x, M0_y, color='red', marker='x', label='M0 (Dolní skupina)')
plt.scatter(M1_x, M1_y, color='blue', marker='x', label='M1 (Horní skupina)')

if res.success:
    # Střední dělící přímka
    plt.plot(x_line, y_line, color='black', linewidth=2, label=f'Optimální přímka (delta={delta:.2f})')
    # Zobrazení "bezpečného pásma" (margin)
    plt.fill_between(x_line, y_lower, y_upper, color='gray', alpha=0.2, label='Margin (delta)')
    plt.plot(x_line, y_lower, '--', color='gray', linewidth=1)
    plt.plot(x_line, y_upper, '--', color='gray', linewidth=1)

plt.title("Separace pomocí Lineárního programování (Max Margin)")
plt.xlabel("Výška [m]")
plt.ylabel("Váha [kg]")
plt.legend()
plt.grid(True)
plt.show()