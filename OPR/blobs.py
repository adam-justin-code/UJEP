import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog

# --- 1. Generování dat "po celém gridu" ---
np.random.seed(42)
n_points = 400

# Náhodná výška 1.4 až 2.2 metru
h = np.random.uniform(1.4, 2.2, n_points)
# Náhodná váha 50 až 130 kg
m = np.random.uniform(50, 130, n_points)

# --- 2. Vytvoření "nutné mezery" pro Hard Margin ---
# Aby rovnice měla řešení, musíme vymazat body, které leží přesně uprostřed.
# Definujeme si cvičnou hranici (např. váha = 100 * výška - 100)
line_true = 100 * h - 100
margin_safety = 5  # Velikost mezery, kterou vyčistíme

# Rozdělíme body na horní a dolní a zahodíme ty, co jsou příliš blízko středu
mask_gap = np.abs(m - line_true) > margin_safety
h = h[mask_gap]
m = m[mask_gap]

# Teď určíme, kdo je nahoře a kdo dole (pro vstup do solveru)
# Skupina 0 (Pod čárou) a Skupina 1 (Nad čárou)
labels = (m > line_true).astype(int)

X = np.column_stack((h, m))
y = labels

# --- 3. Matematický Solver (LP) ---
# Data pro solver
M_bottom = X[y == 0] # Body pod mezerou
M_top = X[y == 1]    # Body nad mezerou

c = [0, 0, -1] # Maximize delta
A_ub = []
b_ub = []

# Nerovnice: Přímka musí být NAD spodními body
for point in M_bottom:
    A_ub.append([-1, -point[0], 1])
    b_ub.append(-point[1])

# Nerovnice: Přímka musí být POD horními body
for point in M_top:
    A_ub.append([1, point[0], 1])
    b_ub.append(point[1])

# Omezení
bounds = [(-2000, 2000), (-2000, 2000), (0, None)]

# Výpočet
res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

# --- 4. Vykreslení ---
plt.figure(figsize=(10, 7))

if res.success:
    beta0, beta1, delta = res.x
    
    # Vykreslení čáry
    x_range = np.linspace(1.35, 2.25, 100)
    y_line = beta0 + beta1 * x_range
    
    # Obarvení bodů podle výsledné čáry
    # Znovu zkontrolujeme polohu bodů vůči vypočítané čáře
    y_calc = beta0 + beta1 * X[:, 0]
    mask_above = X[:, 1] > y_calc
    mask_below = X[:, 1] < y_calc
    
    # Vykreslení bodů rozprostřených po celém gridu
    plt.scatter(X[mask_above, 0], X[mask_above, 1], color='red', marker='x', label='Skupina Nad')
    plt.scatter(X[mask_below, 0], X[mask_below, 1], color='blue', marker='x', label='Skupina Pod')
    
    # Čára a margin
    plt.plot(x_range, y_line, color='green', linewidth=3, label='Optimální přímka')
    plt.fill_between(x_range, y_line - delta, y_line + delta, color='green', alpha=0.1, label='Margin (Delta)')
    
    plt.title(f"Separace dat na celém gridu (Delta={delta:.2f})")
else:
    plt.scatter(X[:, 0], X[:, 1], color='gray')
    plt.title("Řešení nenalezeno (Body se stále prolínají)")

plt.xlabel("Výška [m]")
plt.ylabel("Váha [kg]")
plt.legend()
plt.grid(True)
plt.show()