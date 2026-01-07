import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

# --- ČÁST 1: Jak se taková rovnice řeší v Pythonu ---
# (Tato část je ukázková. Aby fungovala na tvá data, 
# musel bys nahradit 'dummy' data svou maticí C a vektorem m)

def solve_portfolio_optimization():
    # Příklad dat (5 aktiv C1-C5)
    n_assets = 5
    
    # Náhodná kovarianční matice C (symetrická)
    # V reálu: C = data.cov()
    dummy_data = np.random.randn(100, n_assets)
    C = np.cov(dummy_data.T) 
    
    # Náhodné očekávané výnosy m
    # V reálu: m = data.mean()
    m = np.array([0.05, 0.1, 0.12, 0.04, 0.09]) 
    
    # Požadovaný minimální výnos r
    r = 0.08 

    # Počáteční odhad vah (rovnoměrné rozložení)
    x0 = np.ones(n_assets) / n_assets

    # 1. Cílová funkce (Objective): min risk = 0.5 * x.T * C * x
    def objective(x):
        return 0.5 * np.dot(x.T, np.dot(C, x))

    # 2. Omezení (Constraints)
    constraints = (
        # Suma vah musí být 1 (sum(x) = 1)
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        # Výnos musí být alespoň r (sum(m*x) >= r)
        {'type': 'ineq', 'fun': lambda x: np.dot(m, x) - r}
    )

    # 3. Meze (Bounds): 0 <= xi <= 1
    bounds = tuple((0, 1) for _ in range(n_assets))

    # Řešení pomocí SLSQP (vhodné pro kvadratické problémy s omezeními)
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result

# --- ČÁST 2: Vykreslení grafu z tvých výsledků ---

# Tvé výsledky z textového zadání
vysledne_vahy = [
    0.11962057827825744,    # C1
    2.30103264816861e-06,   # C2 (téměř 0)
    0.2814317889490023,     # C3
    2.6672503869561893e-08, # C4 (téměř 0)
    0.5989453050675883      # C5
]

# Další údaje ze zadání
riziko_hodnota = 0.004461277407178562
labels = ['C1', 'C2', 'C3', 'C4', 'C5']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] # Standardní mpl barvy

# Vytvoření koláčového grafu
plt.figure(figsize=(9, 7))

# Vykreslení
# pctdistance: posune procenta trochu dál od středu
# explode: volitelné, kdybychom chtěli zvýraznit nějakou část (zde vypnuto)
wedges, texts, autotexts = plt.pie(
    vysledne_vahy, 
    labels=labels, 
    autopct='%1.2f%%', 
    startangle=140, # Pootočení, aby C5 (největší) byla dole/vlevo, podobně jako na screenu
    colors=colors,
    pctdistance=0.85
)

# Kosmetické úpravy textu
for text in texts:
    text.set_fontsize(12)
for autotext in autotexts:
    autotext.set_fontsize(10)
    autotext.set_color('white') 
    if autotext.get_text() == "0.00%":
        autotext.set_color('black')

# Přidání kruhu doprostřed (pro moderní vzhled "Donut chart" - volitelné, pokud chceš plný koláč, smaž toto)
# centre_circle = plt.Circle((0,0),0.70,fc='white')
# fig = plt.gcf()
# fig.gca().add_artist(centre_circle)

plt.title(f"Rozložení portfolia\nMinimální riziko: {riziko_hodnota:.5f}", fontsize=14)
plt.axis('equal')  # Aby byl kruh opravdu kruh
plt.tight_layout()
plt.show()