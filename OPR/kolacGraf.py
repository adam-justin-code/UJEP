import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

def solve_portfolio_optimization():
    n_assets = 5

    dummy_data = np.random.randn(100, n_assets)
    C = np.cov(dummy_data.T) 
    
    m = np.array([0.05, 0.1, 0.12, 0.04, 0.09]) 
    
    r = 0.08 

    x0 = np.ones(n_assets) / n_assets

    def objective(x):
        return 0.5 * np.dot(x.T, np.dot(C, x))

    # 2. Omezení (Constraints)
    constraints = (
        # Suma vah musí být 1 (sum(x) = 1)
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        # Výnos musí být alespoň r (sum(m*x) >= r)
        {'type': 'ineq', 'fun': lambda x: np.dot(m, x) - r}
    )

    bounds = tuple((0, 1) for _ in range(n_assets))

    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result

vysledne_vahy = [
    0.11962057827825744,    # C1
    2.30103264816861e-06,   # C2
    0.2814317889490023,     # C3
    2.6672503869561893e-08, # C4
    0.5989453050675883      # C5
]

riziko_hodnota = 0.004461277407178562
labels = ['C1', 'C2', 'C3', 'C4', 'C5']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

plt.figure(figsize=(9, 7))

wedges, texts, autotexts = plt.pie(
    vysledne_vahy, 
    labels=labels, 
    autopct='%1.2f%%', 
    startangle=140, 
    colors=colors,
    pctdistance=0.85
)

for text in texts:
    text.set_fontsize(12)
for autotext in autotexts:
    autotext.set_fontsize(10)
    autotext.set_color('white') 
    if autotext.get_text() == "0.00%":
        autotext.set_color('black')

plt.title(f"Rozložení portfolia\nMinimální riziko: {riziko_hodnota:.5f}", fontsize=14)
plt.axis('equal')  
plt.tight_layout()
plt.show()