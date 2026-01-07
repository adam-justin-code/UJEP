import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --- 1. PŘÍPRAVA DAT (Zde bys nahrál svá data) ---
np.random.seed(42)
n_assets = 5

# Náhodné očekávané výnosy (m)
# V reálu: m = data.mean()
means = np.random.uniform(0.05, 0.20, n_assets) 

# Náhodná kovarianční matice (C)
# V reálu: C = data.cov()
A = np.random.randn(n_assets, n_assets)
C = np.dot(A, A.T) # Zajištění, aby byla matice symetrická a pozitivně definitní

# --- 2. FUNKCE PRO OPTIMALIZACI ---
def portfolio_risk(weights, cov_matrix):
    # Tvá rovnice: 1/2 * x.T * C * x
    # Poznámka: Pro graf se často používá směrodatná odchylka (odmocnina z variance)
    # Ale pro solver použijeme přesně tvou rovnici
    var = 0.5 * np.dot(weights.T, np.dot(cov_matrix, weights))
    return var

def portfolio_return(weights, mean_returns):
    return np.sum(weights * mean_returns)

# --- 3. VÝPOČET EFEKTIVNÍ HRANICE (Efficient Frontier) ---
# Rozsah výnosů, pro které budeme hledat minimální riziko
target_returns = np.linspace(min(means), max(means), 50)
efficient_risks = []
efficient_returns = []

# Pro každý cílový výnos najdeme portfolio s nejmenším rizikem
for target in target_returns:
    n = len(means)
    
    # Omezení
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},       # Suma vah = 1
        {'type': 'eq', 'fun': lambda x: portfolio_return(x, means) - target} # Výnos = target
    )
    bounds = tuple((0, 1) for _ in range(n))
    init_guess = np.ones(n) / n
    
    result = minimize(
        portfolio_risk, 
        init_guess, 
        args=(C,), 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints
    )
    
    if result.success:
        # Do grafu dáváme riziko. 
        # Pokud tvůj vzorec (1/2 xCx) je variance, pro graf se hodí spíše Odmocnina (Volatilita/Std Dev)
        # Aby to bylo srovnatelné s osami. Zde pro přesnost s tvým vzorcem nechávám hodnotu funkce.
        risk_val = result.fun 
        efficient_risks.append(risk_val)
        efficient_returns.append(target)

# --- 4. SIMULACE NÁHODNÝCH PORTFOLIÍ (Monte Carlo) ---
# Pro hezké pozadí grafu vygenerujeme tisíce náhodných mixů
n_portfolios = 5000
rand_risks = []
rand_returns = []

for _ in range(n_portfolios):
    w = np.random.random(n_assets)
    w /= np.sum(w) # Normalizace na součet 1
    
    r = portfolio_return(w, means)
    var = portfolio_risk(w, C)
    
    rand_risks.append(var)
    rand_returns.append(r)

# --- 5. VYKRESLENÍ GRAFU ---
plt.figure(figsize=(10, 6))

# A) Náhodná portfolia (šedý mrak)
plt.scatter(rand_risks, rand_returns, c=np.array(rand_returns)/np.array(rand_risks), 
            marker='o', cmap='viridis', s=10, alpha=0.3, label='Náhodná portfolia')

# B) Efektivní hranice (Červená čára)
plt.plot(efficient_risks, efficient_returns, 'r-', linewidth=3, label='Efektivní hranice (Efficient Frontier)')

# Popisky
plt.title('Risk vs Return (Efektivní hranice)', fontsize=14)
plt.xlabel('Riziko (hodnota účelové funkce)', fontsize=12)
plt.ylabel('Očekávaný výnos', fontsize=12)
plt.colorbar(label='Sharpe Ratio (orientační)')
plt.legend()
plt.grid(True, alpha=0.5)

plt.show()