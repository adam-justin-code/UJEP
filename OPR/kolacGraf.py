import matplotlib.pyplot as plt

# 1. Tvá data ze zadání
reseni = [
    0.11962057827825744,   # C1
    2.30103264816861e-06,  # C2 (velmi malé číslo, zobrazí se jako 0.00%)
    0.2814317889490023,    # C3
    2.6672503869561893e-08, # C4 (velmi malé číslo, zobrazí se jako 0.00%)
    0.5989453050675883     # C5 (největší část)
]

riziko = 0.004461277407178562

# 2. Definice popisků
labels = ['C1', 'C2', 'C3', 'C4', 'C5']

# 3. Vytvoření grafu
plt.figure(figsize=(8, 8)) # Nastavení velikosti, aby byl graf hezky čitelný

# Funkce plt.pie:
# - x: tvoje data (reseni)
# - labels: popisky (C1, C2...)
# - autopct: formátování procent ('%1.2f%%' znamená 2 desetinná místa)
# - startangle: pootočení grafu (volitelné, 0 je standardní)
plt.pie(reseni, labels=labels, autopct='%1.2f%%', startangle=0)

# Nastavení titulku s hodnotou rizika
plt.title(f"Riziko {riziko}")

# Zobrazení
plt.show()