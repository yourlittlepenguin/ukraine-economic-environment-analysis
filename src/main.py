import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# ==============================
# ГЛОБАЛЬНІ НАЛАШТУВАННЯ ВИВОДУ
# ==============================

OUTPUT_DIR = "project_result"
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.set_printoptions(precision=4, suppress=True)

print(f"Всі результати будуть збережені в папку: {os.path.abspath(OUTPUT_DIR)}")
print("="*80)

pd.set_option('display.max_rows', None)    # показати всі рядки
pd.set_option('display.max_columns', None) # показати всі стовпці
pd.set_option('display.width', 1000)       # збільшити ширину виводу
pd.set_option('display.float_format', '{:.4f}'.format)  # формат чисел

# =============================================================================
# ЧАСТИНА 1. Аналіз 2019–2024: оцінка A (МНК і Ridge), оцінка B,C, Пікара, Ньютон
# =============================================================================

print("\n=== ЧАСТИНА 1: 2019–2024, оцінка A,B,C, Пікара, Ньютон ===\n")

# -------------------- Вхідні робочі дані --------------------
years1 = np.array([2019, 2020, 2021, 2022, 2023, 2024])
n1 = len(years1)

# Випуск трьох секторів (умовні індекси)
X1 = np.array([
    [100., 100., 100.],  # 2019
    [95.,  90.,  98.],   # 2020
    [98.,  97.,  100.],  # 2021
    [70.,  60.,  85.],   # 2022
    [75.,  68.,  88.],   # 2023
    [80.,  72.,  90.]    # 2024
])

# Істинна матриця A (для генерації даних)
A_true = np.array([
    [0.28, 0.15, 0.05],
    [0.10, 0.25, 0.08],
    [0.06, 0.07, 0.12]
])

# Обчислення Y_t = X_{t+1} - A_true X_t
Y1 = []
for t in range(n1 - 1):
    Yt = X1[t+1] - A_true.dot(X1[t])
    Y1.append(Yt)
Y1 = np.array(Y1)

# Додаємо шум
rng = np.random.default_rng(42)
Y1_noisy = Y1 + rng.normal(scale=0.5, size=Y1.shape)

# Параметри для Z
B_true = np.array([2.25, 0.45, 0.55])
C_true = np.array([0.12, 0.08, 0.10])

# Шоки (тільки 2024)
S1 = np.zeros((n1, 3))
S1[years1.tolist().index(2024)] = np.array([5., 10., 34.])

# -------------------- Оцінка A (LS і Ridge) --------------------
X1_t = X1[:-1]
X1_next_minus_Y = X1[1:] - Y1_noisy

# МНК (LS)
A_est_T = np.linalg.inv(X1_t.T @ X1_t) @ X1_t.T @ X1_next_minus_Y
A_est = A_est_T.T

def estimate_A_ridge(X_t, X_next_minus_Y, lam):
    k = X_t.shape[1]
    A_T = np.linalg.inv(X_t.T @ X_t + lam * np.eye(k)) @ X_t.T @ X_next_minus_Y
    return A_T.T

lambdas = [0.01, 0.1, 0.5, 1.0, 5.0]
A_ridge = {lam: estimate_A_ridge(X1_t, X1_next_minus_Y, lam) for lam in lambdas}

def compute_rmse(A_mat, X, Y_noisy):
    X_hat = X.copy()
    for t in range(len(X) - 1):
        X_hat[t+1] = A_mat.dot(X[t]) + Y_noisy[t]
    err = X_hat[1:] - X[1:]
    rmse = np.sqrt(np.mean(err**2))
    return rmse, X_hat

rmse_ls, Xhat_ls = compute_rmse(A_est, X1, Y1_noisy)
rmse_ridge = {}
for lam, A_r in A_ridge.items():
    rmse, _ = compute_rmse(A_r, X1, Y1_noisy)
    rmse_ridge[lam] = rmse

# -------------------- Динаміка Z та оцінка B, C --------------------
total_ghg_2021 = 341.5
Z1 = np.zeros((n1, 3))
Z1[0] = np.array([0.65, 0.17, 0.18]) * total_ghg_2021

for t in range(n1 - 1):
    Z1[t+1] = B_true * X1[t] - C_true * Z1[t] + S1[t+1]

Z1_obs = Z1 + rng.normal(scale=0.2, size=Z1.shape)

B_est = np.zeros(3)
C_est = np.zeros(3)

for i in range(3):
    M_rows = []
    rhs = []
    for t in range(n1 - 1):
        M_rows.append([X1[t,i], -Z1_obs[t,i]])
        rhs.append(Z1_obs[t+1,i] - S1[t+1,i])
    M = np.array(M_rows)
    rhs = np.array(rhs)
    theta, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    B_est[i] = theta[0]
    C_est[i] = theta[1]

# -------------------- Нелінійні методи Пікара та Ньютона для 2024 --------------------
BX2024 = B_est * X1[-1]
S2024 = S1[years1.tolist().index(2024)]
C_nl = C_est

def picard_relax_nl(BX, Cnl, Svec, Z0, omega=0.05, max_iter=200, tol=1e-9):
    Zk = Z0.copy().astype(float)
    hist = [Zk.copy()]
    diffs = []
    for k in range(max_iter):
        G = BX - Cnl * (Zk**2) + Svec
        Zk1 = (1-omega)*Zk + omega*G
        hist.append(Zk1.copy())
        diff = np.linalg.norm(Zk1 - Zk)
        diffs.append(diff)
        if diff < tol:
            break
        Zk = Zk1
    return np.array(hist), np.array(diffs)

def newton_nl(BX, Cnl, Svec, Z0, max_iter=50, tol=1e-12):
    Z = Z0.copy().astype(float)
    hist = [Z.copy()]
    diffs = []
    for k in range(max_iter):
        F = Z - BX + Cnl * (Z**2) - Svec
        J_diag = 1.0 + 2.0 * Cnl * Z
        delta = F / J_diag
        Z_new = Z - delta
        hist.append(Z_new.copy())
        diff = np.linalg.norm(Z_new - Z)
        diffs.append(diff)
        if diff < tol:
            break
        Z = Z_new
    return np.array(hist), np.array(diffs)

Z0_guess = Z1_obs[-1]
pic_hist, pic_diffs = picard_relax_nl(BX2024, C_nl, S2024, Z0_guess, omega=0.05)
newt_hist, newt_diffs = newton_nl(BX2024, C_nl, S2024, Z0_guess)

# -------------------- Графіки та вивід (Частина 1) --------------------
sector_names1 = ['Енергетика','Металургія','Агро/Ліс']

plt.figure(figsize=(8,5))
for i in range(3):
    plt.plot(years1, X1[:,i], marker='o', label=sector_names1[i])
plt.title('Випуск секторів 2019-2024 (індекси)')
plt.xlabel('Рік'); plt.ylabel('Індекс випуску'); plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'X1_timeseries_2019_2024.png'), dpi=150)
plt.show()

Z1_no_shock = np.zeros_like(Z1)
Z1_with_shock = np.zeros_like(Z1)
Z1_no_shock[0] = Z1[0].copy()
Z1_with_shock[0] = Z1[0].copy()
for t in range(n1 - 1):
    Z1_no_shock[t+1] = B_true * X1[t] - C_true * Z1_no_shock[t]
    Z1_with_shock[t+1] = B_true * X1[t] - C_true * Z1_with_shock[t] + S1[t+1]

plt.figure(figsize=(8,5))
for i in range(3):
    plt.plot(years1, Z1_no_shock[:,i], marker='o', linestyle='--', label=f'{sector_names1[i]} без шоку')
    plt.plot(years1, Z1_with_shock[:,i], marker='s', linestyle='-', label=f'{sector_names1[i]} зі шоком')
plt.title('Викиди Z (MtCO2e) — сценарії без шоку і зі шоком 2024 (2019–2024)')
plt.xlabel('Рік'); plt.ylabel('Z (MtCO2e)'); plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Z1_timeseries_2019_2024.png'), dpi=150)
plt.show()

print("A_true:\n", np.round(A_true,4))
print("\nA_est (LS):\n", np.round(A_est,4))
print("\nA_ridge (λ=0.5):\n", np.round(A_ridge[0.5],4))
print("\nRMSE LS:", np.round(rmse_ls,6))
for lam, rm in rmse_ridge.items():
    print(f"RMSE Ridge λ={lam}: {np.round(rm,6)}")

plt.figure(figsize=(8,5))
for i in range(3):
    plt.plot(np.arange(len(pic_hist)), pic_hist[:,i], marker='o', label=f'Пікара {sector_names1[i]}')
plt.title('Пікара (релаксація) — траєкторії Z^k (рік 2024, нелінійна фіксація)')
plt.xlabel('k'); plt.ylabel('Z'); plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'picard_trajectories_2024.png'), dpi=150)
plt.show()

plt.figure(figsize=(8,4))
plt.plot(np.arange(1,len(pic_diffs)+1), pic_diffs, marker='o'); plt.yscale('log')
plt.title('Пікара: норма різниці між ітераціями (лог шкала)')
plt.xlabel('k'); plt.ylabel('||Z^{k+1}-Z^k||'); plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'picard_convergence.png'), dpi=150)
plt.show()

plt.figure(figsize=(8,5))
for i in range(3):
    plt.plot(np.arange(len(newt_hist)), newt_hist[:,i], marker='o', label=f'Ньютон {sector_names1[i]}')
plt.title('Ньютон — траєкторії Z^k (рік 2024)')
plt.xlabel('k'); plt.ylabel('Z'); plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'newton_trajectories_2024.png'), dpi=150)
plt.show()

plt.figure(figsize=(8,4))
plt.plot(np.arange(1,len(newt_diffs)+1), newt_diffs, marker='o'); plt.yscale('log')
plt.title('Ньютон: норма різниці між ітераціями (лог шкала)')
plt.xlabel('k'); plt.ylabel('||Z^{k+1}-Z^k||'); plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'newton_convergence.png'), dpi=150)
plt.show()

print("\nB_true:", B_true)
print("B_est:", np.round(B_est,4))
print("\nC_true:", C_true)
print("C_est:", np.round(C_est,4))

# Збереження матриць A
sector_names1 = ['Енергетика','Металургія','Агро/Ліс']
df_A_true = pd.DataFrame(A_true, index=sector_names1, columns=sector_names1)
df_A_est = pd.DataFrame(A_est, index=sector_names1, columns=sector_names1)
df_A_ridge = pd.DataFrame(A_ridge[0.5], index=sector_names1, columns=sector_names1)
df_A_true.to_csv(os.path.join(OUTPUT_DIR, 'A_true_part1.csv'), encoding='utf-8-sig')
df_A_est.to_csv(os.path.join(OUTPUT_DIR, 'A_est_LS_part1.csv'), encoding='utf-8-sig')
df_A_ridge.to_csv(os.path.join(OUTPUT_DIR, 'A_ridge_0.5_part1.csv'), encoding='utf-8-sig')

print("\n[Частина 1] Збережено матриці A та графіки для 2019–2024 у папку project_result")

# =============================================================================
# ЧАСТИНА 2. Симуляція Z_{t+1} = B*X_t - C*Z_t + S_t для 2021–2024 
# =============================================================================

print("\n=== ЧАСТИНА 2: 2021–2024, Z з шоком і без шоку ===\n")

years2 = np.array([2021, 2022, 2023, 2024])

A2 = np.array([[0.41, 0.12, 0.09],
               [0.18, 0.27, 0.11],
               [0.05, 0.10, 0.22]])

X2 = np.array([[950., 820., 480.],   # 2021
               [815., 710., 440.],   # 2022
               [845., 735., 455.],   # 2023
               [880., 760., 470.]])  # 2024

Z_2021 = np.array([246.0, 75.1, 20.4])
B2 = np.array([0.259, 0.0916, 0.0425])
C2 = np.array([0.05, 0.03, 0.08])
S2 = np.array([[0.0,  0.0,  0.0 ],
               [4.2,  3.8,  9.1 ],
               [7.3,  6.1, 14.4 ],
               [14.7, 12.3, 22.0]])

n2 = len(years2)
Z2_with_shock = np.zeros((n2, 3))
Z2_no_shock = np.zeros((n2, 3))

Z2_with_shock[0] = Z_2021.copy()
Z2_no_shock[0] = Z_2021.copy()

for t in range(n2 - 1):
    Z2_no_shock[t+1] = B2 * X2[t] - C2 * Z2_no_shock[t]
    Z2_with_shock[t+1] = B2 * X2[t] - C2 * Z2_with_shock[t] + S2[t+1]

df_Z2 = pd.DataFrame({
    'year': years2,
    'Z_E_no_shock': Z2_no_shock[:,0],
    'Z_M_no_shock': Z2_no_shock[:,1],
    'Z_A_no_shock': Z2_no_shock[:,2],
    'Z_E_with_shock': Z2_with_shock[:,0],
    'Z_M_with_shock': Z2_with_shock[:,1],
    'Z_A_with_shock': Z2_with_shock[:,2],
    'S_E': S2[:,0],
    'S_M': S2[:,1],
    'S_A': S2[:,2]
})

df_Z2['Total_no_shock'] = df_Z2[['Z_E_no_shock','Z_M_no_shock','Z_A_no_shock']].sum(axis=1)
df_Z2['Total_with_shock'] = df_Z2[['Z_E_with_shock','Z_M_with_shock','Z_A_with_shock']].sum(axis=1)
df_Z2['Total_S'] = df_Z2[['S_E','S_M','S_A']].sum(axis=1)

df_Z2.to_csv(os.path.join(OUTPUT_DIR, 'Z_trajectory_2021_2024.csv'), index=False, encoding='utf-8-sig')

plt.figure(figsize=(9,5))
plt.plot(years2, Z2_no_shock[:,0], marker='o', linestyle='--', label='Енергетика без шоку')
plt.plot(years2, Z2_with_shock[:,0], marker='o', linestyle='-', label='Енергетика з шоком')
plt.plot(years2, Z2_no_shock[:,1], marker='s', linestyle='--', label='Металургія без шоку')
plt.plot(years2, Z2_with_shock[:,1], marker='s', linestyle='-', label='Металургія з шоком')
plt.plot(years2, Z2_no_shock[:,2], marker='^', linestyle='--', label='Аграрія без шоку')
plt.plot(years2, Z2_with_shock[:,2], marker='^', linestyle='-', label='Аграрія з шоком')
plt.title('Динаміка викидів Z (2021-2024): сценарії без шоку і з шоком')
plt.xlabel('Рік'); plt.ylabel('Z (Mt CO2-eq)'); plt.grid(True)
plt.legend(loc='upper left', bbox_to_anchor=(1.02,1))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Z_timeseries_2021_2024.png'), dpi=150)
plt.show()

df_Z2.round(3).to_csv(os.path.join(OUTPUT_DIR, 'Z_trajectory_2021_2024_table.csv'), index=False, encoding='utf-8-sig')

print("Таблиця Z 2021–2024:")
print(df_Z2.round(3))
print("\n[Частина 2] Збережено: Z_trajectory_2021_2024.csv, Z_trajectory_2021_2024_table.csv, Z_timeseries_2021_2024.png")

# =============================================================================
# ЧАСТИНА 3. Аналіз чутливості до S(2024) (2021–2024, інша агрегація X)
# =============================================================================

print("\n=== ЧАСТИНА 3: Аналіз чутливості S(2024) ===\n")

years3 = np.array([2021, 2022, 2023, 2024])

# Базові X (агреговані)
X3 = np.array([[120., 90., 150.],  # 2021
               [110., 70., 130.],  # 2022
               [100., 75., 140.],  # 2023
               [105., 80., 155.]]) # 2024

B3 = np.array([0.259, 0.0916, 0.0425])
C3 = np.array([0.12, 0.10, 0.15])

S3_base = np.array([[0.0,  0.0,  0.0],
                    [4.2,  3.8,  9.1],
                    [7.3,  6.1, 14.4],
                    [14.7, 12.3, 22.0]])  # базовий розподіл 49 Мт

def simulate_with_S2024(total_S2024, distr=np.array([0.30, 0.25, 0.45])):
    S = S3_base.copy()
    S2024_vec = distr * total_S2024
    S[3] = S2024_vec
    n = len(years3)
    Z = np.zeros((n,3))
    Z[0] = np.array([180.0, 70.5, 91.0])  # базові Z2021
    for t in range(n-1):
        Z[t+1] = B3 * X3[t] - C3 * Z[t] + S[t+1]
    return Z, S

scenarios = {
    'Zero (0 Mt)': 0.0,
    'Optimistic (25 Mt)': 25.0,
    'Base (49 Mt)': 49.0,
    'Pessimistic (70 Mt)': 70.0
}

results = {}
for name, total in scenarios.items():
    Z, S = simulate_with_S2024(total)
    results[name] = {'Z': Z, 'S': S, 'total': total}

# Сумарні викиди за сценаріями
plt.figure(figsize=(8,5))
for name, data in results.items():
    total_Z = data['Z'].sum(axis=1)
    plt.plot(years3, total_Z, marker='o', label=f"{name} (S2024={data['total']} Mt)")
plt.title('Чутливість: сумарні викиди 2021–2024 для різних сценаріїв S(2024)')
plt.xlabel('Рік'); plt.ylabel('Сумарні викиди Z (Mt CO2-eq)'); plt.grid(True); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'sensitivity_total_scenarios.png'), dpi=160)
plt.show()

# По секторах
plt.figure(figsize=(10,6))
for i, sector in enumerate(['Енергетика','Металургія','Агро/Ліс']):
    plt.subplot(3,1,i+1)
    for name in ['Zero (0 Mt)','Base (49 Mt)','Pessimistic (70 Mt)']:
        Z = results[name]['Z']
        plt.plot(years3, Z[:,i], marker='o', label=name)
    plt.title(f'Викиди сектора {sector} у сценаріях')
    plt.ylabel('Z (Mt)'); plt.grid(True)
    if i == 2:
        plt.xlabel('Рік')
    plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'sensitivity_by_sector.png'), dpi=160)
plt.show()

# Чутливість Z_2024 до S(2024)
S_vals = np.linspace(0, 100, 21)
total_Z2024 = []
for s in S_vals:
    Z, _ = simulate_with_S2024(s)
    total_Z2024.append(Z.sum(axis=1)[-1])
total_Z2024 = np.array(total_Z2024)

plt.figure(figsize=(8,5))
plt.plot(S_vals, total_Z2024, marker='o')
plt.title('Чутливість загальних викидів 2024 до величини шоку S(2024)')
plt.xlabel('S(2024) сумарно, Mt CO2-eq'); plt.ylabel('Сумарні викиди у 2024, Mt CO2-eq')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'sensitivity_curve_S2024.png'), dpi=160)
plt.show()

# Збереження таблички результатів
rows = []
for name, data in results.items():
    Z = data['Z']
    for t, year in enumerate(years3):
        rows.append({
            'scenario': name,
            'year': int(year),
            'Z_E': float(Z[t,0]),
            'Z_T': float(Z[t,1]),
            'Z_P': float(Z[t,2]),
            'S_E': float(data['S'][t,0]),
            'S_T': float(data['S'][t,1]),
            'S_P': float(data['S'][t,2])
        })
df_results3 = pd.DataFrame(rows)
df_results3.to_csv(os.path.join(OUTPUT_DIR, 'sensitivity_results_2021_2024.csv'),
                   index=False, encoding='utf-8-sig')

print("Результати аналізу чутливості:")
print(df_results3.head(12))

print('\n[Частина 3] Збережено фігури та таблиці в папку project_result:')
print('- sensitivity_total_scenarios.png')
print('- sensitivity_by_sector.png')
print('- sensitivity_curve_S2024.png')
print('- sensitivity_results_2021_2024.csv')

print("\nГОТОВО")
