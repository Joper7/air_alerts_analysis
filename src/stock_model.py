"""
Модель запасу дронів з прихованими параметрами (продовження src/launches_eda.py):
  S_k = S_{k-1} + P_k − W_k,  P_k = p0·(1+g)^k,  S_k ≥ 0 (фізичне обмеження)
  W_k — запуски за тиждень (ранкові підсумки @kpszsu), p0 / g / S_0 — невідомі, підбираються по сітці.
Прогноз W_{k+1} = a + b·S_k + c·W_k. Перевірка walk-forward: параметри — лише на тижнях < k.
Порівняння: AR(1) без запасу, «як минулого тижня», середнє 4 тижнів.

Запуск: python -m src.stock_model
"""
from src.launches_eda import parse_launches
s = parse_launches(pd.read_parquet("data/raw/telegram/kpszsu.parquet"))
s = s[s.index >= "2026-01-05"]                         # з понеділка
W = s.resample("W-SUN").sum(); W = W[W.index <= "2026-10-11"].to_numpy(float)   # тижні запусків
n = len(W); print("тижнів:", n, "| запусків/тиждень: перші 4 —", W[:4].mean().round(), ", останні 4 —", W[-4:].mean().round())

def stock_path(p0, g, s0, K):
    """Запас на КІНЕЦЬ тижня k: S_k = S_{k-1} + p0·(1+g)^k − W_k."""
    P = p0 * (1 + g) ** np.arange(K)
    return s0 + np.cumsum(P - W[:K]), P

base = W[:4].mean()
grid = list(itertools.product(base * np.array([0.9, 1.0, 1.1, 1.2, 1.35, 1.5]),   # виробництво/тиждень на старті
                              [0, 0.005, 0.01, 0.015, 0.02, 0.03],             # приріст виробництва за тиждень
                              base * np.array([0, 1, 2, 4, 8])))               # початковий запас (у тижнях виробництва)

def fit(K):
    """Найкращі (p0,g,s0) на тижнях < K: регресія W_{k+1} ~ a + b·S_k + c·W_k; запас ≥ 0 обов'язково."""
    best = None
    for p0, g, s0 in grid:
        S, _ = stock_path(p0, g, s0, K)
        if S.min() < 0:                                  # фізично неможливо → параметри відкидаємо
            continue
        X = np.c_[np.ones(K - 1), S[:-1], W[:K - 1]]; y = W[1:K]
        coef, res, *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = ((X @ coef - y) ** 2).sum()
        if best is None or sse < best[0]:
            best = (sse, p0, g, s0, coef)
    return best

rows = []
for K in range(16, n):                                   # прогноз тижня K за даними тижнів < K
    b = fit(K)
    S, _ = stock_path(b[1], b[2], b[3], K)
    pred_stock = b[4] @ [1, S[-1], W[K - 1]]
    Xa = np.c_[np.ones(K - 1), W[:K - 1]]; ca = np.linalg.lstsq(Xa, W[1:K], rcond=None)[0]   # AR(1) без запасу
    rows.append({"k": K, "y": W[K], "stock": pred_stock, "ar1": ca @ [1, W[K - 1]],
                 "last": W[K - 1], "mean4": W[K - 4:K].mean(), "b_stock": b[4][1], "g": b[2]})
R = pd.DataFrame(rows)
mae = {m: (R[m] - R.y).abs().mean() for m in ["stock", "ar1", "last", "mean4"]}
print("MAE прогнозу запусків на наступний тиждень (out-of-sample, тижнів:", len(R), "):")
for m, v in mae.items(): print(f"  {m:6s} {v:7.1f}")
d = (R.stock - R.y).abs() - (R.mean4 - R.y).abs()
rng = np.random.default_rng(0); bs = [rng.choice(d, len(d)).mean() for _ in range(5000)]
print(f"stock − mean4: ΔMAE {d.mean():+.1f}, 95% CI [{np.percentile(bs,2.5):+.1f}; {np.percentile(bs,97.5):+.1f}]  (<0 — запас кращий)")
print("знак коефіцієнта при запасі (мав би бути >0: більше запасу → більше запусків):",
      (R.b_stock > 0).mean().round(2), "частка тижнів; обраний приріст виробництва g:", R.g.value_counts().to_dict())
