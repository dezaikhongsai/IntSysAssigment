# === REQUIREMENT 14 — ONE-CELL, SELF-CONTAINED (JUPYTER) ===
# Task: Multiple Linear Regression (MLR) to predict house prices using the Boston Housing dataset.
# What this cell guarantees:
#  - Tries to load Boston dataset from scikit-learn (deprecated in newer versions, may still exist).
#  - If unavailable, will try to read /mnt/data/boston.csv with columns:
#       CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT, MEDV
#    (MEDV is the target price in $1000s).
#  - If neither is available, it uses a *synthetic Boston-like* dataset (same columns) so the pipeline runs.
#  - Performs train/test split, standardization, fits LinearRegression, reports metrics (R², RMSE, MAE),
#    shows sorted coefficients, and produces ONE chart (Predicted vs Actual on test set).
#  - Saves artifacts for your report.
#
# Dependencies used: numpy, pandas, matplotlib, scikit-learn (if present; if not, a numpy fallback is used).

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

source = ""
X = y = feature_names = None

# ---------- 0) Try to load Boston from scikit-learn ----------
have_sklearn = True
try:
    from sklearn.datasets import load_boston  # may be removed in recent versions
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
except Exception:
    have_sklearn = False

if have_sklearn:
    try:
        boston = load_boston()
        X = pd.DataFrame(boston.data, columns=boston.feature_names)
        y = pd.Series(boston.target, name="MEDV")
        feature_names = list(X.columns)
        source = "sklearn.load_boston()"
    except Exception:
        pass

# ---------- 1) If not loaded, try local CSV ----------
if X is None or y is None:
    csv_path = "/mnt/data/boston.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        required_cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")
        X = df[required_cols[:-1]].copy()
        y = df['MEDV'].copy()
        feature_names = required_cols[:-1]
        source = "local_csv:/mnt/data/boston.csv"

# ---------- 2) If still not available, create a synthetic Boston-like dataset ----------
if X is None or y is None:
    rng = np.random.default_rng(123)
    n = 400
    CRIM = rng.gamma(shape=2.0, scale=2.0, size=n) / 10
    ZN = rng.integers(0, 100, n).astype(float)
    INDUS = rng.uniform(1, 27, n)
    CHAS = rng.integers(0, 2, n).astype(float)
    NOX = rng.uniform(0.3, 0.9, n)
    RM = rng.normal(6.2, 0.7, n)
    AGE = np.clip(rng.normal(70, 20, n), 1, 100)
    DIS = rng.uniform(1, 12, n)
    RAD = rng.integers(1, 24, n).astype(float)
    TAX = rng.normal(400, 100, n)
    PTRATIO = rng.uniform(12, 22, n)
    B = np.clip(rng.normal(350, 50, n), 200, 400)
    LSTAT = np.clip(rng.normal(12, 7, n), 1, 40)

    X = pd.DataFrame({
        'CRIM': CRIM, 'ZN': ZN, 'INDUS': INDUS, 'CHAS': CHAS, 'NOX': NOX, 'RM': RM,
        'AGE': AGE, 'DIS': DIS, 'RAD': RAD, 'TAX': TAX, 'PTRATIO': PTRATIO, 'B': B, 'LSTAT': LSTAT
    })
    # Ground-truth synthetic relation (RM positive, LSTAT negative, NOX negative, CHAS slight positive, etc.)
    y = ( 5.0
          - 1.5*NOX
          + 4.0*RM
          - 0.4*LSTAT
          + 0.3*CHAS
          - 0.01*TAX
          - 0.02*CRIM
          - 0.1*INDUS
          + 0.05*ZN
          + 0.03*B/100
          - 0.05*PTRATIO
          + 0.02*DIS
          - 0.005*AGE
          + rng.normal(0, 1.5, n)
        )
    y = pd.Series(np.clip(y, 5, 50), name="MEDV")  # price in $1000s
    feature_names = list(X.columns)
    source = "synthetic_boston_like"

# ---------- 3) Train/Test split, scaling ----------
if have_sklearn:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)

    r2 = float(r2_score(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))

    coefs = pd.DataFrame({
        "feature": feature_names,
        "coefficient": model.coef_
    }).sort_values("coefficient", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    intercept = float(model.intercept_)
else:
    # Numpy fallback (no sklearn): standardize features, closed-form least squares
    X_values = X.values
    y_values = y.values
    X_mean, X_std = X_values.mean(axis=0), X_values.std(axis=0)
    X_std_adj = np.where(X_std == 0, 1, X_std)
    Xz = (X_values - X_mean) / X_std_adj
    # Add bias column
    X_design = np.c_[np.ones(len(Xz)), Xz]
    beta = np.linalg.pinv(X_design.T @ X_design) @ X_design.T @ y_values
    intercept = float(beta[0]); betas = beta[1:]
    # Simple split for metrics
    n = len(y_values); idx = np.arange(n)
    rng = np.random.default_rng(42); rng.shuffle(idx)
    test_size = int(0.2*n); test_idx = idx[:test_size]; train_idx = idx[test_size:]
    Xz_train, y_train = Xz[train_idx], y_values[train_idx]
    Xz_test,  y_test  = Xz[test_idx],  y_values[test_idx]
    y_pred = (np.c_[np.ones(len(Xz_test)), Xz_test] @ beta)

    ss_res = np.sum((y_test - y_pred)**2)
    ss_tot = np.sum((y_test - y_test.mean())**2)
    r2 = float(1 - ss_res/ss_tot)
    rmse = float(np.sqrt(np.mean((y_test - y_pred)**2)))
    mae = float(np.mean(np.abs(y_test - y_pred)))

    coefs = pd.DataFrame({
        "feature": feature_names,
        "coefficient": betas
    }).sort_values("coefficient", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)

# ---------- 4) Display results ----------
metrics_df = pd.DataFrame([{
    "source": source,
    "R2_test": round(r2, 6),
    "RMSE_test": round(rmse, 6),
    "MAE_test": round(mae, 6),
    "intercept": round(intercept, 6),
}])

try:
    from caas_jupyter_tools import display_dataframe_to_user
    display_dataframe_to_user("Requirement 14 - Test Metrics", metrics_df)
    display_dataframe_to_user("Requirement 14 - Coefficients (sorted by |value|)", coefs)
except Exception:
    print("\nTest Metrics:\n", metrics_df.to_string(index=False))
    print("\nCoefficients (sorted by |value|):\n", coefs.to_string(index=False))

# ---------- 5) One chart: Predicted vs Actual (test set) ----------
plt.figure(figsize=(7,6))
plt.scatter(y_test, y_pred, alpha=0.8, label="Pred vs Actual")  # default colors only
mn, mx = float(min(y_test.min(), y_pred.min())), float(max(y_test.max(), y_pred.max()))
line = np.linspace(mn, mx, 100)
plt.plot(line, line, linestyle="--", label="Ideal y=x")  # reference line

plt.title("Boston Housing: Predicted vs Actual (Test Set)")
plt.xlabel("Actual MEDV ($1000s)")
plt.ylabel("Predicted MEDV ($1000s)")
plt.legend()

save_dir = "/mnt/data"
os.makedirs(save_dir, exist_ok=True)
fig_path = os.path.join(save_dir, "req14_boston_pred_vs_actual.png")
pred_csv = os.path.join(save_dir, "req14_boston_test_predictions.csv")
coef_csv = os.path.join(save_dir, "req14_boston_coefficients.csv")
plt.savefig(fig_path, bbox_inches="tight")
plt.show()

# Save artifacts
pd.DataFrame({"y_test": y_test, "y_pred": y_pred}).to_csv(pred_csv, index=False)
coefs.to_csv(coef_csv, index=False)

(fig_path, pred_csv, coef_csv, source)
