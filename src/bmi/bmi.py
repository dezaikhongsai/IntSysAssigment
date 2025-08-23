# === BMI - ONE-CELL, SELF-CONTAINED ===
import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Iterable, List

# ----- Types -----
Number = Union[int, float, np.number]

# ----- Helpers & Core -----
def _ensure_number(x, name):
    if x is None or isinstance(x, bool):
        raise ValueError(f"{name} không hợp lệ (None/bool).")
    xf = float(x)
    if math.isnan(xf) or math.isinf(xf):
        raise ValueError(f"{name} không hợp lệ (NaN/Inf).")

def calc_bmi(weight_kg: Number, height_m: Number) -> float:
    _ensure_number(weight_kg, "weight_kg")
    _ensure_number(height_m, "height_m")
    if height_m <= 0:
        raise ValueError("height_m phải > 0")
    return float(weight_kg) / (float(height_m) ** 2)

def classify_bmi(bmi: Number) -> str:
    b = float(bmi)
    if b < 18.5:
        return "Underweight"
    if b > 25:
        return "Overweight"
    return "Normal"

# ----- Vectorized -----
def calc_bmi_array(weights_kg: Iterable[Number], heights_m: Iterable[Number]) -> np.ndarray:
    w = np.asarray(list(weights_kg), dtype=float)
    h = np.asarray(list(heights_m), dtype=float)
    if w.shape != h.shape:
        raise ValueError("weights_kg và heights_m phải có cùng độ dài")
    if np.any(h <= 0):
        raise ValueError("Tất cả height_m phải > 0")
    if np.any(~np.isfinite(w)) or np.any(~np.isfinite(h)):
        raise ValueError("Dữ liệu phải là số hữu hạn")
    return w / (h ** 2)

def classify_bmi_array(bmis: Iterable[Number]) -> List[str]:
    return [classify_bmi(b) for b in bmis]

# ----- Build result table -----
def bmi_table(names, heights_m, weights_kg, round_ndigits: int = 2) -> pd.DataFrame:
    bmis = calc_bmi_array(weights_kg, heights_m)
    classes = classify_bmi_array(bmis)
    return pd.DataFrame({
        "name": list(names),
        "height_m": list(heights_m),
        "weight_kg": list(weights_kg),
        "BMI": np.round(bmis, round_ndigits),
        "class": classes
    })

# ----- Plot -----
def plot_bmi_bar(df: pd.DataFrame, save_path: str | None = None):
    plt.figure(figsize=(6,4))
    plt.bar(df["name"], df["BMI"])
    plt.axhline(18.5, linestyle="--", label="18.5")
    plt.axhline(25, linestyle="--", label="25")
    plt.title("BMI Chart")
    plt.xlabel("Name"); plt.ylabel("BMI"); plt.legend()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

# ====== DEMO (đúng dữ liệu đề bài) ======
names = np.array(['Ann','Joe','Mark'])
heights = np.array([1.5, 1.78, 1.6])
weights = np.array([65, 46, 59])

df = bmi_table(names, heights, weights, round_ndigits=2)

# Hiển thị bảng
try:
    from IPython.display import display
    display(df)
except:
    print(df.to_string(index=False))

# In từng người
for n,h,w in zip(names, heights, weights):
    bmi = calc_bmi(w,h)
    print(f"{n}: BMI={bmi:.2f} → {classify_bmi(bmi)}")

# Vẽ & lưu biểu đồ
plot_bmi_bar(df, save_path="../figs/bmi_chart.png")
