#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 17:44:39 2026

@author: serekparowka
#with help from ChatGPT
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import locale

try:
    locale.setlocale(locale.LC_ALL, 'pl_PL.UTF-8')
except locale.Error:
    print("Nie udało się ustawić locale 'pl_PL.UTF-8'. Zostają ustawienia domyślne.")

DATA_PATH = "T9_skoroszyt.xlsx"
SHEET_NAME = "Arkusz1"
SKIPROWS = 15
NROWS = 125

USECOLS = [10, 11]   # pierwsza kolumna = x, druga = y
R2_MIN = 0.999       # próg liniowości


def fmt(x):
    return f"{x:.6g}".replace(".", ",")


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 1.0

    return 1 - ss_res / ss_tot


def find_best_linear_range(x, y, r2_min=0.999):
    best = None
    n = len(x)

    for start in range(n):
        for end in range(start + 3, n + 1):
            x_part = x[start:end]
            y_part = y[start:end]

            coeffs = np.polyfit(x_part, y_part, 1)
            y_fit = np.polyval(coeffs, x_part)

            r2 = r_squared(y_part, y_fit)
            length = end - start

            if r2 >= r2_min:
                if best is None or length > best["length"]:
                    best = {
                        "start_index": start,
                        "end_index": end - 1,
                        "length": length,
                        "r2": r2,
                        "a": coeffs[0],
                        "b": coeffs[1],
                        "x_start": x[start],
                        "x_end": x[end - 1],
                    }

    return best


df = pd.read_excel(
    DATA_PATH,
    sheet_name=SHEET_NAME,
    usecols=USECOLS,
    skiprows=SKIPROWS,
    nrows=NROWS
)

x_all = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
y_all = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(dtype=float)

mask = np.isfinite(x_all) & np.isfinite(y_all)

x = x_all[mask]
y = y_all[mask]

best = find_best_linear_range(x, y, r2_min=R2_MIN)

print("\n=== SZUKANIE NAJDŁUŻSZEGO ZAKRESU LINIOWEGO ===")
print(f"Próg R² >= {fmt(R2_MIN)}")

if best is None:
    print("Nie znaleziono żadnego zakresu spełniającego warunek.")
else:
    excel_start = best["start_index"] + SKIPROWS + 2
    excel_end = best["end_index"] + SKIPROWS + 2

    print("\nNajlepszy zakres:")
    print(f"Wiersz Excela start: {excel_start}")
    print(f"Wiersz Excela koniec: {excel_end}")
    print(f"Indeks tablicy start: {best['start_index']}")
    print(f"Indeks tablicy koniec: {best['end_index']}")
    print(f"Liczba punktów: {best['length']}")
    print(f"R² = {fmt(best['r2'])}")
    print(f"x start = {fmt(best['x_start'])}")
    print(f"x koniec = {fmt(best['x_end'])}")
    print(f"Dopasowanie: y = {fmt(best['a'])}x + {fmt(best['b'])}")
    
