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

DATA_PATH = "/Users/serekparowka/Documents/Studia/I pracownia fizyczna/skoroszyty/T8_skoroszyt.xlsx"
SHEET_NAME = "Dane"
SKIPROWS = 1114
NROWS = 6547

USECOLS = [3, 5]   # pierwsza kolumna = x, druga = y
R2_MIN = 0.9997       # próg liniowości


def fmt(x):
    return f"{x:.6g}".replace(".", ",")


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 1.0

    return 1 - ss_res / ss_tot


def find_best_linear_range(x, y, r2_min=0.9997):
    n = len(x)

    Sx = np.concatenate([[0], np.cumsum(x)])
    Sy = np.concatenate([[0], np.cumsum(y)])
    Sxx = np.concatenate([[0], np.cumsum(x * x)])
    Syy = np.concatenate([[0], np.cumsum(y * y)])
    Sxy = np.concatenate([[0], np.cumsum(x * y)])

    def interval_stats(i, j):
        # przedział [i, j), czyli od i do j-1
        m = j - i

        sx = Sx[j] - Sx[i]
        sy = Sy[j] - Sy[i]
        sxx = Sxx[j] - Sxx[i]
        syy = Syy[j] - Syy[i]
        sxy = Sxy[j] - Sxy[i]

        x_mean = sx / m
        y_mean = sy / m

        Sxx_c = sxx - sx * sx / m
        Syy_c = syy - sy * sy / m
        Sxy_c = sxy - sx * sy / m

        if Sxx_c <= 0 or Syy_c <= 0:
            return None

        a = Sxy_c / Sxx_c
        b = y_mean - a * x_mean

        r2 = (Sxy_c ** 2) / (Sxx_c * Syy_c)

        return r2, a, b

    # szukamy od najdłuższych przedziałów,
    # więc pierwszy znaleziony będzie miał maksymalną długość
    for length in range(n, 2, -1):
        best_for_length = None

        for start in range(0, n - length + 1):
            end = start + length
            stats = interval_stats(start, end)

            if stats is None:
                continue

            r2, a, b = stats

            if r2 >= r2_min:
                if best_for_length is None or r2 > best_for_length["r2"]:
                    best_for_length = {
                        "start_index": start,
                        "end_index": end - 1,
                        "length": length,
                        "r2": r2,
                        "a": a,
                        "b": b,
                        "x_start": x[start],
                        "x_end": x[end - 1],
                    }

        if best_for_length is not None:
            return best_for_length

    return None


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
    
