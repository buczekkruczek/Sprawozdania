#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 17:25:12 2026

@author: serekparowka
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 16:28:23 2026

@author: serekparowka
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
from matplotlib.ticker import MultipleLocator
import locale
from itertools import combinations

try:
    locale.setlocale(locale.LC_ALL, 'pl_PL.UTF-8')
    plt.rcParams['axes.formatter.use_locale'] = True
except locale.Error:
    print("Nie udało się ustawić locale 'pl_PL.UTF-8'. Zostają ustawienia domyślne.")

DATA_PATH = "/Users/serekparowka/Documents/Studia/I pracownia fizyczna/skoroszyty/T8_skoroszyt.xlsx"
SHEET_NAME = "Dane"
SKIPROWS = 795
NROWS = 5751
OUTPUT_DIR = "/Users/serekparowka/Documents/Studia/I pracownia fizyczna/grafiki"
os.makedirs(OUTPUT_DIR, exist_ok=True)
USE_CUSTOM_SERIES = None
CONFIGS = [
    {
        "usecols": [3, 5],
        "title": r"Zlinearyzowana zależność $\Delta T$ [$^\circ$C]  od $\Delta E$ [J]",
        "filename": "T8_plot14.pdf",

        "series": [
            {
                "label": "Seria 1 - wszystkie punkty",
                "y_col_index": 1,
                "row_start": None,
                "row_end": None,
                "fit": True
            },
            
        ]
    },
]

FIT_MODE = "manual"
MANUAL_MODEL_ALL = "Liniowa"
MANUAL_MODELS_PER_SERIES = []

X_LABEL = r"Wartość $\Delta T$ [$^\circ$C]"
Y_LABEL = r"Wartość $\Delta E$ [J]"

SHOW_CLOSEST_POINTS = False
MAX_CLOSEST_POINTS_PER_PAIR = 2
DENSE_POINTS = 2000

def linear(x, a, b):
    return a * x + b

def linear_no_intercept(x, a):
    return a * x

def poly2(x, a, b, c):
    return a * x**2 + b * x + c

def poly3(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def poly4(x, a, b, c, d, e):
    return a * x**4 + b * x**3 + c * x**2 + d * x + e

def poly5(x, a, b, c, d, e, f):
    return a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f

def poly6(x, a, b, c, d, e, f, g):
    return a * x**6 + b * x**5 + c * x**4 + d * x**3 + e * x**2 + f * x + g

def exponential(x, a, b, c):
    return a * np.exp(b * x) + c

def logarithmic(x, a, b):
    return a * np.log(x) + b

def power_law(x, a, b):
    return a * x**b

def fmt(x):
    return f"{x:.5g}".replace(".", ",")

def format_signed(value):
    if value >= 0:
        return f"+ {fmt(value)}"
    return f"- {fmt(abs(value))}"

def polynomial_equation_from_coeffs(coeffs):
    degree = len(coeffs) - 1
    superscripts = {
        0: "",
        1: "x",
        2: "x²",
        3: "x³",
        4: "x⁴",
        5: "x⁵",
        6: "x⁶"
    }

    terms = []
    for i, c in enumerate(coeffs):
        power = degree - i

        if np.isclose(c, 0.0, atol=1e-14):
            continue

        var = superscripts.get(power, f"x^{power}")
        abs_c = abs(c)

        if power == 0:
            term_body = fmt(abs_c)
        elif np.isclose(abs_c, 1.0):
            term_body = var
        else:
            term_body = f"{fmt(abs_c)}{var}"

        if not terms:
            terms.append(f"-{term_body}" if c < 0 else term_body)
        else:
            terms.append(f"- {term_body}" if c < 0 else f"+ {term_body}")

    if not terms:
        return "y = 0"

    return "y = " + " ".join(terms)

models = [
    {
        "name": "Liniowa",
        "func": linear,
        "p0": [1.0, 0.0],
        "degree": 1,
        "equation": lambda p: polynomial_equation_from_coeffs(p)
    },
    {
        "name": "Liniowa bez wyrazu wolnego",
        "func": linear_no_intercept,
        "p0": [1.0],
        "equation": lambda p: f"y = {fmt(p[0])}x"
    },
    {
        "name": "Wielomian 2 stopnia",
        "func": poly2,
        "p0": [1.0, 1.0, 0.0],
        "degree": 2,
        "equation": lambda p: polynomial_equation_from_coeffs(p)
    },
    {
        "name": "Wielomian 3 stopnia",
        "func": poly3,
        "p0": [1.0, 1.0, 1.0, 0.0],
        "degree": 3,
        "equation": lambda p: polynomial_equation_from_coeffs(p)
    },
    {
        "name": "Wielomian 4 stopnia",
        "func": poly4,
        "p0": [1.0, 1.0, 1.0, 1.0, 0.0],
        "degree": 4,
        "equation": lambda p: polynomial_equation_from_coeffs(p)
    },
    {
        "name": "Wielomian 5 stopnia",
        "func": poly5,
        "p0": [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        "degree": 5,
        "equation": lambda p: polynomial_equation_from_coeffs(p)
    },
    {
        "name": "Wielomian 6 stopnia",
        "func": poly6,
        "p0": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        "degree": 6,
        "equation": lambda p: polynomial_equation_from_coeffs(p)
    },
    {
        "name": "Wykładniczy",
        "func": exponential,
        "p0": [1.0, 0.1, 0.0],
        "equation": lambda p: f"y = {fmt(p[0])}·exp({fmt(p[1])}x) {format_signed(p[2])}"
    },
    {
        "name": "Logarytmiczny",
        "func": logarithmic,
        "p0": [1.0, 0.0],
        "equation": lambda p: f"y = {fmt(p[0])}·ln(x) {format_signed(p[1])}"
    },
    {
        "name": "Potęgowy",
        "func": power_law,
        "p0": [1.0, 1.0],
        "equation": lambda p: f"y = {fmt(p[0])}·x^({fmt(p[1])})"
    },
]

def excel_rows_to_mask(x_all, row_start=None, row_end=None):
    n = len(x_all)
    data_indices = np.arange(n)
    excel_rows = data_indices + SKIPROWS + 2

    mask = np.ones(n, dtype=bool)

    if row_start is not None:
        mask &= excel_rows >= row_start

    if row_end is not None:
        mask &= excel_rows <= row_end

    return mask

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 1.0

    return 1 - ss_res / ss_tot

def r_squared_no_intercept(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum(y_true ** 2)

    if ss_tot == 0:
        return 1.0

    return 1 - ss_res / ss_tot

def fit_polynomial_excel_style(x, y, model):
    degree = model["degree"]

    if len(x) <= degree:
        raise ValueError(
            f"Za mało punktów do modelu '{model['name']}'. "
            f"Liczba punktów: {len(x)}, wymagane > {degree}."
        )

    coeffs = np.polyfit(x, y, degree)
    y_fit = np.polyval(coeffs, x)
    score = r_squared(y, y_fit)

    return {
        "name": model["name"],
        "func": lambda xx, *params: np.polyval(params, xx),
        "params": coeffs,
        "r2": score,
        "equation": model["equation"](coeffs)
    }

def fit_nonlinear_model(x, y, model):
    name = model["name"]
    func = model["func"]
    p0 = model["p0"]

    if name == "Logarytmiczny" and np.any(x <= 0):
        raise ValueError(f"Model '{name}' wymaga x > 0.")

    if name == "Potęgowy" and np.any(x <= 0):
        raise ValueError(f"Model '{name}' wymaga x > 0.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params, _ = curve_fit(func, x, y, p0=p0, maxfev=20000)

    y_fit = func(x, *params)
    
    print("MODEL:", name)
    print("R2 zwykłe:", r_squared(y, y_fit))
    print("R2 bez wyrazu wolnego:", r_squared_no_intercept(y, y_fit))

    if not np.all(np.isfinite(y_fit)):
        raise RuntimeError(f"Dopasowanie modelu '{name}' dało niefinitywne wartości.")
        
    if name == "Liniowa bez wyrazu wolnego":
        score = r_squared_no_intercept(y, y_fit)
    else:
        score = r_squared(y, y_fit)


    return {
        "name": name,
        "func": func,
        "params": params,
        "r2": score,
        "equation": model["equation"](params)
    }

def fit_model_object(x, y, model):
    if "degree" in model:
        return fit_polynomial_excel_style(x, y, model)

    return fit_nonlinear_model(x, y, model)

def fit_best_model(x, y):
    best_result = None

    for model in models:
        name = model["name"]
        print(f"Próbuję dopasować: {name}")

        if name == "Logarytmiczny" and np.any(x <= 0):
            continue

        if name == "Potęgowy" and np.any(x <= 0):
            continue

        if "degree" in model and len(x) <= model["degree"]:
            continue

        try:
            result = fit_model_object(x, y, model)

            if (best_result is None) or (result["r2"] > best_result["r2"]):
                best_result = result

        except Exception:
            continue

    return best_result

def fit_selected_model(x, y, selected_model_name):
    selected_model = None

    for model in models:
        if model["name"] == selected_model_name:
            selected_model = model
            break

    if selected_model is None:
        available = [m["name"] for m in models]
        raise ValueError(
            f"Nie znaleziono modelu '{selected_model_name}'. "
            f"Dostępne modele: {available}"
        )

    return fit_model_object(x, y, selected_model)

def fit_model_by_mode(x, y, series_index=None):
    if FIT_MODE == "auto":
        return fit_best_model(x, y)

    if FIT_MODE == "manual":
        if MANUAL_MODELS_PER_SERIES:
            if series_index is None or series_index >= len(MANUAL_MODELS_PER_SERIES):
                raise ValueError(
                    "Brakuje nazwy modelu dla tej serii w MANUAL_MODELS_PER_SERIES."
                )

            chosen_model = MANUAL_MODELS_PER_SERIES[series_index]
        else:
            chosen_model = MANUAL_MODEL_ALL

        return fit_selected_model(x, y, chosen_model)

    raise ValueError("FIT_MODE musi być równe 'auto' albo 'manual'.")

def find_closest_points(series_a, series_b, n_points=2, n_dense=2000):
    x_min = max(np.min(series_a["x"]), np.min(series_b["x"]))
    x_max = min(np.max(series_a["x"]), np.max(series_b["x"]))

    if x_min >= x_max:
        return np.array([])

    x_dense = np.linspace(x_min, x_max, n_dense)

    f_a = series_a["best_model"]["func"](x_dense, *series_a["best_model"]["params"])
    f_b = series_b["best_model"]["func"](x_dense, *series_b["best_model"]["params"])
    diff = np.abs(f_a - f_b)

    min_indices = []

    for i in range(1, len(diff) - 1):
        if diff[i] < diff[i - 1] and diff[i] < diff[i + 1]:
            min_indices.append(i)

    if len(min_indices) == 0:
        min_indices = [int(np.argmin(diff))]

    min_indices = sorted(min_indices, key=lambda i: diff[i])[:n_points]

    return x_dense[min_indices]

for config in CONFIGS:
    USECOLS = config["usecols"]
    SERIES_LABELS = config.get("series_labels", [])
    TITLE = config["title"]
    FILENAME = config["filename"]

    df = pd.read_excel(
        DATA_PATH,
        sheet_name=SHEET_NAME,
        usecols=USECOLS,
        skiprows=SKIPROWS,
        nrows=NROWS
    )

    if df.shape[1] < 2:
        raise ValueError(
            f"Konfiguracja {FILENAME}: musisz wczytać co najmniej 2 kolumny: jedną dla x i co najmniej jedną serię y."
        )

    x_all = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)

    y_columns = [
        pd.to_numeric(df.iloc[:, i], errors="coerce").to_numpy(dtype=float)
        for i in range(1, df.shape[1])
    ]

    n_series = len(y_columns)

    if len(SERIES_LABELS) < n_series:
        SERIES_LABELS = SERIES_LABELS + [
            f"Seria {i+1}" for i in range(len(SERIES_LABELS), n_series)
        ]

    elif len(SERIES_LABELS) > n_series:
        SERIES_LABELS = SERIES_LABELS[:n_series]

    series_data = []

    if USE_CUSTOM_SERIES and "series" in config:
        for s in config["series"]:
            y = y_columns[s["y_col_index"] - 1]

            mask = np.isfinite(x_all) & np.isfinite(y)

            row_mask = excel_rows_to_mask(
                x_all,
                row_start=s.get("row_start"),
                row_end=s.get("row_end")
            )

            mask &= row_mask

            series_data.append({
                "label": s["label"],
                "x": x_all[mask],
                "y": y[mask],
                "fit": s.get("fit", True)
            })

    else:
        for i, y in enumerate(y_columns):
            mask = np.isfinite(x_all) & np.isfinite(y)

            series_data.append({
                "label": SERIES_LABELS[i],
                "x": x_all[mask],
                "y": y[mask],
                "fit": True
            })

    if not series_data:
        raise RuntimeError(
            f"Konfiguracja {FILENAME}: nie znaleziono żadnej poprawnej serii danych."
        )

    for i, series in enumerate(series_data):
        if not series.get("fit", True):
            series["best_model"] = None
            continue

        best = fit_model_by_mode(series["x"], series["y"], series_index=i)

        if best is None:
            raise RuntimeError(
                f"Nie udało się dopasować żadnego modelu do serii '{series['label']}' w pliku {FILENAME}."
            )

        series["best_model"] = best

    print(f"\n===== {FILENAME} =====")

    for series in series_data:
        best = series["best_model"]
        print(f"\n{series['label']}:")

        if best is None:
            print("Dopasowanie: pominięte")
            continue

        print(f"Najlepszy model: {best['name']}")
        print(f"Równanie: {best['equation']}")
        print(f"R² = {fmt(best['r2'])}")

    print("Kształt wczytanej tabeli:", df.shape)

    plt.figure(figsize=(50, 40))
    ax = plt.gca()
    ax.tick_params(axis="both", labelsize=60)

    for i, series in enumerate(series_data):
        best = series["best_model"]

        if "wszystkie punkty" in series["label"]:
            color = "black"
            linestyle = "None"
            linewidth = 2.5
            zorder_points = 1
            zorder_line = 3
        
        elif "zakres roboczy" in series["label"]:
            color = "mediumorchid"
            linestyle = "-"
            linewidth = 20
            zorder_points = 2
            zorder_line = 4
        
        else:
            # fallback – ręcznie ustawiasz dla kolejnych serii
            if i == 0:
                color = "orangered"
            elif i == 1:
                color = "red"
            elif i == 2:
                color = "blue"
            elif i == 3:
                color = "green"
            else:
                color = "black"  # default jak przekroczysz zakres
        
            linestyle = "-"
            linewidth = 15
            zorder_points = 2
            zorder_line = 3

        plt.scatter(
            series["x"],
            series["y"],
            s=100,
            marker="o",
            facecolors=  "skyblue",
            edgecolors="steelblue",
            linewidths=0.5,
            zorder=zorder_points,
            label=f"{series['label']} - pomiary"
        )

        if series.get("fit", True):
            x_smooth_local = np.linspace(
                np.min(series["x"]),
                np.max(series["x"]),
                500
            )

            # obrys
            plt.plot(
                x_smooth_local,
                best["func"](x_smooth_local, *best["params"]),
                color="black",
                linewidth=linewidth + 2,
                zorder=zorder_line
                )

            # właściwa linia
            plt.plot(
                x_smooth_local,
                best["func"](x_smooth_local, *best["params"]),
                color=color,
                linewidth=linewidth,
                zorder=zorder_line + 1,
                label=f"{series['label']} - dopasowanie: {best['equation']}, R² = {fmt(best['r2'])}"
                )

    for series in series_data:
        if "zakres roboczy" in series["label"]:
            x_start = np.min(series["x"])
            x_end = np.max(series["x"])

            plt.axvline(
                x=x_start,
                linestyle=":",
                linewidth=2,
                color="black",
                alpha=0.85,
                label=f"Początek zakresu roboczego, x ≈ {fmt(x_start)}"
            )

            plt.axvline(
                x=x_end,
                linestyle=":",
                linewidth=2,
                color="black",
                alpha=0.85,
                label=f"Koniec zakresu roboczego, x ≈ {fmt(x_end)}"
            )

    if SHOW_CLOSEST_POINTS and sum(s.get("fit", True) for s in series_data) >= 2:
        for series_a, series_b in combinations(series_data, 2):
            if series_a["best_model"] is None or series_b["best_model"] is None:
                continue

            x_close = find_closest_points(
                series_a,
                series_b,
                n_points=MAX_CLOSEST_POINTS_PER_PAIR,
                n_dense=DENSE_POINTS
            )

            for x0 in x_close:
                plt.axvline(
                    x=x0,
                    linestyle=":",
                    linewidth=1.5,
                    color="black",
                    alpha=0.7,
                    label=f"Zbliżenie: {series_a['label']} / {series_b['label']}, x ≈ {fmt(x0)}"
                )

    plt.xlabel(X_LABEL, fontsize=70)
    plt.ylabel(Y_LABEL, fontsize=70)
    plt.title(TITLE, fontsize=80)

    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    
    ax.yaxis.set_major_locator(MultipleLocator(50))
    ax.yaxis.set_minor_locator(MultipleLocator(10))

    ax.grid(which='major', linestyle='-', alpha=0.5)
    ax.grid(which='minor', linestyle=':', alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys(), fontsize=60, markerscale=5)

    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, FILENAME)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Zapisano wykres: {output_path}")
    
