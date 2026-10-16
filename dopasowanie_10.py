#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 18:20:16 2026

@author: serekparowka
"""


"""Dopasowywanie funkcji i wykresy z opcjonalnymi słupkami niepewności."""
#Skrypt powstały i zmieniający się na przestrzeni ostatniego roku, wzbogacający się o funkcję, stworzony dla mnie i moich znajomych 
#z pomocą AI - wstyd generalnie, ale "druga głowa" jest zawsze w cenie.

import os
import warnings
from itertools import combinations

import locale
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from scipy.optimize import curve_fit

try:
    locale.setlocale(locale.LC_ALL, 'pl_PL.UTF-8')
    plt.rcParams['axes.formatter.use_locale'] = True
except locale.Error:
    print("Nie udało się ustawić locale 'pl_PL.UTF-8'. Zostają ustawienia domyślne.")

DATA_PATH = ""
SHEET_NAME = "Arkusz1"
SKIPROWS = 3
NROWS = 13
OUTPUT_DIR = ""
os.makedirs(OUTPUT_DIR, exist_ok=True)

USE_CUSTOM_SERIES = None  


CONFIGS = [
    {
        "usecols": [3, 4],
        "title": r"Zlinearyzowana zależność $U_y$ [V] od $U$ [V]",
        "filename": "spr1-7.pdf",
        "errors": {
            "show": True,
            "x": {"mode": "resolution", "step": 0.01},
            "y": {"mode": "resolution", "step": 0.01},
            "style": {
                "color": "black",
                "linewidth": 3,
                "capsize": 12,
                "capthick": 3,
            },
        },
        "series": [
            {
                "label": "Seria 1 - wszystkie punkty",
                "y_col_index": 1,
                "row_start": None,
                "row_end": None,
                "fit": True,
                
            },
        ],
    },
]

FIT_MODE = "manual"  # manual lub auto
MANUAL_MODEL_ALL = "Liniowa"
MANUAL_MODELS_PER_SERIES = []
X_LABEL = r"$U$ [V]"
Y_LABEL = r"$U_y$ [V]"
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
    superscripts = {0: "", 1: "x", 2: "x²", 3: "x³", 4: "x⁴", 5: "x⁵", 6: "x⁶"}
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
    return "y = " + " ".join(terms) if terms else "y = 0"


models = [
    {"name": "Liniowa", "func": linear, "p0": [1.0, 0.0], "degree": 1,
     "equation": lambda p: polynomial_equation_from_coeffs(p)},
    {"name": "Liniowa bez wyrazu wolnego", "func": linear_no_intercept,
     "p0": [1.0], "equation": lambda p: f"y = {fmt(p[0])}x"},
    {"name": "Wielomian 2 stopnia", "func": poly2, "p0": [1.0, 1.0, 0.0],
     "degree": 2, "equation": lambda p: polynomial_equation_from_coeffs(p)},
    {"name": "Wielomian 3 stopnia", "func": poly3, "p0": [1.0, 1.0, 1.0, 0.0],
     "degree": 3, "equation": lambda p: polynomial_equation_from_coeffs(p)},
    {"name": "Wielomian 4 stopnia", "func": poly4,
     "p0": [1.0, 1.0, 1.0, 1.0, 0.0], "degree": 4,
     "equation": lambda p: polynomial_equation_from_coeffs(p)},
    {"name": "Wielomian 5 stopnia", "func": poly5,
     "p0": [1.0, 1.0, 1.0, 1.0, 1.0, 0.0], "degree": 5,
     "equation": lambda p: polynomial_equation_from_coeffs(p)},
    {"name": "Wielomian 6 stopnia", "func": poly6,
     "p0": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0], "degree": 6,
     "equation": lambda p: polynomial_equation_from_coeffs(p)},
    {"name": "Wykładniczy", "func": exponential, "p0": [1.0, 0.1, 0.0],
     "equation": lambda p: f"y = {fmt(p[0])}·exp({fmt(p[1])}x) {format_signed(p[2])}"},
    {"name": "Logarytmiczny", "func": logarithmic, "p0": [1.0, 0.0],
     "equation": lambda p: f"y = {fmt(p[0])}·ln(x) {format_signed(p[1])}"},
    {"name": "Potęgowy", "func": power_law, "p0": [1.0, 1.0],
     "equation": lambda p: f"y = {fmt(p[0])}·x^({fmt(p[1])})"},
]


def excel_rows_to_mask(x_all, row_start=None, row_end=None):
    excel_rows = np.arange(len(x_all)) + SKIPROWS + 2
    mask = np.ones(len(x_all), dtype=bool)
    if row_start is not None:
        mask &= excel_rows >= row_start
    if row_end is not None:
        mask &= excel_rows <= row_end
    return mask


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1.0 if ss_tot == 0 else 1 - ss_res / ss_tot


def r_squared_no_intercept(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum(y_true**2)
    return 1.0 if ss_tot == 0 else 1 - ss_res / ss_tot


def fit_polynomial_excel_style(x, y, model):
    degree = model["degree"]
    if len(x) <= degree:
        raise ValueError(f"Za mało punktów do modelu '{model['name']}'. ")
    coeffs = np.polyfit(x, y, degree)
    score = r_squared(y, np.polyval(coeffs, x))
    return {"name": model["name"],
            "func": lambda xx, *params: np.polyval(params, xx),
            "params": coeffs, "r2": score,
            "equation": model["equation"](coeffs)}


def fit_nonlinear_model(x, y, model):
    name, func, p0 = model["name"], model["func"], model["p0"]
    if name in ("Logarytmiczny", "Potęgowy") and np.any(x <= 0):
        raise ValueError(f"Model '{name}' wymaga x > 0.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params, _ = curve_fit(func, x, y, p0=p0, maxfev=20000)
    y_fit = func(x, *params)
    if not np.all(np.isfinite(y_fit)):
        raise RuntimeError(f"Model '{name}' zwrócił wartości niefinitywne.")
    print("MODEL:", name)
    print("R2 zwykłe:", r_squared(y, y_fit))
    print("R2 bez wyrazu wolnego:", r_squared_no_intercept(y, y_fit))
    score = (r_squared_no_intercept(y, y_fit)
             if name == "Liniowa bez wyrazu wolnego" else r_squared(y, y_fit))
    return {"name": name, "func": func, "params": params,
            "r2": score, "equation": model["equation"](params)}


def fit_model_object(x, y, model):
    if "degree" in model:
        return fit_polynomial_excel_style(x, y, model)
    return fit_nonlinear_model(x, y, model)


def fit_best_model(x, y):
    best_result = None
    for model in models:
        print(f"Próbuję dopasować: {model['name']}")
        if model["name"] in ("Logarytmiczny", "Potęgowy") and np.any(x <= 0):
            continue
        if "degree" in model and len(x) <= model["degree"]:
            continue
        try:
            result = fit_model_object(x, y, model)
            if best_result is None or result["r2"] > best_result["r2"]:
                best_result = result
        except Exception:
            continue
    return best_result


def fit_selected_model(x, y, selected_model_name):
    selected = next((m for m in models if m["name"] == selected_model_name), None)
    if selected is None:
        raise ValueError(f"Nieznany model '{selected_model_name}'. Dostępne: "
                         f"{[m['name'] for m in models]}")
    return fit_model_object(x, y, selected)


def fit_model_by_mode(x, y, series_index=None):
    if FIT_MODE == "auto":
        return fit_best_model(x, y)
    if FIT_MODE == "manual":
        if MANUAL_MODELS_PER_SERIES:
            if series_index is None or series_index >= len(MANUAL_MODELS_PER_SERIES):
                raise ValueError("Brakuje nazwy modelu w MANUAL_MODELS_PER_SERIES.")
            chosen = MANUAL_MODELS_PER_SERIES[series_index]
        else:
            chosen = MANUAL_MODEL_ALL
        return fit_selected_model(x, y, chosen)
    raise ValueError("FIT_MODE musi być równe 'auto' albo 'manual'.")


def find_closest_points(series_a, series_b, n_points=2, n_dense=2000):
    x_min = max(np.min(series_a["x"]), np.min(series_b["x"]))
    x_max = min(np.max(series_a["x"]), np.max(series_b["x"]))
    if x_min >= x_max:
        return np.array([])
    x_dense = np.linspace(x_min, x_max, n_dense)
    f_a = series_a["best_model"]["func"](
        x_dense, *series_a["best_model"]["params"])
    f_b = series_b["best_model"]["func"](
        x_dense, *series_b["best_model"]["params"])
    diff = np.abs(f_a - f_b)
    minima = [i for i in range(1, len(diff) - 1)
              if diff[i] < diff[i - 1] and diff[i] < diff[i + 1]]
    if not minima:
        minima = [int(np.argmin(diff))]
    return x_dense[sorted(minima, key=lambda i: diff[i])[:n_points]]


def read_numeric_excel_cols(columns, cache):
    """Czyta wskazane kolumny (indeksy 0-based), zachowując kolejność wierszy."""
    columns = tuple(columns)
    if columns not in cache:
        table = pd.read_excel(
            DATA_PATH, sheet_name=SHEET_NAME, usecols=list(columns),
            skiprows=SKIPROWS, nrows=NROWS)
        sorted_columns = sorted(columns)
        values = table.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        cache[columns] = values[:, [sorted_columns.index(c) for c in columns]]
    return cache[columns]


def get_uncertainties(settings, values, mask, cache):
    """Zwraca u dla punktów serii; nie przeprowadza regresji ważonej."""
    settings = settings or {"mode": "none"}
    mode = settings.get("mode", "none")
    if mode == "none":
        return None
    if mode == "manual":
        raw = np.asarray(settings["value"], dtype=float)
        if raw.ndim == 0:
            result = np.full(len(values), float(raw))
        elif raw.ndim == 1 and len(raw) == len(mask):
            result = raw[mask]
        elif raw.ndim == 1 and len(raw) == len(values):
            result = raw.copy()
        else:
            raise ValueError("'manual': liczba niepewności nie pasuje do punktów.")
    elif mode == "excel":
        result = read_numeric_excel_cols([settings["col"]], cache)[:, 0][mask]
    elif mode == "resolution":
        step = float(settings["step"])
        result = np.full(len(values), step / np.sqrt(12))
    elif mode == "relative":
        result = np.abs(values) * float(settings["percentage"]) / 100
    elif mode == "instrument":
        percentage = float(settings["percentage"])
        digits = float(settings.get("digits", 0))
        resolution = float(settings.get("resolution", 0))
        delta = np.abs(values) * percentage / 100 + digits * resolution
        result = delta / np.sqrt(3)
    elif mode == "repeat":
        measurements = read_numeric_excel_cols(settings["cols"], cache)[mask]
        counts = np.sum(np.isfinite(measurements), axis=1)
        if np.any(counts < 2):
            raise ValueError("'repeat': wymagane co najmniej 2 pomiary w każdym wierszu.")
        result = np.nanstd(measurements, axis=1, ddof=1) / np.sqrt(counts)
    else:
        raise ValueError(f"Nieznany tryb niepewności: {mode}")
    if result.shape != values.shape or not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError(f"Niepoprawne niepewności ({mode}).")
    return result


def main():
    for config in CONFIGS:
        usecols = config["usecols"]
        series_labels = list(config.get("series_labels", []))
        title = config["title"]
        filename = config["filename"]
        df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME, usecols=usecols,
                           skiprows=SKIPROWS, nrows=NROWS)
        if df.shape[1] < 2:
            raise ValueError(f"{filename}: potrzebna kolumna x i co najmniej jedna y.")
        x_all = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
        y_columns = [pd.to_numeric(df.iloc[:, i], errors="coerce").to_numpy(dtype=float)
                     for i in range(1, df.shape[1])]
        n_series = len(y_columns)
        series_labels = (series_labels +
                         [f"Seria {i+1}" for i in range(len(series_labels), n_series)])[:n_series]
        series_data = []
        custom = bool(USE_CUSTOM_SERIES) and "series" in config
        if custom:
            for s in config["series"]:
                y = y_columns[s["y_col_index"] - 1]
                mask = (np.isfinite(x_all) & np.isfinite(y) &
                        excel_rows_to_mask(x_all, s.get("row_start"), s.get("row_end")))
                series_data.append({"label": s["label"], "x": x_all[mask],
                                    "y": y[mask], "mask": mask,
                                    "fit": s.get("fit", True),
                                    "errors_override": s.get("errors", {})})
        else:
            for i, y in enumerate(y_columns):
                mask = np.isfinite(x_all) & np.isfinite(y)
                series_data.append({"label": series_labels[i], "x": x_all[mask],
                                    "y": y[mask], "mask": mask, "fit": True,
                                    "errors_override": {}})
        if not series_data or any(len(s["x"]) == 0 for s in series_data):
            raise RuntimeError(f"{filename}: brak prawidłowych punktów w jednej z serii.")

        err_config = config.get("errors", {})
        cache = {}
        for series in series_data:
            local_errors = series["errors_override"]
            for axis in ("x", "y"):
                settings = local_errors.get(axis, err_config.get(axis, {"mode": "none"}))
                series[f"{axis}err"] = get_uncertainties(
                    settings, series[axis], series["mask"], cache)

        for i, series in enumerate(series_data):
            if not series.get("fit", True):
                series["best_model"] = None
                continue
            best = fit_model_by_mode(series["x"], series["y"], series_index=i)
            if best is None:
                raise RuntimeError(f"Nie udało się dopasować: {series['label']}")
            series["best_model"] = best

        print(f"\n===== {filename} =====")
        for series in series_data:
            best = series["best_model"]
            print(f"\n{series['label']}:")
            if best is None:
                print("Dopasowanie: pominięte")
            else:
                print("Najlepszy model:", best["name"])
                print("Równanie:", best["equation"])
                print("R² =", fmt(best["r2"]))
            for axis in ("x", "y"):
                uncertainties = series[f"{axis}err"]
                if uncertainties is not None:
                    print(f"u({axis}) =", np.array2string(uncertainties, precision=5))
        print("Kształt wczytanej tabeli:", df.shape)

        plt.figure(figsize=(50, 40))# Rozmiar wykresu
        ax = plt.gca()
        ax.tick_params(axis="both", labelsize=60)
        for i, series in enumerate(series_data):
            best = series["best_model"]
            if "wszystkie punkty" in series["label"]:
                color, linewidth, zorder_points, zorder_line = "black", 2.5, 1, 3
            elif "zakres roboczy" in series["label"]:
                color, linewidth, zorder_points, zorder_line = "mediumorchid", 20, 2, 4
            else:
                colors = [(205/255, 102/255, 102/255), "red", "blue", "green"]# Kolor linii dopasownia
                color = colors[i] if i < len(colors) else "black"
                linewidth, zorder_points, zorder_line = 10, 2, 3

          
            if err_config.get("show", False) and (series["xerr"] is not None or series["yerr"] is not None):# Konfiguracja wyglądu słupków błedów
                style = err_config.get("style", {})
                plt.errorbar(
                    series["x"], series["y"], xerr=series["xerr"],
                    yerr=series["yerr"], fmt="none",
                    ecolor=style.get("color", "black"),
                    elinewidth=style.get("linewidth", 3),
                    capsize=style.get("capsize", 12),
                    capthick=style.get("capthick", 3),
                    zorder=zorder_points + 0.5,
                )
            plt.scatter(
                series["x"], series["y"], s=1500, marker="o",# Konfiguracja punktów
                facecolors=(1.000, 0.900, 0.600), edgecolors="black",
                linewidths=1.2, zorder=zorder_points + 1,
                label=f"{series['label']} - pomiary",
            )
            if best is not None:
                x_smooth = np.linspace(np.min(series["x"]), np.max(series["x"]), 500)
                y_smooth = best["func"](x_smooth, *best["params"])
                plt.plot(x_smooth, y_smooth, color="black", linewidth=linewidth + 2,
                         zorder=zorder_line)
                plt.plot(x_smooth, y_smooth, color=color, linewidth=linewidth,
                         zorder=zorder_line + 1,
                         label=(f"{series['label']} - dopasowanie: "
                                f"{best['equation']}, R² = {fmt(best['r2'])}"))

        for series in series_data:
            if "zakres roboczy" in series["label"]:
                x_start, x_end = np.min(series["x"]), np.max(series["x"])
                plt.axvline(x=x_start, linestyle=":", linewidth=2, color="black",
                            alpha=0.85, label=f"Początek zakresu roboczego, x ≈ {fmt(x_start)}")
                plt.axvline(x=x_end, linestyle=":", linewidth=2, color="black",
                            alpha=0.85, label=f"Koniec zakresu roboczego, x ≈ {fmt(x_end)}")

        if SHOW_CLOSEST_POINTS and sum(s["best_model"] is not None for s in series_data) >= 2:
            for a, b in combinations(series_data, 2):
                if a["best_model"] is None or b["best_model"] is None:
                    continue
                closest = find_closest_points(a, b, MAX_CLOSEST_POINTS_PER_PAIR, DENSE_POINTS)
                for x0 in closest:
                    plt.axvline(x=x0, linestyle=":", linewidth=1.5, color="black",
                                alpha=0.7,
                                label=f"Zbliżenie: {a['label']} / {b['label']}, x ≈ {fmt(x0)}")

        plt.xlabel(X_LABEL, fontsize=70)
        plt.ylabel(Y_LABEL, fontsize=70)
        plt.title(title, fontsize=80)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax.minorticks_off()  
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.grid(which="major", linestyle="-", alpha=0.6)
        ax.grid(which="minor", linestyle=":", alpha=0.5)
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        plt.legend(unique.values(), unique.keys(), fontsize=60, markerscale=1)
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Zapisano wykres: {output_path}")


if __name__ == "__main__":
    main()