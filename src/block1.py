from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


X_RAW = [23.4, 18.2, 25.1, 21.0, 18.2, 29.5, 23.4, 21.0, 27.8, 18.2,
         23.4, 25.1, 31.0, 21.0, 23.4, 19.8, 27.8, 23.4, 21.0, 25.1]
GRAPHICS = Path(__file__).resolve().parent.parent / "graphics"


def characteristics(data):
    """Числові характеристики незгрупованої вибірки."""
    n = len(data)
    mean = sum(data) / n
    squared_sum = 0.0
    for value in data:
        squared_sum += (value - mean) ** 2
    variance = squared_sum / n
    corrected_variance = squared_sum / (n - 1)
    std = corrected_variance ** 0.5
    return {"n": n, "mean": float(mean), "variance": float(variance),
            "corrected_variance": float(corrected_variance), "std": float(std),
            "median": float(np.median(data)), "range": float(max(data) - min(data)),
            "cv": float(std / mean * 100)}


def save_figure(fig, filename):
    GRAPHICS.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(GRAPHICS / filename, dpi=200)
    plt.close(fig)


def run():
    ordered = sorted(X_RAW)
    counts = Counter(ordered)
    values = list(counts)
    frequencies = []
    weights = []
    for value in values:
        frequencies.append(counts[value])
        weights.append(counts[value] / len(ordered))
    result = characteristics(ordered)
    mode = max(counts, key=counts.get)

    print("\nБЛОК 1. Незгрупована вибірка")
    print("Ранжований ряд:", ordered)
    print("   x_i     n_i      w_i")
    for value, count, weight in zip(values, frequencies, weights):
        print(f"{value:6.1f} {count:7d} {weight:8.3f}")
    for name, value in result.items():
        print(f"{name}: {value:.6f}")
    print(f"Мода: {mode:.1f} мс")

    # У методичці F*(x) = P_емп(X < x): у вузлі беремо значення ДО стрибка.
    cumulative = np.cumsum(frequencies) / len(ordered)
    x_plot = np.concatenate(([values[0] - 2], values, [values[-1] + 2]))
    y_plot = np.concatenate(([0, 0], cumulative))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.step(x_plot, y_plot, where="pre", color="#235b8e")
    before_jump = np.concatenate(([0], cumulative[:-1]))
    ax.scatter(values, before_jump, color="#235b8e", s=25, zorder=3,
               label="Значення F*(x) у вузлі")
    ax.scatter(values, cumulative, facecolors="white", edgecolors="#235b8e",
               s=25, zorder=3)
    ax.set(xlabel="Затримка x, мс", ylabel="F*(x)", ylim=(-0.04, 1.08),
           title="Емпірична функція розподілу для X < x")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    save_figure(fig, "block1_ecdf.png")
    result.update({"ordered": ordered, "values": values, "counts": frequencies,
                   "weights": weights, "mode": mode})
    return result


if __name__ == "__main__":
    run()
