import numpy as np
from block1 import characteristics, save_figure
from block2 import generate_sample
import matplotlib.pyplot as plt


def run():
    original = generate_sample()
    original_stats = characteristics(original)
    threshold = original_stats["mean"] + 5 * original_stats["std"]
    # Поріг обчислено до додавання аномалій; кожне нове значення більше за нього.
    anomalies = threshold + np.arange(1, 11) * original_stats["std"] / 2
    dirty = np.concatenate((original, anomalies))
    q1, q3 = np.percentile(dirty, [25, 75], method="linear")
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    clean = dirty[(dirty >= lower) & (dirty <= upper)]
    dirty_stats = characteristics(dirty)
    clean_stats = characteristics(clean)
    original_removed = sum((original < lower) | (original > upper))
    added_removed = sum((anomalies < lower) | (anomalies > upper))

    print("\nБЛОК 3. Викиди та правило IQR")
    print(f"Поріг original_mean + 5S = {threshold:.6f} мс")
    print("Додані аномалії:", np.round(anomalies, 6))
    print(f"Q1={q1:.6f}; Q3={q3:.6f}; IQR={iqr:.6f}")
    print(f"Межі: [{lower:.6f}; {upper:.6f}] мс")
    print("Вибірка       n    середнє      S        медіана")
    for name, stats in [("Початкова", original_stats), ("Спотворена", dirty_stats),
                        ("Очищена", clean_stats)]:
        print(f"{name:12s} {stats['n']:3d} {stats['mean']:11.6f} "
              f"{stats['std']:10.6f} {stats['median']:11.6f}")
    print(f"Видалено: {len(dirty)-len(clean)}; із початкових: {original_removed}; "
          f"із доданих: {added_removed}")
    print(f"Зміна середнього: {dirty_stats['mean']-original_stats['mean']:.6f} мс")
    print(f"Зміна медіани: {dirty_stats['median']-original_stats['median']:.6f} мс")

    fig, ax = plt.subplots(figsize=(8, 4.1))
    ax.boxplot([original, dirty, clean], orientation="horizontal",
               tick_labels=["Початкова", "Спотворена", "Очищена"], whis=1.5,
               medianprops={"color": "#a14b30", "linewidth": 2})
    ax.axvline(upper, linestyle="--", color="#235b8e", label="Верхня межа IQR спотвореної")
    ax.set(xlabel="Затримка, мс", title="Вплив аномалій і результат фільтрації")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    save_figure(fig, "block3_boxplot.png")
    return {"original": original, "dirty": dirty, "clean": clean, "anomalies": anomalies,
            "threshold": threshold, "q1": q1, "q3": q3, "iqr": iqr,
            "lower": lower, "upper": upper, "original_stats": original_stats,
            "dirty_stats": dirty_stats, "clean_stats": clean_stats,
            "original_removed": int(original_removed), "added_removed": int(added_removed)}


if __name__ == "__main__":
    run()
