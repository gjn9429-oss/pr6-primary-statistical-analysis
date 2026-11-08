import numpy as np
from block1 import characteristics, save_figure
import matplotlib.pyplot as plt


def run():
    # Навчальна модель CPU, а не фактичні логи комп'ютера: U(20%, 90%).
    np.random.seed(43)
    cpu = np.random.uniform(20, 90, 300)
    stats = characteristics(cpu)
    k = 1 + int(3.322 * np.log10(len(cpu)))
    if stats["cv"] > 33:
        conclusion = "Навантаження неоднорідне: V > 33%."
    elif stats["cv"] < 10:
        conclusion = "Розсіювання незначне: V < 10%."
    else:
        conclusion = "Розсіювання помірне: 10% <= V <= 33%."

    print("\nБЛОК 4. Завантаження CPU, варіант 1")
    print("Модельні 300 вимірів: U(20, 90) %, seed=43.")
    for name in ["n", "mean", "std", "median", "cv"]:
        print(f"{name}: {stats[name]:.6f}")
    print(conclusion)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(cpu, bins=k, color="#bfd8e8", edgecolor="#235b8e")
    ax.axvline(stats["mean"], color="#a14b30", label=f"Середнє {stats['mean']:.2f}%")
    ax.axvline(stats["median"], color="#235b8e", linestyle="--",
               label=f"Медіана {stats['median']:.2f}%")
    ax.set(xlabel="Завантаження CPU, %", ylabel="Кількість вимірів",
           title="Модельні виміри CPU: n = 300")
    ax.legend()
    save_figure(fig, "block4_cpu_histogram.png")
    return {"data": cpu, "stats": stats, "k": k, "conclusion": conclusion}


if __name__ == "__main__":
    run()
