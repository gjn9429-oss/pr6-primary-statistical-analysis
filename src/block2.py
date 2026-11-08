import numpy as np
import matplotlib.pyplot as plt

from block1 import characteristics, save_figure


def generate_sample():
    # Рівно 5% із 200 спостережень замінено сплесками, а не додано зверху.
    np.random.seed(42)
    regular = np.random.normal(150, 20, 190)
    spikes = np.random.uniform(200, 300, 10)
    data = np.concatenate((regular, spikes))
    np.random.shuffle(data)
    return data


def run():
    data = generate_sample()
    n = len(data)
    k = 1 + int(3.322 * np.log10(n))
    edges = np.linspace(min(data), max(data), k + 1)
    h = (max(data) - min(data)) / k
    # NumPy включає максимум до останнього, замкненого справа інтервалу.
    counts, _ = np.histogram(data, bins=edges)
    midpoints = (edges[:-1] + edges[1:]) / 2
    weights = counts / n
    density = weights / h
    grouped_mean = sum(counts * midpoints) / n
    grouped_variance = sum(counts * (midpoints - grouped_mean) ** 2) / (n - 1)
    stats = characteristics(data)
    mean_error = abs(grouped_mean - stats["mean"])
    variance_error = abs(grouped_variance - stats["corrected_variance"])

    print("\nБЛОК 2. Затримка мікросервісу A, варіант 1")
    print(f"n={n}; k={k}; h={h:.6f} мс; seed=42")
    print("    a_i     a_i+1    середина   n_i    w_i       p_i")
    for i in range(k):
        print(f"{edges[i]:9.3f} {edges[i+1]:9.3f} {midpoints[i]:9.3f} "
              f"{counts[i]:5d} {weights[i]:7.3f} {density[i]:10.6f}")
    print(f"Сума n_i: {sum(counts)}; сума w_i: {sum(weights):.6f}")
    print(f"Площа гістограми: {sum(density * h):.6f}")
    print(f"Незгруповані: mean={stats['mean']:.6f}; S^2={stats['corrected_variance']:.6f}")
    print(f"Згруповані: mean={grouped_mean:.6f}; S^2={grouped_variance:.6f}")
    print(f"Похибка середнього: {mean_error:.6f} мс ({mean_error/stats['mean']*100:.6f}%)")
    print(f"Похибка дисперсії: {variance_error:.6f} мс^2 "
          f"({variance_error/stats['corrected_variance']*100:.6f}%)")

    # Дві осі потрібні, бо щільність і абсолютна частота мають різні одиниці.
    fig, ax = plt.subplots(figsize=(8, 4.3))
    bars = ax.bar(edges[:-1], density, width=h, align="edge", color="#bfd8e8",
                  edgecolor="#235b8e", label="Гістограма щільності")
    ax.set(xlabel="Затримка, мс", ylabel="Щільність, 1/мс")
    second = ax.twinx()
    line, = second.plot(midpoints, counts, "o-", color="#a14b30", label="Полігон частот")
    second.set_ylabel("Абсолютна частота n_i")
    second.set_ylim(0, max(counts) * 1.2)
    ax.set_ylim(0, max(density) * 1.2)
    ax.set_title("Затримка мікросервісу A: 8 інтервалів")
    ax.legend([bars, line], ["Гістограма щільності", "Полігон частот"])
    save_figure(fig, "block2_histogram_polygon.png")
    return {"data": data, "k": k, "h": h, "edges": edges, "midpoints": midpoints,
            "counts": counts, "weights": weights, "density": density, "stats": stats,
            "grouped_mean": grouped_mean, "grouped_variance": grouped_variance,
            "mean_error": mean_error, "variance_error": variance_error}


if __name__ == "__main__":
    run()
