from matplotlib import pyplot as plt
from datetime import datetime


def sort_xy(x_vals, y_vals):
    # Zip, sort by x, and unzip
    xy_sorted = sorted(zip(x_vals, y_vals), key=lambda pair: pair[0])
    x_sorted, y_sorted = zip(*xy_sorted)
    return x_sorted, y_sorted

def plot_graph(x, y, prob_1, prob_2,):
    

    # Sort each dataset
    x_a, y_a = sort_xy(x[prob_1], y[prob_1])
    x_b, y_b = sort_xy(x[prob_2], y[prob_2])

    plt.figure()

    plt.plot(x_a, y_a, marker='o', label="5%")
    plt.plot(x_b, y_b, marker='o', label="20%")

    plt.title("Performance Comparison for Goal Biasing of 5% vs 20%")
    plt.xlabel("Time")
    plt.ylabel("Cost")
    plt.legend()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"comp_{timestamp}.png"

    plt.savefig(filename, bbox_inches="tight")
    plt.close()  # good practice when saving figures

    print(f"Saved plot as {filename}")
