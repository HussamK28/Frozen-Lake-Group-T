import pandas as pd

WINDOW = 20
THRESHOLD = 0.8

def load_returns(csv_path: str):
    # SB3 Monitor file starts with a JSON line that begins with '#'
    df = pd.read_csv(csv_path, comment="#")
    # 'r' column is episode return
    return df["r"].to_list()

def moving_average(values, window):
    # rolling mean, same idea as np.convolve but easier here
    return pd.Series(values).rolling(window).mean().dropna().to_list()

def episodes_to_threshold(returns, threshold):
    ma = moving_average(returns, WINDOW)
    for i, v in enumerate(ma):
        if v >= threshold:
            return i  # index in the MA array
    return None

for seed in [0, 1]:
    path = f"results/seed_{seed}/monitor.csv"
    returns = load_returns(path)
    ett = episodes_to_threshold(returns, THRESHOLD)
    print(f"Seed {seed}: episodes-to-threshold (MA{WINDOW} >= {THRESHOLD}) = {ett}")

def final_performance_stats(returns, window=20, tail=100):
    ma = pd.Series(returns).rolling(window).mean().dropna()
    tail_ma = ma[-tail:]
    return tail_ma.mean(), tail_ma.std()

print("\nFinal performance statistics (last 100 MA episodes):")

for seed in [0, 1]:
    path = f"results/seed_{seed}/monitor.csv"
    returns = load_returns(path)

    mean, std = final_performance_stats(returns)
    print(f"Seed {seed}: mean={mean:.3f}, std={std:.3f}")


import numpy as np

final_means = []
final_stds = []

for seed in [0, 1]:
    path = f"results/seed_{seed}/monitor.csv"
    returns = load_returns(path)
    mean, std = final_performance_stats(returns)
    final_means.append(mean)
    final_stds.append(std)

print("\nAggregated across seeds:")
print(f"Mean final return = {np.mean(final_means):.3f} ± {np.std(final_means):.3f}")
print(f"Mean stability (std) = {np.mean(final_stds):.3f}")