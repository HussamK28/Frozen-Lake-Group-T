import pandas as pd
import matplotlib.pyplot as plt

def load_returns(path):
    # Monitor CSV has a JSON header line; pandas can skip it with comment="#"
    df = pd.read_csv(path, comment="#")
    # df columns: r (return), l (length), t (time)
    return df["r"].to_numpy()

def moving_average(x, window=20):
    if len(x) < window:
        return x  # not enough episodes yet
    return pd.Series(x).rolling(window).mean().dropna().to_numpy()

import numpy as np

r0 = load_returns("results/baseline/seed_0/monitor.csv")
r1 = load_returns("results/baseline/seed_1/monitor.csv")

m0 = moving_average(r0, window=20)
m1 = moving_average(r1, window=20)

# Make same length
min_len = min(len(m0), len(m1))
m0 = m0[:min_len]
m1 = m1[:min_len]

mean_curve = np.mean([m0, m1], axis=0)
std_curve = np.std([m0, m1], axis=0)

episodes = range(len(mean_curve))

plt.plot(episodes, mean_curve, label="Mean (2 seeds)")
plt.fill_between(
    episodes,
    mean_curve - std_curve,
    mean_curve + std_curve,
    alpha=0.3,
    label="Std"
)

plt.xlabel("Episode")
plt.ylabel("Return (moving avg, window=20)")
plt.title("Learning Curve (PPO on MiniGrid-KeyDoor)")
plt.legend()
plt.tight_layout()
plt.savefig("figure1_baseline_learning.png", dpi=300, bbox_inches="tight")
plt.close()