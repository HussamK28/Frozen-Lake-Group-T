import pandas as pd
import matplotlib.pyplot as plt

WINDOW = 20

def load_returns(path):
    df = pd.read_csv(path, comment="#")
    return df["r"].to_numpy()

def moving_average(x, window):
    return pd.Series(x).rolling(window).mean().dropna().to_numpy()

# Load reward shaping ON and OFF data
r_on = load_returns("results/shaping_on/seed_1/monitor.csv")
r_off = load_returns("results/shaping_off/seed_1/monitor.csv")

ma_on = moving_average(r_on, WINDOW)
ma_off = moving_average(r_off, WINDOW)

# Align lengths
min_len = min(len(ma_on), len(ma_off))
ma_on = ma_on[:min_len]
ma_off = ma_off[:min_len]

# Plot
plt.figure(figsize=(8, 5))
plt.plot(ma_on, label="Reward shaping ON")
plt.plot(ma_off, label="Reward shaping OFF")

plt.xlabel("Episode")
plt.ylabel("Return (moving avg, window=20)")
plt.title("Effect of Reward Shaping on PPO Learning")
plt.legend()
plt.tight_layout()

plt.savefig("figure2_shaping_comparison.png", dpi=300, bbox_inches="tight")
plt.close()