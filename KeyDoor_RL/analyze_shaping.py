import pandas as pd

WINDOW = 20
THRESHOLD = 0.8

def load_returns(csv_path: str):
    df = pd.read_csv(csv_path, comment="#")
    return df["r"].to_list()

def moving_average(values, window):
    return pd.Series(values).rolling(window).mean().dropna().to_list()

def episodes_to_threshold(returns, threshold):
    ma = moving_average(returns, WINDOW)
    for i, v in enumerate(ma):
        if v >= threshold:
            return i
    return None

def final_stats(returns, tail=100):
    ma = pd.Series(returns).rolling(WINDOW).mean().dropna()
    tail_ma = ma[-tail:]
    return tail_ma.mean(), tail_ma.std()

def sustained_threshold(returns, threshold, sustain=20):
    ma = moving_average(returns, WINDOW)
    count = 0
    for i, v in enumerate(ma):
        if v >= threshold:
            count += 1
            if count >= sustain:
                return i - sustain + 1
        else:
            count = 0
    return None

def summarize(name, path):
    r = load_returns(path)
    ett = episodes_to_threshold(r, THRESHOLD)
    mean, std = final_stats(r)
    print(f"{name}")
    print(f"  episodes-to-threshold (MA{WINDOW} >= {THRESHOLD}) = {ett}")
    print(f"  final mean (last 100 MA episodes) = {mean:.3f}")
    print(f"  final std  (last 100 MA episodes) = {std:.3f}")
    sustained = sustained_threshold(r, THRESHOLD)
    print(f"  sustained threshold (>= {THRESHOLD} for 20 episodes) = {sustained}")
    print()

summarize("Reward shaping ON",  "results/shaping_on/seed_1/monitor.csv")
summarize("Reward shaping OFF", "results/shaping_off/seed_1/monitor.csv")