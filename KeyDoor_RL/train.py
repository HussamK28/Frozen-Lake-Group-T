import os
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from minigrid.wrappers import FlatObsWrapper

import keydoor_env  # registers env

REWARD_SHAPING = False  #False for sparse-reward experiment

def make_env(seed=0):
    env = gym.make("MiniGrid-KeyDoor-v0", reward_shaping=REWARD_SHAPING)
    env.reset(seed=seed)
    env = FlatObsWrapper(env)

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))

    return env

env = DummyVecEnv([lambda: make_env(seed=1)])

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    gamma=0.99,
    n_steps=2048,
    batch_size=64,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1
)

suffix = "shaping_on" if REWARD_SHAPING else "shaping_off"
try:
    model.learn(total_timesteps=500_000)
finally:
    model.save(f"ppo_keydoor_{suffix}")