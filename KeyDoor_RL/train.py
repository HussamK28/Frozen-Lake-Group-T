import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from minigrid.wrappers import FlatObsWrapper

import keydoor_env  # registers env

def make_env():
    env = gym.make("MiniGrid-KeyDoor-v0")
    env = FlatObsWrapper(env)
    return env

env = DummyVecEnv([make_env])

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

model.learn(total_timesteps=500_000)
model.save("ppo_keydoor")
