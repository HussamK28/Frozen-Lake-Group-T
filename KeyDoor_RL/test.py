import gymnasium as gym
from stable_baselines3 import PPO
from minigrid.wrappers import FlatObsWrapper

import keydoor_env  # registers env

env = gym.make("MiniGrid-KeyDoor-v0", render_mode="human")
env = FlatObsWrapper(env)

model = PPO.load("ppo_keydoor")

obs, _ = env.reset()
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

env.close()
