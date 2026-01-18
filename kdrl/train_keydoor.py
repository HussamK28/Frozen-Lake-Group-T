import gymnasium as gym
import minigrid

from gymnasium.wrappers import FilterObservation, ResizeObservation
from minigrid.wrappers import ImgObsWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy


# =========================
# CONFIG
# =========================
ENV_ID = "MiniGrid-DoorKey-5x5-v0"
TRAIN_TIMESTEPS = 100_000
EVAL_EPISODES = 10


# =========================
# ENV FACTORY
# =========================
def make_env(render=False):
    def _init():
        env = gym.make(
            ENV_ID,
            render_mode="human" if render else None
        )

        # Convert MiniGrid dict obs -> image only
        env = ImgObsWrapper(env)

        # Resize image so CNN works
        env = ResizeObservation(env, (84, 84))

        env = Monitor(env)
        return env

    return _init


# =========================
# TRAIN
# =========================
print("Creating training environment...")
train_env = DummyVecEnv([make_env(render=False)])

print("Creating PPO model...")
model = PPO(
    policy="CnnPolicy",
    env=train_env,
    learning_rate=3e-4,
    gamma=0.99,
    n_steps=2048,
    verbose=1,
    device="cpu"
)

print("Training...")
model.learn(total_timesteps=TRAIN_TIMESTEPS)

print("Training complete.")


# =========================
# EVALUATE
# =========================
print("\nEvaluating model...")
eval_env = DummyVecEnv([make_env(render=False)])

mean_reward, std_reward = evaluate_policy(
    model,
    eval_env,
    n_eval_episodes=EVAL_EPISODES,
    deterministic=True
)

print(f"PPO | Avg Reward: {mean_reward:.2f} ± {std_reward:.2f}")


# =========================
# RENDER ONE EPISODE
# =========================
print("\nRendering one episode...")
render_env = gym.make(ENV_ID, render_mode="human")
render_env = ImgObsWrapper(render_env)
render_env = ResizeObservation(render_env, (84, 84))

obs, _ = render_env.reset()
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = render_env.step(action)
    done = terminated or truncated

render_env.close()
