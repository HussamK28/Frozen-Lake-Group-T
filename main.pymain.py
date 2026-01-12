
from src.environment import FrozenLake
from src.model_based import policy_iteration, value_iteration, sarsa, q_learning, LinearWrapper, linear_q_learning
from src.dqn import FrozenLakeImageWrapper, deep_q_network_learning

def main():
    lake = [
        ['&', '.', '.', '.'],
        ['.', '#', '.', '#'],
        ['.', '.', '.', '#'],
        ['#', '.', '.', '$']
    ]
    
    env = FrozenLake(lake, slip=0.1, max_steps=100, seed=42)
    
    print('Model-based algorithms')
    print('')
    print('Policy iteration')
    policy, value = policy_iteration(env, gamma=0.9, theta=0.001, max_iterations=128)
    env.render(policy, value)
    
    print('')
    print('Value iteration')
    policy, value = value_iteration(env, gamma=0.9, theta=0.001, max_iterations=128)
    env.render(policy, value)

    print('')
    print('SARSA')
    policy, value, sarsa_returns = sarsa(env, max_episodes=50000, eta=0.07, gamma=0.90, epsilon=1.0, seed=42)
    env.render(policy, value)

    print('')
    print('Q-Learning')
    policy, value, q_returns = q_learning(env, max_episodes=50000, eta=0.07, gamma=0.90, epsilon=1.0, seed=42)
    env.render(policy, value)

    print('')
    print('Deep Q-Network (Point 8)')
    image_env = FrozenLakeImageWrapper(env)
    dqn = deep_q_network_learning(
        image_env,
        max_episodes=4000,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=0.2,
        batch_size=32,
        target_update_frequency=4,
        buffer_size=256,
        kernel_size=3,
        convoutchannels=4,
        fcoutfeatures=8,
        seed=0
    )
    policy, value = image_env.decode_policy(dqn)
    image_env.render(policy, value)

if __name__ == "__main__":
    main()