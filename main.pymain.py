from src.environment import FrozenLake
from src.model_based import policy_iteration, value_iteration, sarsa_learning, q_learning

def main():
    lake = [
        ['&', '.', '.', '.'],
        ['.', '#', '.', '#'],
        ['.', '.', '.', '#'],
        ['#', '.', '.', '$']
    ]
    
    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=0)
    
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
    policy, value = sarsa_learning(env, max_episodes=2000, eta=0.1, gamma=0.99, epsilon=0.1, seed=None)
    env.render(policy, value)

    print('')
    print('Q-Learning')
    policy, value = q_learning(env, max_episodes=2000, eta=0.1, gamma=0.99, epsilon=0.1, seed=None)
    env.render(policy, value)

if __name__ == "__main__":
    main()