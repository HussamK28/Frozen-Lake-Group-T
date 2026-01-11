import numpy as np
from environment import FrozenLake

class LinearWrapper:
    def __init__(self, env):
        self.env = env
        
        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states
        self.n_features = self.n_actions * self.n_states
        
    def encode_state(self, s):
        features = np.zeros((self.n_actions, self.n_features))
        for a in range(self.n_actions):
            i = np.ravel_multi_index((s, a), (self.n_states, self.n_actions))
            features[a, i] = 1.0
          
        return features
    
    def decode_policy(self, theta):
        policy = np.zeros(self.env.n_states, dtype=int)
        value = np.zeros(self.env.n_states)
        
        for s in range(self.n_states):
            features = self.encode_state(s)
            q = features.dot(theta)
            
            policy[s] = np.argmax(q)
            value[s] = np.max(q)
        
        return policy, value
        
    def reset(self):
        return self.encode_state(self.env.reset())
    
    def step(self, action):
        state, reward, done = self.env.step(action)
        
        return self.encode_state(state), reward, done
    
    def render(self, policy=None, value=None):
        self.env.render(policy, value)
        

def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.n_states, dtype=float)
    
    for _ in range(max_iterations):
        delta = 0
        for s in range(env.n_states):
            v = value[s]
            total = 0
            
            # For absorbing state, value should remain 0
            if s == env.absorbing_state:
                continue
            
            # Get action from policy
            a = policy[s]
            
            # Bellman expectation equation
            for ns in range(env.n_states):
                prob = env.p(ns, s, a)
                reward = env.r(ns, s, a)
                total += prob * (reward + gamma * value[ns])
            
            value[s] = total
            delta = max(delta, abs(v - value[s]))
        
        if delta < theta:
            break
            
    return value

def policy_improvement(env, value, gamma):
    policy = np.zeros(env.n_states, dtype=int)
    
    for s in range(env.n_states):
        # Absorbing state - any action is fine (won't be used)
        if s == env.absorbing_state:
            policy[s] = 0
            continue
            
        best_action = 0
        best_value = float('-inf')
        
        # Try all actions
        for a in range(env.n_actions):
            total = 0
            for ns in range(env.n_states):
                prob = env.p(ns, s, a)
                reward = env.r(ns, s, a)
                total += prob * (reward + gamma * value[ns])
            
            # Choose action with highest value
            if total > best_value:
                best_value = total
                best_action = a
        
        policy[s] = best_action
        
    return policy

def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)
    
    for i in range(max_iterations):
        value = policy_evaluation(env, policy, gamma, theta, max_iterations)
        new_policy = policy_improvement(env, value, gamma)
        
        if np.array_equal(policy, new_policy):
            break
            
        policy = new_policy
    
    return policy, value

def value_iteration(env, gamma, theta, max_iterations, value=None):
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=float)
    
    for i in range(max_iterations):
        delta = 0
        for s in range(env.n_states):
            # Skip absorbing state - value should remain 0
            if s == env.absorbing_state:
                continue
                
            v = value[s]
            best_value = float('-inf')
            
            # Find best action value
            for a in range(env.n_actions):
                total = 0
                for ns in range(env.n_states):
                    prob = env.p(ns, s, a)
                    reward = env.r(ns, s, a)
                    total += prob * (reward + gamma * value[ns])
                
                if total > best_value:
                    best_value = total
            
            value[s] = best_value
            delta = max(delta, abs(v - value[s]))
        
        if delta < theta:
            break
    
    # Extract policy from final value function
    policy = policy_improvement(env, value, gamma)
    return policy, value

def e_greedy(q, state, random_state, epsilon):
    if random_state.rand() < epsilon:
        return random_state.randint(q.shape[1])
    else:
        return np.argmax(q[state])


def sarsa(env, max_episodes, eta, gamma, epsilon, seed):
    random_state = np.random.RandomState(seed)
    eta = np.linspace(eta, 0.05, max_episodes)
    epsilon = np.linspace(epsilon, 0.05, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))*0.1
    returns = []

    for i in range(max_episodes):
        s = env.reset()
        a = e_greedy(q, s, random_state, epsilon[i])
        end = False
        episode_return = 0.0
        discount = 1.0

        while not end:
            nextS, r, end = env.step(a)
            episode_return+=discount*r
            discount*=gamma

            if end:
                q[s, a] += eta[i] * (r - q[s, a])
                break

            nextA = e_greedy(q, nextS, random_state, epsilon[i])
            q[s, a] += eta[i] * (r + gamma * q[nextS, nextA] - q[s, a])

            s, a = nextS, nextA
        returns.append(episode_return)

    policy = q.argmax(axis=1)
    value = q.max(axis=1)
    return policy, value


def q_learning(env, max_episodes, eta, gamma, epsilon, seed):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0.05, max_episodes)
    epsilon = np.linspace(epsilon, 0.05, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))
    returns = []

    for i in range(max_episodes):
        s = env.reset()
        done = False
        episode_return = 0.0
        discount = 1.0
        while not done:
            a = e_greedy(q, s, random_state, epsilon[i])
            nextS, r, done = env.step(a)
            episode_return+=discount*r
            discount*=gamma

            q[s, a] += eta[i] * (r + gamma * np.max(q[nextS]) - q[s, a])
            s = nextS
        returns.append(episode_return)


    # Derive policy and value from Q-table
    policy = q.argmax(axis=1)
    value = q.max(axis=1)


    return policy, value

def e_greedy_linear(theta, features, n_actions, random_state, epsilon):
    if random_state.rand() < epsilon:
        return random_state.randint(n_actions)
    else:
        q_values = np.array([features[a].dot(theta) for a in range(n_actions)])
        return np.argmax(q_values)


def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0.05, max_episodes)
    epsilon = np.linspace(epsilon, 0.1, max_episodes)
    
    theta = np.zeros(env.n_features)
    returns = []
    
    for i in range(max_episodes):
        features = env.reset()
        a = e_greedy_linear(theta, features, env.n_actions, random_state, epsilon[i])
        done = False

        episode_return = 0.0
        discount = 1.0

        while not done:
            nextFeatures, r, done = env.step(a)
            episode_return += discount * r
            discount *= gamma
            q = features[a].dot(theta)
            if done:
                theta += eta[i] * (r - q) * features[a]
                break
            nextA = e_greedy_linear(theta, nextFeatures, env.n_actions, random_state, epsilon[i])
            nextQ = nextFeatures[nextA].dot(theta)
            targetError = r + gamma * nextQ - q
            theta += eta[i] * targetError * features[a]
            features = nextFeatures
            a = nextA
        returns.append(episode_return)

    
    return theta




def print_results(lake, policy, value):
    height = len(lake)
    width = len(lake[0])
    
    # Reshape to match lake grid (excluding absorbing state)
    policy_grid = policy[:-1].reshape(height, width)
    value_grid = value[:-1].reshape(height, width)
    
    # Action mapping (adjust based on your environment's action order)
    # Based on your FrozenLake: directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]
    # That's: 0=up, 1=left, 2=down, 3=right
    action_symbols = {0: '^', 1: '<', 2: '_', 3: '>'}
    
    print("Q-Learning Results:")
    print("Lake:")
    for row in lake:
        print(row)
    
    print("\nPolicy:")
    for i in range(height):
        row_policy = []
        for j in range(width):
            state_idx = i * width + j
            if lake[i][j] in ['#', '$']:  # Hole or goal
                row_policy.append(lake[i][j])
            else:
                row_policy.append(action_symbols[policy_grid[i, j]])
        print(row_policy)
    
    print("\nValue:")
    for i in range(height):
        row_values = []
        for j in range(width):
            if lake[i][j] == '#':  # Hole
                row_values.append(0.0)
            else:
                row_values.append(float(value_grid[i, j]))
        # Format to 3 decimal places
        formatted_row = [f"{v:.3f}" for v in row_values]
        print(formatted_row)

def main():
    # Define the lake grid from the assignment
    lake = [
        ['&', ' ', ' ', ' '],
        [' ', '#', ' ', '#'],
        [' ', ' ', ' ', '#'],
        ['#', ' ', ' ', '$']
    ]
    env = FrozenLake(lake, slip=0.1, max_steps=100, seed=42)
    print("LINEAR SARSA CONTROL......")
    linear_env = LinearWrapper(env)
    parameters = linear_sarsa(
        env=linear_env,
        max_episodes=50000,
        eta=0.07,
        gamma=0.96,
        epsilon=0.1,
        seed=42
    )
    policy, value = linear_env.decode_policy(parameters)
    print("Policy:", policy)
    print("Value:", value)

    linear_env.render(policy, value)
    print("SARSA CONTROL......")

    policy, value = sarsa (
        env=env,
        max_episodes=50000,
        eta=0.07,
        gamma=0.92,
        epsilon=1.0,
        seed=42

    )
    print_results(lake, policy, value)
    env.render(policy=policy, value=value)

    print("Q-LEARNING CONTROL......")

    policy, value = q_learning(
        env=env,
        max_episodes=50000,
        eta=0.07,
        gamma=0.92,
        epsilon=1.0,
        seed=42

    )
    print_results(lake, policy, value)
    env.render(policy=policy, value=value)

if __name__ == "__main__":
    main()
    