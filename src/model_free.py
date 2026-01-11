import numpy as np
import random
from environment import FrozenLake

class SARSAAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
        self.env = env
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        
        # Q-table: n_states × n_actions
        self.q_table = np.zeros((env.n_states, env.n_actions))
        
    def choose_action(self, state, epsilon=None):
        """Epsilon-greedy action selection"""
        if epsilon is None:
            epsilon = self.epsilon
            
        # Explore
        if random.random() < epsilon:
            return random.randint(0, self.env.n_actions - 1)
        # Exploit
        else:
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, next_action):
        """SARSA update rule"""
        # Current Q-value
        current_q = self.q_table[state, action]
        
        # Next Q-value (using next_action from policy - that's what makes it SARSA)
        next_q = self.q_table[next_state, next_action]
        
        # SARSA update: Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        td_target = reward + self.gamma * next_q
        td_error = td_target - current_q
        self.q_table[state, action] += self.alpha * td_error
    
    def train(self, episodes=10000):
        """Train the agent using SARSA"""
        for episode in range(episodes):
            state = self.env.reset()
            action = self.choose_action(state)
            done = False
            
            while not done:
                # Take action
                next_state, reward, done = self.env.step(action)
                
                # Choose next action (SARSA: from policy, not greedy)
                next_action = self.choose_action(next_state)
                
                # Update Q-table
                self.update(state, action, reward, next_state, next_action)
                
                # Move to next state
                state = next_state
                action = next_action
                
                # Episode ends when we reach absorbing state or max_steps
                if done:
                    break
    
    def get_policy(self):
        """Extract policy from Q-table"""
        policy = np.zeros(self.env.n_states, dtype=int)
        
        for state in range(self.env.n_states):
            # Skip absorbing state
            if state == self.env.absorbing_state:
                policy[state] = 0  # Arbitrary action for absorbing state
            else:
                policy[state] = np.argmax(self.q_table[state])
        
        return policy
    
    def get_value_function(self):
        """Extract value function V(s) = max_a Q(s,a)"""
        value = np.zeros(self.env.n_states)
        
        for state in range(self.env.n_states):
            value[state] = np.max(self.q_table[state])
        
        return value


def sarsa_learning(lake, slip=0.1, max_steps=16, 
                   learning_rate=0.1, discount_factor=0.99, 
                   epsilon=0.1, episodes=10000, seed=None):
    """
    Run SARSA learning on FrozenLake
    Returns: (policy, value)
    """
    # Create environment
    env = FrozenLake(lake, slip=slip, max_steps=max_steps, seed=seed)
    
    # Create SARSA agent
    agent = SARSAAgent(env, 
                       learning_rate=learning_rate,
                       discount_factor=discount_factor,
                       epsilon=epsilon)
    
    # Train agent
    print(f"Training SARSA for {episodes} episodes...")
    for episode in range(episodes):
        state = env.reset()
        action = agent.choose_action(state)
        done = False
        total_reward = 0
        
        while not done:
            next_state, reward, done = env.step(action)
            total_reward += reward
            next_action = agent.choose_action(next_state)
            agent.update(state, action, reward, next_state, next_action)
            state = next_state
            action = next_action
        
        # Print progress every 1000 episodes
        if episode % 1000 == 0:
            print(f"Episode {episode}: Total reward = {total_reward}")
    
    # Get results
    policy = agent.get_policy()
    value = agent.get_value_function()
    
    # Print some debug info
    print(f"\nQ-table shape: {agent.q_table.shape}")
    print(f"Max Q-value: {np.max(agent.q_table)}")
    print(f"Min Q-value: {np.min(agent.q_table)}")
    
    return policy, value


def print_results(lake, policy, value):
    """Print results in the required format"""
    height = len(lake)
    width = len(lake[0])
    
    # Reshape to match lake grid (excluding absorbing state)
    policy_grid = policy[:-1].reshape(height, width)
    value_grid = value[:-1].reshape(height, width)
    
    # Action mapping (adjust based on your environment's action order)
    # Based on your FrozenLake: directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]
    # That's: 0=up, 1=left, 2=down, 3=right
    action_symbols = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    
    print("SARSA Results:")
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
    """Main function for testing"""
    # Define the lake grid from the assignment
    lake = [
        ['&', ' ', ' ', ' '],
        [' ', '#', ' ', '#'],
        [' ', ' ', ' ', '#'],
        ['#', ' ', ' ', '$']
    ]
    
    # Set parameters for better learning
    slip = 0.1  # Slippery surface
    max_steps = 100  # More steps per episode
    learning_rate = 0.1  # Higher learning rate
    discount_factor = 0.99  # Discount factor
    epsilon = 0.1  # More exploration initially
    episodes = 20000  # More episodes
    
    print("Starting SARSA training...")
    print(f"Parameters: alpha={learning_rate}, gamma={discount_factor}, epsilon={epsilon}")
    
    # Run SARSA
    policy, value = sarsa_learning(
        lake=lake,
        slip=slip,
        max_steps=max_steps,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        episodes=episodes,
        seed=42
    )
    
    # Print results
    print_results(lake, policy, value)
    
    # Also try using the environment's render method
    print("\n\nUsing environment's render method:")
    env = FrozenLake(lake, slip=slip, max_steps=max_steps, seed=42)
    env.render(policy=policy, value=value)


if __name__ == "__main__":
    main()