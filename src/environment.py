import numpy as np
import contextlib

@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)

class EnvironmentModel:
    def __init__(self, n_states, n_actions, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.random_state = np.random.RandomState(seed)

    def p(self, next_state, state, action):
        raise NotImplementedError()

    def r(self, next_state, state, action):
        raise NotImplementedError()

    def draw(self, state, action):
        p = [self.p(ns, state, action) for ns in range(self.n_states)]
        next_state = self.random_state.choice(self.n_states, p=p)
        reward = self.r(next_state, state, action)
        return next_state, reward

class Environment(EnvironmentModel):
    def __init__(self, n_states, n_actions, max_steps, pi, seed=None):
        EnvironmentModel.__init__(self, n_states, n_actions, seed)
        self.max_steps = max_steps
        self.pi = pi
        if self.pi is None:
            self.pi = np.full(n_states, 1.0/n_states)

    def reset(self):
        self.n_steps = 0
        self.state = self.random_state.choice(self.n_states, p=self.pi)
        return self.state

    def step(self, action):
        if action < 0 or action >= self.n_actions:
            raise Exception('Invalid action.')
        self.n_steps += 1
        done = (self.n_steps >= self.max_steps)
        self.state, reward = self.draw(self.state, action)
        return self.state, reward, done

    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)
            if self.state < self.absorbing_state:
                lake[self.state] = '@'
            print(lake.reshape(self.lake.shape))
        else:
            actions = ['^', '<', 'v', '>']
            print('Lake:')
            print(self.lake)
            print('Policy:')
            policy_arrows = np.array([actions[a] for a in policy[:-1]])
            print(policy_arrows.reshape(self.lake.shape))
            print('Value:')
            with printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))

class FrozenLake(Environment):
    def __init__(self, lake, slip, max_steps, seed=None):
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1)
        self.slip = slip
        
        n_states = self.lake.size + 1
        n_actions = 4
        
        pi = np.zeros(n_states, dtype=float)
        start_idx = np.where(self.lake_flat == '&')[0]
        if len(start_idx) > 0:
            pi[start_idx[0]] = 1.0
            
        self.absorbing_state = n_states - 1
        
        self.rows, self.cols = self.lake.shape
        self.directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]  # up, left, down, right
        
        super().__init__(n_states, n_actions, max_steps, pi, seed=seed)

    def p(self, next_state, state, action):
        """
        Transition probability P(next_state | state, action)
        """
        # If in absorbing state, stay there
        if state == self.absorbing_state:
            return 1.0 if next_state == self.absorbing_state else 0.0
        
        # If at goal or hole, go to absorbing state
        if state < len(self.lake_flat):
            tile = self.lake_flat[state]
            if tile == '$' or tile == '#':
                return 1.0 if next_state == self.absorbing_state else 0.0
        
        # Normal movement with slippage
        row, col = state // self.cols, state % self.cols
        total_prob = 0.0
        
        # For each possible direction
        for dir_idx, (dr, dc) in enumerate(self.directions):
            if dir_idx == action:
                # Probability for intended direction
                prob = (1.0 - self.slip) + (self.slip / 4.0)
            else:
                # Probability for other directions (slipping)
                prob = self.slip / 4.0
            
            # Calculate resulting position for this direction
            new_row, new_col = row + dr, col + dc
            
            # Check boundaries - stay in place if out of bounds
            if new_row < 0 or new_row >= self.rows or new_col < 0 or new_col >= self.cols:
                new_row, new_col = row, col
            
            # Calculate state index
            new_state = new_row * self.cols + new_col
            
            # If this direction leads to the requested next_state, add its probability
            if new_state == next_state:
                total_prob += prob
        
        return total_prob

    def r(self, next_state, state, action):
        """
        Return expected reward for transition (state, action) -> next_state.
        Reward is 1 ONLY when taking action at goal state ($).
        """
        # If we're in a goal state, we get reward 1 for any action
        if state < len(self.lake_flat):
            if self.lake_flat[state] == '$':
                return 1.0
        return 0.0

    def step(self, action):
        state, reward, done = super().step(action)
        done = (state == self.absorbing_state) or done
        return state, reward, done