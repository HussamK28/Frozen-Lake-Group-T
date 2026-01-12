import numpy as np
import collections
import torch
import torch.nn.functional as F


class FrozenLakeImageWrapper:
    """
    Wraps FrozenLake into 4-channel image states (4, H, W):
      0: agent location (one-hot)
      1: start tile
      2: holes
      3: goal tile
    """
    def __init__(self, env):
        self.env = env
        lake = self.env.lake  # numpy array (H, W)

        self.n_actions = self.env.n_actions
        self.absorbing_state = self.env.absorbing_state

        H, W = lake.shape
        self.state_shape = (4, H, W)

        # constant channels
        start = (lake == '&').astype(np.float32)
        holes = (lake == '#').astype(np.float32)
        goal = (lake == '$').astype(np.float32)

        self._base = np.stack([start, holes, goal], axis=0)  # (3,H,W)

        # precompute state->image dict
        self.state_image = {}

        # absorbing state: agent channel all zeros
        agent0 = np.zeros((H, W), dtype=np.float32)
        self.state_image[self.absorbing_state] = np.concatenate(
            [agent0[None, :, :], self._base], axis=0
        )

        # normal states
        for s in range(lake.size):
            agent = np.zeros((H, W), dtype=np.float32)
            r, c = np.unravel_index(s, (H, W))
            agent[r, c] = 1.0
            self.state_image[s] = np.concatenate([agent[None, :, :], self._base], axis=0)

    def reset(self):
        s = self.env.reset()
        return self.state_image[s]

    def step(self, action):
        s, r, done = self.env.step(action)
        return self.state_image[s], r, done

    def decode_policy(self, dqn):
        """
        Evaluate Q(s,a) for each state (excluding absorbing), take argmax_a.
        Return policy and value arrays sized env.n_states.
        """
        n_states = self.env.n_states
        policy = np.zeros(n_states, dtype=int)
        value = np.zeros(n_states, dtype=float)

        with torch.no_grad():
            for s in range(n_states):
                x = np.array([self.state_image[s]])  # (1,4,H,W)
                q = dqn(x).cpu().numpy()[0]          # (A,)
                policy[s] = int(np.argmax(q))
                value[s] = float(np.max(q))

        return policy, value

    def render(self, policy, value):
        # reuse base env render
        self.env.render(policy, value)


class DeepQNetwork(torch.nn.Module):
    def __init__(self, state_shape, n_actions, learning_rate,
                 kernel_size=3, convoutchannels=4, fcoutfeatures=8, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        C, H, W = state_shape
        self.n_actions = n_actions

        self.conv_layer = torch.nn.Conv2d(
            in_channels=C,
            out_channels=convoutchannels,
            kernel_size=kernel_size
        )

        # compute conv output size
        H2 = H - kernel_size + 1
        W2 = W - kernel_size + 1
        conv_flat = convoutchannels * H2 * W2

        self.fc_layer = torch.nn.Linear(conv_flat, fcoutfeatures)
        self.out_layer = torch.nn.Linear(fcoutfeatures, n_actions)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        # x: (B,4,H,W)
        x = torch.tensor(x, dtype=torch.float32)
        x = self.conv_layer(x)
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.fc_layer(x)
        x = F.relu(x)
        x = self.out_layer(x)
        return x

    def train_step(self, transitions, gamma, tdqn):
        # unpack batch
        states = np.array([t[0] for t in transitions])
        actions = np.array([t[1] for t in transitions])
        rewards = np.array([t[2] for t in transitions], dtype=np.float32)
        next_states = np.array([t[3] for t in transitions])
        dones = np.array([t[4] for t in transitions], dtype=np.float32)

        # Q(s,a) from online net
        q_all = self(states)  # (B,A)
        q = q_all.gather(
            1, torch.tensor(actions).view(len(transitions), 1).long()
        ).view(len(transitions))

        # target = r + gamma*(1-done)*max_a' Q_target(s',a')
        with torch.no_grad():
            next_q = tdqn(next_states).max(dim=1)[0]
            target = torch.tensor(rewards) + gamma * (1.0 - torch.tensor(dones)) * next_q

        loss = F.mse_loss(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class ReplayBuffer:
    def __init__(self, buffer_size, seed=None):
        self.buffer = collections.deque(maxlen=buffer_size)
        self.random_state = np.random.RandomState(seed)

    def add(self, transition):
        self.buffer.append(transition)

    def __len__(self):
        return len(self.buffer)

    def draw(self, batch_size):
        idx = self.random_state.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[i] for i in idx]


def deep_q_network_learning(env, max_episodes, learning_rate, gamma, epsilon,
                           batch_size, target_update_frequency, buffer_size,
                           kernel_size=3, convoutchannels=4, fcoutfeatures=8,
                           seed=None):
    random_state = np.random.RandomState(seed)

    epsilon = np.linspace(epsilon, 0.0, max_episodes)

    dqn = DeepQNetwork(env.state_shape, env.n_actions, learning_rate,
                       kernel_size=kernel_size, convoutchannels=convoutchannels,
                       fcoutfeatures=fcoutfeatures, seed=seed)

    tdqn = DeepQNetwork(env.state_shape, env.n_actions, learning_rate,
                        kernel_size=kernel_size, convoutchannels=convoutchannels,
                        fcoutfeatures=fcoutfeatures, seed=seed)

    tdqn.load_state_dict(dqn.state_dict())

    buffer = ReplayBuffer(buffer_size=buffer_size, seed=seed)

    for i in range(max_episodes):
        state = env.reset()
        done = False

        while not done:
            # epsilon-greedy action from dqn
            if random_state.rand() < epsilon[i]:
                action = random_state.randint(env.n_actions)
            else:
                with torch.no_grad():
                    q = dqn(np.array([state])).cpu().numpy()[0]
                qmax = np.max(q)
                best = np.where(np.isclose(q, qmax))[0]
                action = int(random_state.choice(best))

            next_state, r, done = env.step(action)

            buffer.add((state, action, r, next_state, done))
            state = next_state

            # train once buffer is ready
            if len(buffer) >= batch_size:
                transitions = buffer.draw(batch_size)
                dqn.train_step(transitions, gamma, tdqn)

        # update target net periodically
        if (i + 1) % target_update_frequency == 0:
            tdqn.load_state_dict(dqn.state_dict())

    return dqn