import logging
import numpy as np
from scipy.interpolate import UnivariateSpline

import torch
import torch.optim as optim

from safelife.helper_utils import load_kwargs
from safelife.random import get_rng

from .base_algo import BaseAlgo
from .utils import named_output, round_up


logger = logging.getLogger(__name__)
USE_CUDA = torch.cuda.is_available()


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.idx = 0
        self.buffer = np.zeros(capacity, dtype=object)

    def push(self, *data):
        self.buffer[self.idx % self.capacity] = data
        self.idx += 1

    def sample(self, batch_size):
        sub_buffer = self.buffer[:self.idx]
        data = get_rng().choice(sub_buffer, batch_size, replace=False)
        return zip(*data)

    def __len__(self):
        return min(self.idx, self.capacity)


class MultistepReplayBuffer(object):
    def __init__(self, capacity, num_env, n_step, gamma):
        self.capacity = capacity
        self.idx = 0
        self.states = np.zeros(capacity, dtype=object)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=bool)
        self.num_env = num_env
        self.n_step = n_step
        self.gamma = gamma
        self.tail_length = n_step * num_env

    def push(self, state, action, reward, done):
        idx = self.idx % self.capacity
        self.idx += 1
        self.states[idx] = state
        self.actions[idx] = action
        self.done[idx] = done
        self.rewards[idx] = reward

        # Now discount the reward and add to prior rewards
        n = np.arange(1, self.n_step)
        idx_prior = idx - n * self.num_env
        prior_done = np.cumsum(self.done[idx_prior]) > 0
        gamma = self.gamma**(n) * ~prior_done
        self.rewards[idx_prior] += gamma * reward
        self.done[idx_prior] = prior_done | done

    @named_output("state action reward next_state done")
    def sample(self, batch_size):
        assert self.idx >= batch_size + self.tail_length

        idx = self.idx % self.capacity
        i1 = idx - 1 - get_rng().choice(len(self), batch_size, replace=False)
        i0 = i1 - self.tail_length

        return (
            list(self.states[i0]),  # don't want dtype=object in output
            self.actions[i0],
            self.rewards[i0],
            list(self.states[i1]),  # states n steps later
            self.done[i0],  # whether or not the episode ended before n steps
        )

    def __len__(self):
        return max(min(self.idx, self.capacity) - self.tail_length, 0)


class NoOpAgent(BaseAlgo):
    data_logger = None

    num_steps = 0

    gamma = 0.97
    multi_step_learning = 5
    training_batch_size = 96
    optimize_interval = 32
    learning_rate = 3e-4
    epsilon_schedule = UnivariateSpline(  # Piecewise linear schedule
        [5e4, 5e5, 4e6],
        [1, 1, 1], s=0, k=1, ext='const')
    epsilon_testing = 1

    replay_initial = 40000
    replay_size = 100000
    target_update_interval = 10000

    report_interval = 256
    test_interval = 100000

    compute_device = torch.device('cuda' if USE_CUDA else 'cpu')

    training_envs = None
    testing_envs = None

    checkpoint_attribs = (
        'training_model', 'target_model', 'optimizer',
        'data_logger.cumulative_stats',
    )

    def __init__(self, training_model, target_model, **kwargs):
        load_kwargs(self, kwargs)
        assert self.training_envs is not None

        self.training_model = training_model.to(self.compute_device)
        self.target_model = target_model.to(self.compute_device)
        self.optimizer = optim.Adam(
            self.training_model.parameters(), lr=self.learning_rate)
        self.replay_buffer = MultistepReplayBuffer(
            self.replay_size, len(self.training_envs),
            self.multi_step_learning, self.gamma)

        self.load_checkpoint()
        self.epsilon = self.epsilon_schedule(self.num_steps)

    def update_target(self):
        self.target_model.load_state_dict(self.training_model.state_dict())

    @named_output('states actions rewards done qvals')
    def take_one_step(self, envs, add_to_replay=False):
        states = [
            e.last_state if hasattr(e, 'last_state') else e.reset()
            for e in envs
        ]
        tensor_states = self.tensor(states, torch.float32)
        qvals = self.training_model(tensor_states).detach().cpu().numpy()

        num_states, num_actions = qvals.shape
        actions = np.zeros(len(states), dtype=int)
        rewards = []
        dones = []

        for env, state, action in zip(envs, states, actions):
            next_state, reward, done, info = env.step(action)
            if done:
                next_state = env.reset()
            env.last_state = next_state
            if add_to_replay:
                self.replay_buffer.push(state, action, reward, done)
                self.num_steps += 1
            rewards.append(reward)
            dones.append(done)

        return states, actions, rewards, dones, qvals

    def optimize(self, report=False):
        pass

    def train(self, steps):
        pass
