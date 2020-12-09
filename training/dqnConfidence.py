import logging
import numpy as np
from scipy.interpolate import UnivariateSpline

import torch
import torch.optim as optim

from safelife.helper_utils import load_kwargs
from safelife.random import get_rng

from .base_algo import BaseAlgo
from .dqn import DQN
from .utils import named_output, round_up

import copy

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


class ConfidenceDQN(DQN):

    def __init__(self, training_model, target_model, **kwargs):
        super().__init__(training_model, target_model, **kwargs)

    def update_target(self):
        super().update_target()

    @named_output('states actions rewards done qvals')
    def take_one_step(self, envs, add_to_replay=False):

        states = [
            e.last_state if hasattr(e, 'last_state') else e.reset()
            for e in envs
        ]
        tensor_states = self.tensor(states, torch.float32)
        qvals = self.training_model(tensor_states).detach().cpu().numpy()
        confidences = []

        for ind, qvali in enumerate(qvals):

            qvaliCopy = copy.deepcopy(qvali)
            e_qvals = np.exp(qvaliCopy - np.max(qvaliCopy))
            softmax = e_qvals/e_qvals.sum()
            confidences.append(softmax)
            confident = max(softmax) > 0.3
            if not confident:
                qvaliCopy[5]=-9999
                qvaliCopy[6]=-9999
                qvaliCopy[7]=-9999
                qvaliCopy[8]=-9999
                qvals[ind]=qvaliCopy   
        # print(qvals)
        num_states, num_actions = qvals.shape
        actions = np.argmax(qvals, axis=-1)
        random_actions = get_rng().integers(num_actions, size=num_states)
        use_random = get_rng().random(num_states) < self.epsilon
        actions = np.choose(use_random, [actions, random_actions])
        rewards = []
        dones = []

        for env, state, action, conf in zip(envs, states, actions, confidences):
            next_state, reward, done, info = env.step(action, conf)
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
        super().optimize(report=False)
    def train(self, steps):
        super().train(steps)
