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

from safelife.safelife_game import CellTypes

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


class DQNAvoidGreen(DQN):
   
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
        #print(envs)
        
        games = [
            e.game for e in envs
        ]

        adjacents = [
            (s.board[s.relative_loc(2)] == 1033,
             s.board[s.relative_loc(0,2)] == 1033,
             s.board[s.relative_loc(-2)] == 1033,
             s.board[s.relative_loc(0,-2)] == 1033)
             for s in games
        ] 

        directionPositions = [[0,1,2,3], [3,0,1,2], [2,3,0,1], [1,2,3,0]]
        #print(adjacents)
        
        adjacentPositions = [
            tuple(adjacents[ind][j] for j in directionPositions[s.orientation])                  
        for ind, s in enumerate(games)
        ]     

        adjAnys = []
        adjAny = False
        for s in games:
            adjAny=False
            for forPos in [-4,-3,-2,-1,0,1,2,3,4]:
                for rightPos in [-4,-3,-2,-1,0,1,2,3,4]:
                    if not (forPos==0 and rightPos==0): 
                        adjAny = adjAny or s.board[s.relative_loc(forPos,rightPos)]==1033
            adjAnys.append(adjAny)


        tensor_states = self.tensor(states, torch.float32)
        qvals = self.training_model(tensor_states).detach().cpu().numpy()
      
        wrappedqvals = copy.deepcopy(qvals)  
        
        #print(qvals)
        for i in range(0, len(qvals)):
        #    adj = adjacentPositions[i]
         #   wrappedqvals[i]
          #  if adj[0]:
           #     wrappedqvals[i][5]=99999
            #if adj[1]:
           #     wrappedqvals[i][6]=99999
           # if adj[2]:
           #     wrappedqvals[i][7]=99999
           # if adj[3]:
            #    wrappedqvals[i][8]=99999
        #print(wrappedqvals)

            if adjAnys[i]:
                wrappedqvals[i][5]=-99999
                wrappedqvals[i][6]=-99999
                wrappedqvals[i][7]=-99999
                wrappedqvals[i][8]=-99999

        num_states, num_actions = qvals.shape
        actions = np.argmax(wrappedqvals, axis=-1)
         #print(np.argmax(wrappedqvals, axis=-1), np.max(wrappedqvals, axis=-1))
        random_actions = get_rng().integers(num_actions, size=num_states)
        use_random = get_rng().random(num_states) < self.epsilon
        actions = np.choose(use_random, [actions, random_actions])
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
        super().optimize(report=False)

  
    def train(self, steps):
        super().train(steps)
     
