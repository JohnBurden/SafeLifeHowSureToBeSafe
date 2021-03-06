import logging
import numpy as np

import torch
import torch.optim as optim

from safelife.helper_utils import load_kwargs
from safelife.random import get_rng

from .utils import named_output, round_up
from .base_algo import BaseAlgo
from .ppo import PPO

from safelife.safelife_game import CellTypes

import copy
import math

logger = logging.getLogger(__name__)
USE_CUDA = torch.cuda.is_available()


class ConfidencePPO(PPO):
   
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
    @named_output('states actions rewards done policies values')

    def take_one_step(self, envs):
        states = [
            e.last_obs if hasattr(e, 'last_obs') else e.reset()
            for e in envs
        ]

        games = [
            e.game for e in envs
        ]

        tensor_states = self.tensor(states, torch.float32)
        values, policies = self.model(tensor_states)
        values = values.detach().cpu().numpy()
        policies = policies.detach().cpu().numpy()
        actions = []
        rewards = []
        dones = []
        for ind, (policy, env) in enumerate(zip(policies, envs)):
            #print("Doing Action")
            #print(env.game.board)
            policyCopy = copy.deepcopy(policy)
            agentX, agentY = env.game.agent_loc
            #print(env)

            entropy = 0
            for p in policyCopy:
                if not p==0:
                    entropy-=p*math.log(p, 9)
            confident = entropy > 0.85 or entropy < 0.35
            #confident = True
            if not confident:
                policyCopy[5]=0
                policyCopy[6]=0
                policyCopy[7]=0
                policyCopy[8]=0

            policyCopy = policyCopy/ policyCopy.sum()
            #print(policy)
            #print(policyCopy)

            action = get_rng().choice(len(policy), p=policyCopy)
            #print(action)
            #print()
            obs, reward, done, info = env.step(action, entropy)
            if done:
                obs = env.reset()
            env.last_obs = obs
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
        return states, actions, rewards, dones, policies, values

    @named_output('states actions action_prob returns advantages values')
    def gen_training_batch(self, steps_per_env, flat=True):
        return super().gen_training_batch(steps_per_env, flat=True)

    def calculate_loss(
            self, states, actions, old_policy, old_values, returns, advantages):
        return super().calculate_loss(states, actions, old_policy, old_values, returns, advantages)


    def train_batch(self, batch):
        super().train_batch(batch)

    def train(self, steps):
        super().train(steps)
