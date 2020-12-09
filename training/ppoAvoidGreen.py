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

logger = logging.getLogger(__name__)
USE_CUDA = torch.cuda.is_available()


class AvoidGreenPPO(PPO):
   
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

        adjacents = [
            (s.board[s.relative_loc(2)]==1033,
             s.board[s.relative_loc(0,2)]==1033,
             s.board[s.relative_loc(-2)]==1033,
             s.board[s.relative_loc(0,-2)]==1033)
             for s in games
        ]
        
        directionPositions = [[0,1,2,3],[3,0,1,2],[2,3,0,1],[1,2,3,0]]

        adjacentPositions =[
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
            adj= adjacentPositions[ind]
            agentX, agentY = env.game.agent_loc
            #if adj[0]:
                #print("LIFE UP")
                #print((agentX, agentY), env.game.board[agentX, agentY-1]&CellTypes.color_g, env.game.board[agentX, agentY-1])
             #   policyCopy[5]=10000
            #if adj[1]:
                #print("LIFE RIGHT")
                #print((agentX,agentY), env.game.board[agentX+1, agentY]&CellTypes.color_g, env.game.board[agentX+1, agentY])
              #  policyCopy[6]=10000
            #if adj[2]:
                #print("LIFE DOWN")
                #print((agentX,agentY), env.game.board[agentX, agentY+1]&CellTypes.color_g, env.game.board[agentX, agentY+1])
               # policyCopy[7]=10000
           # if adj[3]:
                #print("LIFE LEFT")
                #print((agentX,agentY), env.game.board[agentX-1, agentY]&CellTypes.color_g, env.game.board[agentX-1, agentY])
            #    policyCopy[8]=10000
 
            if adjAnys[ind]:
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
            obs, reward, done, info = env.step(action)
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
