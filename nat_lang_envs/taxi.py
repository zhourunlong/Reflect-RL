import copy
import glob
import json
import os
import pdb
import random
import time
from queue import Queue
from typing import Optional, Tuple

import diskcache
import gym
from gym.envs.toy_text.taxi import MAP
from termcolor import colored
from transformers import GenerationConfig

from nat_lang_envs.base_env import (NaturalLanguageEnvironment,
                                    cache_function_call)

ACTION_SPACE_TAXI =   """
The actions you can take now is:
Action 0: move south
Action 1: move north
Action 2: move east
Action 3: move west
Action 4: pickup passenger
Action 5: drop off passenger
"""

ACTION_DICT = {
    0: "move south",
    1: "move north",
    2: "move east",
    3: "move west",
    4: "pickup passenger",
    5: "drop off passenger",
}




class TaxiNatLangEnv(NaturalLanguageEnvironment):
    """
    Environment for taxi in OpenAI gym.
    """
    def __init__(self,
                 with_prompt: Optional[bool] = False,
                 with_history: Optional[bool] = False,
                 horizon: Optional[int] = None,
                 generation_config: Optional[GenerationConfig] = None,
                 **kwargs):
        # del generation_config
        self.generation_config = generation_config
        self.env = gym.make('Taxi-v3')

        self.with_prompt = with_prompt

        if with_prompt:
            self._prompt_template =  open(
            "data_gen/prompt_templates/toy_text/taxi.md",
            "r").read()
        else:
            self._prompt_template = ""

        self.with_history = with_history

        self.horizon = horizon

        self._obs_history = []
        self._action_history = []
        self.obs = None
        self.pickup_rewarded = False


    @property
    def max_action_space(self):
        # return [str(a) for a in range(self.env.action_space.n)]
        return [str(a) for a in range(6)]

    @property
    def grid(self):
        return self.convert_state(self.obs)

    @property
    def prompt(self):
        state = self._prompt_template
        
        if self.with_history:
            state += "--- Initial State ---\n"
            state += self.convert_state(self._initial_obs) + "\n\n"
            state += "\n\nHere is the previous path:\n"
            for step, (prev_action, prev_obs) in enumerate(
                    zip(self._action_history, self._obs_history)):
                state += f"--- Step: {step} ---\n"
                # prev_obs = self.convert_state(prev_obs)
                state += f"Action: {prev_action}\nObservation:\n{prev_obs}\n\n"
        else:
            state += "You have already completed the following actions:\n"
            state += "\n".join(self._action_history)

        state += "\n\n--- Current State ---\n\n"
        state += self.grid
        state += ACTION_SPACE_TAXI
        self.state = state # alias for prompt

        return state
    
    @property
    def action_space(self):
        # return [str(a) for a in range(self.env.action_space.n)]
        return [str(a) for a in range(6)]

    def convert_state(self, obs: int) -> str:
        if isinstance(obs, str):
            return obs

        colors = ["R", "G", "Y", "B"]
        locations = {"R": (0, 0), "G": (0, 4), "Y": (4, 0), "B": (4, 3)}

        taxi_row, taxi_col, passenger_location, destination = self.env.decode(obs)

        _grid = [[" " for j in range(5)] for i in range(5)]

        dest_row, dest_col = locations[colors[destination]]
        _grid[dest_row][dest_col] = "D"

        # Mark the passenger's location
        if passenger_location < 4:
            pass_row, pass_col = locations[colors[passenger_location]]
            _grid[pass_row][pass_col] = "P"
            _grid[taxi_row][taxi_col] = "T"

            if pass_row == taxi_row and pass_col == taxi_col:
                _grid[pass_row][pass_col] = "O"
        else:
            # Passenger is in the taxi
            _grid[taxi_row][taxi_col] = "X"
        
        grid = [[y for y in x] for x in MAP]
        for i in range(5):
            for j in range(5):
                grid[i + 1][2 * j + 1] = _grid[i][j]

        return "\n".join([("".join(row)) for row in grid])

    def step(self, action: str) -> Tuple[str, float, bool, bool, dict]:
        self._action_history.append(ACTION_DICT[int(action)])
        obs, reward, done, truncated, info = self.env.step(
            int(action) + self.env.action_space.start)
        self.obs = obs
        self._obs_history.append(self.convert_state(obs))
        # self._obs_history.append(obs)
        self.time_step += 1
        if self.time_step >= self.horizon:
            truncated = True
    
        if self.curriculum in ["pickup", "full"]:
            if "X" in self.grid:
                if not self.pickup_rewarded:
                    reward = 20
                    self.pickup_rewarded = True
                if self.curriculum == "pickup":
                    truncated = True

        if len(self._obs_history) >= 2 and self._obs_history[-1] == self._obs_history[-2]:
            reward = -10
            truncated = True
            done = True

        return (self.prompt, reward / 20, done, truncated, info)

    def reset(
        self,
        data: str = None,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        **kwargs
    ) -> Tuple[str, dict]:
        del options, kwargs
        self.curriculum = data
        random.seed(seed)
        obs, info = self.env.reset()
        self.obs = obs
        self._initial_obs = obs
        self.time_step = 0
        self._obs_history = []
        self.pickup_rewarded = False

        if data == "dropoff":
            action_sequence = self.oracle_action
            assert action_sequence != [], "No action sequence found."
            for action in action_sequence:
                self.obs, reward, done, truncated, info = self.env.step(action)
                state = self.convert_state(self.obs)
                if "X" in state:
                    break

        return (self.prompt, info)

    @property
    def oracle_action(self) -> list:
        """Returns the oracle of the environment."""
        """
        Find the minimum number of steps to reach the goal in an OpenAI Gym environment using BFS.

        Args:
            env (gym.Env): The OpenAI Gym environment.

        Returns:
            list: a list of actions to reach the goal.
        """
        queue = Queue()
        visited = set([self.obs])
        queue.put((self.obs, copy.deepcopy(self.env), []))

        while not queue.empty():
            obs, env, action_sequence = queue.get()
            for _action in self.action_space:
                action = int(_action)
                step_env = copy.deepcopy(env)
                new_obs, reward, done, truncated, _info = step_env.step(action)

                if done or truncated:
                    if reward > 0:
                        action_sequence = action_sequence + [action]
                        return action_sequence
                elif new_obs not in visited:
                    visited.add(new_obs)
                    queue.put((new_obs, step_env, action_sequence + [action]))

        return []
