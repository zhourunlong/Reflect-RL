"""
How to setup: you need to clone the alfworld repo and then download the data.

```bash
git clone https://github.com/alfworld/alfworld
cd alfworld
pip install -e .  # install the package
python scripts/alfworld-download --extra    # download extra data (aka, oracle path)
```
"""

import glob
import json
import os
import pdb
import random
from typing import Optional, Tuple

import textworld
import textworld.gym
from alfworld.agents.environment.alfred_tw_env import (AlfredDemangler,
                                                       AlfredExpert,
                                                       AlfredExpertType)
from alfworld.agents.utils.misc import add_task_to_grammar
from alfworld.info import ALFWORLD_DATA
from transformers import GenerationConfig

from constants import ACTION_TEMPLATE, CHOICES
from nat_lang_envs.base_env import NaturalLanguageEnvironment
from utils import debug_msg, timeout


class AlfworldEnv(NaturalLanguageEnvironment):
    """
    Environment for ALFWorld.
    """

    def __init__(
            self,
            discretize_actions: bool = True,
            with_prompt: Optional[bool] = True,
            with_history_in_prompt: Optional[bool] = True,
            horizon: Optional[int] = None,
            generation_config: Optional[GenerationConfig] = None,
            disable_alfworld: bool = False,
            **kwargs):

        if discretize_actions == False:
            raise NotImplementedError("Currently only support discretize_actions=True")

        self.horizon = horizon
        self.discretize_actions = discretize_actions
        self.with_prompt = with_prompt
        self.with_history_in_prompt = with_history_in_prompt
        self.generation_config = generation_config
        self.disable_alfworld = disable_alfworld

        self.env = None

        # self.reset() # Move this to the first step in the loop

    def __del__(self):
        if hasattr(self, "env"):
            del self.env

    @property
    def max_action_space(self):
        return CHOICES

    @property
    def action_space(self):
        if self.discretize_actions:
            return self.choices
        else:
            return self.admissible_commands

    @property
    def admissible_commands(self):
        return self.infos.get("admissible_commands", [])

    @property
    def oracle_action(self) -> str:
        expert_plan = self.infos.get("extra.expert_plan", [])

        if len(expert_plan) == 0:
            return "TERMINATE"

        action = expert_plan[0]
        if self.discretize_actions:
            return str(self.admissible_commands.index(action))
        else:
            return action

    @property
    def prompt(self) -> str:
        state = self._task

        if len(self._action_history):
            if self.with_history_in_prompt:
                state += "\n\nHere is the previous path:\n"
                for step, (prev_action, prev_obs) in enumerate(
                        zip(self._action_history, self._obs_history)):
                    state += f"--- Step: {step} ---\n"
                    state += f"Action: {prev_action}\nObservation: {prev_obs}\n\n"
            else:
                state += "You have already completed the following actions:\n"
                state += "\n".join(self._action_history)
                state += f"\nYour last observation is: {self.obs}"

        state += "\n\n"
        if self.discretize_actions:
            sorted_pairs = sorted(zip(self.action_space, self.
            admissible_commands), key=lambda x: CHOICES.index(x[0]))
            state += "The actions you can take now is:\n" + "\n".join([
                f"Action {num}: {action}" for num, action in sorted_pairs
            ])
            # state += "\nGive me the numeric Action ID: "
        else:
            state += "The actions you can take now is:\n" + str(
                self.admissible_commands)
            state += "\nChoice an action from the list: "

        return state
    
    def update_choices(self):
        self.choices = CHOICES[:len(self.admissible_commands)]
        if self.shuffle_action:
            random.shuffle(self.choices)

    @property
    def state(self) -> str:
        return self.prompt    # give an alias to prompt

    def step(self, action: str) -> Tuple[str, float, bool, bool, dict]:
        # convert action back into string if discretize_actions
        self.time_step += 1
        truncated = self.time_step >= self.horizon

        if self.discretize_actions:
            if action not in self.choices:
                # Invalid action. Prompt again
                return (self.prompt, 0, False, truncated, self.infos)
            action = self.admissible_commands[self.action_space.index(action)]

        self._action_history.append(action)

        debug_msg("Env step start")

        @timeout(100000) # 100 seconds
        def timed_step(action):
            return self.env.step(action)
        try:
            obs, score, done, infos = timed_step(action)
        except Exception as e:
            print(e)
            write_black_list(self._problem)
            self.env = None
            return (self.prompt, 0, False, True, self.infos)

        debug_msg("Env step finish")

        self._obs_history.append(obs)
        self.obs = obs
        self.infos = infos
        self._total_reward += score

        self.update_choices()

        return (self.prompt, score, done, truncated, infos)

    def reset(self,
              problem: str,
              shuffle_action: bool = False,
              **kwargs) -> Tuple[str, dict]:
        self.time_step = 0
        self.shuffle_action = shuffle_action

        debug_msg("Resetting environment start")

        # Environment Setup
        if problem is None:
            debug_msg("Read black list begin")

            black_list = read_black_list()

            debug_msg("Read black list finish")

            while True:
                problem = os.path.dirname(random.choice(self._filtered_problems))
                if problem not in black_list:
                    break
        self._problem = problem    # str to the problem setup dir

        try:
            debug_msg(self._problem)
            self.env = init_textworld(self._problem, self.disable_alfworld)
            self.obs, self.infos = self.env.reset()
        except Exception as e:
            pdb.set_trace()
            write_black_list(self._problem)
            print(e)

            self.env, self.obs, self.infos = None, None, None
            pass
        
        debug_msg("Resetting environment finish")

        self.update_choices()

        # General Setup
        self._action_history = []
        self._obs_history = []

        # if self.with_prompt:
        #     self._task = self.obs
        # else:
        #     self._task = self.obs[self.obs.find("Your task is to:"):]
        self._task = self.obs

        self._total_reward = 0
        return self.prompt, self.infos


def init_textworld(problem: str, disable_alfworld: bool):
    domain = os.path.join(ALFWORLD_DATA, "logic", "alfred.pddl")
    grammar = os.path.join(ALFWORLD_DATA, "logic", "alfred.twl2")
    GAME_LOGIC = {
        "pddl_domain": open(domain).read(),
        "grammar": open(grammar).read(),
    }

    # load state and trajectory files
    pddl_file = os.path.join(problem, 'initial_state.pddl')
    json_file = os.path.join(problem, 'traj_data.json')
    with open(json_file, 'r') as f:
        traj_data = json.load(f)
    GAME_LOGIC['grammar'] = add_task_to_grammar(GAME_LOGIC['grammar'],
                                                traj_data)

    # dump game file
    gamedata = dict(**GAME_LOGIC, pddl_problem=open(pddl_file).read())
    gamefile = os.path.join(os.path.dirname(pddl_file), 'game.tw-pddl')
    json.dump(gamedata, open(gamefile, "w"))

    if not disable_alfworld:
        expert = AlfredExpert(expert_type=AlfredExpertType.PLANNER)
        # expert = AlfredExpert(expert_type=AlfredExpertType.HANDCODED)
        wrappers = [AlfredDemangler(), expert]
    else:
        expert = None
        wrappers = [AlfredDemangler()]

    # register a new Gym environment.
    request_infos = textworld.EnvInfos(won=True,
                                       admissible_commands=True,
                                       score=True,
                                       max_score=True,
                                       intermediate_reward=True,
                                       extras=["expert_plan"])
    env_id = textworld.gym.register_game(gamefile,
                                         request_infos,
                                         max_episode_steps=1000000,
                                         wrappers=wrappers)

    env = textworld.gym.make(env_id)

    return env

def read_black_list():
    if os.path.exists("alfworld_blacklist.txt"):
        with open("alfworld_blacklist.txt", "r") as f:
            return [line.strip() for line in f.readlines()]
    else:
        return []

def write_black_list(problem):
    problem = problem[problem.find("json_2.1.1"):]
    with open("alfworld_blacklist.txt", "a") as f:
        f.write(f"{problem}\n")
