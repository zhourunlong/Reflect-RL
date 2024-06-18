
import copy
import glob
import json
import os
import pdb
import random
import time
from queue import Queue

import autogen
from autogen import AssistantAgent, UserProxyAgent
from termcolor import colored

from nat_lang_envs.base_env import NaturalLanguageEnvironment
from nat_lang_envs.taxi import TaxiNatLangEnv

TRAJ_ALGO = "random"    # one of ["oracle", "oracle + random 0.3", "gpt"]
GEN_REASON = False
GEN_REFLECTION = True
CURRICULUM = "pickup" # pickup or dropoff


def find_min_steps_to_goal_bfs(env):
    """
    Find the minimum number of steps to reach the goal in an OpenAI Gym environment using BFS.

    Args:
        env (gym.Env): The OpenAI Gym environment.

    Returns:
        tuple: A tuple containing the minimum number of steps and the sequence of actions.
    """
    queue = Queue()
    visited = set()
    queue.put((copy.deepcopy(env), 0, []))  # (state, step_count, action_sequence)

    while not queue.empty():
        current_env, step_count, action_sequence = queue.get()

        # Check if this state has been visited
        if current_env.grid in visited:
            continue
        visited.add(current_env.grid)

        space = copy.deepcopy(current_env.action_space)
        random.shuffle(space)
        for action in space:
            # if "O" in current_env.grid and action == "4":
            #     pdb.set_trace()
            step_env = copy.deepcopy(current_env)
            new_state, reward, done, truncated, _info = step_env.step(action)
            # print(action_sequence + [action])

            if (done or truncated) and reward > 0:
                print(colored("Success!!!! after", "green"), step_count + 1, "steps:", action_sequence + [action])
                return step_count + 1, action_sequence + [action]

            if not done and not truncated:
                queue.put((step_env, step_count + 1, action_sequence + [action]))

    return None  # No solution found

# Example usage
OUT_FILE = f"data/taxi/{CURRICULUM}/oracle.json"

if os.path.exists(OUT_FILE):
    DATA = json.load(open(OUT_FILE, "r"))
else:
    DATA = []

for _ in range(1000):
    env = TaxiNatLangEnv(with_prompt=True, horizon=30, with_history_in_prompt=True)
    env.reset(data=f"reward_{CURRICULUM}")
    print(env.grid)

    min_steps, action_seq = find_min_steps_to_goal_bfs(copy.deepcopy(env))
    print(f"Minimum steps: {min_steps}, Action sequence: {action_seq}")

    
    for action in action_seq:
        env.step(action)
        time.sleep(1)

        DATA.append({
                "state": env.prompt,
                "reflection": "",
                "action": action,
                "reason": "",
                "model": "oracle"
        })
        json.dump(DATA, open(OUT_FILE, "w"), indent=2)

