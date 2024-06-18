import glob
import os
import pdb
import random
import time

import autogen
from termcolor import colored

from agent.ag_agent import EnvAgent, Player, RandomPlayer
from agent.react_agent import ReActAgent

try:
    from alfworld.info import ALFWORLD_DATA

    from nat_lang_envs.alfworld_env import AlfworldEnv
except:
    AlfworldEnv = None
import argparse

from nat_lang_envs.frozen_lake import FrozenLakeNatLangEnv
from nat_lang_envs.taxi import TaxiNatLangEnv

parser = argparse.ArgumentParser(description="Run the agent in a natural language environment.")
parser.add_argument("--scenario", type=str, default="taxi", help="The scenario to run the agent in.")
parser.add_argument("--extra", type=str, default="pickup", help="Extra arguments for the scenario.")
parser.add_argument("--random", type=bool, default=False, help="If True, use a random-action agent")
parser.add_argument("--horizon", type=int, default=15, help="The horizon of the game.")
parser.add_argument("--shuffle_action", type=bool, default=False, help="Shuffle the order of action or not. "
                    "This flag is only valid for `alfworld` scenario")
parser.add_argument("--react", type=bool, default=True, help="If True, use ReAct prompt.")
parser.add_argument("--runs", type=int, default=10, help="The number of runs")
args = parser.parse_args()

scenario = args.scenario
extra = args.extra
num_success = 0
runs = args.runs


config_list = [
    {
    "model": "gpt-3.5-turbo",
    "api_key": os.getenv("OPENAI_KEY", "YOUR_API_KEY"),
    "base_url": os.getenv("BASE_URL", "https://api.openai.com"),
    }
]

llm_config = {"config_list": config_list, "cache_seed": 43, "max_tokens": 2000}

random.seed(1)

if args.react:
    # See the agent/react_agent.py for details. 
    # AutoGen will handle the reflection and action prompts separately.
    system_message = "Perform reflection or give Action ID."
else:
    system_message = "Think step by step. Reply Reflection, Reason, Plan, Action within 100 words. You MUST reply the action ID, such as 'Action 2'"

model_name = config_list[0]["model"]


for i in range(runs):
    random.seed(i)
    if scenario == "alfworld":
        problems = glob.glob(os.path.join(ALFWORLD_DATA, "json_2.1.1", "valid*",
                                    "**", "initial_state.pddl"),
                        recursive=True)
        filtered_problems = [p.replace("initial_state.pddl", "") for p in problems if "pick_and_place_simple-Tomato-None-Microwave" in p]
        problem = random.choice(filtered_problems)
        env = AlfworldEnv(discretize_actions=True,
                          with_prompt=True,
                          with_history_in_prompt=True,
                          horizon=args.horizon)
        env.reset(problem, shuffle_action=args.shuffle_action)
    elif scenario == "taxi":
        env = TaxiNatLangEnv(
            with_prompt=True,
            horizon=args.horizon,
            with_history_in_prompt=True,
            generation_config=None
        )
        env.reset("reward_pickup" if extra == "pickup" else extra)
    print("PROMPT:", env.prompt)

    if args.react:
        env_agent = ReActAgent(
            env=env,
            name="Env",
        )
    else:
        env_agent = EnvAgent(
            env=env,
            name="Env",
        )

    if args.random:
        player = RandomPlayer(name="Random", max_consecutive_auto_reply=args.horizon)
    else:
        player = Player(name=model_name, llm_config=llm_config, system_message=system_message)

    env_agent.initiate_chat(recipient=player,
                            clear_history=True,
                            message=env_agent.initial_state,
                            silent=False)

    if scenario == "taxi" and extra == "dropoff":
        is_success = env_agent._total_reward > 20
    else:
        is_success = env_agent._total_reward > 0


    open("result_episode.txt", "a").write(
        f"""
    ------------------------------------------------
    {args}
    Current Success: {is_success}
    Success So Far: {num_success} out of {i + 1} runs.

    ------------------------------------------------
    """)
    num_success += int(is_success)
    del env, env_agent, player

    print("Total Success", num_success, f"out of {i + 1} runs")


print("Total Success", num_success, f"out of {runs} runs")


open("result.txt", "a").write(
    f"""
------------------------------------------------
{args}
Total Success: {num_success} out of {runs} runs.


------------------------------------------------
""")



