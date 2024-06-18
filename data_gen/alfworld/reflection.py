import glob
import json
import os
import random
import time

import autogen
from alfworld.info import ALFWORLD_DATA
from autogen import AssistantAgent, UserProxyAgent
from termcolor import colored

from nat_lang_envs.alfworld_env import AlfworldEnv

MODEL_TYPE = "OpenAI"
MODEL = "gpt-4"

TRAJ_ALGO = "oracle + random 0.7"    # one of ["oracle", "oracle + random 0.3", "gpt"]
GEN_REFLECTION = True
random.seed(5)

if MODEL_TYPE == "OpenAI":
    config_list = [
        {
            "model": MODEL,
            "api_key": os.environ["OPENAI_API_KEY"],
        }
    ]
elif MODEL_TYPE == "AzureOpenAI":
    config_list = [
        {
            "model": MODEL,
            "api_key": os.environ["AZURE_OPENAI_KEY"],
            "base_url": os.environ["AZURE_BASE_URL"].replace("/v1", "")
        }
    ]


os.makedirs("data/alfworld/", exist_ok=True)
if GEN_REFLECTION:
    OUT_FILE = "data/alfworld/data_reflection.json"
else:
    OUT_FILE = "data/alfworld/data.json"

if os.path.exists(OUT_FILE):
    DATA = json.load(open(OUT_FILE, "r"))
else:
    DATA = []

# filter out invalid data, generated because of API error
# DATA = [data for data in DATA if data["reason"].find("--- State ---") < 0]
# DATA = [data for data in DATA if data["state"].find("Observation") >= 0]
llm_config = {"config_list": config_list, "cache_seed": 42}

###

DONE_PROBLEMS = [_.get("problem", "") for _ in DATA]
DONE_PROBLEMS = set(DONE_PROBLEMS)
HORIZON = 15

num_success = 0
runs = 100
scenario = "alfworld"

# For GPT models to generate reason and action
planner = None

# For non-GPT models to generate reason
reasoner = AssistantAgent(
    name="reasoner",
    system_message="""Given a problem state and a selected action (from you).
You need to give reason why you take this action.
Remember to give a reason that is consistent with the action, within 100 words.
Even if you feel like the action is wrong (in hindsight),
you still decided to take this action, and explain why.


For instance,
Because ..., so I ...
The task is to,... I
I found ... So...
etc.
""",
    llm_config=llm_config,
    max_consecutive_auto_reply=1,
    code_execution_config={"use_docker": False})

reflectioner = AssistantAgent(
    name="reflectioner",
    system_message=
    """Given a problem state, the actions you have taken, and the observations you have.

You need to give reflection on your actions, such as:
- What is the consequence of your previous action?
- How is your previous action? Good or bad? Why?
- What is the next action you want to take if possible? Why?

Keep your reflection concise within 100 words.

For instance,
Because ..., so I ...
The task is to,... I
I found ... So...
etc.
""",
    llm_config=llm_config,
    max_consecutive_auto_reply=1,
    code_execution_config={"use_docker": False})

user = UserProxyAgent(name="user", max_consecutive_auto_reply=1,code_execution_config={"use_docker": False})

REFLECTION_PROMPT = "------ State ------\n{state}\n\n------ Give me a two-sentence reflection ---"
FIRST_STEP_REFLECTION_PROMPT = "------ State ------\n{state}\n\n------ Give me a one-sentence analysis on what I should do  ---"


# @timeout(60*1000)
def create_env(problems):
    env = AlfworldEnv(discretize_actions=True,
                      with_prompt=True,
                      with_history_in_prompt=True,
                      horizon=HORIZON,
                      filtered_problems=problems,
                      mode="train")
    return env


# @timeout(60*1000)
def user_send(user, reason_prompt, reasoner):
    user.send(message=reason_prompt,
              recipient=reasoner,
              request_reply=True,
              silent=False)
    return user


# @timeout(600*1000)
def env_step(env, action):
    return env.step(action)


DONE_PROBLEMS = set()

problems = glob.glob(os.path.join(ALFWORLD_DATA, "json_2.1.1", "train",
                                      "**", "initial_state.pddl"),
                         recursive=True)
white_list = [p.replace("initial_state.pddl", "") for p in problems if "pick_and_place_simple-Tomato-None-Microwave" in p]


while True:
    for problem in white_list:
        try:
            env = create_env(problems=[problem])
            env.reset(problem=problem, shuffle_action=False)
        except Exception as e:
            print(e)
            env = None

        if env is None or not isinstance(env, AlfworldEnv) or env.env is None:
            print("Env error. the env is:", env)
            continue

        done = False
        steps = 0
        while not done:
            # Option 1: use oracle
            if TRAJ_ALGO == "oracle":
                action = env.oracle_action
            # Option 2: use random
            elif TRAJ_ALGO.find("oracle + random") >= 0:
                prob = float(TRAJ_ALGO.split("oracle + random ")[-1])
                if random.random() > prob:
                    action = env.oracle_action
                else:
                    action = random.choice(env.action_space)
            # Option 3: Let GPT decide
            elif TRAJ_ALGO == "gpt":
                # Planner needs to perform ReAct
                action = planner.act(env.prompt, env.action_space)

            if action == "TERMINATE":
                # Some issue with the oracle path. Possibly missing Oracle path.
                break

            if GEN_REFLECTION:
                if steps == 0:
                    reflection_prompt = FIRST_STEP_REFLECTION_PROMPT.format(
                        state=env.prompt[:env.prompt.
                                        find("The actions you can take now is")])
                else:
                    # we have observation now
                    reflection_prompt = REFLECTION_PROMPT.format(
                        state=env.prompt[:env.prompt.
                                        find("The actions you can take now is")])
                
                while len(user._oai_messages[reflectioner]) < 2:
                    user.clear_history()
                    try:
                        rst = user_send(user=user,
                                        reason_prompt=reflection_prompt,
                                        reasoner=reflectioner)
                        if rst is None:
                            continue
                    except Exception as e:
                        print(e)
                        continue
                    reflection = user.last_message()["content"]
                    time.sleep(10) # sleep for 10 seconds because of API busy
                print(colored(reflection, "blue"))
            else:
                reflection = None

            if GEN_REFLECTION:
                DATA.append({
                    "state": env.prompt,
                    "reflection": reflection,
                    "action": action,
                    "reason": "",
                    "problem": env._problem,
                    "model": config_list[0].get("model", "")
                })
                json.dump(DATA, open(OUT_FILE, "w"), indent=2)

            user.clear_history()
            reasoner.clear_history()
            reflectioner.clear_history()

            action_result = env_step(env=env, action=action)
            steps += 1
            # The environment crashed... Let's forget about this environment.
            if action_result is None:
                break
            state, reward, done, truncated, info = action_result

            if truncated or done:
                break

        del env
