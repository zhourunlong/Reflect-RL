import json
import os
import pdb
import random

import tqdm
from transformers import AutoTokenizer, GenerationConfig, HfArgumentParser

from auto_explore_copilot import AutoExploreCopilot
from auto_explore_sandbox import RepoCache
from constants import GPT_ACTION_PROMPT, GPT_WRONG_CHOICE_PROMPT
from functions.cost import KeywordCost, NumTokenCost, StepCost, SynthesizedCost

# inclusively
MAX_LEVEL = 400000 # infinite
MIN_LEVEL = 0

INPUT_FILE = "data/auto_explore/tasks_filtered/data.json"
OUTPUT_FILE = "data/auto_explore/tasks_filtered/data_reflection_negative.json"

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

REPO_DIR = "data/auto_explore/repos_filtered/"
MODEL_TYPE = "remote_OpenAI"
# MODEL_TYPE = "remote_AzureOpenAI"
MODEL_NAME = "gpt-4"
# MODEL_NAME = "gpt-3.5-turbo"
NEGATIVE_NUM = 2
MARKOV = False
    
if os.path.exists(OUTPUT_FILE):
    dataset = json.load(open(OUTPUT_FILE, "r"))
else:
    dataset = json.load(open(INPUT_FILE, "r"))

repo_cache = RepoCache(original_root=REPO_DIR,
                       dir="/tmp/",
                       file_save_path="changed_files/")

generation_config = GenerationConfig(
    max_length=32767,
    temperature=0.7,
    top_p=0.9,
)

opt_cmds_col_name = "optimal_path"

print("Generating GPT trajectory with reflection")

# Create `cost_func` for AutoExploreCopilot
tokenizer = AutoTokenizer.from_pretrained("gpt2-xl",
                                              trust_remote_code=True,
                                              cache_dir="model/")
num_token_cost = NumTokenCost(tokenizer)
keyword_cost = KeywordCost(keywords=["Error", "Warning"], costs=[2, 1])
step_cost = StepCost()
cost_func =  SynthesizedCost(
        cost_functions=[num_token_cost, keyword_cost, step_cost], weights=[0, 1, 1])

for data in tqdm.tqdm(dataset):
    if "trajectories" not in data:
        data["trajectories"] = []
    if MODEL_NAME in [t["model_name"] for t in data["trajectories"] if t["info"].startswith("positive")]:
        continue

    if data["n_level"] > MAX_LEVEL or data["n_level"] < MIN_LEVEL:
        continue
        
    copilot = AutoExploreCopilot(
        repo_root=repo_cache.cache_repo(data["root"]),
        sandbox_dir=repo_cache.cache_dir,
        file_save_path=repo_cache.file_save_path,
        horizon=15,
        generation_config=generation_config,
        interaction_type="gen_data",
        model_type=MODEL_TYPE,
        model_name=MODEL_NAME,
        shuffle_action=False,
        need_output_msgs=False,
        cost_function=cost_func,
        markov=MARKOV,
    )

    # Let gpt-4 generate the command list
    try:
        copilot.answer(question=data["question"],
                       target_files=[data["filename"]])
    except Exception as e:
        print(e, "Failed to generate GPT trajectory for question %s" % data["question"])
        continue

    trajectory = []
    for cmd, msg in zip(copilot.cmd_history, copilot.whole_msgs):
        trajectory.append({"reflection": msg[-3][1], "cmd": cmd})
    gpt_file = None
    if len(copilot._ided_files) > 0:
        gpt_file = copilot._ided_files[-1]
        if gpt_file[0] == ".":
            gpt_file = gpt_file[1:]
    data["trajectories"].append({
        "model_name": MODEL_NAME,
        "trajectory": trajectory,
        "filename": gpt_file,
        "info": "positive"})
    
    json.dump(dataset, open(OUTPUT_FILE, "w"), indent=2)

print("Generating negative data by perturbing optimal path")
for data in tqdm.tqdm(dataset):
    if "trajectories" not in data:
        data["trajectories"] = []
    negative_keys = [t["info"] for t in data["trajectories"] if t["info"].startswith("negative")]
    if len(negative_keys) > NEGATIVE_NUM:
        continue

    if data["n_level"] > MAX_LEVEL or data["n_level"] < MIN_LEVEL:
        continue

    copilot = AutoExploreCopilot(
        repo_root=repo_cache.cache_repo(data["root"]),
        sandbox_dir=repo_cache.cache_dir,
        file_save_path=repo_cache.file_save_path,
        horizon=min(15, (MAX_LEVEL + 1) * 4),
        generation_config=generation_config,
        interaction_type="gen_data",
        model_type=MODEL_TYPE,
        model_name=MODEL_NAME,
        shuffle_action=False,
        need_output_msgs=False,
        markov=MARKOV,
    )

    if negative_keys == []:
        key = 0
    else:
        key = max([int(k[len("negative"):]) for k in negative_keys]) + 1

    for i in range(len(data[opt_cmds_col_name])):
        if data[opt_cmds_col_name][i].startswith("cat"):
            for _ in range(NEGATIVE_NUM):
                copilot.set_question(data["question"])
                copilot.ans_cmds = []

                # perturb cat command
                for j in range(i):
                    copilot.build_cur_msgs(gen_data=False)
                    response = copilot.choices[copilot.cmd_list.index(data[opt_cmds_col_name][j])]
                    copilot.act_with_response(response)

                copilot.build_cur_msgs(gen_data=False)
                filtered_cmd_list = [cmd for cmd in copilot.cmd_list if cmd != data[opt_cmds_col_name][i] and cmd.startswith("cat")]
                
                if filtered_cmd_list == []:
                    continue 

                cmd = random.choice(filtered_cmd_list)
                response = copilot.choices[copilot.cmd_list.index(cmd)]
                copilot.act_with_response(response)

                try:
                    copilot.build_cur_msgs(gen_data=True, wrong=True)
                    try:
                        response = copilot.get_response()
                    except Exception as e:
                        print(e)
                        raise e
                    
                    copilot.act_with_response(response)

                except Exception as e:
                    print(e, "Failed to generate negative trajectory for question %s" % data["question"])
                    continue

                trajectory = []
                for cmd, msg in zip(copilot.cmd_history, copilot.whole_msgs):
                    if len(msg) < 3:
                        trajectory.append({"reflection": "", "cmd": cmd})
                    else:
                        trajectory.append({"reflection": msg[-3][1], "cmd": cmd})
                
                gpt_file = None
                if len(copilot._ided_files) > 0:
                    gpt_file = copilot._ided_files[-1]
                    if gpt_file[0] == ".":
                        gpt_file = gpt_file[1:]
                data["trajectories"].append({
                    "model_name": MODEL_NAME,
                    "trajectory": trajectory,
                    "filename": gpt_file,
                    "info": "negative%d" % key})
                key += 1
                
                json.dump(dataset, open(OUTPUT_FILE, "w"), indent=2)

                if key > NEGATIVE_NUM:
                    break
