import json
import os

from tqdm import tqdm
from transformers import HfArgumentParser

from experiment_args import ScriptArguments
from utils import load_dataset, unwrap_path, wrap_path

def get_num(trajectories, prefix):
    return sum([1 for t in trajectories if t["info"].startswith(prefix)])

NEW_REPO_DIR = "data/auto_explore/repos_filtered/"
NEW_TASK_DIR = "data/auto_explore/tasks_filtered/"
# NEW_TASK_DIR = "data/auto_explore/tasks_filtered_test/"

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

dataset = load_dataset(script_args.task_file)
task_dir = os.path.dirname(script_args.task_file)

opt_cmds_col_name = "optimal_path"
del_col_names = ["copilot_cmds", "reasons", "reason", "commands_reason", "commands_copilot_cmds", "commands"]

data_dict = {}

for data in tqdm(dataset):
    if data["question"].strip() == "":
        continue

    root = data["root"]
    root = os.path.join(NEW_REPO_DIR, root)

    if not os.path.exists(root):
        continue

    target_file_path = os.path.join(root, data["filename"])
    if not os.path.exists(target_file_path):
        continue

    # remove unnecessary columns
    for del_name in del_col_names:
        if del_name in data:
            del data[del_name]
    
    # reformat optimal commands
    opt_cmds = []
    for cmd in data[opt_cmds_col_name]:
        if cmd == "ls":
            continue
        for op in ["cd", "cat"]:
            if cmd.startswith(op):
                file = unwrap_path(cmd[len(op):].strip())
                cmd = op + " " + wrap_path(file)
                break
        opt_cmds.append(cmd)
    if opt_cmds[-1].startswith("cat"):
        opt_cmds.append("id" + opt_cmds[-1][len("cat"):])

    # rewrite 
    data[opt_cmds_col_name] = opt_cmds

    dict_key = root + "@" + data["filename"] + "@" + data["question"]
    if dict_key not in data_dict:
        data_dict[dict_key] = data
    else:
        if "trajectories" not in data_dict[dict_key]:
            data_dict[dict_key]["trajectories"] = []
        if "trajectories" in data:
            for traj in data["trajectories"]:
                if traj["info"].startswith("positive"):
                    traj["info"] = "positive%d" % get_num(data_dict[dict_key]["trajectories"], "positive")
                    data_dict[dict_key]["trajectories"].append(traj)
                else:
                    traj["info"] = "negative%d" % get_num(data_dict[dict_key]["trajectories"], "negative")
                    data_dict[dict_key]["trajectories"].append(traj)

filtered_data = list(data_dict.values())

os.makedirs(NEW_TASK_DIR, exist_ok=True)
with open(os.path.join(NEW_TASK_DIR, "data.json"), 'w') as file:
    json.dump(filtered_data, file, indent=2)
