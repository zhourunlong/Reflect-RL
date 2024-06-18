"""Generate SFT data for taxi env.
1. sft.json: state, action
2. sft_reflection.json: state, reflection
3. sft_policy.json: state, reflection, action

Oracle is also included in every file, where reflection is "". 
"""
import copy
import json
import os
import pdb
import re

from transformers import AutoTokenizer, GenerationConfig, HfArgumentParser

from constants import ACTION_TEMPLATE, ANALYZE_TEMPLATE
from experiment_args import ScriptArguments
from nat_lang_envs import ACTION_HEADS, TaxiNatLangEnv

DEL_STR_LIST = ["\nGive me the numeric Action ID: ", "# Response:\n"]
ACTION_HEAD = ACTION_HEADS[TaxiNatLangEnv]

def dump(data: list, filename: str, mode:str = "w"):
    with open(filename, mode) as f:
        f.write("\n".join([json.dumps(d) for d in data]))
    
def load_dataset(task_file):
    dataset = []
    if task_file.endswith(".json"):
        task_files = [task_file]
    else:
        task_files = [
            os.path.join(task_file, f)
            for f in os.listdir(task_file)
            if f.endswith(".json")
        ]
    for task_file in task_files:
        dataset += json.load(open(task_file, "r"))
    return dataset


def create_dataset(task_file):
    dataset = load_dataset(task_file)
    sft_dataset, sft_reflection_dataset, sft_policy_dataset = [], [], []

    skip_count = 0
    for data in dataset:

        prompt = copy.deepcopy(data["state"])
        if "--- Initial State ---" in prompt:
            prompt = prompt[prompt.find("--- Initial State ---"):]
        elif "--- Current State ---" in prompt:
            prompt = prompt[prompt.find("--- Current State ---"):]
        else:
            skip_count += 1
            continue

        # Remove line with cheating info
        # prompt = re.sub(r"\nNote that .*optimal\n", "\n", prompt)  
        prompt = re.sub(r"\nNote that .*optimal.*\n", "\n", prompt)
        prompt = re.sub(r"\nSpoiler: .*\n", "\n", prompt)

        # if "dropoff" in task_file and "optimal" in data["state"]:
        #     pdb.set_trace()

        # Remove "# Response" from the prompt if exists
        for string in DEL_STR_LIST:
            if string in prompt:
                prompt = prompt.replace(string, "")
        prompt = prompt.strip()

        before_action, after_action = prompt.split(ACTION_HEAD)

        text = prompt + ACTION_TEMPLATE + str(data["action"])
        reflection_text = before_action + "\n" + ANALYZE_TEMPLATE + data["reflection"]
        policy_text = reflection_text + ACTION_HEAD + after_action + ACTION_TEMPLATE + str(data["action"])

        sft_dataset.append({"text": text})
        sft_reflection_dataset.append({"text": reflection_text})
        sft_policy_dataset.append({"text": policy_text})

    print(f"Total {skip_count} of {len(dataset)} reflections are skipped because of their format is in older version.")
    return sft_dataset, sft_reflection_dataset, sft_policy_dataset 

if __name__ == "__main__":
    output_dir = "data/taxi/"
    
    # Handle "Reflection" data
    task_file = output_dir + "data_reflection.json"
    sft_dataset, sft_reflection_dataset, sft_policy_dataset = create_dataset(task_file)
    dump(sft_dataset, 
        os.path.join(output_dir, "sft.jsonl"), "w")
    dump(sft_reflection_dataset, 
        os.path.join(output_dir, "sft_reflection.jsonl"), "w")
    dump(sft_policy_dataset, 
        os.path.join(output_dir, "sft_policy.jsonl"), "w") 
    del sft_dataset, sft_reflection_dataset, sft_policy_dataset 
