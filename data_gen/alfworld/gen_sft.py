import json
import os
import pdb

from transformers import AutoTokenizer, GenerationConfig, HfArgumentParser

from constants import ACTION_TEMPLATE, ANALYZE_TEMPLATE
from experiment_args import ScriptArguments
from nat_lang_envs import ACTION_HEADS, AlfworldEnv

PROBLEM_FILTER = ["pick_and_place_simple-Tomato-None-Microwave"]
DEL_STR_LIST = ["\nGive me the numeric Action ID: ", "# Response:\n"]
ACTION_HEAD = ACTION_HEADS[AlfworldEnv]

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

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

tokenizer = AutoTokenizer.from_pretrained(script_args.model_name,
                                          trust_remote_code=True,
                                          cache_dir=script_args.cache_dir)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset(script_args.task_file)

generation_config = GenerationConfig(
    max_length=script_args.max_seq_length,
    max_new_tokens=script_args.max_new_tokens,
    do_sample=True,
    num_beams=1,
    temperature=script_args.temperature,
    top_p=script_args.top_p,
    top_k=script_args.top_k,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

sft_dataset, sft_reflection_dataset, sft_policy_dataset = [], [], []

for data in dataset:
    # acceptable_prob = True
    try:
        acceptable_prob = False
        for filter in PROBLEM_FILTER:
            if filter in data["problem"]:
                acceptable_prob = True
                break
    except:
        acceptable_prob = True

    # pdb.set_trace()

    if not acceptable_prob:
        continue

    if data["reflection"] is None or data["reflection"] == "":
        continue

    prompt = data["state"]
    for string in DEL_STR_LIST:
        if string in prompt:
            prompt = prompt.replace(string, "")
    prompt = prompt.strip()

    before_action, after_action = prompt.split(ACTION_HEAD)

    text = prompt + ACTION_TEMPLATE + str(data["action"])
    reflection_text = before_action + ANALYZE_TEMPLATE + data["reflection"]
    policy_text = reflection_text + ACTION_HEAD + after_action + ACTION_TEMPLATE + str(data["action"])

    sft_dataset.append({"text": text})
    sft_reflection_dataset.append({"text": reflection_text})
    sft_policy_dataset.append({"text": policy_text})


dump(sft_dataset, 
    os.path.join(os.path.dirname(script_args.task_file), "sft.jsonl"))
dump(sft_reflection_dataset, 
    os.path.join(os.path.dirname(script_args.task_file), "sft_reflectioner.jsonl"))
dump(sft_policy_dataset, 
    os.path.join(os.path.dirname(script_args.task_file), "sft_policy.jsonl"))

