import os
import json
import random

from transformers import HfArgumentParser

from experiment_args import ScriptArguments

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

if os.path.isdir(script_args.task_file):
    files = [os.path.join(script_args.task_file, f) for f in os.listdir(script_args.task_file) if f.endswith(".json")]
    dir = script_args.task_file
else:
    files = [script_args.task_file]
    dir = os.path.dirname(script_args.task_file)

total_length = 0
file_lengths = {}

repo_data = {}

# Calculate total data length
for file in files:
    data = read_json_file(file)
    for d in data:
        repo = d['root']
        if d["question"] == "":
            continue
        
        if repo not in repo_data:
            repo_data[repo] = []
        repo_data[repo].append(d)

for data in repo_data.values():
    length = len(data)    # Assuming data is a list
    total_length += length

# Calculate lengths for each split
lengths = {
    "train": total_length * 0.7,
    "validate": total_length * 0.2,
    "test": total_length * 0.1, # all remaining data goes to test
}

# Distribute files
distributions = {'train': [], 'validate': [], 'test': []}
current_lengths = {'train': 0, 'validate': 0, 'test': 0}

for repo, data in repo_data.items():
    for d in data:
        split = random.choices(
            ["train", "validate", "test"],
            weights=[0.7, 0.2, 0.1],
            k=1)[0]
        distributions[split].append(d)
        current_lengths[split] += 1

for split, data in distributions.items():
    split_file = os.path.join(dir, split + ".json")
    with open(split_file, 'w') as file:
        json.dump(data, file, indent=2)
