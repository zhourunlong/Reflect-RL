import os
import json
import pdb
import yaml

import matplotlib.pyplot as plt
import numpy as np

from experiment_args import ScriptArguments

def ensemble_logs(expdir: str) -> list:
    cur_dir = expdir
    limit = 999999999
    trace_stack_logs, trace_stack_gradaccumustep, trace_stack_offsets = [], [], []
    while True:
        script_args = ScriptArguments()
        script_args.load(os.path.join(cur_dir, "setting.yml"))

        cur_logs = []
        for checkpoint_folder in os.listdir(cur_dir):
            checkpoint_path = os.path.join(cur_dir, checkpoint_folder, 'logs.json')
            # Check if logs.json exists
            if os.path.exists(checkpoint_path):
                with open(checkpoint_path, 'r') as file:
                    logs = json.load(file)
                    cur_logs += [lg for lg in logs if lg["iter"] < limit]
        trace_stack_logs.append(cur_logs)
        trace_stack_gradaccumustep.append(script_args.gradient_accumulation_steps)
        
        cur_dir = script_args.load_dir
        if script_args.load_dir is None or "rl_finetune" not in script_args.load_dir:
            break
        limit = int(cur_dir.split("checkpoint-")[-1][:-1])
        cur_dir = os.path.join(cur_dir, "..")
        trace_stack_offsets.append(limit / script_args.gradient_accumulation_steps)
    
    trace_stack_offsets.append(0)
    for i in range(len(trace_stack_offsets) - 2, -1, -1):
        trace_stack_offsets[i] = trace_stack_offsets[i] / trace_stack_gradaccumustep[i + 1] + trace_stack_offsets[i + 1]
    
    whole_logs = []
    for i in range(len(trace_stack_logs)):
        for j in range(len(trace_stack_logs[i])):
            trace_stack_logs[i][j]["iter"] = trace_stack_logs[i][j]["iter"] / trace_stack_gradaccumustep[i] + trace_stack_offsets[i]
        whole_logs += trace_stack_logs[i]
    whole_logs.sort(key=lambda x: x["iter"])
    
    return whole_logs

if __name__ == "__main__":
    # Specify the latest experiment directories, this script will trace back previous ones
    expdirs = [
        "results/531_rl_finetune/",
        "results/536_rl_finetune/",
    ]

    # Process each experiment directory
    for expdir in expdirs:
        whole_logs = ensemble_logs(expdir)

        exp_name = expdir.split("/")[-2]
        with open(exp_name + ".json", "w") as file:
            json.dump(whole_logs, file, indent=2)