import os
import json
import pdb

import matplotlib.pyplot as plt
import numpy as np

from visualization.ensemble_logs import ensemble_logs

window_size = 99
max_steps = 99999
KEYS = ["succ", "cost", "loss"]

# Function to calculate moving average with handling null values
def moving_average(data, window_size):
    if isinstance(data[0][1], list):
        data = sum([[(x[0], x[1][0]), (x[0], x[1][1])] for x in data], [])


    result = []
    right = max([i for i, x in enumerate(data) if x[0] <= window_size // 2]) + 1
    left = 0
    interval_sum = sum([x[1] for x in data[:right] if x[1] is not None])

    for i in range(len(data)):
        result.append(interval_sum / (right - left))

        while right < len(data) and data[right][0] <= data[i][0] + window_size // 2:
            interval_sum += data[right][1] if data[right][1] is not None else 0
            right += 1
        
        while left < len(data) and data[left][0] < data[i][0] - window_size // 2:
            interval_sum -= data[left][1] if data[left][1] is not None else 0
            left += 1

    return np.array([x[0] for x in data]), np.array(result)


# Experiment directories
expdirs = [
    "results/561_rl_finetune/",
    "results/587_rl_finetune/",
]

# Initialize dictionaries to store data for plotting
data_map = {key: {} for key in KEYS}

# Process each experiment directory
for expdir in expdirs:
    for key in KEYS:
        data_map[key][expdir] = []

    logs = ensemble_logs(expdir)
    # Extract 'iter', 'loss', and 'cost' values
    for log in logs:
        iter = log['iter']
        for key in KEYS:
            val = log[key] if key in log and log[key] is not None else None
            
            # Append data for plotting
            if iter < max_steps:
                data_map[key][expdir].append((iter, val))

    for key in KEYS:
        data_map[key][expdir].sort(key=lambda x: x[0])

# Create plots
fig, ax = plt.subplots(len(KEYS), 1, figsize=(10, 12))

for i, key in enumerate(KEYS):
    min_val, max_val = float('inf'), -float('inf')
    for expdir, data in data_map[key].items():
        iters, smoothed_vals = moving_average(data, window_size)

        indexes = np.where(smoothed_vals != None)[0] # DONT USE 'is not None' HERE

        min_val = min(min_val, min(smoothed_vals[indexes]))
        max_val = max(max_val, max(smoothed_vals[indexes]))

        ax[i].plot(iters[indexes],
                   smoothed_vals[indexes],
                   label=f'{expdir.split("/")[-2]} - Smoothed',
                   linestyle='--')
    
    lower_bound = min(0, min_val * 1.1)
    upper_bound = max(0, max_val * 1.1)
    
    ax[i].set_ylim(lower_bound, upper_bound)

    ax[i].set_xlabel('Iteration')
    ax[i].set_ylabel(key.capitalize())
    ax[i].set_title(f'{key.capitalize()} curve')
    ax[i].legend()

plt.tight_layout()
plt.savefig("visualization.pdf", format="pdf", bbox_inches="tight")
