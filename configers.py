import glob
import os
import pdb

from auto_explore_sandbox import RepoCache
from functions.cost import KeywordCost, NumTokenCost, StepCost, SynthesizedCost
from nat_lang_envs import AutoExploreEnv, TaxiNatLangEnv

try:
    from alfworld.info import ALFWORLD_DATA

    from nat_lang_envs import AlfworldEnv
    from nat_lang_envs.alfworld_env import read_black_list
except:
    AlfworldEnv = None

from utils import build_curriculum_and_schedule, load_dataset


def auto_explore_configer(args, tokenizer, **kwargs):
    # Build the cost function
    num_token_cost = NumTokenCost(tokenizer)
    keyword_cost = KeywordCost(keywords=["Error", "Warning"], costs=[2, 1])
    step_cost = StepCost()
    synthesized_cost = SynthesizedCost(
        cost_functions=[num_token_cost, keyword_cost, step_cost], weights=[0, 1, 1])

    # Init repo cache
    repo_cache = RepoCache(original_root=args.repo_dir,
                           dir=args.sandbox_dir,
                           file_save_path="changed_files/")

    # Build dataset
    _dataset = load_dataset(args.task_file)

    dataset = []
    for d in _dataset:
        gpt_files = [traj["filename"] for traj in d["trajectories"] if traj["filename"] is not None]
        
        if gpt_files != []:
            d["filename"] = list(set(gpt_files + [d["filename"]]))
            dataset.append(d)
    
    dataset, trigger_set = build_curriculum_and_schedule(dataset, args)

    return (dataset, trigger_set, AutoExploreEnv,
            {"repo_cache": repo_cache,
             "cost_function": synthesized_cost,
             "succ_thresholds": [0.01] * len(dataset),})

def taxi_configer(args, **kwargs):
    bs = args.per_device_train_batch_size
    if args.taxi_curriculum:
        dataset = [["pickup" for _ in range(bs)],
                   ["dropoff" for _ in range(bs)],
                   ["full" for _ in range(bs)]]
        trigger_set = [0, args.max_steps // 3, 2 * args.max_steps // 3]
        succ_thresholds = [0, 0, -1]

    else:
        dataset = [[None for _ in range(args.per_device_train_batch_size)]]
        trigger_set = [0]
        succ_thresholds = [-1]
    
    if args.curriculum_index != -1:
        dataset = [dataset[args.curriculum_index]]
        trigger_set = [0]
        succ_thresholds = [succ_thresholds[args.curriculum_index]]

    return dataset, trigger_set, TaxiNatLangEnv, {"succ_thresholds": succ_thresholds}

def alfworld_configer(args, mode, **kwargs):
    if "train" in mode:
        _mode = "train"
    elif "test" in mode:
        _mode = "valid*"

    problems = glob.glob(os.path.join(ALFWORLD_DATA, "json_2.1.1", _mode,
                                      "**", "initial_state.pddl"),
                         recursive=True)
    # filtered_problems = [p for p in problems if "movable_recep" not in p]
    filtered_problems = [p.replace("initial_state.pddl", "") for p in problems if "pick_and_place_simple-Tomato-None-Microwave" in p]
    black_list = tuple(read_black_list())
    filtered_problems = [p for p in filtered_problems if not p.endswith(black_list)]

    dataset = [filtered_problems]
    trigger_set = [0]

    if args.few_data:
        dataset = [dataset[0][:args.few_data]]
        trigger_set = [0]

    return dataset, trigger_set, AlfworldEnv, {}


CONFIGERS = {
    "auto_explore": auto_explore_configer,
    "taxi": taxi_configer,
    "alfworld": alfworld_configer,
}