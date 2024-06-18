import json
import os
import pdb

from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig, HfArgumentParser

from auto_explore_copilot import AutoExploreCopilot
from constants import ACTION_TEMPLATE, ANALYZE_TEMPLATE
from experiment_args import ScriptArguments
from functions.cost import StepCost
from functions.terminate import AnytimeTerminate
from nat_lang_envs import ACTION_HEADS, AutoExploreEnv
from utils import load_dataset

ACTION_HEAD = ACTION_HEADS[AutoExploreEnv]

def dump(data: list, filename: str):
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps(d) for d in data]))

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

tokenizer = AutoTokenizer.from_pretrained(script_args.model_name,
                                          trust_remote_code=True,
                                          cache_dir=script_args.cache_dir)
tokenizer.pad_token = tokenizer.eos_token

sft_dataset, sft_reflection_dataset, sft_policy_dataset, sft_reflection_positive_dataset, sft_policy_positive_dataset = [], [], [], [], []

dataset = load_dataset(script_args.task_file)
task_dir = os.path.dirname(script_args.task_file)

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

copilot_col_name = "commands_copilot_cmds"
reason_col_name = "commands_reason"

for data in tqdm(dataset):
    root = data["root"]
    root = os.path.join(script_args.repo_dir, root)

    if "trajectories" not in data:
        continue

    for traj in data["trajectories"]:
        copilot = AutoExploreCopilot(repo_root=root,
                                     sandbox_dir=script_args.sandbox_dir,
                                     horizon=15,
                                     generation_config=generation_config,
                                     file_save_path="new_and_changed_files/",
                                     interaction_type="debug",
                                     model_type="null",
                                     model=None,
                                     tokenizer=None,
                                     cost_function=StepCost(),
                                     terminate_criteria=AnytimeTerminate(),
                                     leaveout_prob=0,
                                     shuffle_action=False,
                                     need_output_msgs=False)

        cmds = [step["cmd"] for step in traj["trajectory"]]
        reflections = [step["reflection"] for step in traj["trajectory"]]

        try:
            copilot.answer(question=data["question"],
                           target_files=[data["filename"]],
                           ans_cmds=cmds + ["exit"])
        except Exception as e:
            print(e)
            continue

        whole_msgs = copilot.whole_msgs

        if "negative" in traj["info"]:
            whole_msgs = whole_msgs[-1:]
            reflections = reflections[-1:]

        for reflection, msgs in zip(reflections, whole_msgs):
            prompt = "\n".join([msg[1] for msg in msgs[:-1]])
            before_action, after_action = prompt.split(ACTION_HEAD)

            text = prompt + ACTION_TEMPLATE + msgs[-1][1]
            reflection_text = before_action + ANALYZE_TEMPLATE + reflection
            policy_text = reflection_text + ACTION_HEAD + after_action + ACTION_TEMPLATE + msgs[-1][1]

            sft_dataset.append({"text": text})
            sft_reflection_dataset.append({"text": reflection_text})
            sft_policy_dataset.append({"text": policy_text})

            if "negative" not in traj["info"]:
                sft_reflection_positive_dataset.append({"text": reflection_text})
                sft_policy_positive_dataset.append({"text": policy_text})

dump(sft_dataset, 
        os.path.join(os.path.dirname(script_args.task_file), "sft.jsonl"))
dump(sft_reflection_dataset, 
    os.path.join(os.path.dirname(script_args.task_file), "sft_reflection.jsonl"))
dump(sft_policy_dataset, 
    os.path.join(os.path.dirname(script_args.task_file), "sft_policy.jsonl"))
dump(sft_reflection_positive_dataset,
    os.path.join(os.path.dirname(script_args.task_file), "sft_reflection_positive.jsonl"))
dump(sft_policy_positive_dataset,
    os.path.join(os.path.dirname(script_args.task_file), "sft_policy_positive.jsonl"))
