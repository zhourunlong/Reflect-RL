import json
import os
import random
from statistics import mean

import torch
from termcolor import colored
from tqdm import tqdm
from transformers import GenerationConfig, HfArgumentParser

from configers import CONFIGERS
from evaluation.evaluate import batched_answer, calc_Q_values
from experiment_args import ScriptArguments
from model_utils import (CriticModel, create_and_prepare_model,
                         load_reflect_model)
from nat_lang_envs import ENVS
from trainers import TRAINERS
from utils import (ReplayBuffer, get_exp_id, load_script_args)

LOG_KEYS = ["step", "Q_value", "prob", "entropy", "cost", "prompt", "generation"]

def calc_avg(arr):
    _arr = arr.copy()
    if isinstance(arr[0], list):
        _arr = sum(_arr, [])
    filtered = [x for x in _arr if x is not None]
    return mean(filtered) if filtered != [] else 0

is_main_process = True
if "RANK" in os.environ:
    is_main_process = int(os.environ["RANK"]) == 0

parser = HfArgumentParser(ScriptArguments)
script_args = load_script_args(parser.parse_args_into_dataclasses()[0])
script_args.mode = "train_RLFT"

assert script_args.trainer in TRAINERS, f"Invalid trainer: {script_args.trainer}, must be one of {list(TRAINERS.keys())}."

assert script_args.env in ENVS, f"Invalid env: {script_args.env}, must be one of {list(ENVS.keys())}."

if script_args.disable_dropout:
    if script_args.lora_dropout != 0:
        print(colored("disable_dropout is set to True. lora_dropout is overridden to 0.", "yellow"))     
        script_args.lora_dropout = 0

# Setup policy network
tokenizer, peft_config, model, special_decoder = create_and_prepare_model(script_args)
reflect_tokenizer, reflect_model = load_reflect_model(script_args)

if is_main_process:
    exp_id = get_exp_id(script_args.ckpt_path)
    output_dir = script_args.ckpt_path + exp_id + "_rl_finetune/"
    os.makedirs(output_dir, exist_ok=True)

    # Saving the arguments for reference in the future
    script_args.dump(os.path.join(output_dir, "setting.yml"))

    print(colored("Experiment directory: " + output_dir, "green"))

optimizer = torch.optim.Adam(model.parameters(),
                             lr=script_args.learning_rate,
                             weight_decay=script_args.weight_decay)

if script_args.use_critic:
    # Setup value network, sharing the main body with policy network
    if script_args.shared_critic:
        critic_model = CriticModel(main_model=model,
                                   layer_type=script_args.critic_layer_type)
        critic_optimizer = torch.optim.Adam(critic_model.score.parameters(),
                                            lr=script_args.learning_rate,
                                            weight_decay=script_args.weight_decay)
    else:
        create_and_prepare_model(script_args)
        
else:
    critic_model, critic_optimizer = None, None

dataset, trigger_set, env_type, env_kwargs = CONFIGERS[script_args.env](script_args, tokenizer=tokenizer, **vars(script_args))

if "succ_thresholds" in env_kwargs:
    succ_thresholds = env_kwargs["succ_thresholds"]
else:
    succ_thresholds = [0] * len(dataset)

# Init curriculum
curriculum_idx = -1
cur_dataset = []

# Logs
losses, succs, critic_losses, costs = [], [], [], []
logs, msgs = [], []
train_logs, critic_train_logs = [], []

replay_buffer = ReplayBuffer(script_args.replay_buffer_size)

# Setup trainer
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

trainer_kwargs = {
    "model": model,
    "tokenizer": tokenizer,
    "optimizer": optimizer,
    "generation_config": generation_config,
    "critic_model": critic_model,
    "critic_optimizer": critic_optimizer,
    "ppo_clip_coef": script_args.ppo_clip_coef,
    "ppo_update_iter": script_args.ppo_update_iter,
    "max_grad_norm": script_args.max_grad_norm,
    "batch_size": script_args.per_device_train_batch_size,
    "entropy_coef": script_args.entropy_coef,
    "gradient_accumulation_steps": script_args.gradient_accumulation_steps,
    "critic_update_freq": script_args.critic_update_freq,
    "critic_update_iter": script_args.critic_update_iter,
}

trainer = TRAINERS[script_args.trainer](**trainer_kwargs)

for iter in (pbar := tqdm(range(script_args.max_steps), desc="Iter")):
    # Move on to the next curriculum
    if iter in trigger_set:
        curriculum_idx += 1
        # Replace dataset
        cur_dataset = dataset[curriculum_idx]
        idx = 0
        random.shuffle(cur_dataset)
        replay_buffer.clear()

    # get current batch
    batch = cur_dataset[idx:idx + script_args.per_device_eval_batch_size]
    idx += script_args.per_device_eval_batch_size
    if idx >= len(cur_dataset):
        idx = 0
        random.shuffle(cur_dataset)

    cur_logs = batched_answer(
        env_type=env_type,
        batch=batch,
        model=model,
        tokenizer=tokenizer,
        special_decoder=special_decoder,
        generation_config=generation_config,
        reflect_model=reflect_model,
        reflect_tokenizer=reflect_tokenizer,
        succ_threshold=succ_thresholds[curriculum_idx],
        **vars(script_args),
        **env_kwargs,
    )

    calc_Q_values(cur_logs, script_args.entropy_coef)

    # print(cur_logs)

    logs.append(cur_logs)

    # print("succ_threshold:", succ_thresholds[curriculum_idx])
    cost = mean([log[0]["Q_value"] for log in cur_logs])
    succ = mean([log[0]["Q_value"] < succ_thresholds[curriculum_idx] for log in cur_logs])
    costs.append(cost)
    succs.append(succ)

    for log in cur_logs:
        if log[0]["Q_value"] < succ_thresholds[curriculum_idx]:
            replay_buffer.add([{
                "data": log,
                "weight": 1 # exp(-log[0]["Q_value"] / script_args.horizon)
            }])
        #     print("Replay Buffer added:")
        # elif any(log[i]["cost"]  < 0 for i in range(len(log))):
        #     # We received rewards in some steps, but this episode is not successful
        #     # in this case, we should remove the last few steps that are wrong, and then
        #     # add to the buffer. Aka, we only add the meaningful steps into the buffer.
        #     cut_id = -1
        #     for i in range(len(log) - 1, 0, -1):
        #         if log[i]["cost"] < 0: 
        #             # we found the (partial) success point!
        #             cut_id = i + 1
        #             break
        #     if cut_id > 0:
        #         # Recalculate Q-value
        #         log_copy = [copy.deepcopy(log[0:cut_id])]
        #         calc_Q_values(log_copy, script_args.entropy_coef)
        #         print(colored("Selected Q-values:", "green"), 
        #               [step["Q_value"] for step in log_copy[0]])
        #         # pdb.set_trace()
        #         replay_buffer.add([{
        #             "data": log_copy[0],
        #             "weight": 0.1
        #         }])
        #         print(colored("Partial replay Buffer added:", "blue"))
        #         replay_buffer.print()
    
    # Train
    cur_loss, cur_critic_loss = [], []
    datas = [cur_logs, replay_buffer.sample(script_args.per_device_train_batch_size)]
    
    for data in datas:
        train_result = trainer.train(data)
        loss, critic_loss = train_result["loss"], train_result["critic_loss"]
        cur_loss.append(loss)
        cur_critic_loss.append(critic_loss)
    losses.append(cur_loss)
    critic_losses.append(cur_critic_loss)

    # Update tqdm
    avg_cost = calc_avg(costs[-script_args.logging_steps:])
    avg_succ = calc_avg(succs[-script_args.logging_steps:])
    avg_loss = calc_avg(losses[-script_args.logging_steps:])
    avg_critic_loss = calc_avg(critic_losses[-script_args.logging_steps:])
    pbar.set_description(
        "Cost: %.2f Succ Rate: %.2f Loss: %.2f Critic Loss: %.2f Iter:" %
        (avg_cost, avg_succ, avg_loss, avg_critic_loss))

    if is_main_process and (iter + 1) % script_args.save_steps == 0:
        ckpt_path = output_dir + "checkpoint-" + str(iter + 1) + "/"
        os.makedirs(ckpt_path, exist_ok=True)

        # dump the model
        model.save_pretrained(save_directory=ckpt_path)
        tokenizer.save_pretrained(save_directory=ckpt_path)
        if script_args.use_critic:
            critic_path = ckpt_path + "critic/"
            os.makedirs(critic_path, exist_ok=True)
            torch.save(critic_model.score.state_dict(),
                       critic_path + "score.pt")

        # dump the logs
        save_file = []

        for i in range(iter + 1 - script_args.save_steps, iter + 1):
            save_file.append({
                "iter":
                    i,
                "loss":
                    losses[i],
                "critic_loss":
                    critic_losses[i],
                "cost":
                    costs[i],
                "succ":
                    succs[i],
                "log": [{
                    "batch":
                        b,
                    "detail": [{k: lg[k] for k in LOG_KEYS
                    } for lg in ll]
                } for b, ll in enumerate(logs[i])],
            })

        with open(ckpt_path + "logs.json", "w") as file:
            json.dump(save_file, file, indent=2)

