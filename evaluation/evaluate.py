import random
from statistics import mean
from typing import  Tuple

from peft import PeftModel
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig, HfArgumentParser

from agent.ag_agent import ask_autogen
from configers import CONFIGERS
from constants import ACTION_TEMPLATE, ANALYZE_TEMPLATE
from experiment_args import ScriptArguments
from model_utils import (create_and_prepare_model, load_reflect_model,
                         transformer_text_completion)
from nat_lang_envs import ACTION_HEADS, TOY_TEXT_ENVS
from utils import (debug_msg, load_script_args)

ASK_AUTOGEN_REFLECTION = False
def batched_answer(
    env_type: type,
    batch: list,
    model: PeftModel,
    tokenizer: AutoTokenizer,
    special_decoder: list,
    reflect: bool,
    reflect_prob: float,
    reflect_model: PeftModel,
    reflect_tokenizer: AutoTokenizer,
    keywords: list = [],
    **kwargs,
 ) -> Tuple[list, list]:
    # init environemtns
    envs, prompts = [], []
    for i in range(len(batch)):
        envs.append(env_type(**kwargs))
        prompts.append(envs[i].reset(batch[i], **kwargs)[0])

    analyze_instruction = ""
    if env_type in TOY_TEXT_ENVS.values():
        analyze_instruction = open("data_gen/prompt_templates/toy_text/analyze_instruction.md", "r").read()
    
    action_head = ACTION_HEADS[env_type]

    logs = [[] for _ in range(len(batch))]
    step = 0
    while True:
        if reflect:
            _prompts = [analyze_instruction + p[:p.find(action_head)] + ANALYZE_TEMPLATE if p else None for p in prompts]

            generation_config = GenerationConfig(**vars(envs[0].generation_config))
            generation_config.max_new_tokens = 100

            # Reflection~
            if random.random() < reflect_prob:
                debug_msg("Reasoning start")
                debug_msg(_prompts[0], "white")

                if ASK_AUTOGEN_REFLECTION:
                    ret = []
                    for p in _prompts:
                        ans = ask_autogen(question=p + 
                                        "\nGive me one-sentence and concrete action: where I should move to?", 
                                        sys_msg="You should reflect and analyze the situation.")
                        ret.append({"generation": {"content": ans}})
                else:
                    ret = transformer_text_completion(
                        model=reflect_model,
                        tokenizer=reflect_tokenizer,
                        special_decoder=None,
                        prompts=_prompts,
                        generation_config=generation_config)
                    
                debug_msg("Reasoning finish")
                debug_msg(ret[0]["generation"]["content"], "green")
            else:
                ret = [{"generation": {"content": ""}} for p in _prompts]
            
            # print(_prompts[0])
            # print(ret[0]["generation"]["content"])
            
            # for i in range(len(ret)):
            #     assign_reward(ret[i], keywords[i])

            prompts_with_reflection = []
            for (_p, p, r) in zip(_prompts, prompts, ret):
                reflect_result = r["generation"]["content"]
                if p is None:
                    prompts_with_reflection.append(None)
                    continue

                if reflect_result.find(ACTION_TEMPLATE.strip()) != -1:
                    reflect_result = reflect_result[:reflect_result.find(ACTION_TEMPLATE.strip())]
                prompts_with_reflection.append(_p + reflect_result + p[p.find(action_head):])
            prompts = prompts_with_reflection
        
        prompts = [(p + ACTION_TEMPLATE) if p else None for p in prompts]

        # print("*" * 10)
        # print(prompts[0])

        debug_msg("Action start")

        ret = transformer_text_completion(
            model=model,
            tokenizer=tokenizer,
            special_decoder=special_decoder,
            prompts=prompts,
            generation_config=envs[0].generation_config)

        debug_msg("Action finish")
        
        prompts, dones = [], []
        for i in range(len(batch)):
            action = ret[i]["generation"]["content"]

            if action is None:
                prompts.append(None)
                dones.append(True)
                continue

            obs, reward, done, truncated, info = envs[i].step(ret[i]["generation"]["content"])
            if done or truncated:
                prompts.append(None)
                dones.append(True)
            else:
                prompts.append(obs)
                dones.append(False)
            
            ret[i].update({"cost": -reward, "step": step})
            logs[i].append(ret[i])

        step += 1
        
        if kwargs["first_step"] or all(dones):
            break

    return logs


def evalutate(
    dataset: list,
    per_device_eval_batch_size: int,
    env_type: type,
    model: PeftModel,
    tokenizer: AutoTokenizer,
    special_decoder: list,
    generation_config: GenerationConfig,
    reflect_model: PeftModel,
    reflect_tokenizer: AutoTokenizer,
    succ_threshold: float,
    **kwargs
):
    costs, succs = [], []
    batch_size = per_device_eval_batch_size

    for idx in tqdm(
            range(0, len(dataset), batch_size)):
        batch = dataset[idx:idx + batch_size]

        logs = batched_answer(
            batch=batch,
            env_type=env_type,
            model=model,
            tokenizer=tokenizer,
            special_decoder=special_decoder,
            generation_config=generation_config,
            reflect_model=reflect_model,
            reflect_tokenizer=reflect_tokenizer,
            **kwargs
        )

        calc_Q_values(logs)

        costs += [log[0]["Q_value"] for log in logs]
        succs += [log[0]["Q_value"] < succ_threshold for log in logs]

    return mean(costs), mean(succs)


def calc_Q_values(logs, entropy_coef=0):
    for log in logs:
        tot_cost = 0
        for i in range(len(log) - 1, -1, -1):
            tot_cost += log[i]["cost"] - entropy_coef * log[i]["entropy"]
            # tot_cost += log[i]["cost"] + entropy_coef * log[i]["log_prob"]
            log[i]["Q_value"] = tot_cost


if __name__ == "__main__":
    EVAL_LOAD_LIST = [
        {"env": "auto_explore",
         "load_dir": "results/587_rl_finetune/checkpoint-6500/",
         "reflect_load_dir": "results/582_supervised_pretrain/checkpoint-20000/",
        },
        # {"env": "auto_explore",
        #  "load_dir": "results/561_rl_finetune/checkpoint-5000/",
        #  "reflect_load_dir": None,
        #  },
    ]

    parser = HfArgumentParser(ScriptArguments)
    script_args = load_script_args(parser.parse_args_into_dataclasses()[0])
    script_args.mode = "test"
    script_args.leaveout_prob = 0

    if len(EVAL_LOAD_LIST) == 0:
        EVAL_LOAD_LIST = [
            {"env": script_args.env,
             "load_dir": script_args.load_dir,
             "reflect_load_dir": script_args.reflect_load_dir,},
        ]

    for load_dict in EVAL_LOAD_LIST:
        args = ScriptArguments(**vars(script_args))
        for k, v in load_dict.items():
            setattr(args, k, v)
        if args.reflect_load_dir is not None:
            args.reflect = True
            args.reflect_prob = 1
        
        print(f"Evaluating {args.load_dir}")

        # Setup policy network
        tokenizer, peft_config, model, special_decoder = create_and_prepare_model(args)
        reflect_tokenizer, reflect_model = load_reflect_model(args)
        model.eval()
        if reflect_model:
            reflect_model.eval()

        dataset, trigger_set, env_type, env_kwargs = CONFIGERS[args.env](script_args, tokenizer=tokenizer, **vars(args))

        if "succ_thresholds" in env_kwargs:
            succ_thresholds = env_kwargs["succ_thresholds"]
        else:
            succ_thresholds = [0] * len(dataset)

        generation_config = GenerationConfig(
            max_length=args.max_seq_length,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            num_beams=1,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        for i in range(len(dataset)):
            current_dataset = dataset[i] * args.eval_reps
            random.shuffle(current_dataset)
            cost, succ = evalutate(env_type=env_type,
                                    dataset=current_dataset,
                                    model=model,
                                    tokenizer=tokenizer,
                                    special_decoder=special_decoder,
                                    generation_config=generation_config,
                                    reflect_model=reflect_model,
                                    reflect_tokenizer=reflect_tokenizer,
                                    succ_threshold=succ_thresholds[i],
                                    **vars(args),
                                    **env_kwargs,)
                
            print("Curriculum %d: avg cost = %.2f, avg succ = %.2f" % (i, cost, succ))
