import os
import pdb

from datasets import load_dataset
from transformers import HfArgumentParser, MistralConfig, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

from constants import ACTION_TEMPLATE, ANALYZE_TEMPLATE
from experiment_args import ScriptArguments
from model_utils import create_and_prepare_model
from utils import get_exp_id, load_script_args

parser = HfArgumentParser(ScriptArguments)
script_args = load_script_args(parser.parse_args_into_dataclasses()[0])
script_args.mode = "train_SPT"
script_args.shrink_head = False

assert script_args.shrink_head == False, "For SPT, shrink_head must be False."

script_args.save_steps = min(script_args.save_steps, script_args.max_steps)

exp_id = get_exp_id(script_args.ckpt_path)

training_arguments = TrainingArguments(
    output_dir=script_args.ckpt_path + exp_id + "_supervised_pretrain/",
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optim=script_args.optim,
    save_steps=script_args.save_steps,
    save_total_limit=script_args.save_total_limit,
    logging_steps=script_args.logging_steps,
    learning_rate=script_args.learning_rate,
    fp16=script_args.fp16,
    bf16=script_args.bf16,
    max_grad_norm=script_args.max_grad_norm,
    max_steps=script_args.max_steps,
    warmup_ratio=script_args.warmup_ratio,
    group_by_length=script_args.group_by_length,
    lr_scheduler_type=script_args.lr_scheduler_type,
    ddp_find_unused_parameters=False,
    gradient_checkpointing=script_args.gradient_checkpointing,
)

# Saving the arguments for reference in the future
os.makedirs(training_arguments.output_dir, exist_ok=True)
script_args.dump(os.path.join(training_arguments.output_dir, "setting.yml"))

tokenizer, peft_config, model, _ = create_and_prepare_model(script_args)
model.config.use_cache = False

dataset = load_dataset(
    "json",
    data_files=script_args.task_file,
    split="train").shuffle()

response_template = ANALYZE_TEMPLATE if script_args.reflect else ACTION_TEMPLATE
if isinstance(model.config, MistralConfig) or "mistral" in script_args.model_name.lower():
    response_template = response_template.lstrip()

collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer)

# pdb.set_trace()
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    peft_config=peft_config,
    max_seq_length=script_args.max_seq_length,
    tokenizer=tokenizer,
    data_collator=collator,
    args=training_arguments,
)

trainer.train()

del model, trainer
