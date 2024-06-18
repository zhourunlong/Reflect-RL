from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class ScriptArguments:
    """
    These arguments vary depending on model location, data location,
    what their capacity, features, etc.
    """

    per_device_train_batch_size: Optional[int] = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=4)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    eval_reps: Optional[int] = field(default=10)
    mode: Optional[str] = field(default="train")
    learning_rate: Optional[float] = field(default=2e-4)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[int] = field(default=0.001)
    ppo_clip_coef: Optional[float] = field(default=0.1)
    ppo_update_iter: Optional[int] = field(default=3)
    entropy_coef: Optional[float] = field(default=0.0)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    disable_dropout: Optional[bool] = field(default=False)
    lora_r: Optional[int] = field(default=64)
    max_seq_length: Optional[int] = field(default=1024)
    max_new_tokens: Optional[int] = field(default=1)
    temperature: Optional[float] = field(default=1)
    top_p: Optional[float] = field(default=1)
    top_k: Optional[int] = field(default=99999)
    model_name: Optional[str] = field(
        default="gpt2-xl",
        metadata={
            "help": "The model that you want to train from the Hugging "
                    "Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    load_dir: Optional[str] = field(
        default=None,
        metadata={
            "help":
                "Where to load the pretrained models. None for no loading. "
                "latest for latest checkpoint. directory for loading from a "
                "directory."
        })
    reflect_model_name: Optional[str] = field(
        default="gpt2-xl",
        metadata={
            "help": "The model that you want to use as reflectioner."
        },
    )
    reflect_load_dir: Optional[str] = field(
        default=None,
        metadata={
            "help":
                "Where to load the pretrained reflectioner. None for no loading. "
                "latest for latest checkpoint. directory for loading from a "
                "directory."
        })
    ckpt_path: Optional[str] = field(
        default="results/",
        metadata={
            "help": "The location to save the experiment checkpoints. It "
                    " should be the folder with all experiments."
        })
    use_8bit: Optional[bool] = field(
        default=True,
        metadata={"help": "Activate 8bit precision base model loading"},
    )
    use_4bit: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate 4bit precision base model loading"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={
            "help": "The number of training epochs for the reward model."
        },
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables bf16 training."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="paged_adamw_32bit",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={
            "help": "Learning rate schedule. Constant a bit better than "
                    "cosine, and has advantage for analysis"
        },
    )
    max_steps: int = field(
        default=20000,
        metadata={"help": "How many optimizer "
                          "update steps to take"})
    warmup_ratio: float = field(
        default=0.03,
        metadata={"help": "Fraction of "
                          "steps to do a warmup for"})
    group_by_length: bool = field(
        default=False,
        metadata={
            "help": "Group sequences into batches with same length. Saves "
                    "memory and speeds up training considerably."
        },
    )
    save_steps: int = field(
        default=100,
        metadata={"help": "Save checkpoint "
                          "every X updates steps."})
    save_total_limit: int = field(
        default=10,
        metadata={
            "help": "Limit the total amount of checkpoints. "
                    "Deletes the older checkpoints."
        })
    logging_steps: int = field(default=10,
                               metadata={"help": "Log every X updates steps."})
    cache_dir: Optional[str] = field(
        default="model/",
        metadata={"help": "Where to store the pretrained models."})
    
    # Environment
    env: str = field(
        default="auto_explore",
        metadata={
            "help": "The env to run in. Could be auto_explore, frozen_lake, taxi, alfworld."
        })

    # Trainer
    trainer: Optional[str] = field(
        default="pg",
        metadata={
            "help": "The RL trainer to use. Could be pg (policy gradient), ppo "
                    "(proximal policy optimization)."
        },
    )
    replay_buffer_size: Optional[int] = field(default=50)


    # Critic
    use_critic: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether use critic in RL finetuning."})
    critic_update_freq: Optional[int] = field(
        default=5,
        metadata={"help": "Update critic model after X model update steps."})
    critic_update_iter: Optional[int] = field(
        default=5, metadata={"help": "Update critic model X times per update."})
    shared_critic: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to share the critic model with the actor."})
    critic_layer_type: Optional[str] = field(
        default="linear",
        metadata={
            "help": "The type of critic layer. Could be linear or mlp."
        },
    )

    # Model customization
    shrink_head: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to shrink the final LM head of the model."})
    
    ###############
    # Frozen Lake #
    ###############
    map_size: Optional[int] = field(
        default=4,
        metadata={"help": "The size of the map in frozen lake."})
    random_map: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use random map in frozen lake."})
    is_slippery: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use slippery mode in frozen lake."})

    ################
    # Auto Explore #
    ################
    # Copilot arguments
    horizon: Optional[int] = field(
        default=15,
        metadata={
            "help": "The horizon (number of interactions) for each episode."
        },
    )
    task_file: Optional[str] = field(
        default="data/auto_explore/tasks_filtered/train.json",
        metadata={
            "help": "The path to the task file. Could be a directory or a "
                    "specific file. All files should contain the path of "
                    "associated repositories."
        },
    )
    repo_dir: Optional[str] = field(
        default="data/auto_explore/repos_filtered/",
        metadata={
            "help": "The path to the directory containing the repositories."
        },
    )
    sandbox_dir: Optional[str] = field(
        default="/dev/shm/",
        metadata={
            "help": "The path to the directory for sandbox temporary files."
        },
    )

    ############
    # Alfworld #
    ############
    discretize_actions: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to discretize the actions in Alfworld."})
    disable_alfworld: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to disable Alfworld."})

    # Curriculum Learning
    first_step: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether only train the first step, viewing as a contextual"
                    " bandit problem."
        })
    curriculum_index: Optional[int] = field(
        default=-1,
        metadata={
            "help":
                "The index of the curriculum to use, starting from 0. -1 for"
                " all curriculum."
        })
    few_data: Optional[int] = field(
        default=0,
        metadata={"help": "Whether only use a small portion of fixed data."})
    ## Auto Explore
    easy: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether use easy task (file finding)."})
    reflect: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether use reflection."})
    reflect_prob: Optional[float] = field(
        default=1,
        metadata={"help": "The probability to use reflection."})
    
    leaveout_prob: Optional[float] = field(
        default=0.5,
        metadata={
            "help":
                "The probability to leave out unrelated files when training."
        })
    shuffle_action: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to shuffle the actions in the copilot."})
    depth_curriculum: Optional[bool] = field(
        default=False,
        metadata={
            "help":
                "Whether use depth curriculum: sort the target files by their"
                " depth, and train in increasing order."
        })
    merge_first_two: Optional[bool] = field(
        default=False,
        metadata={
            "help":
                "Whether to merge the first two steps in the curriculum."
        })
    merge_after_first_k: Optional[int] = field(
        default=3,
        metadata={
            "help":
                "Whether to merge steps after the first k steps in the "
                "curriculum."
        })
    ## Toy Text
    with_prompt: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use prompt in the OpenAI gym tasks."})
    with_history: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use history in the prompt."})
    ### Taxi
    taxi_curriculum: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to set a curriculum for taxi."})
    

    def load(self, yaml_file: str):
        with open(yaml_file, 'r') as file:
            yaml_data = yaml.safe_load(file)

        for key, value in yaml_data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def dump(self, filename: str):
        with open(filename, 'w') as file:
            yaml.dump(self.__dict__, file)
