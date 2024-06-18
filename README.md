# Reflect-RL: Two-Player Online RL Fine-Tuning for LMs

Authors: Runlong Zhou, Simon S. Du, and Beibin Li

We introduce Reflect-RL, a novel approach for fine-tuning language models (LMs) through online reinforcement learning (RL) in two-player interactive settings. By combining supervised fine-tuning (SFT) with an innovative online RL methodology, where a static reflection model aids the policy model, we enhance the LM's performance in complex, multi-round interaction tasks. Our empirical results demonstrate that GPT-2 XL models fine-tuned using Reflect-RL surpass both traditional SFT and larger, un-fine-tuned models in these scenarios.

Read our paper [here](https://arxiv.org/abs/2402.12621).

Before running the code, create and activate the environment:
```bash
conda env create -f environment.yml
conda activate reflectrl
```

## Part 1: Data Generation

We provide zipped data under the folder `data/`. If you would like to run our data generation pipeline to generate your own data, the first thing to do is to [configure your OpenAI API key](https://platform.openai.com/docs/quickstart).

### AutoExplore

AutoExplore is a standalone benchmark. Construction of the benchmark contains four parts: repositories, tasks, trajectories with reflections, and train-test split and SFT data.

We've already filtered 11 repos and generated the relevant data. They are compressed in `data/auto_explore.zip`, so you can use it off-the-shelf.

You are free to choose your own set of repos.

#### Repositories

Place the repos at `<REPO_DIR>`, e.g., `data/auto_explore/repos/`.
Filter out unsupported repo files using
```bash
python -m data_gen.auto_explore.filter_repo --repo_dir=<REPO_DIR>
```
The filtered repos will be saved at a fixed location: `data/auto_explore/repos_filtered/`.

#### Tasks

Run
```bash
python -m data_gen.auto_explore.gen_task
```
By default, it uses GPT-4 to generate tasks. You can manually change this setting in `data_gen/auto_explore/gen_task.py`. You can also customize this file to use AzureOpenAI endpoints or local models.
The task files will be put at a fixed location `data/auto_explore/tasks`, with separate `.json` files for each repo.
Filter tasks
```bash
python -m data_gen.auto_explore.filter_task --task_file=data/auto_explore/tasks/
```
The filtered tasks will be saved in a fixed single file `data/auto_explore/tasks_filtered/data.json`.

#### Trajectories with reflections

To generate trajectories with reflections for your own data, you can run
```bash
python -m data_gen.auto_explore.reflection_negative
```
It will save the trajectories to a fixed file `data/auto_explore/tasks_filtered/data_reflection_negative.json`.

#### Train-test split and SFT data
Split data into train, validation and test with proportion $7:2:1$ by running
```bash
python -m data_gen.auto_explore.split_data --task_file=data/auto_explore/tasks_filtered/data_reflection_negative.json
```
Finally generate SFT data using
```bash
python -m data_gen.auto_explore.gen_sft --task_file=data/auto_explore/tasks_filtered/train.json
```
The final SFT data will be put at `data/auto_explore/tasks_filtered/*.jsonl`.
`sft.jsonl` is for SFT only and SFT + RL (without reflection).
`sft_reflection.jsonl` and `sft_policy.jsonl` are for Reflect-RL reflection model and policy model, respectively.
`sft_reflection_positive.jsonl` and `sft_policy_positive.jsonl` are for Reflect-RL (without negative examples) reflection model and policy model, respectively.


### DangerousTaxi

To generate trajectories with reflections for your own data, you can run
```bash
python -m data_gen.taxi.reflection
```
Finally generate SFT data using
```bash
python -m data_gen.taxi.gen_sft
```

### ALFWorld

You need to clone the ALFWorld repo and then download the data.
```bash
git clone https://github.com/alfworld/alfworld
cd alfworld
pip install -e .
python scripts/alfworld-download --extra
```
To generate trajectories with reflections for your own data, you can run
```bash
python -m data_gen.alfworld.reflection
```
Finally generate SFT data using
```bash
python -m data_gen.alfworld.gen_sft
```

## Part 2: Training

### SFT

If you want single-GPU training, simply run
```bash
python supervised_pretrain.py <ARGS>
```
The definition of `<ARGS>` can be found in `experiment_args.py`.
For example in AutoExplore task, to use the default GPT-2 XL model with `bf16` and `8_bit` enabled, you can run the following commands for the reflection model and policy model:
```bash
python supervised_pretrain.py --max_steps=2000 --task_file=data/auto_explore/tasks_filtered/sft_reflection.jsonl
python supervised_pretrain.py --max_steps=2000 --task_file=data/auto_explore/tasks_filtered/sft_policy.jsonl
```
There is an output directory shown after each command. Record the directory for reflection model as `<REFLECT_DIR>` and that for policy model as `<POLICY_DIR>`.

We support `accelerate` for parallel training. Make sure to configure `accelerate` before each run if you want multiple-GPU training.

### RLFT

After SFT, run
```bash
python rl_finetune.py --env=<ENV> --load_dir=<POLICY_DIR> --reflect_load_dir=<REFLECT_DIR> <OTHER_ARGS>
```
For example, to RLFT a GPT2-XL model for AutoExplore and use curriculum learning, you can run
```bash
python rl_finetune.py --env=auto_explore --task_file=data/auto_explore/tasks_filtered/train.json --load_dir=results/123_supervised_pretrain/checkpoint-2000/ --reflectload_dir=results/124_supervised_pretrain/checkpoint-2000/ --save_steps=100 --shuffle_action=True --shrink_head --depth_curriculum --merge_first_two
```

In the current version, the behavior of using `accelerate` is nondetermined. We suggest using a single GPU for RLFT.

### Visualization

To plot the training curve in RLFT stage, edit `expdirs` in `visualization/plot.py`. Include all the **latest** directories, namely if you continue to RLFT directory `a` and the final directory is `b`, then include only `b`. Then run
```bash
python -m visualization.plot 
```

## Part 3: Evaluation

### Reflect-RL and baselines

`evaluation/evaluate.py` can evaluate several configurations **for the same environment** in a single run. You need to edit `EVAL_LOAD_LIST` in the main function then run
```bash
python -m evaluation.evaluate --env=<ENV> <OTHER_ARGS>
```
For example, to evaluate several configurations for AutoExplore, run
```bash
python -m evaluation.evaluate --env=auto_explore --task_file=data/auto_explore/tasks_filtered/test.json --per_device_eval_batch_size=10 --eval_reps=100 --shrink_head
```

### Open-source models and GPT

Evaluation of open-source models and GPT is slightly different from that of Reflect-RL and baselines, in that the prompts are more detailed to improve the performance for in-context inference.
To evaluate GPT, [configure your OpenAI API key](https://platform.openai.com/docs/quickstart). To evaluate an open-source model, we suggest hosting it using **OpenAI-compatible API** at `http://localhost:1234/v1`, e.g., using [LM Studio](https://lmstudio.ai/).
You can customize this address in `utils.py`:
```python
    elif api == "local":
        client = OpenAI(api_key="no key needed",
                        base_url="http://localhost:1234/v1")
```

#### AutoExplore

For example, you can evaluate your selected model by
```bash
python -m evaluation.auto_explore_gpt --task_file=data/auto_explore/tasks_filtered/test.json --eval_reps=10
```

#### DangerousTaxi and ALFWorld

For example, you can evaluate your selected model by
```bash
python -m evaluation.evaluate_gpt --scenario=alfworld --runs=100
```

## Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex
@misc{zhou2024reflectrl,
      title={Reflect-RL: Two-Player Online RL Fine-Tuning for LMs}, 
      author={Runlong Zhou and Simon S. Du and Beibin Li},
      year={2024},
      eprint={2402.12621},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```