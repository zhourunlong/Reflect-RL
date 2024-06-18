import os
from tqdm import tqdm

from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig, HfArgumentParser

from auto_explore_copilot import AutoExploreCopilot
from configers import CONFIGERS
from experiment_args import ScriptArguments
from functions.terminate import IdentifyFileTerminate
from utils import load_script_args

MODEL_TYPE = "remote_OpenAI"
# MODEL_TYPE = "remote_local"
# MODEL_NAME = "gpt-3.5-turbo"
MODEL_NAME = "gpt-4"

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = load_script_args(parser.parse_args_into_dataclasses()[0])
    script_args.mode = "test"
    script_args.leaveout_prob = 0

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name,
                                              trust_remote_code=True,
                                              cache_dir=script_args.cache_dir)

    dataset, trigger_set, env_type, env_kwargs = CONFIGERS["auto_explore"](script_args, tokenizer=tokenizer, **vars(script_args))

    generation_config = GenerationConfig(
        max_length=32767,
        max_new_tokens=script_args.max_new_tokens,
        do_sample=True,
        num_beams=1,
        temperature=script_args.temperature,
        top_p=script_args.top_p,
        top_k=script_args.top_k,
    )

    for i in range(len(dataset)):
        succ = 0
        current_dataset = dataset[i] * script_args.eval_reps
        for data in tqdm(current_dataset):
            root = data["root"]
            root = os.path.join(script_args.repo_dir, root)
            copilot = AutoExploreCopilot(
                repo_root=root,
                sandbox_dir=script_args.sandbox_dir,
                file_save_path="tmp",
                horizon=15,
                generation_config=generation_config,
                interaction_type="gen_data",
                model_type=MODEL_TYPE,
                model_name=MODEL_NAME,
                cost_function=env_kwargs["cost_function"],
                terminate_criteria=IdentifyFileTerminate(data["filename"]),
                shuffle_action=False,
                need_output_msgs=False,
                markov=False,
            )

            try:
                copilot.answer(question=data["question"],
                               target_files=data["filename"])
            except Exception as e:
                print(e, "Failed to generate GPT trajectory for question %s" % data["question"])
                continue
            
            if sum(copilot._costs) <= 0:
                succ += 1

        print("Level %d: %.2f" % (i, succ / len(current_dataset) * 100))
