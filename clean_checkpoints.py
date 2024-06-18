import os

DELETE_FILES = ["adapter_model.bin", "merges.txt", "tokenizer.json", "vocab.json", "adapter_model.safetensors", "optimizer.pt"]

# clear all the checkpoints
exps = os.listdir("results")
exps = [e for e in exps if "rl_finetune" in e]
for exp in exps:
    checkpoints = os.listdir("results/" + exp)
    checkpoints = [c for c in checkpoints if c.startswith("checkpoint")]
    # get the latest checkpoint by system time
    checkpoints.sort(key=lambda x: os.path.getmtime("results/" + exp + "/" + x))
    checkpoints = checkpoints[:-1]

    for checkpoint in checkpoints:
        for file in DELETE_FILES:
            path = "results/" + exp + "/" + checkpoint + "/" + file
            if os.path.exists(path):
                os.remove(path)
