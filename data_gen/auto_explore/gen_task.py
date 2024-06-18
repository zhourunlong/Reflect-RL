"""Generate exploration problem set (Chat history).
Find a file, write a question, and create a path"""
import glob
import json
import os
import pdb
import random
import re
import string
import time
from typing import List, Tuple

from termcolor import colored
from tqdm import tqdm

from constants import (ALLOWED_FILE_SUFFIXES, CODE_SUFFIXES, DATA_SUFFIXES,
                       EXEC_SUFFIXES, TEXT_SUFFIXES)
from utils import reply, wrap_path


MAX_LEVEL = 2

API = "OpenAI"
MODEL_NAME = "gpt-4"
# MODEL_NAME = "gpt-3.5-turbo"

# API = "AzureOpenAI"
# MODEL_NAME = "gpt-4"
# MODEL_NAME = "gpt-35-turbo"

# API = "local"
# MODEL_NAME = "mistral"


TEMPLATE_PATH = "data_gen/prompt_templates/auto_explore/"
CODE_PROMPT_TEMPLATE = open(TEMPLATE_PATH + "code_question.md", "r").read()
TEXT_PROMPT_TEMPLATE = open(TEMPLATE_PATH + "text_question.md", "r").read()
DATA_PROMPT_TEMPLATE = open(TEMPLATE_PATH + "data_question.md", "r").read()

REPOS_ROOT = "data/auto_explore/repos_filtered/"
OUT_DIR = "data/auto_explore/tasks/"

MAX_COMMAND_STEPS = 100

MIN_TOKENS = 50
MAX_TOKENS = 2000

def extract_QAs(response: str) -> list:
    """Extract the question and answer from the response.

    Args:
        response (str): The response from GPT.

    Returns:
        list: A list of tuples, where each tuple is a question and its answer.
    """
    # Split the response into lines
    lines = response.split("\n")

    # Initialize an empty list to store the questions and answers
    QAs = []

    # Initialize a variable to store the current question
    curr_question = ""

    # Loop through the lines
    for line in lines:
        # Check if the line starts with "QUESTION:"
        if "QUERY" in line[:10]:
            i = line.find("QUERY") + len("QUERY")
            while line[i] in (list(string.digits) + [" ", ":"]):
                i += 1
            curr_question = line[i:]
        # Check if the line starts with "ANSWER:"
        elif "ANSWER" in line[:10]:
            i = line.find("ANSWER") + len("ANSWER")
            while line[i] in (list(string.digits) + [" ", ":"]):
                i += 1
            QAs.append((curr_question, line[i:]))

    return QAs


if MODEL_NAME == "mistral":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

    def num_words(sentence):
        tokens = tokenizer([sentence], return_tensors="pt").to("cpu")
        return int(tokens["input_ids"].shape[1])
elif "gpt-" in MODEL_NAME:
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")

    def num_words(sentence):
        return len(encoding.encode(sentence))
else:
    raise NotImplementedError


def random_files(root: str,
                 suffix: list,
                 ignore_regex: list,
                 n_files=1000) -> str:
    # Initialize an empty list to store the filenames that
    # match the suffix criteria
    matching_files = []

    # Traverse through the directory tree starting from the root
    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            # Check if the file has one of the defined suffixes
            if any(filename.endswith(s) for s in suffix):
                # The file extension matches one of the suffixes
                dirpath = dirpath.replace("\\", "/")
                folder_paths = dirpath.split("/")
                if any(f.startswith(".") for f in folder_paths):
                    # Skip: the file is in a hidden folder
                    continue

                fname = os.path.join(dirpath, filename)
                if any(re.search(regex, fname) for regex in ignore_regex):
                    # The file name matches one of the ignore regexes
                    continue
                # Append the full path of the file to the list
                matching_files.append(fname)
    if not matching_files:
        return []

    # Randomly select a file from the list of matching files
    return random.choices(matching_files, k=min(n_files, len(matching_files)))


def random_block(filename: str,
                 max_tokens: int = -1,
                 first_k_lines: int = -1) -> str:
    """
    Given a large file, return a random chunk of text (split by lines)
    containing up to n_max_words words.

    Parameters:
        filename (str): The path to the file from which to read.
        n_max_words (int): The maximum number of words that the random block of
            text should contain.

    Returns:
        str: A string containing the random block of text from the file.
    """
    # Initialize an empty list to store all lines from the file
    all_lines = []

    # Read all lines from the file and store them in the list
    with open(filename, 'r') as f:
        all_lines = f.readlines()

    # If the file is empty or contains no lines, return an empty string
    if not all_lines:
        return ""

    if first_k_lines > 0:
        all_lines = all_lines[:first_k_lines]

    num_tokens = [num_words(line) for line in all_lines]
    tot_tokens = sum(num_tokens)

    # File too short
    if tot_tokens < MIN_TOKENS:
        return ""
    if tot_tokens <= max_tokens:
        return "\n".join(all_lines)

    tot = 0
    for end_line in range(len(all_lines) - 1, -1, -1):
        tot += num_tokens[end_line]
        if tot >= max_tokens:
            break

    # Randomly choose a starting line index
    start_idx = random.randint(0, end_line)

    # Initialize variables to keep track of the number of words and the
    # selected lines
    n_words = 0
    selected_lines = []

    # Loop to collect lines until n_max_words is reached or the end of
    # the file is reached
    for i in range(start_idx, len(all_lines)):
        n_words += num_tokens[i]

        if n_words > max_tokens:
            break

        selected_lines.append(all_lines[i])

    return '\n'.join(selected_lines)


def optimal_path(start: str, destination: str) -> List[str]:
    """Use Linux's "ls", "cd", and "cat command to explore the path,
    from `start` to the `destination`.

    Note that you are unfamiliar with the path, so you may need to "ls" to see
    the content inside a folder.

    It is guaranteed that the start and destination exist.

    Args:
        start (str): a path
        destination (str): filename, a file which we want to find.
        folder_find_acc (float, optional): the probability of finding a
            correct folder. Defaults to 0.8.

    Returns:
        commands (list): a list of commands
    """
    assert os.path.isdir(start)
    assert os.path.isfile(destination)

    folders = os.path.relpath(destination, start).split("/")
    commands = [
        cmd for dirname in folders[:-1]
        for cmd in ["ls", f"cd {wrap_path(dirname)}"]
    ] + ["ls", f"cat {wrap_path(folders[-1])}"]

    return commands


def gen_question(root: str, filename: str) -> Tuple[str, str]:
    if "." in filename:
        suffix = "." + filename.split(".")[-1]
        if suffix in (CODE_SUFFIXES + EXEC_SUFFIXES):
            file_type = "code"
        elif suffix in TEXT_SUFFIXES:
            file_type = "text"
        elif suffix in DATA_SUFFIXES:
            file_type = "data"
    else:
        file_type = "text"

    if file_type == "code":
        block = random_block(filename, MAX_TOKENS)
        prompt = CODE_PROMPT_TEMPLATE.format(NAME=os.path.relpath(
            filename, root),
                                             CONTENT=block)
    elif file_type == "text":
        block = random_block(filename, MAX_TOKENS)
        prompt = TEXT_PROMPT_TEMPLATE.format(NAME=os.path.relpath(
            filename, root),
                                             CONTENT=block)
    elif file_type == "data":
        block = random_block(filename, MAX_TOKENS, first_k_lines=20)
        prompt = DATA_PROMPT_TEMPLATE.format(NAME=os.path.relpath(
            filename, root),
                                             CONTENT=block)

    if block == "":
        return []

    chat_history = [
        {
            "role": "system",
            "content": "You are a helpful assistant to help me generate data."
        },
        {
            "role": "user",
            "content": prompt
        },
    ]

    response = reply(
        api=API,
        chat_history=chat_history,
        model_name=MODEL_NAME,
    )
    QAs = extract_QAs(response)

    return QAs


def gen(root: str, outname: str):
    root = root.replace("\\", "/")

    if os.path.exists(outname):
        rst = json.load(open(outname, "r"))
        if rst == []:
            os.remove(outname)
            random.seed(time.time())
        else:
            print(f"Loaded {len(rst)} results from {outname}, continue...")

    rst = []
    file_list = random_files(
        root=root,
        suffix=ALLOWED_FILE_SUFFIXES,
        ignore_regex=[".*out.*", ".*\.git.*", ".*test.*", "__init__.py"],
        n_files=100)

    for filename in file_list:
        filename = filename.replace("\\", "/")

        existing_files = [r["filename"] for r in rst]
        if filename in existing_files:
            # Don't generate questions for the same file again
            continue

        optimal = optimal_path(root, filename)    # the optimal path
        n_level = len([cmd for cmd in optimal if cmd.startswith("cd ")])
        if n_level > MAX_LEVEL:
            continue

        print("Generating questions for file:", colored(filename, "green"))
        try:
            pairs = gen_question(root, filename)
        except Exception as e:
            print(e)
            print(colored("Fail to generate questions for file: ", "red"),
                  filename)
            continue
        
        if len(pairs) == 0:
            # use empty answer
            print(colored("NO questions generated by GPT!", "red"), filename)
            continue

        for question, answer in pairs:
            rst.append({
                "question": question,
                "answer": answer,
                "optimal_path": optimal,
                "filename": os.path.relpath(filename, root),
                "root": os.path.relpath(root, REPOS_ROOT),
                "n_level": n_level,
                "model": MODEL_NAME
            })

            # dump `rst` to json
        with open(outname, "w") as f:
            json.dump(rst, f, indent=2)


if __name__ == "__main__":
    random.seed(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    dirs = os.listdir(REPOS_ROOT)
    random.shuffle(dirs)

    random.seed(1)
    for dirname in tqdm(dirs):
        gen(os.path.join(REPOS_ROOT, dirname),
            outname=os.path.join(OUT_DIR, f"{dirname}.json"))

    # Combine all /*.json into one json file
    all_data = []
    for filename in glob.glob(os.path.join(OUT_DIR, "*.json")):
        all_data += json.load(open(filename, "r"))

    json.dump(all_data, open(os.path.join(OUT_DIR, "data.json"), "w"), indent=2)
