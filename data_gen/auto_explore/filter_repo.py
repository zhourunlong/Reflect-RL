import os
import pdb
import shutil

from tqdm import tqdm
from transformers import HfArgumentParser

from constants import ALLOWED_FILE_SUFFIXES
from experiment_args import ScriptArguments
from utils import ascii_fraction

NEW_REPO_DIR = "data/auto_explore/repos_filtered/"
KEYWORDS_TO_IGNORE = ["LICENSE", "Makefile", "conf.py", "artwork", "requirements", "ci", "tests", "examples"]
MIN_FILE_LENGTH = 10

def ignore(dir, files):
    ignored = []
    for f in files:
        # hidden files or folders
        if f[0] in [".", "_"]:
            ignored.append(f)
            continue

        # usually not informative files
        if any([(keyw in f) for keyw in KEYWORDS_TO_IGNORE]):
            ignored.append(f)
            continue

        # non-ascii files or folder names
        if ascii_fraction(f) < 1:
            ignored.append(f)
            continue

        # files
        if "." in f:
            # not supported file types
            if not f.endswith(tuple(ALLOWED_FILE_SUFFIXES)):
                ignored.append(f)
                continue

            # non-ascii files
            try:
                content = open(os.path.join(dir, f), "r", encoding="utf-8").read()
            except:
                print("Error reading file", os.path.join(dir, f))
                ignored.append(f)
                continue

            # empty files
            if len(content) < MIN_FILE_LENGTH:
                ignored.append(f)
                continue

            if ascii_fraction(content) < 0.9:
                ignored.append(f)
                continue

    return ignored


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

repos = os.listdir(script_args.repo_dir)
os.makedirs(NEW_REPO_DIR, exist_ok=True)

for repo in tqdm(repos):
    if not os.path.isdir(os.path.join(script_args.repo_dir, repo)):
        continue
    shutil.copytree(os.path.join(script_args.repo_dir, repo), os.path.join(NEW_REPO_DIR, repo), ignore=ignore, dirs_exist_ok=True)
