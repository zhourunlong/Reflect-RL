import os
import random
import shutil
import subprocess
import tempfile

from constants import SUPPORTED_CMDS
from utils import (display_files_recursively, list_files, hide_root, trunc_text,
                   get_file_names, handle_ls, unwrap_path)

SAFE_MESSAGE = "SAFE"


def remove_dir(dir: str):
    if os.name == "nt":
        os.system('rmdir /S /Q "{}"'.format(dir))
    else:
        os.system('rm -rf "{}"'.format(dir))


class RepoCache:

    def __init__(self, original_root: str, dir: str, file_save_path: str):
        """
        A cache for the files used in training and evaluation.

        Args:
        - `original_root` (str): path to the original repos.
        - `dir` (str): path to the directory containing the cache folder.
        """
        self.original_root = os.path.abspath(original_root).replace("\\", "/")
        # The directory containing cached repos as well as other temp files
        self.cache_dir = os.path.abspath(tempfile.mkdtemp(dir=dir)).replace(
            "\\", "/") + "/"
        self.cache_root = self.cache_dir + "cached_repos/"
        os.makedirs(self.cache_root, exist_ok=True)
        self.cached_repos = []
        self.file_save_path = file_save_path

    def __del__(self):
        remove_dir(self.cache_dir)

    def cache_repo(self, repo: str):
        """
        Copy the repo to the cache directory if not already cached.

        Args:
        - `repo` (str): The repo to be cached.

        Returns:
        - str: The path to the cached repo.
        """
        cache_repo_dir = os.path.join(self.cache_root, repo)

        if repo not in self.cached_repos:
            shutil.copytree(os.path.join(self.original_root, repo),
                            cache_repo_dir)
            # print("Cache %s to %s" % (root, cached_root))
            self.cached_repos.append(repo)

        return cache_repo_dir


class LeaveoutOption:

    def __init__(self, files_must_contain: list, leaveout_prob: float):
        """
        Args:
        - `files_must_contain` (list): The list of files that must be included.
        - `leaveout_prob` (float): The probability of leaving out unrelated
        files.
        """
        self.files_must_contain = files_must_contain
        self.leaveout_prob = leaveout_prob

    def __call__(self, file: str) -> bool:
        """
        Args:
        - `file` (str): The file to be left out.

        Returns:
        - bool: True if and only if the file is not in `files_must_contain` and
        randomly chosen to be left out.
        """
        for file_must_contain in self.files_must_contain:
            if file_must_contain.startswith(file):
                return False
        return random.random() < self.leaveout_prob


class AutoExploreSandbox:

    def __init__(
        self,
        dataset_path: str,
        sandbox_path: str,
        supported_cmds: list = SUPPORTED_CMDS,
        leaveout_option: LeaveoutOption = None,
    ):
        """
        Wraps the dataset folder.
        Prevents any system commands from ruining the original dataset.

        Args:
        - `dataset_path` (str): The path to the dataset.
        - `sandbox_path` (str): The path to the sandbox.
        - `supported_cmds` (list): The list of supported commands. Must be a
        subset of SUPPORTED_CMDS.
        - `leaveout_fraction` (float): The probability of leaving out unrelated
        files.
        """
        assert set(supported_cmds).issubset(
            set(SUPPORTED_CMDS
               )), "supported_cmds must be a subset of SUPPORTED_CMDS."
        self.supported_cmds = supported_cmds

        # Use absolute path
        self.dataset_path = os.path.abspath(dataset_path).replace("\\",
                                                                  "/") + "/"
        # Copy dataset to a temporary directory in the working directory
        self.sandbox_dir = os.path.abspath(
            tempfile.mkdtemp(dir=sandbox_path)).replace("\\", "/") + "/"

        self.leaveout_option = leaveout_option

        # Ignore hidden files and directories, optionally leave out unrelated
        # files
        def ignore(directory, filenames):
            return [
                fn for fn in filenames
                if fn.startswith('.') or self.leaveout_option(
                    os.path.join(directory[len(self.dataset_path):], fn))
            ]

        shutil.copytree(self.dataset_path,
                        self.sandbox_dir,
                        ignore=ignore,
                        dirs_exist_ok=True)

        # print(
        #     colored(f"Data copied to temporary directory: {self.sandbox_dir}",
        #             "yellow"))

        # Checkpoint cwd to avoid outside changes
        self.cwd = self.sandbox_dir

    def __del__(self):
        remove_dir(self.sandbox_dir)

    def safety_check(self, cmd: list) -> str:
        """
        Return "SAFE" if and only if the cmd is safe to run.
        Otherwise, return error message.

        Args:
        - `cmd` (list): A single command splitted into a list of arguments.

        Returns:
        """
        # Restrict command type
        if cmd[0] == "exit":
            raise NotImplementedError(
                "exit should be handled outside of run_command().")
        if cmd[0] not in self.supported_cmds:
            return f"Error: You can only use {', '.join(self.supported_cmds[:-1])}."

        # Test if the target file/dir is inside the sandbox
        target_dirs = self._get_relative_target_dirs(cmd)
        for target_dir in target_dirs:
            if target_dir.startswith("Error:"):
                return target_dir
            if target_dir.startswith(".."):
                return (
                    f"Error: You cannot access file {target_dir} "
                    f"outside the repo! You are now at {self._get_relative_path('.')}"
                )

        return SAFE_MESSAGE

    def run_command(self, cmd: list) -> (str, dict):
        """Wrapper function for self._run_command().
        Run a bash command in the dataset sandbox.

        The supported tools are:
        "cd", "ls", "cat", "head", "tail", "echo", "python", "pip".
        "exit" is handled outside of this function.

        Args:
        - `cmd` (list): A single command splitted into a list of arguments.

        Returns:
        - str: The execution result of the given command. If any errors
        occurred, then just return the error message.
        - dict: State updates to the sandbox.
            - 'identified_file': The file identified using 'id'.
        """
        # Restore to the checkpointed cwd
        # Use absolute path to store cwd!
        system_cwd = os.path.abspath(os.getcwd()).replace('\\', '/') + "/"
        os.chdir(self.cwd)

        safety_check_result = self.safety_check(cmd)

        if safety_check_result != SAFE_MESSAGE:
            ret = (safety_check_result, {})
        else:
            ret = self._run_command(cmd)

        # Checkpoint cwd
        self.cwd = os.path.abspath(os.getcwd()).replace('\\', '/') + "/"
        os.chdir(system_cwd)

        return ret

    def _run_command(self, cmd: list) -> (str, dict):
        # Check if echo outputs to a file
        if cmd[0] == "echo" and len(cmd) == 3:
            return ("Warning: echo command without output file, ignored.", {})

        # Run the command
        try:
            if cmd[0] == "cd":
                # cd cannot be handled by subprocess
                os.chdir(unwrap_path(cmd[1]))
                return ("Success: Now at " + self._get_relative_path("."), {})
            elif cmd[0] == "id":
                abs_path = self._get_absolute_path(unwrap_path(cmd[1]))
                if abs_path.startswith("Error:"):
                    return (abs_path, {})
                elif not os.path.isfile(abs_path):
                    return (f"Error: {cmd[1]} does not exist!", {})
                return ("Success: Identified file " + unwrap_path(cmd[1]), {
                    "identified_file":
                        self._get_relative_path(unwrap_path(cmd[1]))
                })
            else:
                result = subprocess.run(' '.join(cmd),
                                        shell=True,
                                        capture_output=True)
                return (self.respond_cmd(cmd, result), {})
        except Exception as e:
            return ("Error: " + str(e), {})

    def respond_cmd(self, cmd: list, result) -> str:
        """
        Generate the response for the result of a command.

        Args:
        - `cmd` (list): A single command splitted into a list of arguments.
        - `result` (subprocess.CompletedProcess): The result of the command.

        Returns:
        - str: The response for the result of the command.
        """
        rstdout = result.stdout.decode('utf-8')
        rstderr = hide_root(result.stderr.decode('utf-8'), self.sandbox_dir)

        if rstderr != "":
            return f"Error: {rstderr}"

        if cmd[0] == "ls":
            return "Success: The result of ls is:\n" + handle_ls(rstdout)
        elif cmd[0] in ["cat", "head", "tail"]:
            fn = get_file_names(cmd)[0]
            return (f"Success: The content of {fn} is:\n" +
                    trunc_text(fn, rstdout))
        elif cmd[0] == "echo":
            return f"Success: echoed to {cmd[-1]}"
        elif cmd[0] == "python":
            return f"Success: The output of python is:\n{rstdout}"
        elif cmd[0] == "pip":
            return "Success: pip succeeded"
        else:
            raise NotImplementedError(f"Does not support command: {cmd[0]}")

    def get_changed_files(self) -> dict:
        """
        Return the name and content of changed files in the sandbox.

        Returns:
        - dict: Key is relative file path, value is the content in bytes.
        """
        original_files = set(list_files(self.dataset_path))
        current_files = set(list_files(self.sandbox_dir))
        changed_files = list(current_files - original_files)

        common_files = current_files.intersection(original_files)

        for file in common_files:
            file = file.replace("\\", "/")

            original_file_path = self.dataset_path + file
            current_file_path = self.sandbox_dir + file

            original_file_content = open(original_file_path, "rb").read()
            current_file_content = open(current_file_path, "rb").read()

            if original_file_content != current_file_content:
                changed_files.append(file)

        # print(colored("List of changed files:", "yellow"))
        # print(changed_files)

        return {
            file: open(self.sandbox_dir + file, "rb").read()
            for file in changed_files
        }

    def _get_relative_target_dirs(self, cmd: list) -> list:
        """
        Get the relative directories of the target files/dirs from a command.

        Args:
        - cmd (list): a single command splitted into a list of arguments.

        Returns:
        - list: A list of directories.
        """
        # Get the files
        files = get_file_names(cmd)
        target_dirs = []

        for file in files:
            target_dirs.append(self._get_relative_path(file))

        return target_dirs

    def _get_absolute_path(self, path) -> str:
        """
        Get the absolute path of a given path relative to cwd in the
        sandbox.

        Args:
        - `path` (str): The path with respect to cwd to be converted to
        absolute path.

        Returns:
        - str: The absolute path.
        """
        # Use absolute path to store cwd!
        original_cwd = os.path.abspath(os.getcwd()).replace('\\', '/')

        base_name = os.path.basename(path)
        if base_name in [".", ".."]:
            base_name = ""
        elif base_name == path:
            path = "."
        else:
            path = path[:len(path) - len(base_name)]

        try:
            os.chdir(path)
        except Exception as e:
            os.chdir(original_cwd)
            return "Error: " + str(e)

        absolute_path = os.path.abspath(os.getcwd()).replace(
            '\\', '/') + "/" + base_name

        os.chdir(original_cwd)

        return absolute_path

    def _get_relative_path(self, path: str) -> str:
        """
        Get the relative path with respect to the sandbox directory of a given
        path relative to cwd in the sandbox.

        Args:
        - `path` (str): The path with respect to cwd to be converted to
        relative path.

        Returns:
        - str: The relative path.
        """

        absolute_path = self._get_absolute_path(path)

        if absolute_path.startswith("Error:"):
            return absolute_path

        relative_path = os.path.relpath(absolute_path, self.sandbox_dir)

        return relative_path


if __name__ == "__main__":
    sandbox = AutoExploreSandbox(
        dataset_path="/home/vectorzhou/Coffee_Roasting_Dataset/data/",
        leaveout_option=LeaveoutOption(files_must_contain=["public/cafe.md"],
                                       leaveout_fraction=0.5),
    )

    print(sandbox.run_command(["cat", "a.txt"], "zrl"))

    print(display_files_recursively(sandbox.sandbox_dir))

    del sandbox
