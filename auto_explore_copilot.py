import argparse
import os
import pdb
import random
import string

from peft import PeftModel
from transformers import AutoTokenizer, GenerationConfig

from auto_explore_sandbox import AutoExploreSandbox, LeaveoutOption
from constants import (CHOICES, SUPPORTED_CMDS,
                       GPT_ANALYZE_PROMPT, GPT_ACTION_PROMPT,
                       GPT_WRONG_CHOICE_PROMPT)
from functions.cost import AutoExploreCostFunction
from functions.terminate import AnytimeTerminate, AutoExploreTerminateCriteria
# from model_utils import transformer_text_completion
from utils import (colored_string, extract_commands, list_all_actions, reply,
                   wrap_path)

class AutoExploreCopilot():

    def __init__(
            self,
            repo_root: str,
            sandbox_dir: str,
            file_save_path: str,
            horizon: int,
            generation_config: GenerationConfig,
            interaction_type: str,
            model_type: str,
            model_name: str = None,
            model: PeftModel = None,
            tokenizer: AutoTokenizer = None,
            cost_function: AutoExploreCostFunction = None,
            terminate_criteria: AutoExploreTerminateCriteria = AnytimeTerminate(
            ),
            leaveout_prob: float = 0,
            shuffle_action: bool = False,
            need_output_msgs: bool = True,
            markov: bool = False):
        """
        A copilot to help language models explore a repo.

        Args:
        - `repo_root` (str): The root directory of the repo.
        - `sandbox_dir` (str): The directory to store the sandbox.
        - `file_save_path` (str): The path to save the new or changed files.
        - `horizon` (int): The horizon (number of interactions) for each episode.
        - `temperature` (float): The temperature of the language model.
        - `top_p` (float): The top_p of the language model.
        - `top_k` (int): The top_k of the language model.
        - `max_token_length` (int): The maximum total token length for chat completion.
        - `max_new_tokens` (int): The maximum new tokens.
        - `interaction_type` (str): The type of the interaction, with choices
        in ['train', 'inference', 'debug', 'gen_data'].
        - `model_type` (str): The type of the model to use, with choices
        in ['local', 'remote_<API_type>', 'null']. If `interaction_type` is 'train', then
        must be 'local'.
        - `model_name` (str): The name of the model to use. Only used when
        `model_type` is 'remote'.
        - `model` (PeftModel): The model to use, only support Llama 2.
        Only used when `model_type` is 'local'.
        - `tokenizer` (AutoTokenizer): The tokenizer to use. Only used when
        `model_type` is 'local'.
        - `cost_function` (AutoExploreCostFunction): The cost function to use.
        Input is the list of messages, output is the cost. Only used when
        `interaction_type` is 'train'.
        - `terminate_criteria` (AutoExploreTerminateCriteria): The terminate
        criteria for an interaction. Input is the list of messages, output is
        True / False.
        - `leaveout_prob` (float): The probability of leaving out unrelated
        files. Only used when `interaction_type` is 'train', and passed to the
        sandbox.
        - `shuffle_action` (bool): Whether to shuffle the actions.
        - `need_output_msgs` (bool): Whether to output the messages after each act.
        """
        assert interaction_type in [
            "train", "inference", "debug", "gen_data"
        ], ("Only support interaction type in ['train', 'inference', 'debug', 'gen_data]."
           )
        assert model_type.startswith("remote") or model_type in [
            "local", "null"
        ], ("Only support model ype in ['local', 'remote_<API_type>', 'null'].")

        if interaction_type == "train":
            assert model_type == "local", "Only support local model for training."
        if interaction_type in ["inference", "gen_data"]:
            assert model_type != "null", "Must provide a model for inference."

        if model_type == "local":
            # assert (model is not None and tokenizer is not None), ("For local model, provide the model and the tokenizer.")
            if interaction_type == "train":
                assert cost_function is not None, ("For training, provide the "
                                                   "cost function.")
        elif model_type.startswith("remote"):
            assert model_name is not None, ("For remote model, provide the "
                                            "model name.")

        # replace all paths with absolute paths
        self.repo_root = os.path.abspath(repo_root).replace('\\', '/')
        self.sandbox_dir = os.path.abspath(sandbox_dir).replace('\\', '/')
        self.file_save_path = os.path.abspath(
            os.path.join(self.sandbox_dir, file_save_path)).replace('\\', '/')

        self.horizon = horizon
        self.generation_config = generation_config

        self.interaction_type = interaction_type
        self.model_type = model_type
        if model_type == "local":
            self.model = model
            self.tokenizer = tokenizer
        elif model_type.startswith("remote"):
            self.model_name = model_name

        self.cost_function = cost_function
        self.terminate_criteria = terminate_criteria
        self.leaveout_prob = leaveout_prob
        self.shuffle_action = shuffle_action
        self.need_output_msgs = need_output_msgs
        self.markov = markov

        # Read system instructions
        self.gpt_instruction = open(
            "data_gen/prompt_templates/auto_explore/gpt_instruction.md", "r").read()

    def set_question(self, question: str,
                     target_files: str = "",
                     ans_cmds: list = []):
        """
        Set the question to answer in the copilot.

        Args:
        - `question` (str): The question to answer.
        - `target_file` (str): The target file to answer the question. Only used
        when `self.interaction_type` is 'train'.
        """

        self.question = question

        self.prompt_template = open(
            "data_gen/prompt_templates/auto_explore/prompt.md",
            "r").read()

        # Store the generation logs for training
        self._sys_infos = []
        self._costs = []
        self._whole_msgs = []
        self._cmd_history = []

        # Initialize the files that have been visited for command filtering
        self._catted_files = []
        self._ided_files = []

        # Create sandbox environment
        if self.interaction_type == "train":
            self.supported_cmds = [
                "cd", "ls", "cat", "head", "tail", "id", "exit"
            ]
        else:
            self.supported_cmds = SUPPORTED_CMDS
        self.sandbox = AutoExploreSandbox(
            dataset_path=self.repo_root,
            sandbox_path=self.sandbox_dir,
            supported_cmds=self.supported_cmds,
            leaveout_option=LeaveoutOption(target_files, self.leaveout_prob))

        self.ans_cmds = ans_cmds.copy()

        self.ans_cmd = ""

        self.is_finished = False
        self.step = 1

    def set_answer(self, ans_cmd: str):
        """
        For first step training only
        """
        self.ans_cmd = ans_cmd

    def answer(self, question: str, target_files: list = [], ans_cmds: list = []):
        """
        Answer a question about the repo by autonomous exploration.

        Args:
        - `question` (str): The question to answer.
        - `target_file` (str): The target file to answer the question. Only used
        when `self.interaction_type` is 'train'.
        - `ans_cmds` (list): The commands of answer, can be either optimal or
        random (but still correct). Only used when debug.
        """
        self.set_question(question=question, target_files=target_files, ans_cmds=ans_cmds)

        self.continue_answer()
    
    def continue_answer(self):
        while not self.is_finished:
            if len(self.ans_cmds) == 1 and self.ans_cmds[0] == "exit":
                break

            self.build_cur_msgs(gen_data=(self.interaction_type == "gen_data"))

            if self.ans_cmds == []:
                try:
                    response = self.get_response()
                except Exception as e:
                    print(e)
                    continue
            else:
                cmd = self.ans_cmds.pop(0)
                response = self.choices[self.cmd_list.index(cmd)]

            self.act_with_response(response)

        self.wrap_up()
    
    def build_cur_msgs(self, gen_data: bool = False, wrong: bool = False):
        """
        Build current messages to send to the language model.
        """
        self.cwd = os.path.relpath(self.sandbox.cwd, self.sandbox.sandbox_dir)
        if self.cwd == ".":
            self.cwd = ""
        else:
            self.cwd = self.cwd.replace("\\", "/") + "/"

        files_under_cwd = os.listdir(self.sandbox.cwd)
        all_cmd = list_all_actions(root=self.sandbox.sandbox_dir,
                                   curr_dir=self.sandbox.cwd)
        self.cmd_list = self._filter_commands(sandbox_cwd=self.cwd,
                                              commands=all_cmd)

        if self.shuffle_action:
            self.choices = random.sample(CHOICES, len(self.cmd_list))
        else:
            self.choices = CHOICES[:len(self.cmd_list)]

        self._cur_msgs = [
            ("user",
             self.prompt_template.format(
                 TASK=self.question,
                 CWD=self.cwd,
                 FILES_UNDER_CWD="\n".join(
                     [wrap_path(f) for f in files_under_cwd]),
                 CMD_HIST="\n".join(self._cmd_history),
                 EXEC_RES="\n".join([r[1] for r in self._sys_infos]),
                 CMD_LIST="\n".join([
                     self.choices[i] + ". " + cmd
                     for i, cmd in enumerate(self.cmd_list)
                 ]))),
        ]
        if gen_data:
            p = self._cur_msgs[0][1]
            a = p[p.find("# Choose from below your command"):]
            p = p[:p.find("# Choose from below your command")] + (GPT_WRONG_CHOICE_PROMPT if wrong else GPT_ANALYZE_PROMPT)
            self._cur_msgs[0] = ("user", p)

            response = self.get_response()

            self._cur_msgs += [("assistant", response),
                                ("user", a + GPT_ACTION_PROMPT)]
            
        self._sys_infos = []

        if self.need_output_msgs:
            print(colored_string(self._cur_msgs[0]))

    def get_response(self) -> str:
        # Case 1: in debug mode, commands are provided
        if self.interaction_type == "debug":
            if self.ans_cmds == []:
                response = input("Input a command:")
            else:
                cmd = self.ans_cmds.pop(0)
                response = self.choices[self.cmd_list.index(cmd)]

        # Case 2: in other modes, use the language model
        # Case 2.1: using local model, e.g. GPT-2, Llama 2
        if self.model_type == "local":
            # Get response from local model
            # Use multinomial sampling to generate the next token:
            # Set do_sample = True, num_beams = 1
            # ret = transformer_text_completion(
            #     model=self.model,
            #     tokenizer=self.tokenizer,
            #     prompts=["\n".join([msg[1] for msg in self._cur_msgs])],
            #     generation_config=self.generation_config)[0]
            # response = self.use_lm_ret(ret)

            raise NotImplementedError
        # Case 2.2: using remote model, e.g. GPT-4
        elif self.model_type.startswith("remote"):
            # Get response from remote model
            msgs = [[("system", self.gpt_instruction)]]
            if not self.markov:
                msgs += [m[:] for m in self._whole_msgs]

            # remove redundant history messages
            for msg in msgs:
                for i in range(len(msg)):
                    if msg[i][1].find("# Choose from below your command") != -1:
                        msg[i] = (msg[i][0], msg[i][1][:msg[i][1].find("# Choose from below your command")])

            msgs.append(self._cur_msgs)
            msgs = sum(msgs, [])
            messages = [{
                "role": msg[0],
                "content": msg[1]
            } for msg in msgs]
            response = reply(api=self.model_type[len("remote_"):],
                             chat_history=messages,
                             model_name=self.model_name,
                             temperature=self.generation_config.temperature,
                             top_p=self.generation_config.top_p)

        return response

    def act_with_response(self, response: str) -> str:
        self._costs.append(0)
        try:
            ret = self._act_with_response(response)
        except Exception as e:
            ret = "Continue"
            self._sys_infos.append(("system", f"Runtime Error: {e}"))

        self._costs[-1] = self.cost_function.call(user_msgs=self._cur_msgs +
                                                    self._sys_infos)

        ### For first step training only
        if self.ans_cmd in self._cmd_history:
            self._costs[-1] = -100 - self.horizon

        if self._ided_files != [] or self.step == self.horizon:
            self.is_finished = True

        self.step += 1

    def _act_with_response(self, response: str) -> str:
        # Extract the numerical response
        response = response.strip(" \n")
        if response in self.cmd_list:
            response = self.choices[self.cmd_list.index(response)]
        else:
            response = response.strip(".")
            if "." in response:
                response = response[:response.find(".")]
        
        # Record the numerical response
        self._cur_msgs.append(("assistant", response))
        self._whole_msgs.append(self._cur_msgs)

        if response in self.choices:
            idx = self.choices.index(response)
            if idx >= len(self.cmd_list):
                self._sys_infos.append(("user", "Error: Invalid choice."))
                return "Continue"
            commands = extract_commands(f"```bash\n{self.cmd_list[idx]}\n```",
                                        only_first=True)
        else:
            commands = []

        for cmd in commands:
            self._sys_infos.append(("user", "Executing: " + " ".join(cmd)))
            self._update_cmd_history(self.cwd, " ".join(cmd))

            if cmd[0] == "exit":
                if len(commands) > 1:
                    self._sys_infos.append((
                        "user", "Error: There are other commands. "
                        "You could only use exit standalone in a single response."
                    ))
                else:
                    try:
                        self.flush_msgs()
                    except Exception:
                        return "Exit"

                    return "Exit"
            else:
                command_output, status = self.sandbox.run_command(cmd)
                self.terminate_criteria.update_status(**status)

                self._sys_infos.append(("user", command_output))

        if commands == []:
            self._sys_infos.append(
                ("user", "Warning: You didn't give me any command. "
                 "Further explore the repo by sending me system commands: "
                 f"{', '.join(self.supported_cmds)}."))

        return "Continue"

    def wrap_up(self):
        """
        Wrap up after answering a question by assigning final cost, computing Q
        values, saving the new or changed files, and deleting the sandbox.
        """
        if self.terminate_criteria.can_terminate():
            self._costs[-1] -= self.horizon
        else:
            self._costs[-1] += self.horizon

        # Save the new or changed files
        os.makedirs(self.file_save_path, exist_ok=True)
        for file_name, content in self.sandbox.get_changed_files().items():
            os.makedirs(self.file_save_path + os.path.dirname(file_name),
                        exist_ok=True)
            with open(self.file_save_path + file_name, "wb") as f:
                f.write(content)

        # Cleanup sandbox and environment
        del self.sandbox

    @property
    def costs(self):
        """
        Get the costs for training.

        Returns:
        - list: A list of costs.
        """
        return self._costs

    @property
    def cur_msgs(self):
        """
        Get the current messages.

        Returns:
        - list: A list of messages, each message following format:
        (role, content)
        """
        return self._cur_msgs

    @property
    def whole_msgs(self):
        """
        Get the whole messages.

        Returns:
        - list: A list of messages, each message following format:
        (role, content)
        """
        return self._whole_msgs
    
    @property
    def cmd_history(self):
        """
        Get the command history.

        Returns:
        - list: A list of commands.
        """
        return self._cmd_history

    @property
    def cmd_history(self):
        """
        Get the command history.

        Returns:
        - list: A list of commands.
        """
        return self._cmd_history

    def _filter_commands(self, sandbox_cwd: str, commands: list) -> list:
        """
        Filter out available commands based on the cwd in the sandbox and
        command history. Prevents repeated access of a same file.

        Args:
        - `sandbox_cwd` (str): The cwd in the sandbox.
        - `commands` (list): The commands to filter.

        Returns:
        - list: The available commands.
        """
        ret = []
        for command in commands:
            if command.startswith("cat"):
                file = sandbox_cwd + command[4:]
                if file not in self._catted_files:
                    ret.append(command)
            elif command.startswith("id"):
                file = sandbox_cwd + command[3:]
                # if file not in self._ided_files:
                #     ret.append(command)
                ret.append(command)
            elif command == "exit":
                pass
            else:
                ret.append(command)
        return ret

    def _update_cmd_history(self, sandbox_cwd: str, command: string):
        """
        Maintain command history.

        Args:
        - `sandbox_cwd` (str): The cwd in the sandbox.
        - `command` (str): The command to execute.
        """
        self._cmd_history.append(command)
        if command.startswith("cat"):
            file = sandbox_cwd + command[4:]
            self._catted_files.append(file)
        elif command.startswith("id"):
            file = sandbox_cwd + command[3:]
            self._ided_files.append(file)
