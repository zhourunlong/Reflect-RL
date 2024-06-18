import random

from typing import Optional, Tuple

from transformers import GenerationConfig


from auto_explore_copilot import AutoExploreCopilot
from auto_explore_sandbox import RepoCache
from constants import CHOICES
from functions.cost import AutoExploreCostFunction
from functions.terminate import IdentifyFileTerminate
from nat_lang_envs.base_env import NaturalLanguageEnvironment


class AutoExploreEnv(NaturalLanguageEnvironment):
    """
    Environment for autonomous exploration of a file system.
    """

    def __init__(
        self,
        repo_cache: Optional[RepoCache] = None,
        horizon: Optional[int] = None,
        generation_config: Optional[GenerationConfig] = None,
        cost_function: Optional[AutoExploreCostFunction] = None,
        **kwargs,
    ):
        self.repo_cache = repo_cache
        self.horizon = horizon
        self.generation_config = generation_config
        self.cost_function = cost_function

    @property
    def max_action_space(self):
        return CHOICES

    @property
    def action_space(self):
        return CHOICES

    def step(self, action: str) -> Tuple[str, float, bool, bool, dict]:
        if self.copilot.is_finished:
            return (None, None, True, False, {"msg": None})
        action = action.strip(" ")
        if action == "":
            action = " "
        self.copilot.act_with_response(action)
        if self.copilot.is_finished:
            self.copilot.wrap_up()
            obs = None
        else:
            self.copilot.build_cur_msgs()
            obs = "\n".join([msg[1] for msg in self.copilot.cur_msgs])
        return (
            obs,    # observation
            -self.copilot.costs[-1],    # reward
            self.copilot.is_finished,    # terminated
            False,    # truncated
            {
                "msg": self.copilot.whole_msgs[-1]
            },    # info
        )

    def reset(
        self,
        data: dict,
        leaveout_prob: float,
        shuffle_action: bool,
        easy: bool,
        first_step: bool,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        **kwargs,
    ) -> Tuple[str, dict]:
        if easy:
            target_files = [random.choice(data["filename"])]
            question = f"Find {target_files[0]}"
        else:
            target_files = data["filename"]
            question = data["question"]

        self.copilot = AutoExploreCopilot(
            repo_root=self.repo_cache.cache_repo(data["root"]),
            sandbox_dir=self.repo_cache.cache_dir,
            file_save_path=self.repo_cache.file_save_path,
            horizon=self.horizon,
            generation_config=self.generation_config,
            interaction_type="train",
            model_type="local",
            cost_function=self.cost_function,
            terminate_criteria=IdentifyFileTerminate(target_files),
            leaveout_prob=leaveout_prob,
            shuffle_action=shuffle_action,
            need_output_msgs=False,
        )
        self.copilot.set_question(question=question,
                                  target_files=target_files)

        if first_step:
            self.copilot.set_answer(data["optimal_path"][0])

        self.copilot.build_cur_msgs()
        obs = "\n".join([msg[1] for msg in self.copilot.cur_msgs])

        return (
            obs,    # observation
            None,    # info
        )
