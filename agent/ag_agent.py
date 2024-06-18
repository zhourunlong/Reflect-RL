import os
import random
import re
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import autogen
from autogen import Agent, AssistantAgent, OpenAIWrapper, UserProxyAgent

config_list_gpt3 = [
    {
    "model": "gpt-3.5-turbo",
    "api_key": os.getenv("GCR_OPENAI_KEY", "YOUR_API"),
    "base_url": os.getenv("GCR_BASE_URL", "https://api.openai.com"),
    }
]

def ask_autogen(question: str, sys_msg: str="Answer.") -> str:
    """Ask autogen to generate a reply."""
    agent = AssistantAgent(name="agent",
                           system_message=sys_msg,
                           llm_config=
                           {"config_list": config_list_gpt3, 
                            "cache_seed": 42, 
                            "max_tokens": 200}
                            )
    user = autogen.UserProxyAgent(
        name="User",
        system_message="A human user.",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0
    )
    user.send(message=question, recipient=agent, request_reply=True, silent=True)
    extracted = agent.last_message()["content"]
    return extracted

def ask_autogen(*args, **kwargs):
    raise NotImplementedError("ask_autogen is not implemented")

def extract_action(msg: str, action_space: list) -> str:
    for regex in [r"action:\s*(\d+)", r"action\s*(\d+)", r"\d+"]:
        action_substr = re.findall(regex, msg.lower(), re.DOTALL)
        if len(action_substr):
            action = action_substr[0].strip().rstrip()
            break
        else:
            action = msg

    if action in action_space:
        return action

    # If we cannot extract the action,
    # Find the one in the action space
    first_action, first_action_idx = None, 1e10
    for valid_action in action_space:
        _idx = action.find(valid_action)
        if _idx >= 0 and _idx < first_action_idx:
            first_action = valid_action
            first_action_idx = _idx

    if first_action is not None:
        return first_action

    return random.choice(action_space)    # we are unable to extract action


class EnvAgent(UserProxyAgent):

    def __init__(
        self,
        env: object,
        name: str,
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        code_execution_config: Optional[Union[Dict, Literal[False]]] = None,
        default_auto_reply: Optional[Union[str, Dict, None]] = "",
        system_message: Optional[Union[str, List]] = "",
        description: Optional[str] = None,
    ):

        self._env = env    # store the environment
        super().__init__(name=name,
                         system_message=system_message,
                         is_termination_msg=is_termination_msg,
                         code_execution_config=code_execution_config,
                         default_auto_reply=default_auto_reply,
                         description=description,
                         max_consecutive_auto_reply=None)
        self._reply_func_list = []
        self.register_reply([Agent, None], EnvAgent._generate_env_reply)

        # keep track of the env
        self._total_reward = 0
        self._steps = 0
        self.initial_state = self._env.prompt

    def _generate_env_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[OpenAIWrapper] = None,
    ) -> Tuple[bool, Union[str, Dict, None]]:
        """Generate a reply using autogen.oai."""

        if messages is None:
            messages = self._oai_messages[sender]

        sender.clear_history()    # markov, clear previous history
        message = messages[
            -1]    # for environment, only the last action matters.
        
        if message["content"].strip().rstrip() == "":
            return True, self._env.prompt # API error. let's redo the query

        action = extract_action(message["content"],
                                action_space=self._env.action_space)

        state, reward, done, truncated, info = self._env.step(action)

        self._total_reward += reward
        self._steps += 1
        if done:
            reply = "TERMINATE"
        elif truncated:
            reply = "TERMINATE"
        else:
            reply = state

        return True, reply

    def final_info(self):
        result = f"Total Reward: {self._total_reward}\n"
        result += f"Num Steps: {self._steps}"
        return result


class Player(AssistantAgent):

    def __init__(
        self,
        name: str = "player",
        system_message: Optional[str] = "Play the game.",
        llm_config: Optional[Union[Dict, Literal[False]]] = None,
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        code_execution_config: Optional[Union[Dict, Literal[False]]] = False,
        description: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            system_message=system_message,
            is_termination_msg=is_termination_msg,
            code_execution_config=code_execution_config,
            llm_config=llm_config,
            description=description,
            max_consecutive_auto_reply=None,
            **kwargs,
        )


class RandomPlayer(AssistantAgent):

    def __init__(
        self,
        name: str = "random player",
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        description: Optional[str] = None,
        max_consecutive_auto_reply=None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            system_message="a random player",
            is_termination_msg=is_termination_msg,
            code_execution_config=False,
            llm_config=None,
            description=description,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            **kwargs,
        )
        self._reply_func_list = []
        self.register_reply([Agent, None], RandomPlayer._generate_random_reply)

    def _generate_random_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[OpenAIWrapper] = None,
    ) -> Tuple[bool, Union[str, Dict, None]]:
        """Generate a reply using autogen.oai."""
        if messages is None:
            messages = self._oai_messages[sender]

        sender.clear_history()    # markov, clear previous history
        message = messages[
            -1]    # for environment, only the last action matters.
        
        if message["content"].strip().rstrip() == "TERMINATE":
            return True, None # None means stop the conversation
        

        return True, "I don't know. Choose a random action."
