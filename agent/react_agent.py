import pdb
import random
import re
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

from autogen import Agent, OpenAIWrapper, UserProxyAgent

from .ag_agent import extract_action


class ReActAgent(UserProxyAgent):

    def __init__(
        self,
        env: object,
        name: str,
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        default_auto_reply: Optional[Union[str, Dict, None]] = "",
        system_message: Optional[Union[str, List]] = "",
        description: Optional[str] = None,
    ):

        self._env = env    # store the environment
        super().__init__(name=name,
                         system_message=system_message,
                         is_termination_msg=is_termination_msg,
                         code_execution_config=False,
                         default_auto_reply=default_auto_reply,
                         description=description,
                         max_consecutive_auto_reply=None)
        self._reply_func_list = []
        self.register_reply([Agent, None], ReActAgent._generate_env_reply)

        # keep track of the env
        self._total_reward = 0
        self._steps = 0

        self.initial_state = self.reflect_prompt

        self.wait_reflect = True

    @property
    def reflect_prompt(self):
        # Retrieve the stepped state
        state = self._env.state
        idx = state.find("The actions you can take now is:")
        assert idx > 0
        state_no_action = state[:idx]

        # B.2 request reflection
        reply = state_no_action + "\n\nGive me the analysis and reflection on the previous actions and observations."

        return reply

    def _generate_env_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[OpenAIWrapper] = None,
    ) -> Tuple[bool, Union[str, Dict, None]]:
        """Generate a reply using autogen.oai."""

        # Clean the received message.
        if messages is None:
            messages = self._oai_messages[sender]

        sender.clear_history()    # markov, clear previous history
        message = messages[
            -1]    # for environment, only the last action matters.
        
        if message["content"].strip().rstrip() == "":
            return True, self._env.prompt # API error. let's redo the query

        if self.wait_reflect:
            # A. If the input message is reflection, ask for Action
            reflection: str = message["content"].strip().rstrip()

            # Retrieve the current state
            state = self._env.state
            idx = state.find("The actions you can take now is:")
            assert idx > 0
            state_no_action = state[:idx]
            action_prompt = state[idx:]

            reply = state_no_action + "\nYour Reflection is:\n" + reflection + "\n\n\n" + action_prompt
            reply += "\n\nNow, give me the action to take. You MUST reply the action ID, such as 'Action 2': "


            self.wait_reflect = False
            return True, reply
        else:
            # B. If the input message is action, then step, and ask for next reflection.

            # B.1 step
            action = extract_action(message["content"],
                                    action_space=self._env.action_space)
            state, reward, done, truncated, info = self._env.step(action)
            
            # B.2 request reflection
            self.wait_reflect = True

            self._total_reward += reward
            self._steps += 1
            if done:
                reply = "TERMINATE"
            elif truncated:
                reply = "TERMINATE"
            else:
                reply = self.reflect_prompt

            return True, reply

    def final_info(self):
        result = f"Total Reward: {self._total_reward}\n"
        result += f"Num Steps: {self._steps}"
        return result

