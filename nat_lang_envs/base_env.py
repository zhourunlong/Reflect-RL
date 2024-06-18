from typing import Optional, Tuple

import diskcache


class NaturalLanguageEnvironment:
    """
    Base class for all natural language environments. Logic similar to OpenAI Gym.

    The main API methods that users of this class need to know are:

    - :meth:`step` - Takes a step in the environment using an action returning the next observation, reward,
      if the environment terminated and observation information.
    - :meth:`reset` - Resets the environment to an initial state, returning the initial observation and observation information.

    And set the following attributes:

    - :attr:`action_space` - The Space object corresponding to valid actions
    - :attr:`observation_space` - The Space object corresponding to valid observations
    - :attr:`reward_range` - A tuple corresponding to the minimum and maximum possible rewards
    - :attr:`spec` - An environment spec that contains the information used to initialise the environment from `gym.make`
    - :attr:`metadata` - The metadata of the environment, i.e. render modes
    - :attr:`np_random` - The random number generator for the environment

    """
    def __init__(self, **kwargs):
        raise NotImplementedError

    def step(self, action: str) -> Tuple[str, float, bool, bool, dict]:
        """Run one timestep of the environment's dynamics.

        When end of episode is reached, you are responsible for calling :meth:`reset` to reset this environment's state.
        Accepts an action and returns either a tuple `(observation, reward, terminated, truncated, info)`.

        Args:
        - `action` (str): an action provided by the agent.

        Returns:
        - `observation` (str): the next step observation.
        - `reward` (float): The amount of reward returned as a result of taking the action.
        - `terminated` (bool): whether a `terminal state` (as defined under the MDP of the task) is reached.
        In this case further step() calls could return undefined results.
        - `truncated` (bool): whether a truncation condition outside the scope of the MDP is satisfied.
        Typically a timelimit, but could also be used to indicate agent physically going out of bounds.
        Can be used to end the episode prematurely before a `terminal state` is reached.
        - `info` (dict): `info` contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
        This might, for instance, contain: metrics that describe the agent's performance state, variables that are
        hidden from observations, or individual reward terms that are combined to produce the total reward.
        """
        raise NotImplementedError

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        **kwargs
    ) -> Tuple[str, dict]:
        """Resets the environment to an initial state and returns the initial observation.

        This method can reset the environment's random number generator(s) if ``seed`` is an integer or
        if the environment has not yet initialized a random number generator.
        If the environment already has a random number generator and :meth:`reset` is called with ``seed=None``,
        the RNG should not be reset. Moreover, :meth:`reset` should (in the typical use case) be called with an
        integer seed right after initialization and then never again.

        Args:
        - `seed` (optional int): The seed that is used to initialize the environment's PRNG.
            If the environment does not already have a PRNG and ``seed=None`` (the default option) is passed,
            a seed will be chosen from some source of entropy (e.g. timestamp or /dev/urandom).
            However, if the environment already has a PRNG and ``seed=None`` is passed, the PRNG will *not* be reset.
            If you pass an integer, the PRNG will be reset even if it already exists.
            Usually, you want to pass an integer *right after the environment has been initialized and then never again*.
            Please refer to the minimal example above to see this paradigm in action.
        - `options` (optional dict): Additional information to specify how the environment is reset (optional,
        depending on the specific environment)


        Returns:
        - `observation` (str): Observation of the initial state, analogous to the observation returned by :meth:`step`.
        - `info` (dictionary):  This dictionary contains auxiliary information complementing ``observation``. It should be analogous to
            the ``info`` returned by :meth:`step`.
        """
        raise NotImplementedError




def cache_function_call(func):
    def wrapper(self):
        cache = diskcache.Cache(".oracle_cache")
        key = f"{self.grid}"
        if key in cache:
            return cache[key]
        else:
            result = func(self)
            cache[key] = result
            return result
    return wrapper