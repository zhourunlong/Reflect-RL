from transformers import AutoTokenizer


class AutoExploreCostFunction:
    """
    Cost function for auto explore.
    """

    def __init__(self, *kwargs):
        pass

    def call(self, **kwargs) -> float:
        pass


class StepCost(AutoExploreCostFunction):
    """
    Whatever the command is, the cost is always 1.
    """

    def call(self, **kwargs) -> float:
        """
        Returns:
        - float: 1.
        """
        return 1


class NumTokenCost(AutoExploreCostFunction):
    """
    Calculate the number of tokens in the user messages.
    """

    def __init__(self, tokenizer: AutoTokenizer):
        """
        Args:
        - `tokenizer` (AutoTokenizer): The tokenizer used to tokenize the user
        messages.
        """
        self.tokenizer = tokenizer

    def call(self, **kwargs) -> float:
        """
        Args:
        - `kwargs` (dict):
            - 'user_msgs' (list): The user messages.

        Returns:
        - float: The number of tokens in the user messages.
        """
        return sum(
            [len(self.tokenizer.encode(msg[1])) for msg in kwargs["user_msgs"]])


class KeywordCost(AutoExploreCostFunction):
    """
    Calculate the weighted sum of keywords in the user messages.
    Can be used to calculate the number of invalid commands.
    """

    def __init__(self, keywords: list, costs: list):
        """
        Args:
        - `keywords` (list): A list of keywords.
        - `costs` (list): A list of costs for the keywords.
        """
        assert len(keywords) == len(costs), (
            "The length of keywords and costs should be the same.")

        self.keywords = keywords
        self.costs = costs

    def call(self, **kwargs) -> float:
        """
        Args:
        - `kwargs` (dict):
            - 'user_msgs' (list): The user messages.

        Returns:
        - float: The weighted sum of keywords in the user messages.
        """
        return sum([
            sum([cost * msg.count(keyword)
                 for msg in kwargs["user_msgs"]])
            for (keyword, cost) in zip(self.keywords, self.costs)
        ])


class SynthesizedCost(AutoExploreCostFunction):
    """
    Synthesize a list of cost functions into a single one.
    """

    def __init__(self, cost_functions: list, weights: list):
        """
        Args:
        - `cost_functions` (list): A list of cost functions.
        - `weights` (list): A list of weights for the cost functions.
        """
        assert len(cost_functions) == len(weights), (
            "The length of cost_functions and weights should be the same.")

        self.cost_functions = cost_functions
        self.weights = weights

    def call(self, **kwargs) -> float:
        """
        Args:
        - `kwargs` (dict):
            - 'user_msgs' (list): The user messages.

        Returns:
        - float: The weighted sum of costs.
        """
        return sum([
            w * f.call(**kwargs)
            for w, f in zip(self.weights, self.cost_functions)
        ])
