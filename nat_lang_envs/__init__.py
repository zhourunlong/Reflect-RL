
try:
    from nat_lang_envs.alfworld_env import AlfworldEnv
except:
    AlfworldEnv = None
from nat_lang_envs.auto_explore import AutoExploreEnv
from nat_lang_envs.taxi import TaxiNatLangEnv

TOY_TEXT_ENVS = {
    "taxi": TaxiNatLangEnv,
}

OUR_ENVS = {
    "alfworld": AlfworldEnv,
    "auto_explore": AutoExploreEnv,
}

ENVS = {**TOY_TEXT_ENVS, **OUR_ENVS}


ACTION_HEADS = {
    AutoExploreEnv: "\n# Choose from below your command",
    AlfworldEnv: "\nThe actions you can take now is:",
    TaxiNatLangEnv: "\nThe actions you can take now is:",
}