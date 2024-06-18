import string

DEBUG = False

CHOICES = [str(i) for i in range(200)] + list(string.ascii_letters)
ANALYZE_TEMPLATE = " # Analyze:\n"
ACTION_TEMPLATE = " # Action:\n"

GPT_WRONG_CHOICE_PROMPT = "The last command opens a wrong folder or file, which is a suboptimal move. Give reasons why the folder or file is incorrect and plan what to do next using 50 words. Don't give the choice yet."
GPT_ANALYZE_PROMPT = "Now analyze the current situation and plan what to do next using 50 words. Don't give the choice yet. If you have identified the correct file in previous steps, you should exit at this step."
GPT_ACTION_PROMPT = "Give me your choice."



OVERRIDE_KEYS = ["model_name", "lora_r", "bf16", "fp16", "use_8bit", "use_4bit"]

DROPOUT_KEYS = ["resid_pdrop", "embd_pdrop", "attn_pdrop", "summary_first_dropout"]

DISABLE_DROPOUT_KWARGS = {k: 0 for k in DROPOUT_KEYS}



# exit should always be the last
SUPPORTED_CMDS = [
    "cd", "ls", "cat", "head", "tail", "echo", "python", "pip", "id", "exit"
]
FULL_CMDS = SUPPORTED_CMDS + [
    "pwd",
    "mkdir",
    "rmdir",
    "touch",
    "rm",
    "cp",
    "mv",
    "less",
    "grep",
    "find",
    "who",
    "w",
    "ps",
    "top",
    "kill",
    "tar",
    "chmod",
    "chown",
    "df",
    "du",
    "ifconfig",
    "ping",
    "netstat",
    "ssh",
    "scp",
]

# Common programming language suffixes
CODE_SUFFIXES = [
    ".py", ".c", ".cpp", ".cxx", ".cc", ".h", ".hpp", ".hxx", ".cs", ".java",
    ".go", ".ipynb"
]

# Common data file suffixes
DATA_SUFFIXES = [".csv", ".tsv"]

# Common text file suffixes
TEXT_SUFFIXES = [".txt", ".md", ".rst", ".json", ".yaml", ".yml"]

# Executable file suffixes
EXEC_SUFFIXES = [".sh", ".bash", ".zsh"]

ALLOWED_FILE_SUFFIXES = CODE_SUFFIXES + DATA_SUFFIXES + TEXT_SUFFIXES + EXEC_SUFFIXES
