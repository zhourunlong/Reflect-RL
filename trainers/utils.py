import pdb
import random
import torch

from statistics import mean

MAX_VAL = 100

def compile_from_log(generation_result_steps, pad_token_id):
    STACK_KEYS = [("tokens", torch.long, pad_token_id),
                       ("generated_mask", torch.bool, 0),
                       ("attention_mask", torch.bool, 0),]
    TENSOR_KEYS = [("Q_value", torch.float32), ("advantage", torch.float32),
                   ("prob", torch.float32), ("log_prob", torch.float32)]
    LIST_KEYS = ["prompt", "scores"]
    KEYS = [k[0] for k in (STACK_KEYS + TENSOR_KEYS)] + LIST_KEYS

    ret = {k: [] for k in KEYS}
    for step in generation_result_steps:
        for key in KEYS:
            if key in step:
                ret[key].append(step[key])

    # Must contain 'tokens'
    max_len = max([x.shape[0] for x in ret["tokens"]])

    for key, dtype, pad_val in STACK_KEYS:
        if key in ret:
            for i in range(len(ret["tokens"])):
                cur_len = ret[key][i].shape[0]
                # Pad to same length
                ret[key][i] = torch.cat((torch.full((max_len - cur_len,), pad_val, dtype=dtype),
                                     ret[key][i]))
            ret[key] = torch.stack(ret[key])

    for key, dtype in TENSOR_KEYS:
        if key in ret:
            ret[key] = torch.tensor(ret[key], dtype=dtype)

    return ret


def compute_advantage(data, batch_size, critic_model, tokenizer):
    for i in range(0, len(data), batch_size):
        # Get the input batch for this step
        keyword_dict = compile_from_log(data[i:i + batch_size],
                                        tokenizer.pad_token_id)
        prompts = keyword_dict["prompt"]

        if critic_model is not None:
            value_inputs = tokenizer.batch_encode_plus(
                prompts,
                truncation=True,
                padding=True,
                max_length=critic_model.config.max_length,
                return_tensors="pt")
            value_inputs = {
                k: v.to(critic_model.device) for k, v in value_inputs.items()
            }

            with torch.no_grad():
                values = critic_model(**value_inputs).logits.squeeze(-1)
        else:
            values = torch.zeros(len(prompts))

        j = 0
        for d in data[i:i + batch_size]:
            d["advantage"] = d["Q_value"] - values[j].item()
            j += 1


def update_critic(data, critic_model, critic_optimizer, tokenizer, batch_size,
                  max_grad_norm, gradient_accumulation_steps, update_iter):
    losses = []

    accumulated_steps = 0
    critic_optimizer.zero_grad()

    for iter in range(update_iter):
        random.shuffle(data)
        for i in range(0, len(data), batch_size):
            # Get the input batch for this step
            keyword_dict = compile_from_log(data[i:i + batch_size],
                                            tokenizer.pad_token_id)
            Q_values = keyword_dict["Q_value"]
            prompts = keyword_dict["prompt"]

            value_inputs = tokenizer.batch_encode_plus(
                prompts,
                truncation=True,
                padding=True,
                max_length=critic_model.config.max_length,
                return_tensors="pt")
            value_inputs = {
                k: v.to(critic_model.device) for k, v in value_inputs.items()
            }
            values = critic_model(**value_inputs).logits.squeeze(-1)

            critic_optimizer.zero_grad()

            loss = torch.nn.MSELoss()(Q_values.to(critic_model.device) / MAX_VAL,
                                      values / MAX_VAL) / gradient_accumulation_steps
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                pdb.set_trace()
            loss.backward()
            losses.append(loss.item())
            accumulated_steps += 1

            if accumulated_steps == gradient_accumulation_steps:
                torch.nn.utils.clip_grad_norm_(critic_model.score.parameters(),
                                               max_grad_norm)
                critic_optimizer.step()
                accumulated_steps = 0
                critic_optimizer.zero_grad()

    if accumulated_steps:
        torch.nn.utils.clip_grad_norm_(critic_model.score.parameters(), max_grad_norm)
        critic_optimizer.step()

    return mean(losses)


class PolicyTrainer:

    def __init__(self, model, tokenizer, optimizer, gradient_accumulation_steps,
                 generation_config, critic_model, critic_optimizer,
                 critic_update_freq, critic_update_iter, batch_size,
                 max_grad_norm, **kwargs):
        """
        Compute gradient for proximal policy optimization.

        Args:
        - `model` (PeftModel): the model to be updated
        - `tokenizer` (AutoTokenizer): the tokenizer for `model`
        - `optimizer` (torch.optim.Optimizer): the optimizer for `model`
        - `generation_config` (GenerationConfig): the generation config used to
        generate the dialog
        - `generation_results` (list): the generation result, which is a list
        consisting of `batch_size` lists. Each inner list contains dicts in the
        following format:
        {
            "tokens": torch.Tensor,
            "generated_mask": list,
            "attention_mask": torch.Tensor,
            "cost": float,
            "Q_value": float,
            "step": int
        }
        - `critic_model` (PeftModel): the value model to be updated
        - `critic_tokenizer` (AutoTokenizer): the tokenizer for `critic_model`
        - `critic_optimizer` (torch.optim.Optimizer): the optimizer for
        `critic_model`
        - `clip_coef` (float): the clipping coefficient for PPO
        - `max_grad_norm` (float): the maximum gradient norm for gradient clipping
        """

        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.generation_config = generation_config
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.critic_model = critic_model
        self.critic_optimizer = critic_optimizer
        self.use_critic = critic_model is not None

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.data = []
        self.gradient_accumulated_steps = 0

        self.critic_update_freq = critic_update_freq
        self.critic_update_iter = critic_update_iter
        self.critic_data = []
        self.critic_accumulated_steps = 0