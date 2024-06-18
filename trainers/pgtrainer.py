import pdb
import random
import torch

from peft import PeftModel
from torch import nn
from transformers import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from model_utils import calc_probs_log_probs
from utils import debug_msg

from trainers.utils import PolicyTrainer, compute_advantage, compile_from_log, update_critic

def calc_pg_loss(
    advantages: torch.Tensor,
    model: PeftModel,
    tokens: torch.Tensor,
    attention_mask: torch.Tensor,
    generated_mask: torch.Tensor,
    generation_config: GenerationConfig,
):
    tokens = tokens.to(model.device)
    attention_mask = attention_mask.to(model.device)
    generated_mask = generated_mask.to(model.device)

    batch_size = tokens.shape[0]
    axis = torch.arange(batch_size, device=model.device)

    loss = 0

    for pos in range(1, tokens.shape[1]):
        if not generated_mask[:, pos].any():
            continue        

        model_kwargs = {
            "attention_mask": attention_mask[:, :pos],
            "use_cache": True,
        }

        logits_processor = model._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=tokens.shape[1],
            encoder_input_ids=tokens,
            prefix_allowed_tokens_fn=None,
            logits_processor=LogitsProcessorList(),
            model_kwargs=model_kwargs,
            negative_prompt_ids=None,
            negative_prompt_attention_mask=None,
        )
        logits_warper = model._get_logits_warper(generation_config)

        model_inputs = model.prepare_inputs_for_generation(tokens[:, :pos], **model_kwargs)

        outputs = model(**model_inputs, return_dict=True)

        # pre-process distribution
        scores = logits_processor(tokens, outputs.logits[:, pos - 1, :])
        scores = logits_warper(tokens, scores)
    
        step_log_probs = nn.functional.log_softmax(scores, dim=-1)[axis, tokens[:, pos]]

        loss += torch.dot(generated_mask[:, pos].float() * step_log_probs,
                          advantages)

    return loss


class PGTrainer(PolicyTrainer):

    def train(self, generation_results):
        if generation_results == []:
            return {"loss": None, "critic_loss": None}

        self.data += generation_results
        self.gradient_accumulated_steps += 1
        if self.gradient_accumulated_steps < self.gradient_accumulation_steps:
            return {"loss": None, "critic_loss": None}

        debug_msg("PGTrainer.train start")

        self.critic_data += generation_results
        self.critic_accumulated_steps += 1

        self.optimizer.zero_grad()

        losses = []
        data = [x for r in self.data for x in r]

        compute_advantage(data=data,
                          batch_size=self.batch_size,
                          critic_model=self.critic_model,
                          tokenizer=self.tokenizer)

        random.shuffle(data)
        for i in range(0, len(data), self.batch_size):
            # Get the input batch for this step
            keyword_dict = compile_from_log(
                generation_result_steps=data[i:i + self.batch_size],
                pad_token_id=self.tokenizer.pad_token_id)
            input_tokens = keyword_dict["tokens"]
            attention_mask = keyword_dict["attention_mask"]
            generated_mask = keyword_dict["generated_mask"]
            advantages = keyword_dict["advantage"].to(self.model.device)

            loss = calc_pg_loss(
                advantages=advantages,
                model=self.model,
                tokens=input_tokens,
                attention_mask=attention_mask,
                generated_mask=generated_mask,
                generation_config=self.generation_config) / len(data)

            if torch.isinf(loss).any():
                continue
            
            loss.backward()
            losses.append(loss.item())

        # Policy network update
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                       self.max_grad_norm)
            
        self.optimizer.step()

        self.gradient_accumulated_steps = 0
        self.data = []

        critic_loss = None
        if (self.use_critic
                and self.critic_accumulated_steps == self.critic_update_freq):
            critic_loss = update_critic(
                data=[x for r in self.critic_data for x in r],
                critic_model=self.critic_model,
                critic_optimizer=self.critic_optimizer,
                tokenizer=self.tokenizer,
                batch_size=self.batch_size,
                max_grad_norm=self.max_grad_norm,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                update_iter=self.critic_update_iter)

            self.critic_accumulated_steps = 0
            self.critic_data = []

        debug_msg("PGTrainer.train finish")

        # print(losses)

        # Equal to average per step loss
        return {"loss": sum(losses), "critic_loss": critic_loss}
