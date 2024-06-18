import pdb
import random
import torch

from model_utils import calc_probs_log_probs
from trainers.utils import PolicyTrainer, compute_advantage, compile_from_log, update_critic

class PPOTrainer(PolicyTrainer):

    def __init__(self, ppo_clip_coef, ppo_update_iter, **kwargs):
        super().__init__(**kwargs)

        assert self.use_critic, "For PPO, critic model must be provided."

        self.ppo_clip_coef = ppo_clip_coef
        self.ppo_update_iter = ppo_update_iter

    def train(self, generation_results):
        if generation_results == []:
            return {"loss": None, "critic_loss": None}

        self.data += generation_results
        self.gradient_accumulated_steps += 1
        if self.gradient_accumulated_steps < self.gradient_accumulation_steps:
            return {"loss": None, "critic_loss": None}

        self.critic_data += generation_results
        self.critic_accumulated_steps += 1

        losses = []
        data = [x for r in self.data for x in r]

        compute_advantage(data=data,
                          batch_size=self.batch_size,
                          critic_model=self.critic_model,
                          tokenizer=self.tokenizer)

        for iter in range(self.ppo_update_iter):
            self.optimizer.zero_grad()

            random.shuffle(data)
            for i in range(0, len(data), self.batch_size):
                # Get the input batch for this step
                keyword_dict = compile_from_log(
                    generation_result_steps=data[i:i + self.batch_size],
                    pad_token_id=self.tokenizer.pad_token_id)
                input_tokens = keyword_dict["tokens"]
                attention_mask = keyword_dict["attention_mask"]
                generated_mask = keyword_dict["generated_mask"]
                advantages = keyword_dict["advantage"]
                old_probs = keyword_dict["prob"]

                # PPO uses probs
                probs_log_probs = calc_probs_log_probs(
                    model=self.model,
                    tokens=input_tokens,
                    attention_mask=attention_mask,
                    generated_mask=generated_mask,
                    generation_config=self.generation_config,
                    calc_probs=True,
                    calc_log_probs=False)
                probs = probs_log_probs["probs"]

                # Advantage for minimizing cost is negative of maximizing reward
                loss1 = probs / old_probs * (-advantages)
                loss2 = torch.clamp(probs / old_probs, 1 - self.ppo_clip_coef,
                                    1 + self.ppo_clip_coef) * (-advantages)
                loss = torch.sum(-torch.min(loss1, loss2)) / len(data)
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    pdb.set_trace()
                losses.append(loss.item())

                loss.backward()

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

        # Equal to average per step loss
        return {
            "loss": sum(losses) / self.ppo_update_iter,
            "critic_loss": critic_loss
        }
