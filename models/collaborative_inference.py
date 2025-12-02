import torch
from transformers import AutoTokenizer
from utils.config_loader import load_config
from utils.replay_buffer import ReplayBuffer
import torch.nn.functional as F
from utils.compute_conf import compute_confidence_features

class CollaborativeInference:
    def __init__(self, large_model, small_model, weight_network, tokenizer, device):
        self.config = load_config()
        self.large_model = large_model
        self.small_model = small_model
        self.weight_network = weight_network.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.eos_token_id = tokenizer.eos_token_id 
        self.replay_buffer = ReplayBuffer(max_size=1000) 
        self.llm_past_key_values = None
        self.slm_past_key_values = None

        self.invalid_token_strings = [
            self.tokenizer.pad_token,
            "<|im_start|>",
            "<|im_end|>",
            "<s>", "</s>", 
        ]
        self.invalid_token_ids = set()
        for token in self.invalid_token_strings:
            ids = self.tokenizer(token, add_special_tokens=False).input_ids
            self.invalid_token_ids.update(ids)

    def get_outputs(self, input_ids, attention_mask, llm_past_key_values=None, slm_past_key_values=None, use_past=True):
        with torch.no_grad():
            large_out = self.large_model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                use_cache=use_past, 
                output_hidden_states=False,
                past_key_values=self.llm_past_key_values if use_past else None)

        with torch.no_grad():
            small_out = self.small_model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                use_cache=use_past, 
                output_hidden_states=True,
                past_key_values=self.slm_past_key_values if use_past else None)
        
        return large_out, small_out
    
    def forward(self, input_ids, attention_mask, labels=None, use_past=True):
        batch_size = input_ids.size(0)
        current_ids = input_ids
        current_labels = self.tokenizer.batch_decode(current_ids)
        input_length = input_ids.size(1)
        max_length = self.config["base"]["max_length"]
        finished = torch.zeros(batch_size, dtype=torch.bool).to(self.device)
        top_k = self.config["base"]["top_k"]
        temperature = self.config["base"]["temperature"]
        current_mask = attention_mask

        self.llm_past_key_values = None
        self.slm_past_key_values = None

        logits_sequence = []
        for step in range(max_length):
            active = ~finished
            if not active.any():
                break
            
            if use_past:
                if self.llm_past_key_values is None:
                    llm_inputs = current_ids
                    llm_masks = current_mask
                else:
                    llm_inputs = current_ids[:, -1:]
                    llm_masks = current_mask[:, -1:]

                if self.slm_past_key_values is None:
                    slm_inputs = current_ids
                    slm_masks = current_mask
                else:
                    slm_inputs = current_ids[:, -1:]
                    slm_masks = current_mask[:, -1:]
                
            else:
                llm_inputs = current_ids
                llm_masks = current_mask

            llm_outputs, slm_outputs = self.get_outputs(
                input_ids=llm_inputs,
                attention_mask=llm_masks,
                llm_past_key_values=self.llm_past_key_values,
                slm_past_key_values=self.slm_past_key_values,
                use_past=use_past
            )

            self.llm_past_key_values = llm_outputs.past_key_values
            self.slm_past_key_values = slm_outputs.past_key_values

            llm_logits = llm_outputs.logits.to(torch.float32)
            slm_logits = slm_outputs.logits.to(torch.float32)

            llm_last_token = llm_logits[:, -1, :]
            slm_last_token = slm_logits[:, -1, :] 

            probs_s = F.softmax(slm_last_token, dim=-1)
            probs_l = F.softmax(llm_last_token, dim=-1)

            entropy_s = -torch.sum(probs_s * torch.log(probs_s + 1e-8), dim=-1)  # [B]
            entropy_l = -torch.sum(probs_l * torch.log(probs_l + 1e-8), dim=-1)  # [B]

            last_hidden_state = slm_outputs.hidden_states[-1]  
            slm_hidden_states = last_hidden_state[:, -1, :].to(torch.float32) 

            conf_feat = compute_confidence_features(slm_last_token, llm_last_token, topk=1)  # [B, 5]

            # weights = self.weight_network(llm_last_token, slm_last_token)
            weights = self.weight_network(slm_hidden_states, conf_feat)
            
            weights_llm = weights
            weights_slm = 1 - weights


            combined_logits = (weights_llm * llm_last_token) + (weights_slm * slm_last_token)

            if labels is not None:
                next_token = labels[:, step] 

                # if self.replay_buffer is not None:
                #     for b in range(batch_size):
                #         label_token = labels[b, step].item()
                #         if label_token in self.invalid_token_ids:
                #             continue 
                #         sample = {
                #             "input_ids": current_ids[b].detach().cpu(),    
                #             "attention_mask": current_mask[b].detach().cpu(),
                #             "label_token_id": labels[b, step].item()
                #         }
                #         balancing_weight = self.config["replay"]["balancing_weight"]
                #         score = abs(entropy_l[b] - entropy_s[b]).item()
                #         self.replay_buffer.add(sample, score)
                if self.replay_buffer is not None:
                    alpha = float(self.config["replay"].get("balancing_weight", 0.5))

                    # 取出需要的三列：熵_s、熵_l、JS
                    entropy_s = conf_feat[:, 0]           # [B]
                    entropy_l = conf_feat[:, 1]           # [B]
                    js        = conf_feat[:, 4]           # [B]

                    # 计算每个样本的争议度得分（向量化）
                    score_vec = alpha * (entropy_s - entropy_l).abs() + (1.0 - alpha) * js
                    score_vec = torch.nan_to_num(score_vec, nan=0.0)  # 避免极端数值

                    for b in range(batch_size):
                        label_token = labels[b, step].item()
                        if label_token in self.invalid_token_ids:
                            continue

                        sample = {
                            "input_ids":      current_ids[b].detach().cpu(),
                            "attention_mask": current_mask[b].detach().cpu(),
                            "label_token_id": label_token,
                        }

                        # 使用新的分数
                        score = float(score_vec[b].detach().cpu())
                        print(f"Score for sample in batch {b}, step {step}: {score:.4f}")
                        self.replay_buffer.add(sample, score)
                        

            else:
                next_logits = combined_logits

                next_logits /= temperature
                if top_k > 0:
                    top_logits, top_indices = torch.topk(next_logits, top_k)
                    next_logits = torch.full_like(next_logits, -float("Inf"))
                    next_logits.scatter_(1, top_indices, top_logits)

                probs = torch.nn.functional.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

            finished = finished | (next_token == self.eos_token_id)

            next_token_all = next_token.unsqueeze(1)

            current_ids = torch.cat([current_ids, next_token_all], dim=1)

            new_attention = torch.where(
                finished.unsqueeze(1),
                torch.tensor(0, dtype=torch.long, device=self.device),
                torch.tensor(1, dtype=torch.long, device=self.device)
            ).expand(batch_size, 1) 

            current_mask = torch.cat([current_mask, new_attention], dim=1)

            padded_combined_logits = combined_logits.unsqueeze(1)  # [batch_size, 1, vocab_size]
            logits_sequence.append(padded_combined_logits)

        total_max_length = input_length + max_length
        if len(logits_sequence) < max_length:
            pad_logits = torch.zeros(
                batch_size, 1, combined_logits.size(-1),
                dtype=torch.float32,
                device=self.device
            )
            for _ in range(max_length - len(logits_sequence)):
                logits_sequence.append(pad_logits)

        combined_logits = torch.cat(logits_sequence, dim=1)

        if current_ids.size(1) < total_max_length:
            pad_length = total_max_length - current_ids.size(1)
            pad_tokens = torch.full(
                (batch_size, pad_length),
                self.eos_token_id,
                dtype=torch.long,
                device=self.device
            )
            current_ids = torch.cat([current_ids, pad_tokens], dim=1)
        else:
            current_ids = current_ids[:, :total_max_length]

        generated_tokens = current_ids[:, input_length:]

        return {
            "output_ids": current_ids,
            "combined_logits": combined_logits,
            "generated_tokens": generated_tokens  
        }

    def generate(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

        attention_mask = torch.ones_like(inputs)
        outputs = self.forward(inputs, attention_mask)


        output_ids = outputs["output_ids"]
        
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)
