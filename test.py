import json
import pandas as pd
from transformers import AutoTokenizer
from models.tokenizer import Tokenizer
from utils.config_loader import load_config

config = load_config()
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B-Chat")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
        # tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')
tokenizer.pad_token_id = pad_token_id

text = tokenizer.apply_chat_template([
            {
                "role": "system",
                "content": "You are an academic assistant. Your task is to generate a concise and accurate paper title **only**, based on the user's abstract. The title should: 1) Output **only the title** (no explanations, formatting, or extra text); 2) capture the core innovation; 3) include key technical terms; 4) be under 20 words.",
            },
            {"role": "user", "content": "Stochastic computing is a novel approach to real arithmetic, offering better error tolerance and lower hardware costs over the conventional implementations. Stochastic modules are digital systems that process random bit streams representing real values in the unit interval. Stochastic modules based on finite state machines (FSMs) have been shown to realize complicated arithmetic functions much more efficiently than combinational stochastic modules. However, a general approach to synthesize FSMs for realizing arbitrary functions has been elusive. We describe a systematic procedure to design FSMs that implement arbitrary real-valued functions in the unit interval using the Taylor series approximation."},
            {"role": "assistant", "content": "Stochastic Functions Using Sequential Logic"}
        ], tokenize=False, add_generation_prompt=False, add_special_tokens=True)

print(text)

print(tokenizer.encode(text))

print(tokenizer.encode("<|im_start|>assistant\n"))

print(tokenizer.decode([151644, 77091, 198]))