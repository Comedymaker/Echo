from transformers import AutoTokenizer
from utils.config_loader import load_config

class Tokenizer:
    @staticmethod
    def load_tokenizer():
        config = load_config()
        tokenizer = AutoTokenizer.from_pretrained(config["base"]["large_model_id"])
        # tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')
        tokenizer.pad_token_id = pad_token_id
        

        return tokenizer
