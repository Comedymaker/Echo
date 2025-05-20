from datasets import Dataset
import json
import pandas as pd
from models.tokenizer import Tokenizer
from utils.config_loader import load_config

class PaperDataset:
    @staticmethod
    def format_data():
        config = load_config()
        dataset_path = config["base"]["lamp5_path"]
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)

        tokenizer = Tokenizer.load_tokenizer()

        df["text"] = df[["abstract", "title"]].apply(lambda x: 
            tokenizer.apply_chat_template([
            {
                "role": "system",
                "content": "You are an academic assistant. Your task is to generate a concise and accurate paper title **only**, based on the user's abstract. The title should: 1) Output **only the title** (no explanations, formatting, or extra text); 2) capture the core innovation; 3) include key technical terms; 4) be under 20 words.",
            },
            {"role": "user", "content": x["abstract"]},
            {"role": "assistant", "content": x["title"]}
        ], tokenize=False, add_generation_prompt=False, add_special_tokens=True), axis=1)

        dataset = Dataset.from_pandas(df)

        return dataset.train_test_split(seed=1057, test_size=0.2)

    def format_data_combModel():
        config = load_config()
        dataset_path = config["base"]["lamp5_path"]
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)

        tokenizer = Tokenizer.load_tokenizer()

        df["text"] = df["abstract"].apply(lambda x:
            tokenizer.apply_chat_template([
            {
                "role": "system",
                "content": "You are an academic assistant who always generate a concise and accurate paper title based on the abstract provided by the user, without any explanations or formatting. The title should: 1) capture the core innovation; 2) include key technical terms; 3) be under 20 words.",
            },
            {"role": "user", "content": x}
        ], tokenize=False, add_generation_prompt=True, add_special_tokens=True))

        dataset = Dataset.from_pandas(df)

        return dataset.train_test_split(seed=42, test_size=0.2)

    def preprocess_function(examples):
        inputs = ["Generate a concise academic paper title based on the abstract: " + doc for doc in examples["abstract"]]  # 输入文本
        targets = examples["title"]     

        tokenizer = Tokenizer.load_tokenizer()

        model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=512)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, truncation=True, padding="max_length", max_length=512)

        model_inputs["labels"] = labels["input_ids"]

        print(f"input_ids: {model_inputs["input_ids"]}")
        print(f"labels: {model_inputs["labels"]}")

        return model_inputs

class NewsDataset:
    @staticmethod
    def format_data():
        config = load_config()
        dataset_path = config["base"]["lamp4_path"]
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)

        tokenizer = Tokenizer.load_tokenizer()

        df["text"] = df[["abstract", "title"]].apply(lambda x: 
            tokenizer.apply_chat_template([
            {
                "role": "system",
                "content": "You are a news assistant. Your task is to generate a concise and accurate news headline only, based on the provided news article. The headline should: 1) Output only the headline (no explanations, formatting, or extra text); 2) capture the core event or key focus; 3) include critical terms or locations; 4) remain under 15 words.",
            },
            {"role": "user", "content": x["abstract"]},
            {"role": "assistant", "content": x["title"]}
        ], tokenize=False, add_generation_prompt=False, add_special_tokens=True), axis=1)

        dataset = Dataset.from_pandas(df)

        return dataset.train_test_split(seed=1057, test_size=0.1)

    def format_data_combModel():
        config = load_config()
        dataset_path = config["base"]["lamp4_path"]
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)

        tokenizer = Tokenizer.load_tokenizer()

        df["text"] = df["abstract"].apply(lambda x:
            tokenizer.apply_chat_template([
            {
                "role": "system",
                "content": "You are a news assistant. Your task is to generate a concise and accurate news headline only, based on the provided news article. The headline should: 1) Output only the headline (no explanations, formatting, or extra text); 2) capture the core event or key focus; 3) include critical terms or locations; 4) remain under 15 words.",
            },
            {"role": "user", "content": x}
        ], tokenize=False, add_generation_prompt=True, add_special_tokens=True))

        dataset = Dataset.from_pandas(df)

        return dataset.train_test_split(seed=42, test_size=0.1)

class RatingDataset:
    @staticmethod
    def format_data():
        config = load_config()
        dataset_path = config["base"]["lamp3_path"]
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)

        tokenizer = Tokenizer.load_tokenizer()

        df["text"] = df[["abstract", "title"]].apply(lambda x: 
            tokenizer.apply_chat_template([
            {
                "role": "system",
                "content": "You are a helpful assistant that only outputs a rating from 1 to 5 for the given user review. Do not explain with any word. Only respond with a single number.",
            },
            {"role": "user", "content": x["abstract"]},
            {"role": "assistant", "content": x["title"]}
        ], tokenize=False, add_generation_prompt=False, add_special_tokens=True), axis=1)

        dataset = Dataset.from_pandas(df)

        return dataset.train_test_split(seed=1057, test_size=0.2)

    def format_data_combModel():
        config = load_config()
        dataset_path = config["base"]["lamp3_path"]
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)

        tokenizer = Tokenizer.load_tokenizer()

        df["text"] = df["abstract"].apply(lambda x:
            tokenizer.apply_chat_template([
            {
                "role": "system",
                "content": "You are a helpful assistant that only outputs a rating from 1 to 5 for the given user review. Do not explain with any word. Only respond with a single number.",
            },
            {"role": "user", "content": x}
        ], tokenize=False, add_generation_prompt=True, add_special_tokens=True))

        dataset = Dataset.from_pandas(df)

        return dataset.train_test_split(seed=42, test_size=0.2)