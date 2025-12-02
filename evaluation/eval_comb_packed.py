import torch
import os
import sys
import numpy as np
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from torch.utils.data import DataLoader
from models.model import TinyModelLoader, LargeModelLoader
from utils.config_loader import load_config
from models.tokenizer import Tokenizer
from dataloader.dataset import PaperDataset, NewsDataset
from models.weight_network import WeightNetwork
from models.collaborative_inference import CollaborativeInference
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def set_random_seed(seed_num):
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)

def collate_fn(batch):
    """自定义数据整理函数"""
    texts = [item["text"] for item in batch]
    titles = [item["title"] for item in batch]
    text_encodings = tokenizer(
        texts,
        padding="longest",
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False,
        padding_side="left"
    )
    title_encodings = tokenizer(
        titles,
        padding="max_length",
        truncation=True,
        max_length=35,
        return_tensors="pt"
    )
    # 获取 <eos> token 的 ID
    eos_token_id = tokenizer.eos_token_id
    
    # 确保每个标题序列包含 <eos> token
    labels = title_encodings["input_ids"]
    
    for i in range(labels.size(0)):  # 遍历每个样本
        if labels[i, -1] != eos_token_id:  # 如果最后一个 token 不是 <eos> token
            labels[i, -1] = eos_token_id  # 将最后一个 token 替换为 <eos> token
    
    # 返回字典，包括输入文本的 input_ids、attention_mask 和标签 labels
    return {
        "input_ids": text_encodings["input_ids"].to(device),
        "attention_mask": text_encodings["attention_mask"].to(device),
        "labels": labels.to(device)  # 修改后的标签
    }

def calculate_bleu(predictions, references):
    """计算 BLEU 评分"""
    smoothing_function = SmoothingFunction().method4
    bleu_scores = []

    for pred, ref in zip(predictions, references):
        ref = [ref.split()]
        pred = pred.split()
        bleu_score = sentence_bleu(ref, pred, smoothing_function=smoothing_function)
        bleu_scores.append(bleu_score)

    return sum(bleu_scores) / len(bleu_scores)

def calculate_rouge(predictions, references):
    """计算 ROUGE 评分"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {
        'rouge1': 0.0,
        'rouge2': 0.0,
        'rougeL': 0.0
    }

    for pred, ref in zip(predictions, references):
        score = scorer.score(pred, ref)
        rouge_scores['rouge1'] += score['rouge1'].fmeasure
        rouge_scores['rouge2'] += score['rouge2'].fmeasure
        rouge_scores['rougeL'] += score['rougeL'].fmeasure

    num_samples = len(predictions)
    for key in rouge_scores:
        rouge_scores[key] /= num_samples

    return rouge_scores

def evaluate_comb_model(checkpoint_path: str, dataset_type: str = "paper"):
    tokenizer = Tokenizer.load_tokenizer()
    set_random_seed(1057)

    config = load_config()
    os.environ["CUDA_VISIBLE_DEVICES"] = config["base"]["device_id"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型和 checkpoint
    checkpoint = torch.load(checkpoint_path)
    tiny_model = TinyModelLoader.load_finetuned_model()
    large_model = LargeModelLoader.load_finetuned_model()
    tiny_model.eval()
    large_model.eval()

    # 加载数据
    if dataset_type == "paper":
        data = PaperDataset.format_data_combModel()
    elif dataset_type == "news":
        data = NewsDataset.format_data_combModel()
    else:
        raise ValueError("Unsupported dataset_type. Use 'paper' or 'news'.")

    ctx_dim = 1024 if config["base"]["tiny_model_id"] == "Qwen/Qwen1.5-0.5B-Chat" else 2048
    weight_network = WeightNetwork(vocab_size=len(tokenizer), hidden_dims=[512, 512], ctx_dim=ctx_dim)
    weight_network.load_state_dict(checkpoint["model_state"])
    weight_network.eval()

    collaborative_inference = CollaborativeInference(large_model, tiny_model, weight_network, tokenizer, device)

    test_loader = DataLoader(
        data["test"],
        batch_size=1,
        shuffle=True,
        collate_fn=lambda batch: {
            "input_ids": tokenizer(
                [item["text"] for item in batch],
                padding="longest",
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
                padding_side="left"
            )["input_ids"].to(device),
            "attention_mask": tokenizer(
                [item["text"] for item in batch],
                padding="longest",
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
                padding_side="left"
            )["attention_mask"].to(device),
            "labels": (
                lambda enc: (
                    enc["input_ids"].masked_fill(
                        (enc["input_ids"][:, -1] != tokenizer.eos_token_id).unsqueeze(1).expand(-1, enc["input_ids"].shape[1]),
                        tokenizer.eos_token_id
                    ) if enc["input_ids"].shape[1] > 0 else enc["input_ids"]
                )
            )(
                tokenizer(
                    [item["title"] for item in batch],
                    padding="max_length",
                    truncation=True,
                    max_length=35,
                    return_tensors="pt"
                )
            ).to(device)
        }
    )

    all_predictions = []
    all_references = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            attention_mask = batch["attention_mask"]

            outputs = collaborative_inference.forward(input_ids, attention_mask, use_past=False)
            predictions = tokenizer.batch_decode(outputs["generated_tokens"], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            references = tokenizer.batch_decode(labels, skip_special_tokens=True)

            all_predictions.extend(predictions)
            all_references.extend(references)

            print(f"\nSample {len(all_predictions)}:")
            print(f"Prediction: {predictions[0]}")
            print(f"Reference:  {references[0]}")

    # 打分
    rouge_scores = calculate_rouge(all_predictions, all_references)
    bleu_scores = calculate_bleu(all_predictions, all_references)

    print(f"\nROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
    print(f"BLEU:    {bleu_scores:.4f}")

    return {
        "rouge1": rouge_scores['rouge1'],
        "rouge2": rouge_scores['rouge2'],
        "rougeL": rouge_scores['rougeL'],
        "bleu": bleu_scores
    }