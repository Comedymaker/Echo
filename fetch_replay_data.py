import torch
import os
import sys
import numpy as np
import random
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
from datetime import datetime


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

tokenizer = Tokenizer.load_tokenizer()

set_random_seed(1057)



config = load_config()

tiny_model = TinyModelLoader.load_finetuned_model()
large_model = LargeModelLoader.load_finetuned_model()
tiny_model.eval()
large_model.eval()
# large_model.resize_token_embeddings(len(tokenizer))



os.environ["CUDA_VISIBLE_DEVICES"]=config["base"]["device_id"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取数据集
data = PaperDataset.format_data_combModel()
# data = NewsDataset.format_data_combModel()

if(config["base"]["tiny_model_id"] == "Qwen/Qwen1.5-0.5B-Chat"):
    ctx_dim = 1024
elif(config["base"]["tiny_model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"):
    ctx_dim = 896
else:
    ctx_dim = 2048



test_loader = DataLoader(
    data["train"],
    batch_size=1,
    shuffle=True,
    collate_fn=collate_fn  # 修改为测试数据整理函数
)



path = config["base"]["fusion_network_path"]
checkpoint = torch.load(path)
weight_network = WeightNetwork(vocab_size=len(tokenizer), hidden_dims=[512, 512], ctx_dim=ctx_dim)

collaborative_inference = CollaborativeInference(large_model, tiny_model, weight_network, tokenizer, device)

weight_network.load_state_dict(checkpoint["model_state"])

weight_network.eval()  # 设置模型为评估模式

total_loss = 0.0
num_batches = 0
all_predictions = []
all_references = []

with torch.no_grad():  # 关闭梯度计算
    for batch in test_loader:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]

        # 前向传播
        outputs = collaborative_inference.forward(input_ids, attention_mask, labels, use_past=True)
        weighted_logits = outputs["combined_logits"]
        # 计算损失
        # loss = calculate_loss(weighted_logits, labels)

        # total_loss += loss.item()
        num_batches += 1

        # 获取预测和参考文本
        predictions = tokenizer.batch_decode(outputs["generated_tokens"], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        references = tokenizer.batch_decode(labels, skip_special_tokens=True)
                
        all_predictions.extend(predictions)
        all_references.extend(references)

        print(f"\nSample {len(all_predictions)}:")
        print(f"Prediction: {predictions[0]}")
        print(f"Reference:  {references[0]}")

    # avg_loss = total_loss / num_batches
    # print(f"Test Loss: {avg_loss}")

    # 计算 ROUGE 和 BLEU 评分
    rouge_scores = calculate_rouge(all_predictions, all_references)
    bleu_scores = calculate_bleu(all_predictions, all_references)

    print(f"ROUGE-1: {rouge_scores['rouge1']}")
    print(f"ROUGE-2: {rouge_scores['rouge2']}")
    print(f"ROUGE-L: {rouge_scores['rougeL']}")
    print(f"BLEU: {bleu_scores}")

replay_path = f"replay_data/{datetime.now().strftime('%Y%m%d%H%M%S')}"
os.makedirs(replay_path, exist_ok=True)
collaborative_inference.replay_buffer.save(f"{replay_path}/replay.pt")
print(f"Replay data saved to {replay_path}/replay.pt")