import torch
import os
import sys
import numpy as np
import random
import sklearn
import math
import re
from sklearn.metrics import mean_absolute_error, mean_squared_error
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from torch.utils.data import DataLoader
from models.model import TinyModelLoader, LargeModelLoader
from utils.config_loader import load_config
from models.tokenizer import Tokenizer
from dataloader.dataset import PaperDataset, NewsDataset, RatingDataset
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

def calculate_metrics(predictions, references):
    predictions = [int(p) for p in predictions]
    references = [int(r) for r in references]

    mae = mean_absolute_error(references, predictions)
    mse = mean_squared_error(references, predictions)
    rmse = math.sqrt(mse)

    return mae, rmse

tokenizer = Tokenizer.load_tokenizer()

set_random_seed(1057)

path = "../autodl-tmp/results/models/combModel/20250513095352/checkpoint_epoch4.pt"
checkpoint = torch.load(path)

config = load_config()

tiny_model = TinyModelLoader.load_finetuned_model()
large_model = LargeModelLoader.load_finetuned_model()
tiny_model.eval()
large_model.eval()
# large_model.resize_token_embeddings(len(tokenizer))



os.environ["CUDA_VISIBLE_DEVICES"]=config["base"]["device_id"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取数据集
# data = PaperDataset.format_data_combModel()
# data = NewsDataset.format_data_combModel()
data = RatingDataset.format_data_combModel()

if(config["base"]["tiny_model_id"] == "Qwen/Qwen1.5-0.5B-Chat"):
    ctx_dim = 1024
else:
    ctx_dim = 2048

weight_network = WeightNetwork(vocab_size=len(tokenizer), hidden_dims=[512, 512], ctx_dim=ctx_dim)
# weight_network = WeightNetwork(vocab_size=len(tokenizer), hidden_dims=[512, 512])

collaborative_inference = CollaborativeInference(large_model, tiny_model, weight_network, tokenizer, device)

weight_network.load_state_dict(checkpoint["model_state"])

test_loader = DataLoader(
    data["test"],
    batch_size=1,
    shuffle=True,
    collate_fn=collate_fn  # 修改为测试数据整理函数
)

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
        outputs = collaborative_inference.forward(input_ids, attention_mask, use_past=False)
        weighted_logits = outputs["combined_logits"]
        # 计算损失
        # loss = calculate_loss(weighted_logits, labels)

        # total_loss += loss.item()
        num_batches += 1

        # 获取预测和参考文本
        predictions = tokenizer.batch_decode(outputs["generated_tokens"], skip_special_tokens=True)
        references = tokenizer.batch_decode(labels, skip_special_tokens=True)

        if config["base"]["tiny_model_id"] == "TinyLlama/TinyLlama-1.1B-Chat-v1.0":
            predictions = [prediction.split("[/INST]")[1].strip() if "[/INST]" in prediction else prediction for prediction in predictions]
        else:
            predictions = [re.findall(r"\b[1-5]\b", pred)[-1] if re.findall(r"\b[1-5]\b", pred) else "3" for pred in predictions]


        # 打印当前批次的预测和标签（可选：仅打印前 2 个样本）
        # print("\n--- 当前批次示例 ---")
        # for pred, ref in zip(predictions[:4], references[:4]):
        #     print(f"预测文本: {pred}")
        #     print(f"标签文本: {ref}")
        #     print("-------------------")
        # 打印示例
        print(f"\nSample {len(all_predictions)}:")
        print(f"Prediction: {predictions[0]}")
        print(f"Reference:  {references[0]}")

        # 处理预测内容，确保只包含[/INST]符号之后的部分
        
                
        all_predictions.extend(predictions)
        all_references.extend(references)

    # avg_loss = total_loss / num_batches
    # print(f"Test Loss: {avg_loss}")

    # 计算 ROUGE 和 BLEU 评分
    mae, rmse = calculate_metrics(all_predictions, all_references)
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")