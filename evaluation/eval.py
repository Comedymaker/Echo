import torch
import os
import sys
import numpy as np
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformers import AutoTokenizer, pipeline
from torch.utils.data import DataLoader
from models.model import TinyModelLoader, LargeModelLoader  # 匹配你的模型加载器路径
from models.tokenizer import Tokenizer
from dataloader.dataset import PaperDataset, NewsDataset, RatingDataset
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from utils.config_loader import load_config

def set_random_seed(seed_num):
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)

set_random_seed(1057)

# 设置路径和设备
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# 加载配置和分词器
config = load_config()  # 确保该函数已定义
tokenizer = Tokenizer.load_tokenizer()

os.environ["CUDA_VISIBLE_DEVICES"]=config["base"]["device_id"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_fn(batch):
    """调整后的数据整理函数（支持单模型测试）"""
    texts = [item["text"] for item in batch]
    titles = [item["title"] for item in batch]
    
    # 输入文本编码（保持与训练时一致的参数）
    text_encodings = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
        add_special_tokens=True,
        padding_side="left"
    )
    
    # 标签编码（标题）
    title_encodings = tokenizer(
        titles,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
        padding_side="left"
    )

    # 确保每个标题序列包含 <eos> token
    labels = title_encodings["input_ids"]
    
    
    return {
        "input_ids": text_encodings["input_ids"].to(device),
        "attention_mask": text_encodings["attention_mask"].to(device),
        "labels": labels.to(device)
    }

def calculate_loss(logits, labels):
    """定义损失计算函数（假设使用交叉熵）"""
    ce_loss = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    # 展开为 (batch*seq_len, vocab_size)
    logits_flat = logits.view(-1, logits.size(-1))
    labels_flat = labels.view(-1)
    return ce_loss(logits_flat, labels_flat)

def calculate_metrics(predictions, references):
    """合并ROUGE和BLEU计算"""
    # 计算ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {
        'rouge1': 0.0,
        'rouge2': 0.0,
        'rougeL': 0.0
    }
    for pred, ref in zip(predictions, references):
        scores = scorer.score(pred, ref)
        rouge_scores['rouge1'] += scores['rouge1'].fmeasure
        rouge_scores['rouge2'] += scores['rouge2'].fmeasure
        rouge_scores['rougeL'] += scores['rougeL'].fmeasure
    for k in rouge_scores:
        rouge_scores[k] /= len(predictions)
    
    # 计算BLEU
    smoothing = SmoothingFunction().method4
    bleu = sum(sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothing)
              for pred, ref in zip(predictions, references)) / len(predictions)
    
    return rouge_scores, bleu

# 主测试流程
if __name__ == "__main__":
    # 指定模型路径（从配置文件读取）
    # 加载模型（使用你的加载器）
    # model = TinyModelLoader.load_finetuned_model()  # 自动加载到正确设备
    model = LargeModelLoader.load_finetuned_model()  # 自动加载到正确设备
    # model.resize_token_embeddings(len(tokenizer))
    model.eval()  # 设置为评估模式
    
    # 加载数据集（假设PaperDataset支持test_split方法）
    dataset = PaperDataset.format_data_combModel()  
    # dataset = NewsDataset.format_data_combModel()
    # dataset = RatingDataset.format_data_combModel()
    test_loader = DataLoader(
        dataset["test"],
        batch_size=1,          # 单条测试更稳定
        shuffle=False,
        collate_fn=collate_fn
    )

    all_predictions = []
    all_references = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            # 前向传播（注意：需要显式传递labels）
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                # labels=labels
            )
            
            # 获取logits和预测文本
            logits = outputs.logits  # 直接访问模型输出属性
            # 生成文本需要单独调用generate方法
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=config["base"]["max_length"],
                do_sample=True,          # 启用采样（非贪心解码）
                temperature=0.1,         # 控制输出多样性（0.5-1.0常见）
                top_k=50,               # 核采样（top-p sampling）阈值
                eos_token_id=tokenizer.eos_token_id  # 确保正确结束符
            )

            # 解码预测和标签
            predictions = tokenizer.batch_decode(
                generated_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )
            references = tokenizer.batch_decode(
                labels, 
                skip_special_tokens=True
            )
            
            # 处理预测内容，确保只包含[/INST]符号之后的部分
            if config["base"]["tiny_model_id"] == "TinyLlama/TinyLlama-1.1B-Chat-v1.0":
                predictions = [prediction.split("[/INST]")[1].strip() if "[/INST]" in prediction else prediction for prediction in predictions]
            else:
                predictions = [prediction.split("assistant\n")[1].strip() if "assistant\n" in prediction else prediction for prediction in predictions]


            # 保存结果
            all_predictions.extend(predictions)
            all_references.extend(references)
            
            # 计算损失
            loss = calculate_loss(logits, labels)
            total_loss += loss.item()
            
            # 打印示例
            print(f"\nSample {len(all_predictions)}:")
            print(f"Prediction: {predictions[0]}")
            print(f"Reference:  {references[0]}")

        # 计算平均指标
        avg_loss = total_loss / len(test_loader)
        rouge, bleu = calculate_metrics(all_predictions, all_references)

        # 输出结果
        print(f"\nAverage Loss: {avg_loss:.4f}")
        print(f"ROUGE-1: {rouge['rouge1']:.4f}")
        print(f"ROUGE-2: {rouge['rouge2']:.4f}")
        print(f"ROUGE-L: {rouge['rougeL']:.4f}")
        print(f"BLEU: {bleu:.4f}")