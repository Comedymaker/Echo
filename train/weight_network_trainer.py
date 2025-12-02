# train_weight_network.py
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.config_loader import load_config
from models.model import TinyModelLoader, LargeModelLoader
from models.tokenizer import Tokenizer
from dataloader.dataset import PaperDataset, NewsDataset, RatingDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch
import os
from models.weight_network import WeightNetwork
from models.collaborative_inference import CollaborativeInference
from torch.optim import AdamW
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from datetime import datetime



class WeightNetworkTrainer:
    def __init__(self):
        self.config = load_config()
        self.tokenizer = Tokenizer.load_tokenizer() 
        
        os.environ["CUDA_VISIBLE_DEVICES"]=self.config["base"]["device_id"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.path = f"{self.config['combModel_training']['output_dir']}/{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        os.makedirs(self.path, exist_ok=True)

        self.num_epochs = self.config["combModel_training"]["num_epochs"]

        self.large_model = LargeModelLoader.load_finetuned_model()
        
        # self.large_model.resize_token_embeddings(len(self.tokenizer))

        # self.large_model = TinyModelLoader.load_finetuned_model()
        self.tiny_model = TinyModelLoader.load_finetuned_model()
        
        if(self.config["base"]["tiny_model_id"] == "Qwen/Qwen1.5-0.5B-Chat"):
            ctx_dim = 1024
        elif(self.config["base"]["tiny_model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"):
            ctx_dim = 896
        else:
            ctx_dim = 2048

        self.weight_network = WeightNetwork(vocab_size=len(self.tokenizer), hidden_dims=[512, 512], ctx_dim=ctx_dim)
        # self.weight_network = WeightNetwork(vocab_size=len(self.tokenizer), hidden_dims=[512, 512])
        self.data = PaperDataset.format_data_combModel()
        # self.data = NewsDataset.format_data_combModel()
        # self.data = RatingDataset.format_data_combModel()

        self.collaborative_inference = CollaborativeInference(self.large_model, self.tiny_model, self.weight_network, self.tokenizer, self.device)

        self.train_loader = DataLoader(
            self.data["train"],
            batch_size=self.config["combModel_training"]["batch_size"],
            shuffle=True,
            collate_fn=self.collate_fn 
        )

        self.test_loader = DataLoader(
            self.data["test"],
            batch_size=self.config["combModel_training"]["batch_size"],
            shuffle=True,
            collate_fn=self.collate_fn
        )
        
        self.optimizer = AdamW(
            self.weight_network.parameters(),
            lr=self.config["combModel_training"]["lr"],
            weight_decay=0.01
        )
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=100,
            num_training_steps=len(self.train_loader)*self.config["combModel_training"]["num_epochs"]
        )


    def collate_fn(self, batch):
        texts = [item["text"] for item in batch]
        titles = [item["title"] for item in batch]
        text_encodings = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=512,
            return_tensors="pt",
            add_special_tokens=False,
            padding_side="left"
        )
        title_encodings = self.tokenizer(
            titles,
            padding="max_length",
            truncation=True,
            max_length=self.config["base"]["max_length"],
            return_tensors="pt"
        )
        eos_token_id = self.tokenizer.eos_token_id
        
        labels = title_encodings["input_ids"]
        
        for i in range(labels.size(0)):
            if labels[i, -1] != eos_token_id:
                labels[i, -1] = eos_token_id 
        
        return {
            "input_ids": text_encodings["input_ids"].to(self.device),
            "attention_mask": text_encodings["attention_mask"].to(self.device),
            "labels": labels.to(self.device)
        }

    def calculate_loss(self, logits, labels):
        flat_logits = logits.view(-1, logits.size(-1))  # (batch_size * output_length, vocab_size)
        flat_labels = labels.view(-1)                   # (batch_size * output_length)
        
        return F.cross_entropy(
            flat_logits,
            flat_labels,
            ignore_index=self.tokenizer.pad_token_id
        )

    def evaluate(self):
        self.weight_network.eval() 
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_references = []

        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch["input_ids"]
                labels = batch["labels"]
                attention_mask = batch["attention_mask"]

                outputs = self.collaborative_inference.forward(input_ids, attention_mask, use_past=False)
                weighted_logits = outputs["combined_logits"]

                loss = self.calculate_loss(weighted_logits, labels)

                total_loss += loss.item()
                num_batches += 1

                predictions = self.tokenizer.batch_decode(outputs["generated_tokens"], skip_special_tokens=True)
                references = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
                
                all_predictions.extend(predictions)
                all_references.extend(references)

        avg_loss = total_loss / num_batches
        print(f"Test Loss: {avg_loss}")

        rouge_scores = self.calculate_rouge(all_predictions, all_references)
        bleu_scores = self.calculate_bleu(all_predictions, all_references)

        print(f"ROUGE-1: {rouge_scores['rouge1']}")
        print(f"ROUGE-2: {rouge_scores['rouge2']}")
        print(f"ROUGE-L: {rouge_scores['rougeL']}")
        print(f"BLEU: {bleu_scores}")

        return avg_loss, rouge_scores, bleu_scores

    def calculate_bleu(self, predictions, references):
        smoothing_function = SmoothingFunction().method4
        bleu_scores = []

        for pred, ref in zip(predictions, references):
            ref = [ref.split()]
            pred = pred.split()
            bleu_score = sentence_bleu(ref, pred, smoothing_function=smoothing_function)
            bleu_scores.append(bleu_score)

        return sum(bleu_scores) / len(bleu_scores)

    def calculate_rouge(self, predictions, references):
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
    
    def freeze_models(self):
        for param in self.tiny_model.parameters():
            param.requires_grad = False

        for param in self.large_model.parameters():
            param.requires_grad = False

    def train_step(self, batch):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]
        
        self.optimizer.zero_grad()

        outputs = self.collaborative_inference.forward(input_ids, attention_mask, labels, use_past=True)
        
        weighted_logits = outputs["combined_logits"] 

        print("Logits shape:", weighted_logits.shape)

        for pos in [0, 1]:
            logits = weighted_logits[0, pos] 
            probs = F.softmax(logits, dim=-1)
            topk = torch.topk(probs, k=10)
            
            print(f"\n🧠 Token Position {pos}:")
            for i, (token, prob) in enumerate(zip(
                self.tokenizer.convert_ids_to_tokens(topk.indices.tolist()),
                topk.values.tolist()
            )):
                print(f"Top {i+1}: Token = {token}, Probability = {prob:.4f}")

        decoded_labels = [self.tokenizer.decode(label_ids, skip_special_tokens=False) for label_ids in labels]
        print(decoded_labels)

        loss = self.calculate_loss(weighted_logits, labels)
        
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.weight_network.parameters(), 1)
        
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        return loss.item()

    def train_weight_network(self):
        self.weight_network.train()
        self.freeze_models()
        num_epochs = self.num_epochs
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_idx, batch in enumerate(self.train_loader):
                loss = self.train_step(batch)
                total_loss += loss
                
                if batch_idx % 10 == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {avg_loss:.4f}")
            
            self.save_checkpoint(epoch)
            print(f"Epoch {epoch} saved in {self.path}/checkpoint_epoch{epoch}.pt")
        

        replay_path = f"replay_data/{datetime.now().strftime('%Y%m%d%H%M%S')}"
        os.makedirs(replay_path, exist_ok=True)
        self.collaborative_inference.replay_buffer.save(f"{replay_path}/replay.pt")
        print(f"Replay data saved to {replay_path}/replay.pt")

    def save_checkpoint(self, epoch):
        """Save intermediate results"""
        checkpoint = {
            "epoch": epoch,
            "model_state": self.weight_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict()
        }
        full_path = os.path.join(self.path, f"checkpoint_epoch{epoch}.pt")
        torch.save(checkpoint, full_path)
