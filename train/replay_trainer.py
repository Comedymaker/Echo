import torch
import torch.nn.functional as F
import os
from utils.config_loader import load_config
from torch.nn.utils.rnn import pad_sequence
from models.model import TinyModelLoader, LargeModelLoader
from utils.replay_buffer import ReplayBuffer
from models.weight_network import WeightNetwork
from models.tokenizer import Tokenizer
from datetime import datetime
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from utils.compute_conf import compute_confidence_features

class CombinerReplayTrainer:
    def freeze_models(self):
        for param in self.tiny_model.parameters():
            param.requires_grad = False

        for param in self.large_model.parameters():
            param.requires_grad = False

    def __init__(self):
        self.config = load_config()
        self.tokenizer = Tokenizer.load_tokenizer()
        
        os.environ["CUDA_VISIBLE_DEVICES"]=self.config["base"]["device_id"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.path = f"{self.config['combModel_training']['output_dir']}/{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        os.makedirs(self.path, exist_ok=True)

        self.large_model = LargeModelLoader.load_finetuned_model()
        
        # self.large_model.resize_token_embeddings(len(self.tokenizer))

        # self.large_model = TinyModelLoader.load_finetuned_model()
        self.tiny_model = TinyModelLoader.load_finetuned_model()
        
        path = self.config["base"]["fusion_network_path"]
        checkpoint = torch.load(path)
        if(self.config["base"]["tiny_model_id"] == "Qwen/Qwen1.5-0.5B-Chat"):
            ctx_dim = 1024
        else:
            ctx_dim = 2048
        self.weight_network = WeightNetwork(vocab_size=len(self.tokenizer), hidden_dims=[512, 512], ctx_dim=ctx_dim)
        self.weight_network.load_state_dict(checkpoint["model_state"])
        print(f"Loaded weight network from {path}")
        self.weight_network.to(self.device)


        self.optimizer = AdamW(
            self.weight_network.parameters(),
            lr=0.0002,
            weight_decay=0.01
        )
    
    def save_checkpoint(self, epoch):
        checkpoint = {
            "epoch": epoch,
            "model_state": self.weight_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        full_path = os.path.join(self.path, f"checkpoint_epoch{epoch}.pt")
        torch.save(checkpoint, full_path)

    def refit(self, replay_buffer, num_epochs=1, batch_size=4):
        self.weight_network.train()
        self.freeze_models()

        samples = replay_buffer.get_all()
        print(f"[Replay] Loaded {len(samples)} samples.")

        for epoch in range(num_epochs):
            total_loss = 0.0
            for i in range(0, len(samples), batch_size):
                batch = samples[i:i+batch_size]

                input_ids = torch.stack([s["input_ids"] for s in batch]).to(self.device)
                attention_mask = torch.stack([s["attention_mask"] for s in batch]).to(self.device)

                labels = torch.tensor([s["label_token_id"] for s in batch], dtype=torch.long).to(self.device)

                with torch.no_grad():
                    llm_outputs = self.large_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
                    slm_outputs = self.tiny_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

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

                    conf_feat = compute_confidence_features(slm_last_token, llm_last_token, topk=3)  # [B, 5]

                weights = self.weight_network(slm_hidden_states, conf_feat)
                weights_llm = weights
                weights_slm = 1 - weights
                combined_logits = (weights_llm * llm_last_token) + (weights_slm * slm_last_token)

                print(f"Combined logits shape: {combined_logits.shape}")

                loss = F.cross_entropy(combined_logits, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / (len(samples) // batch_size + 1)
            print(f"[Replay][Epoch {epoch+1}] Loss: {avg_loss:.4f}")
            self.save_checkpoint(epoch)
            print(f"Epoch {epoch} saved in {self.path}/checkpoint_epoch{epoch}.pt")
