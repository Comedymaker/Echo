from typing import Any, Dict, List, Tuple, Union
import numpy as np
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
from utils.config_loader import load_config
from models.model import TinyModelLoader, LargeModelLoader
from models.tokenizer import Tokenizer
from dataloader.dataset import PaperDataset, NewsDataset, RatingDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch
import os
import torch.nn as nn
from peft import LoraConfig, AutoPeftModelForCausalLM, PeftModel


BEGIN_KEY = f"<|system|>"
RESPONSE_KEY = f"<|assistant|>"

class CustomSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super(CustomSFTTrainer, self).__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        tokenizer = Tokenizer.load_tokenizer()

        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits  #  (batch_size, seq_length, vocab_size)

        first_sample_logits = logits[0]
        first_sample_labels = labels[0] 

        predicted_ids = torch.argmax(first_sample_logits, dim=-1) 

        predicted_ids_list = predicted_ids.cpu().numpy().tolist()
        label_ids_list = first_sample_labels.cpu().numpy().tolist()

        predicted_text = tokenizer.decode(predicted_ids_list, skip_special_tokens=True)
        label_text = tokenizer.decode(
            [id if id != -100 else tokenizer.pad_token_id for id in label_ids_list],
            skip_special_tokens=True
        )

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            logits.view(-1, logits.size(-1)),  
            labels.view(-1)
        )

        return (loss, outputs) if return_outputs else loss

class DataCollatorForCompletionLM(DataCollatorForLanguageModeling):    
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        
        batch = super().torch_call(examples)

        labels = batch["labels"].clone()

        # print(labels[0].tolist())

        begin_token_ids = self.tokenizer.encode(BEGIN_KEY)
        config = load_config()
        if(config["base"]["tiny_model_id"] == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
            response_token_ids = [29914, 25580, 29962]
        else:
            response_token_ids = [151644, 77091, 198]

        for i in range(len(examples)):
            response_token_ids_start_idx = None

            for idx in range(len(batch["labels"][i]) - len(response_token_ids) + 1):
                current_window = batch["labels"][i][idx:idx + len(response_token_ids)].tolist()

                if current_window == response_token_ids:
                    response_token_ids_start_idx = idx
                    break

            if response_token_ids_start_idx is None:
                token_ids = batch["labels"][i].tolist()
                raise RuntimeError(
                    f'Could not find response key {response_token_ids} in token IDs {token_ids}'
                )

            response_token_ids_end_idx = response_token_ids_start_idx + len(response_token_ids)

            labels[i, :response_token_ids_start_idx] = -100 

        batch["labels"] = labels

        return batch

class FineTuner:
    def __init__(self): 
        self.config = load_config()
        self.tokenizer = Tokenizer.load_tokenizer()
        if(self.config["tinyModel_training"]["train_type"] == "tiny"):
            self.model = TinyModelLoader.load_model()
        else:
            self.model = LargeModelLoader.load_model()
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.data = PaperDataset.format_data()
        # self.data = NewsDataset.format_data()
        # self.data = RatingDataset.format_data()
        self.data_collator = DataCollatorForCompletionLM(tokenizer=self.tokenizer, mlm=False, return_tensors="pt")
        
    def _get_training_args(self):
        return TrainingArguments(
            output_dir=self._get_output_dir(),
            per_device_train_batch_size=self.config["tinyModel_training"]["batch_size"],
            gradient_accumulation_steps=self.config["tinyModel_training"]["gradient_accumulation_steps"],
            learning_rate=self.config["tinyModel_training"]["learning_rate"],
            num_train_epochs=self.config["tinyModel_training"]["num_epochs"],
            max_steps=self.config["tinyModel_training"]["max_steps"],
            fp16=self.config["tinyModel_training"]["fp16"],
            save_strategy="steps",
            save_total_limit=1,
            logging_steps=10,
            optim="paged_adamw_32bit",
            lr_scheduler_type="cosine",
            report_to="wandb"
        )
    
    def _get_output_dir(self):
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if(self.config["tinyModel_training"]["train_type"] == "tiny"):
            return f"{self.config['tinyModel_training']['output_dir']}/{timestamp}_{self.config['base']['tiny_model_id'].split('/')[-1]}"
        else:
            return f"{self.config['tinyModel_training']['output_dir']}/{timestamp}_{self.config['base']['large_model_id'].split('/')[-1]}"

    def run(self):
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.data["train"],
            eval_dataset=self.data["test"],
            peft_config=self._get_lora_config(),
            args=self._get_training_args(),
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            max_seq_length=512,
        )
        trainer.train()
        trainer.evaluate()

        self._save_model(trainer)

    def _get_lora_config(self):
        from peft import LoraConfig
        return LoraConfig(
            r=self.config["lora"]["r"],
            lora_alpha=self.config["lora"]["alpha"],
            lora_dropout=self.config["lora"]["dropout"],
            target_modules=self.config["lora"]["target_modules"],
            bias="lora_only",
            task_type="CAUSAL_LM"
        )

    def _save_model(self, trainer):
        del trainer.model
        torch.cuda.empty_cache()  
        if(self.config["tinyModel_training"]["train_type"] == "tiny"):
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config['base']['tiny_model_id'],
                load_in_8bit=False,
                device_map="cpu",
                torch_dtype=torch.float16
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config['base']['large_model_id'],
                load_in_8bit=False,
                device_map="cpu",
                torch_dtype=torch.float16
            )
        base_model.resize_token_embeddings(len(self.tokenizer))
        peft_model = PeftModel.from_pretrained(
            base_model,
            f"{trainer.args.output_dir}/checkpoint-{self.config['tinyModel_training']['max_steps']}", 
            device_map="auto",
            from_transformers=True
        )

        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained(trainer.args.output_dir + "_merged")
        self.tokenizer.save_pretrained(trainer.args.output_dir + "_merged")

        print(f"Final model saved in: {trainer.args.output_dir}_merged")
