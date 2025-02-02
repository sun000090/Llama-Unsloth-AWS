import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from accelerate import init_empty_weights, infer_auto_device_map, disk_offload
from unsloth import FastLanguageModel
from logger import Logger

logs = Logger.logger_init()

class TrainModel:
    # Prepare the prompt template
    def generate_prompt(data_point):
        try:
            result = f"[INST] {data_point['prompt']} [/INST] {data_point['response']}"
            logs.info('Data generated')
            return result
        except Exception as e:
            logs.info(f'Data generation failed {e}')
            return None
        
    # Function to add a 'text' column based on the generated prompt
    def add_text(example):
        try:
            example["text"] = TrainModel.generate_prompt(example)
            # Optionally remove the original columns to clean up the dataset
            example.pop("prompt", None)
            example.pop("response", None)
            logs.info('Data generated')
            return example
        except Exception as e:
            logs.info(f'Data generation failed {e}')
            return None

    def train_save_model(model_name_,dataset):
    # Initialize the model
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name_,
            max_seq_length = 8192,
            dtype = torch.float16,
            load_in_4bit = True,
            )
            logs.info('Model and tokenizer loaded')
        except Exception as e:
            logs.info(f'Model and tokenizer loading failed {e}')

        # LoRA config
        try:
            model = FastLanguageModel.get_peft_model(
            model = model,
            r = 8, # LoRA rank
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
            lora_alpha = 16,
            lora_dropout = 0.05, # LoRA dropout
            bias = "none", # LoRA bias ("none","all", "lora_only")
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 100,
            use_rslora = False,  # Rank stabilized LoRA
            loftq_config = None, # And LoftQ
            )
            logs.info('Lora generation done')

            training_arguments = TrainingArguments(
            output_dir="./train_logs",
            fp16=True,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=1,
            eval_accumulation_steps=1,
            evaluation_strategy='epoch',
            optim="paged_lion_32bit",
            learning_rate=1e-3,
            lr_scheduler_type="cosine",
            max_grad_norm=0.3,
            warmup_ratio=0.01,
            weight_decay=0.001,
            save_strategy='epoch',
            group_by_length=False,
            report_to="none"
            )

            sft_trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            dataset_text_field="text",
            args=training_arguments,
            max_seq_length=4096,
            packing=False,
            compute_metrics=None
            )
            logs.info('SFT Trainer loaded')

        except Exception as e:
            logs.info(f'Training parameters generation pipeline error {e}')

    # Train and save model
        try:
            logs.info('Training started')
            sft_trainer.train()
            logs.info('Training completed')
            # Save the model and merge the weights
            model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit")
            logs.info('Model saved')
            return
        except Exception as e:
            logs.info(f'Training error {e}')
            return None
