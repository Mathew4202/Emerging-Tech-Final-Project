from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import torch

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR.parent.parent / "data" / "model_train.jsonl"
OUTPUT_DIR = "training/lora-output"

print("Using data file:", DATA_FILE)
print("File exists:", DATA_FILE.exists())
dataset = load_dataset("json", data_files=str(DATA_FILE), split="train")

def format_example(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    }

formatted_dataset = dataset.map(format_example)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

if torch.cuda.is_available():
    model = model.to("cuda")
print("CUDA available:", torch.cuda.is_available())
print("Model device:", next(model.parameters()).device)

lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    num_train_epochs=8,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    fp16=torch.cuda.is_available()
)

trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    args=training_args,
    processing_class=tokenizer
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("LoRA training complete")