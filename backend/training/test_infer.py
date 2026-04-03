from httpcore import request
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_PATH = "training/lora-output"

tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
model = PeftModel.from_pretrained(base_model, LORA_PATH)

question = "What does NSCC stand for?"
prompt = f"<|system|>\nYou are an NSCC assistant. Answer only using the training information.\n<|user|>\n{question}\n<|assistant|>\n"

inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=50)

print(tokenizer.decode(output[0], skip_special_tokens=True))