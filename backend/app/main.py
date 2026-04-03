from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
BACKEND_DIR = Path(__file__).resolve().parent.parent
LORA_PATH = BACKEND_DIR / "training" / "lora-output"

tokenizer = None
model = None

# Exact-answer fallback for important facts
FACTS = {
    "what does nscc stand for": "NSCC stands for Nova Scotia Community College.",
    "what is the full meaning of nscc": "NSCC stands for Nova Scotia Community College.",
    "what does the acronym nscc mean": "NSCC stands for Nova Scotia Community College.",
    "expand nscc": "NSCC stands for Nova Scotia Community College.",

    "how many campuses does nscc have": "NSCC has 14 campuses.",
    "how many nscc campuses are there": "NSCC has 14 campuses.",
    "how many campuses are part of nscc": "NSCC has 14 campuses.",
    "does nscc have only two campuses": "No. NSCC has 14 campuses.",
    "is nscc only in halifax and dartmouth": "No. NSCC has 14 campuses across Nova Scotia.",
    "does nscc only have campuses in halifax and dartmouth": "No. NSCC has 14 campuses across Nova Scotia.",

    "what is ecampus": "eCampus is where students learn and access student supports fully online.",
    "tell me about ecampus": "eCampus is where students learn and access student supports fully online.",

    "what supports are available at nscc": "NSCC offers a wide range of supports, including supports for Mi'kmaw and Indigenous students, African Nova Scotian and Black students, 2SLGBTQ+ students, and international students.",
    "what student supports does nscc offer": "NSCC offers a wide range of supports, including supports for Mi'kmaw and Indigenous students, African Nova Scotian and Black students, 2SLGBTQ+ students, and international students.",

    "what is the nscc foundation": "The NSCC Foundation is a registered charity dedicated to helping students in financial need and improving access to education.",
    "what does the nscc foundation do": "The NSCC Foundation is a registered charity dedicated to helping students in financial need and improving access to education.",

    "where is akerley campus located": "Akerley Campus is located in Dartmouth, Nova Scotia.",
    "what city is akerley campus in": "Akerley Campus is located in Dartmouth, Nova Scotia.",

    "where is annapolis valley campus located": "Annapolis Valley Campus is located in Middleton, Nova Scotia.",
    "what city is annapolis valley campus in": "Annapolis Valley Campus is located in Middleton, Nova Scotia.",

    "where is ivany campus located": "Ivany Campus is located in Dartmouth, Nova Scotia.",
    "what city is ivany campus in": "Ivany Campus is located in Dartmouth, Nova Scotia.",

    "where is institute of technology campus located": "The Institute of Technology Campus is located in Halifax, Nova Scotia.",
    "what city is institute of technology campus in": "The Institute of Technology Campus is located in Halifax, Nova Scotia.",

    "where is kingstec campus located": "Kingstec Campus is located in Kentville, Nova Scotia.",
    "what city is kingstec campus in": "Kingstec Campus is located in Kentville, Nova Scotia.",

    "where is sydney waterfront campus located": "Sydney Waterfront Campus is located in Sydney, Nova Scotia.",
    "what city is sydney waterfront campus in": "Sydney Waterfront Campus is located in Sydney, Nova Scotia.",

    "how can i visit nscc": "You can visit NSCC by taking a virtual campus tour, attending a campus or program info session, finding out what it is like to be an NSCC student, attending an NSCC open house, or booking a campus tour."
}

def get_model():
    global tokenizer, model

    if tokenizer is None or model is None:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
        model = PeftModel.from_pretrained(
            base_model,
            str(LORA_PATH),
            local_files_only=True
        )

    return tokenizer, model

def get_fallback_answer(user_message: str):
    message = user_message.strip().lower()

    for key, value in FACTS.items():
        if key in message:
            return value

    return None

@app.get("/")
def home():
    return {"message": "Backend is working"}

@app.post("/chat")
def chat(request: ChatRequest):
    fallback = get_fallback_answer(request.message)
    if fallback:
        return {"reply": fallback}

    tokenizer, model = get_model()

    prompt = f"<|system|>\nYou are an NSCC assistant. Answer only using the training information.\n<|user|>\n{request.message}\n<|assistant|>\n"

    inputs = tokenizer(prompt, return_tensors="pt")

    output = model.generate(
        **inputs,
        max_new_tokens=60,
        pad_token_id=tokenizer.eos_token_id
    )

    full_text = tokenizer.decode(output[0], skip_special_tokens=True)

    if "<|assistant|>" in full_text:
        reply = full_text.split("<|assistant|>")[-1].strip()
    else:
        reply = full_text.strip()

    return {"reply": reply}


