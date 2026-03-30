from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def home():
    return {"message": "Backend is working"}

@app.post("/chat")
def chat(request: ChatRequest):
    return {"reply": f"You said: {request.message}"}