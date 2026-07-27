import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

app = FastAPI()

# Permitir conexiones desde cualquier origen (útil para conectar con tu interfaz web/móvil)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar cliente de Groq (tomará la API key de las variables de entorno)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

class Message(BaseModel):
    prompt: str

@app.post("/chat")
async def chat_with_ai(message: Message):
    try:
        # Llamada a la API ultrarrápida de Groq con Llama 3
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "Eres un asistente virtual amable, conciso y servicial, similar a Bixby de Samsung.",
                },
                {
                    "role": "user",
                    "content": message.prompt,
                }
            ],
            model="llama3-8b-8192",  # Modelo gratuito y rápido de Groq
        )
        
        reply = chat_completion.choices[0].message.content
        return {"response": reply}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"status": "El asistente está en línea"}
