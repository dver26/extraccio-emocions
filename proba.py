import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai

MODEL = "gemini-3.5-flash-lite"

# Cargar el .env de la carpeta padre
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

print("Modelos disponibles para generar contenido:\n")

for model in client.models.list():
    if "generateContent" in (model.supported_actions or []):
        print(f"Nombre: {model.name}")
        print(f"Descripción: {model.description}")
        print("-" * 50)

t0 = time.time()

response = client.models.generate_content(
    model=MODEL, contents="Hola, Gemini. Responde en una sola frase."
)
print(time.time() - t0, "segons de resposta")
print(response.text)
