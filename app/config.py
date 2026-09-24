"""Configuración central del proyecto (rutas, modelos y parámetros del RAG)."""
import os
from pathlib import Path

from dotenv import load_dotenv

# Carpeta raíz del proyecto (chatBots/), sin importar desde dónde se ejecute
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# --- Rutas ---
PDF_DIR = BASE_DIR / "pdfs"
CHROMA_DIR = BASE_DIR / "chroma"
COLLECTION_NAME = "mis_programas"

# --- Embeddings locales (CPU) ---
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# --- Chunking ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- Recuperación ---
TOP_K = 10  # fragmentos que se recuperan por pregunta

# --- LLM (Groq) ---
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
TEMPERATURE = 0.0


def get_groq_api_key() -> str:
    """Lee la API key desde el archivo .env (o variable de entorno)."""
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key or key == "gsk_PEGA_TU_KEY_AQUI":
        raise RuntimeError(
            "Falta la GROQ_API_KEY. Crea el archivo .env (copia .env.example) "
            "y pega tu key de https://console.groq.com"
        )
    return key
