# chatBots — RAG + Chatbot con Groq

Chatbot que responde preguntas sobre tus PDFs usando ChromaDB (búsqueda vectorial) y Groq (LLM).

## Pasos (VS Code, Windows)

1. Abre la carpeta `chatBots` en VS Code y una terminal (Ctrl + ñ).
2. Crea y activa el entorno virtual:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. Instala dependencias:
   ```
   pip install -r requirements.txt
   ```
4. Copia `.env.example` a `.env` y pega tu key de https://console.groq.com
5. Pon tus PDFs en la carpeta `pdfs/`.
6. Indexa los PDFs (solo la primera vez, o cuando cambies los PDFs):
   ```
   python -m app.ingest
   ```
7. Inicia la interfaz web (se abre sola en el navegador):
   ```
   streamlit run interfaz.py
   ```
   Alternativa en terminal: `python chatbot.py` (con `--fuentes` muestra los fragmentos usados).

## Estructura

- `app/config.py` — rutas, modelos y parámetros
- `app/ingest.py` — carga, chunking, embeddings y ChromaDB
- `app/rag.py` — recuperación, prompt aumentado y Groq
- `interfaz.py` — interfaz web de chat (Streamlit)
- `chatbot.py` — chat en terminal
