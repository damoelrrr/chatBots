"""Servidor web Flask del chatbot RAG.

Local:       python servidor.py          (abre http://127.0.0.1:5000)
Producción:  gunicorn servidor:app       (lo hace el Dockerfile)
"""
import os
import threading

from flask import Flask, jsonify, render_template, request

from app import config
from app.rag import RagChatbot

app = Flask(__name__)

MAX_PREGUNTA = 1000  # caracteres

# --- El RAG se carga UNA sola vez (embeddings + ChromaDB + Groq) ---
_bot = None
_lock = threading.Lock()


def get_bot() -> RagChatbot:
    global _bot
    if _bot is None:
        with _lock:
            if _bot is None:
                _bot = RagChatbot()
    return _bot


def _precalentar() -> None:
    """Carga el modelo en segundo plano para que la 1.ª pregunta no espere."""
    try:
        get_bot()
    except Exception as e:  # se volverá a intentar cuando llegue una pregunta
        print(f"[!] No se pudo precargar el RAG: {e}")


threading.Thread(target=_precalentar, daemon=True).start()


@app.get("/")
def index():
    return render_template("index.html", modelo=config.GROQ_MODEL)


@app.get("/salud")
def salud():
    return jsonify(estado="ok")


@app.post("/chat")
def chat():
    datos = request.get_json(silent=True) or {}
    pregunta = (datos.get("mensaje") or "").strip()

    if not pregunta:
        return jsonify(error="Escribe una pregunta."), 400
    if len(pregunta) > MAX_PREGUNTA:
        return jsonify(error=f"La pregunta es muy larga (máx. {MAX_PREGUNTA} caracteres)."), 400

    try:
        resultado = get_bot().preguntar(pregunta)
    except (FileNotFoundError, RuntimeError) as e:  # sin índice o sin API key
        return jsonify(error=str(e)), 503
    except Exception:  # límite de Groq, sin internet, etc.
        app.logger.exception("Error al consultar el RAG")
        return jsonify(
            error="No pude consultar el modelo (¿límite de Groq o sin conexión?). Intenta de nuevo."
        ), 502

    fuentes = [
        {"fuente": fuente, "texto": doc.page_content}
        for fuente, doc in zip(resultado["fuentes"], resultado["fragmentos"])
    ]
    return jsonify(respuesta=resultado["respuesta"], fuentes=fuentes)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", 5000)), debug=False)