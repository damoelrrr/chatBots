"""RAG: recuperación (ChromaDB) -> prompt aumentado -> respuesta (Groq)."""
import os

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from app import config

PROMPT_TEMPLATE = """Eres un asistente legal experto en reglamentos y normas.
Responde la pregunta usando ÚNICAMENTE la información del contexto proporcionado. Al final de cada parte de la respuesta, incluye entre paréntesis la fuente y la página del fragmento de donde proviene la información, por ejemplo: (Fuente: NombreDocumento.pdf, Pág. X).
Si la respuesta no está en el contexto, indica exactamente: "No encontré información sobre esto en la base de conocimientos."

Contexto recuperado del documento:
{context}

Pregunta del usuario: {question}

Respuesta:"""


def _pagina(doc) -> str:
    """PyPDFLoader numera las páginas desde 0; se muestra la página real."""
    pagina = doc.metadata.get("page")
    return str(pagina + 1) if isinstance(pagina, int) else "?"


def _fuente(doc) -> str:
    return os.path.basename(doc.metadata.get("source", "?"))


class RagChatbot:
    """Carga los modelos una sola vez y responde preguntas."""

    def __init__(self, k: int = config.TOP_K):
        if not config.CHROMA_DIR.exists():
            raise FileNotFoundError(
                "No existe la base vectorial. Primero indexa los PDFs con:\n"
                "    python -m app.ingest"
            )

        self.k = k
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vector_store = Chroma(
            persist_directory=str(config.CHROMA_DIR),
            embedding_function=embeddings,
            collection_name=config.COLLECTION_NAME,
        )
        self.prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        self.llm = ChatGroq(
            model=config.GROQ_MODEL,
            temperature=config.TEMPERATURE,
            api_key=config.get_groq_api_key(),
        )

    def preguntar(self, pregunta: str, k: int | None = None) -> dict:
        # Paso 5 — Recuperación (búsqueda por similitud en ChromaDB)
        docs = self.vector_store.similarity_search(pregunta, k=k or self.k)

        # Paso 6 — Prompt aumentado
        contexto = "\n\n---\n\n".join(
            f"[Fuente: {_fuente(d)} — Pág. {_pagina(d)}]\n{d.page_content}"
            for d in docs
        )
        prompt = self.prompt_template.invoke(
            {"context": contexto, "question": pregunta}
        )

        # Paso 7 — Generación con Groq
        respuesta = self.llm.invoke(prompt).content

        return {
            "pregunta": pregunta,
            "fragmentos": docs,
            "contexto": contexto,
            "respuesta": respuesta,
            "fuentes": [f"{_fuente(d)} — Pág. {_pagina(d)}" for d in docs],
        }
