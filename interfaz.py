"""Interfaz web del chatbot RAG (Streamlit).

Uso:
    streamlit run interfaz.py
"""
import streamlit as st

from app import config
from app.rag import RagChatbot

st.set_page_config(page_title="Chatbot RAG", page_icon="🤖", layout="centered")


@st.cache_resource(show_spinner="Cargando modelos (la primera vez tarda un poco)...")
def cargar_bot() -> RagChatbot:
    return RagChatbot()


def mostrar_fuentes(fragmentos: list[dict]) -> None:
    with st.expander(f"Fuentes usadas ({len(fragmentos)})"):
        for i, f in enumerate(fragmentos, 1):
            st.markdown(f"**[{i}] {f['fuente']}**")
            st.caption(f["texto"])


# ---------- Barra lateral ----------
with st.sidebar:
    st.header("Ajustes")
    k = st.slider("Fragmentos a recuperar (k)", 1, 15, config.TOP_K)
    ver_fuentes = st.toggle("Mostrar fuentes", value=True)
    if st.button("Limpiar chat", use_container_width=True):
        st.session_state.mensajes = []
        st.rerun()
    st.divider()
    st.caption(f"LLM: {config.GROQ_MODEL} (Groq)")
    st.caption(f"Embeddings: {config.EMBEDDING_MODEL}")

# ---------- Cabecera ----------
st.title("🤖 Chatbot RAG")
st.caption("Pregunta sobre el contenido de tus PDFs.")

# ---------- Carga del bot ----------
try:
    bot = cargar_bot()
except (FileNotFoundError, RuntimeError) as e:
    st.error(str(e))
    st.stop()

# ---------- Historial ----------
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

for m in st.session_state.mensajes:
    with st.chat_message(m["rol"]):
        st.markdown(m["contenido"])
        if m["rol"] == "assistant" and ver_fuentes and m.get("fragmentos"):
            mostrar_fuentes(m["fragmentos"])

# ---------- Entrada del usuario ----------
pregunta = st.chat_input("Escribe tu pregunta...")

if pregunta:
    st.session_state.mensajes.append({"rol": "user", "contenido": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Buscando en los documentos..."):
                resultado = bot.preguntar(pregunta, k=k)
        except Exception as e:  # p. ej. límite de Groq o sin internet
            st.error(f"Error al consultar el modelo: {e}")
            st.stop()

        fragmentos = [
            {"fuente": fuente, "texto": doc.page_content}
            for fuente, doc in zip(resultado["fuentes"], resultado["fragmentos"])
        ]
        st.markdown(resultado["respuesta"])
        if ver_fuentes:
            mostrar_fuentes(fragmentos)

    st.session_state.mensajes.append(
        {
            "rol": "assistant",
            "contenido": resultado["respuesta"],
            "fragmentos": fragmentos,
        }
    )
