"""Indexación: PDFs -> fragmentos -> embeddings -> ChromaDB.

Se ejecuta UNA vez (o cada vez que cambies los PDFs):
    python -m app.ingest
"""
import shutil

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config


def cargar_pdfs():
    """Paso 1 — Carga de documentos PDF."""
    if not config.PDF_DIR.exists():
        raise FileNotFoundError(f"No existe la carpeta de PDFs: {config.PDF_DIR}")

    pdf_files = sorted(config.PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No hay archivos .pdf en {config.PDF_DIR}")

    documents = []
    print(f"PDFs encontrados en '{config.PDF_DIR.name}/' ({len(pdf_files)}):")
    for pdf in pdf_files:
        pages = PyPDFLoader(str(pdf)).load()
        documents.extend(pages)
        print(f"  {pdf.name}: {len(pages)} páginas")
    print(f"Total de páginas cargadas: {len(documents)}\n")
    return documents


def dividir(documents):
    """Paso 2 — División en fragmentos (chunking)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(documents)
    print(f"Fragmentos generados: {len(chunks)}\n")
    return chunks


def indexar(chunks):
    """Pasos 3 y 4 — Embeddings locales + almacenamiento en ChromaDB."""
    print(f"Cargando modelo de embeddings: {config.EMBEDDING_MODEL}")
    print("(la primera vez se descarga, ~120 MB)")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    if config.CHROMA_DIR.exists():
        shutil.rmtree(config.CHROMA_DIR)
        print(f"Base anterior eliminada: {config.CHROMA_DIR}")

    print(f"Indexando {len(chunks)} fragmentos en ChromaDB...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(config.CHROMA_DIR),
        collection_name=config.COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"},
    )
    total = vector_store._collection.count()
    print(f"\n[OK] Base vectorial creada en {config.CHROMA_DIR} ({total} fragmentos)")


def main():
    documents = cargar_pdfs()
    chunks = dividir(documents)
    indexar(chunks)


if __name__ == "__main__":
    main()
