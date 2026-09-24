FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface

# Usuario no-root (Hugging Face Spaces ejecuta el contenedor con uid 1000)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /home/user/app

# torch solo-CPU (mucho mas liviano que el paquete por defecto con CUDA)
RUN pip install --user torch --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user requirements.txt .
RUN pip install --user -r requirements.txt

COPY --chown=user . .

# Indexa los PDFs al construir la imagen: deja listo chroma/ y descarga el
# modelo de embeddings, asi el arranque es rapido.
RUN python -m app.ingest

# HF Spaces usa el puerto 7860; Render inyecta PORT por su cuenta.
ENV PORT=7860
EXPOSE 7860

# 1 worker (el modelo ocupa RAM) + varios hilos para atender peticiones a la vez
CMD ["sh", "-c", "gunicorn servidor:app --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 180"]