"""Chatbot de terminal sobre tus PDFs (RAG + Groq).

Uso:
    python chatbot.py              # chat normal
    python chatbot.py --fuentes    # además muestra los fragmentos usados
"""
import argparse
import sys

# Para que las tildes y ñ se vean bien en la consola de Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.rag import RagChatbot  # noqa: E402

SALIR = {"salir", "exit", "quit", "q"}


def main():
    parser = argparse.ArgumentParser(description="Chatbot RAG sobre PDFs")
    parser.add_argument(
        "--fuentes", action="store_true", help="mostrar fuentes y páginas usadas"
    )
    args = parser.parse_args()

    print("Cargando modelos (la primera vez puede tardar)...")
    try:
        bot = RagChatbot()
    except (FileNotFoundError, RuntimeError) as e:
        print(f"\n[!] {e}")
        sys.exit(1)

    print("\nChatbot listo. Escribe tu pregunta (o 'salir' para terminar).\n")

    while True:
        try:
            pregunta = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego.")
            break

        if not pregunta:
            continue
        if pregunta.lower() in SALIR:
            print("Hasta luego.")
            break

        try:
            resultado = bot.preguntar(pregunta)
        except Exception as e:  # p. ej. límite de Groq o sin internet
            print(f"[!] Error al consultar el modelo: {e}\n")
            continue

        print(f"\nBot: {resultado['respuesta']}\n")

        if args.fuentes:
            print("Fragmentos usados:")
            for i, fuente in enumerate(resultado["fuentes"], 1):
                print(f"  [{i}] {fuente}")
            print()


if __name__ == "__main__":
    main()
