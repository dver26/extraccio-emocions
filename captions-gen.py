import base64
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai._gaos.lib.compat_errors import BadRequestError

SAMPLE_N = 20

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ENV_PATH)

client_gemini = None


def init_worker():
    global client_gemini
    client_gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def loop_process(image_name, i):
    t0 = time.perf_counter()
    print(image_name)
    with open(f"../../images/FLICKR-AES-001/40K/{image_name}", "rb") as fitxer:
        image_bytes = fitxer.read()
        print(f"Comença el procés del fitxer {i} en paral·lel")
    fitxer.close()

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    try:
        interaction = client_gemini.interactions.create(  # pyright: ignore
            model="gemini-3.5-flash-lite",
            input=[
                {
                    "type": "text",
                    "text": "Eres un experto en describir imágenes. Describe la imagen lo mejor que puedas, muy al detalle todo, que se pueda, al detalleeee.",
                },
                {"type": "image", "data": image_base64, "mime_type": "image/jpeg"},
            ],
        )
        texto = interaction.output_text  # pyright: ignore[reportAttributeAccessIssue]
    except BadRequestError as e:
        print(f"La IA no ha volgut analitzar: {e}")
    print(f"Acaba el procés del fitxer {i} en paral·lel")
    print(f"[{i}] llamada: {time.perf_counter() - t0:.1f}  ss")


def main():

    df_labels_worekrs = pd.read_csv("labels_by_worker.csv")
    df_scores = pd.read_csv("scores.csv")

    print(
        f"El dataset va amb puntuacions desde {df_labels_worekrs['score'].min()} a {df_labels_worekrs['score'].max()}"
    )
    print(
        f"El dataset va amb puntuacions desde {df_scores['score'].min()} a {df_scores['score'].max()}"
    )

    client_gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    inici_seq = time.time()
    for i, image_name in enumerate(df_scores["image"]):
        if i == SAMPLE_N:
            break
        print(image_name)
        with open(f"../../images/FLICKR-AES-001/40K/{image_name}", "rb") as fitxer:
            image_bytes = fitxer.read()
            print(f"Hola des del fitxer {i}")
        fitxer.close()

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        try:
            interaction = client_gemini.interactions.create(
                model="gemini-3.5-flash-lite",
                input=[
                    {
                        "type": "text",
                        "text": "Eres un experto en describir imágenes. Describe la imagen lo mejor que puedas, muy al detalle todo, que se pueda, al detalleeee.",
                    },
                    {"type": "image", "data": image_base64, "mime_type": "image/jpeg"},
                ],
            )
        except BadRequestError as e:
            print(f"La IA no ha volgut analitzar: {e}")
    final_seq = time.time()

    print(f"Temps en seqüencial: {final_seq - inici_seq}")

    image_names = df_scores["image"][:SAMPLE_N]

    inici_parallel = time.time()
    t_pool = time.perf_counter()

    with ProcessPoolExecutor(max_workers=4, initializer=init_worker) as ex:
        futuros = [
            ex.submit(loop_process, *c) for c in zip(image_names, range(SAMPLE_N))
        ]

        print(f"Pool listo y tareas enviadas: {time.perf_counter() - t_pool:.1f} s")

        for fut in as_completed(futuros):
            fut.result()

    final_parallel = time.time()

    print(f"Temps en paral·lel: {final_parallel - inici_parallel}")


if __name__ == "__main__":
    main()
