import logging
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai._gaos.lib.compat_errors import BadRequestError
from PIL import Image

SAMPLE_N = 10
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
MODEL_GEMINI = "gemini-3.5-flash-lite"

ruta_images = "/home/dver26/dev/cvc/images/FLICKR-AES-001/40K/"
ruta_captions = "/home/dver26/dev/cvc/codi/extraccio-emocions/captions/"
ruta_imatges_generades = (
    "/home/dver26/dev/cvc/codi/extraccio-emocions/imatges_generades/"
)

load_dotenv(ENV_PATH)

# logging

# 1. Crear el logger principal con nivel DEBUG (para que acepte todo y no filtre nada de entrada)
logger = logging.getLogger("mi_programa")
logger.setLevel(logging.DEBUG)

# Evitar duplicar handlers si ejecutas el script varias veces en el mismo entorno
if not logger.handlers:
    # 2. Manejador para el ARCHIVO: nivel DEBUG (guarda absolutamente todo)
    file_handler = logging.FileHandler("app.log", encoding="utf-8", mode="w")
    file_handler.setLevel(logging.DEBUG)

    # 3. Manejador para la CONSOLA: nivel WARNING (solo muestra warnings, errors y critical)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    # 4. Definir un formato común para ambos
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # 5. Añadir los manejadores al logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def main():

    df_scores = pd.read_csv("scores.csv")

    client_gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    inici_seq = time.time()
    for i, image_name in enumerate(df_scores["image"]):
        logger.info(f"Entrant a imatge: {image_name}")
        if i == SAMPLE_N:
            break

        im = Image.open(f"{ruta_images}{image_name}")

        try:
            response = client_gemini.models.generate_content(
                model=MODEL_GEMINI, contents=[im, "Describe la imagen"]
            )
        except BadRequestError as e:
            print(f"La IA no ha volgut analitzar: {e}")

        caption = response.text  # type: ignore

        if caption is not None:
            with open(f"{ruta_captions}caption_{image_name}_caption", "w") as fitxer:
                fitxer.write(caption)
        else:
            logger.error(
                f"La caption per la imatge numero {i} de nom {image_name} és None"
            )

    final_seq = time.time()

    print(f"Temps en seqüencial: {final_seq - inici_seq}")


if __name__ == "__main__":
    main()
