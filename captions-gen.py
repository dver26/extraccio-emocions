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
MODEL = "gemini-3.1-flash-lite"
ruta_images = "/home/dver26/dev/cvc/images/FLICKR-AES-001/40K/"

load_dotenv(ENV_PATH)


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
        im = Image.open(f"{ruta_images}{image_name}")

        try:
            response = client_gemini.models.generate_content(
                model=MODEL,
                contents=[im, "Describe la imagen"],
            )
        except BadRequestError as e:
            print(f"La IA no ha volgut analitzar: {e}")

        print(response.text)  # type: ignore
    final_seq = time.time()

    print(f"Temps en seqüencial: {final_seq - inici_seq}")


if __name__ == "__main__":
    main()
