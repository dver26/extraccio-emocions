from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

for model in ["gemini-3.5-flash-lite", "gemini-2.5-flash-image"]:
    try:
        resp = client.models.generate_content(
            model=model,
            contents=["a red fox in a snowy forest"],
        )
        for part in resp.parts:
            if part.inline_data:
                part.as_image().save(f"{model}.png")
                print(model, "OK")
    except Exception as e:
        print(model, "ERROR:", str(e)[:200])
