from fastapi import FastAPI
from app import translate

app = FastAPI()


@app.get("/")
def home():

    return {
        "message": "Transformer Translator Running"
    }


@app.get("/translate")
def translate_api(text: str):

    result = translate(text)

    return {
        "input": text,
        "translation": result
    }