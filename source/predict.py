import pickle

from langdetect import detect
import nltk

from source.models.xgbhi import XGBoostHierarchicalModel
from sentence_transformers import SentenceTransformer
from source import config
from source.files import preprocess_text, chunk_book_text, embed

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("omw-1.4")


language_codes = {
    "sq": "albanian",
    "nl": "dutch",
    "ca": "catalan",
    "de": "german",
    "sl": "slovene",
    "hi": "hinglish",
    "hu": "hungarian",
    "ro": "romanian",
    "kk": "kazakh",
    "tr": "turkish",
    "ru": "russian",
    "it": "italian",
    "en": "english",
    "el": "greek",
    "tg": "tajik",
    "no": "norwegian",
    "pt": "portuguese",
    "he": "hebrew",
    "fi": "finnish",
    "da": "danish",
    "fr": "french",
    "sv": "swedish",
    "be": "belarusian",
    "az": "azerbaijani",
    "es": "spanish",
    "ta": "tamil",
    "zh": "chinese",
    "id": "indonesian",
    "ar": "arabic",
    "ne": "nepali",
    "uk": "ukrainian",
    "bn": "bengali",
    "eu": "basque",
}


def preprocess(texts: list[str]) -> list[str]:
    return embed(
        [
            chunk_book_text(
                preprocess_text(text, "ukrainian"), num_chunks=10, tokens_per_chunk=500
            )
            for text in texts
        ],
        SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ),
    )


def predict(
    texts: list[str], model: XGBoostHierarchicalModel | None = None
) -> list[list[str]]:
    if model is None:
        model_name = "multi_3717bcad110b9c5f3e9279d4d5eacf65"
        model_path = config.METRICS_PATH / "xgbhi" / model_name / f"{model_name}.pkl"
        with model_path.open("rb") as f:
            model = pickle.load(f)

    texts = preprocess(texts)
    return model.predict(texts)


if __name__ == "__main__":
    with open("source/models/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("source/models/text.txt", "r") as f:
        text = f.read()
    with open("source/models/text1.txt", "r") as f:
        text1 = f.read()
    print(predict([text, text1], model))
