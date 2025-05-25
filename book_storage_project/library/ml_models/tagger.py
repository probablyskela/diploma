import os
import pickle
import re
import sys

import nltk
import numpy as np
from django.conf import settings
from langdetect import detect
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer


global_ml_model = None
global_sentence_transformer = None

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE_PATH = os.path.join(MODEL_DIR, "model.pkl")
WORKSPACE_ROOT = os.path.abspath(os.path.join(MODEL_DIR, "..", "..", ".."))


def load_trained_model():
    global global_ml_model, global_sentence_transformer

    if global_ml_model is None:
        if os.path.exists(MODEL_FILE_PATH):
            if WORKSPACE_ROOT not in sys.path:
                sys.path.insert(0, WORKSPACE_ROOT)
            try:
                with open(MODEL_FILE_PATH, "rb") as f:
                    global_ml_model = pickle.load(f)
                print("ML Model loaded successfully.")
            except Exception as e:
                print(f"Error loading ML model from {MODEL_FILE_PATH}: {e}")
                global_ml_model = None
            finally:
                if WORKSPACE_ROOT == sys.path[0]:
                    sys.path.pop(0)
        else:
            print(
                f"ML Model file not found at {MODEL_FILE_PATH}. Tagging will not work."
            )
            global_ml_model = None

    if global_sentence_transformer is None:
        try:
            nltk.download("stopwords", quiet=True)
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            global_sentence_transformer = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            print("SentenceTransformer model loaded successfully.")
        except Exception as e:
            print(f"Error loading SentenceTransformer model: {e}")
            global_sentence_transformer = None

    return global_ml_model


def preprocess_text(text: str, language: str = "english") -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)

    try:
        tokens = word_tokenize(text, language=language)
    except Exception:
        tokens = text.split()

    stop_words = set(stopwords.words(language))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    if language == "english":
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    else:
        lemmatized_tokens = filtered_tokens

    return " ".join(lemmatized_tokens)


def chunk_book_text(
    text_content: str, num_chunks: int = 10, tokens_per_chunk: int = 500
) -> str:
    all_tokens = text_content.split()
    total_tokens = len(all_tokens) - tokens_per_chunk

    if total_tokens == 0 or num_chunks <= 0 or tokens_per_chunk <= 0:
        return ""

    selected_tokens_for_all_chunks = []
    for i in range(num_chunks):
        start_token_index = int(i * total_tokens / num_chunks + tokens_per_chunk)

        chunk = all_tokens[
            start_token_index : min(start_token_index + tokens_per_chunk, total_tokens)
        ]
        selected_tokens_for_all_chunks.extend(chunk)

    return " ".join(selected_tokens_for_all_chunks)


def embed(X, model: SentenceTransformer):
    """
    X = preprocess_text(text) -> get_book_text(etext_number, 10, 500)
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    """
    step, overlap = 100, 10
    book_encodings = []
    for i, book in enumerate(X):
        parts = []

        tokens = book.split()
        for part in range(0, len(tokens), step):
            if tokens[part : min(part + step + overlap, len(tokens))]:
                parts.append(
                    " ".join(tokens[part : min(part + step + overlap, len(tokens))])
                )

        if not parts:
            book_encodings.append(np.zeros(384))
        else:
            book_encodings.append(np.array(model.encode(parts)).mean(axis=0))

    return book_encodings


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
    languages = []
    for text in texts:
        try:
            language_code = detect(text[:10000])
            languages.append(language_codes.get(language_code, "english"))
        except Exception as e:
            print(f"Error detecting language for a text: {e}. Defaulting to English.")
            languages.append("english")

    st_model = global_sentence_transformer
    if st_model is None:
        print(
            "SentenceTransformer model is not available in preprocess. Attempting to load."
        )
        load_trained_model()
        st_model = global_sentence_transformer
        if st_model is None:
            print(
                "SentenceTransformer model still not loaded. Returning zero embeddings."
            )
            return [np.zeros(384) for _ in texts]

    return embed(
        [
            chunk_book_text(preprocess_text(text, languages[i]))
            for i, text in enumerate(texts)
        ],
        st_model,
    )


def get_tags_for_texts(list_of_texts: list[str]) -> list[list[str]]:
    model = load_trained_model()

    if model is None or global_sentence_transformer is None:
        print("ML Model or SentenceTransformer is not loaded. Returning default tags.")
        return [["O"] for _ in list_of_texts]

    try:
        processed_texts_embeddings = preprocess(list_of_texts)

        is_all_zeros = all(np.all(emb == 0) for emb in processed_texts_embeddings)
        if is_all_zeros and len(list_of_texts) > 0:
            print(
                "Embeddings are all zeros, likely due to SentenceTransformer model issue. Returning default tags."
            )
            return [["O"] for _ in list_of_texts]

        return model.predict(processed_texts_embeddings)
    except Exception as e:
        print(f"Error during model prediction: {e}")
        return [["O"] for _ in list_of_texts]
