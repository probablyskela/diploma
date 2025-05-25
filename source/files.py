import pathlib
import re

import nltk
import numpy as np
import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

from source import config
from source.data import preprocess_dataset

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("omw-1.4")


def preprocess_text(text: str, language: str = "english") -> str:
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r"[^\w\s]", "", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # Tokenize text
    try:
        tokens = word_tokenize(text, language=language)
    except Exception as e:
        tokens = text.split()
        print(f"Error tokenizing text: {e}")

    # Remove stop words
    stop_words = set(stopwords.words(language))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize tokens
    if language == "english":
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    else:
        lemmatized_tokens = filtered_tokens

    return " ".join(lemmatized_tokens)


def preprocess_text_old(text: str) -> str:
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r"[^\w\s]", "", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # Tokenize text
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    return " ".join(lemmatized_tokens)


def chunk_book_text(text_content: str, num_chunks: int, tokens_per_chunk: int) -> str:
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


def get_book_text_full(etext_number: str) -> str:
    filepath = config.BOOKS_DIR / f"{etext_number}.txt"
    with filepath.open("r", encoding="utf-8") as f:
        return f.read()


def get_book_text(etext_number: str, num_chunks: int, tokens_per_chunk: int) -> str:
    """Reads a summarized version of the book text by sampling chunks using get_fixed_size_text_from_file."""
    filepath = config.BOOKS_DIR / f"{etext_number}.txt"

    with filepath.open("r", encoding="utf-8") as f:
        text_content = f.read()

    return chunk_book_text(text_content, num_chunks, tokens_per_chunk)


def get_embedding(etext_number: str) -> np.ndarray:
    filepath = config.EMBEDDINGS_DIR / f"{etext_number}"
    with filepath.open("r") as f:
        return np.array(list(map(float, f.read().split())))


def embed(X, model: SentenceTransformer):
    """
    X = preprocess_text(text) -> get_book_text(etext_number, 10, 500)
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    """
    step, overlap = 100, 10
    book_encodings = []
    for i, book in tqdm.tqdm(enumerate(X)):
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


# def main():
#     output = pathlib.Path("books")
#     output.mkdir(parents=True, exist_ok=True)

#     df = pd.read_csv(config.DATASET_DIR / "gutenberg_metadata.csv")

#     df = preprocess_dataset(config.DATASET_DIR)(df)

#     for filename in tqdm.tqdm(df["Etext Number"].unique()):
#         filepath = config.DATASET_DIR / filename
#         outpath = output.joinpath(f"{filename}.txt")

#         if not filepath.exists():
#             print(f"File {filename} does not exist")
#             continue

#         with filepath.open("r") as f:
#             text = f.read()

#         text = preprocess_text(text)

#         with outpath.open("w") as f:
#             f.write(text)


# def main():
#     for book in tqdm.tqdm(config.BOOKS_DIR.glob("*.txt")):
#         filename = book.stem

#         text = get_book_text(book.stem, 10, 500)

#         with open(config.PARTS_DIR / f"{filename}.txt", "w") as f:
#             f.write(text)


if __name__ == "__main__":
    # print(get_embedding("2"))
    pass
