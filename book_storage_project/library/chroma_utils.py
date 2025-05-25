from django.conf import settings
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_chroma_client():
    return chromadb.PersistentClient(path=str(settings.CHROMA_DB_PATH))


def get_book_quotes_collection():
    client = get_chroma_client()

    from chromadb.utils import embedding_functions

    chroma_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    collection = client.get_or_create_collection(
        name="book_quotes_multilingual",
        embedding_function=chroma_ef,
    )
    return collection


def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True,
    )
