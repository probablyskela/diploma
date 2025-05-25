import logging
from .chroma_utils import get_book_quotes_collection, get_text_splitter
from .models import Book

logger = logging.getLogger(__name__)


def index_book(book_instance: Book, all_tag_ids: list[str]):
    collection = get_book_quotes_collection()
    text_splitter = get_text_splitter()

    if not book_instance.extracted_text:
        logger.warning(
            f"Book UUID {book_instance.uuid} (Title: {book_instance.title}) has no extracted text. Skipping indexing."
        )
        return

    book_uuid_str = str(book_instance.uuid)

    try:
        collection.delete(where={"book_id": book_uuid_str})
        logger.info(
            f"Deleted existing chunks for book UUID {book_uuid_str} before re-indexing."
        )
    except Exception as e:
        logger.error(f"Error deleting chunks for book UUID {book_uuid_str}: {e}")

    chunks = text_splitter.split_text(book_instance.extracted_text)

    if not chunks:
        logger.info(
            f"No text chunks generated for book UUID {book_uuid_str}. Nothing to index."
        )
        return

    documents_to_add = []
    metadatas_to_add = []
    ids_to_add = []

    current_book_tag_ids = set(book_instance.tags.values_list("id", flat=True))

    for i, chunk_text in enumerate(chunks):
        chunk_id = f"{book_uuid_str}_chunk_{i}"

        metadata = {
            "book_id": book_uuid_str,
        }

        for tag_id in all_tag_ids:
            metadata[f"tag_{tag_id}"] = tag_id in current_book_tag_ids

        documents_to_add.append(chunk_text)
        metadatas_to_add.append(metadata)
        ids_to_add.append(chunk_id)

    if documents_to_add:
        try:
            collection.add(
                ids=ids_to_add, documents=documents_to_add, metadatas=metadatas_to_add
            )
            logger.info(
                f"Successfully indexed {len(documents_to_add)} chunks for book UUID {book_uuid_str} (Title: {book_instance.title})."
            )
        except Exception as e:
            logger.error(
                f"Error adding chunks to ChromaDB for book UUID {book_uuid_str}: {e}"
            )
    else:
        logger.info(
            f"No documents to add for book UUID {book_uuid_str} after processing."
        )
