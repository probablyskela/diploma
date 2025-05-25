import fitz
import ebooklib
from ebooklib import epub
import os


def extract_text_from_file(file_path_on_server, file_extension_with_dot):
    extracted_text = ""
    file_extension = file_extension_with_dot.lower()

    try:
        if file_extension == ".pdf":
            doc = fitz.open(file_path_on_server)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                extracted_text += page.get_text() or ""
            doc.close()
        elif file_extension == ".epub":
            book = epub.read_epub(file_path_on_server)
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    try:
                        content_str = item.content.decode("utf-8", errors="ignore")
                    except UnicodeDecodeError:
                        content_str = item.content.decode("latin-1", errors="ignore")

                    import re

                    content_str = re.sub("<[^<]+?>", "", content_str)
                    extracted_text += content_str + "\n"
        else:
            print(f"Unsupported file type for text extraction: {file_extension}")
            return None

        return extracted_text.strip() if extracted_text else None
    except Exception as e:
        print(f"Error extracting text from {file_path_on_server}: {e}")
        return None
