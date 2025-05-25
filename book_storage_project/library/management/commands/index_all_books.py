from django.core.management.base import BaseCommand, CommandError
from library.models import Book, Tag
from library.indexing import index_book
import logging

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Indexes books into ChromaDB for semantic search. Can re-index all books or a specific book."

    def add_arguments(self, parser):
        parser.add_argument(
            "--reindex_all",
            action="store_true",
            help="Deletes existing index data for all books and re-indexes them.",
        )
        parser.add_argument(
            "--book_uuid", type=str, help="UUID of a specific book to index/re-index."
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("Starting book indexing process..."))

        all_tag_ids = list(Tag.objects.values_list("id", flat=True))
        if not all_tag_ids:
            self.stdout.write(
                self.style.WARNING(
                    "No tags found in the database. Tag-based metadata will be empty."
                )
            )

        books_to_index = []

        if options["book_uuid"]:
            book_uuid = options["book_uuid"]
            try:
                book = Book.objects.get(uuid=book_uuid)
                books_to_index.append(book)
                self.stdout.write(
                    self.style.SUCCESS(f"Found book with UUID: {book_uuid}")
                )
            except Book.DoesNotExist:
                raise CommandError(f'Book with UUID "{book_uuid}" does not exist.')
        elif options["reindex_all"]:
            books_to_index = list(Book.objects.all())
            if not books_to_index:
                self.stdout.write(
                    self.style.WARNING("No books found in the database to index.")
                )
                return
            self.stdout.write(
                self.style.SUCCESS(f"Found {len(books_to_index)} books to re-index.")
            )
        else:
            self.stdout.write(
                self.style.NOTICE(
                    "No specific indexing option chosen. Use --reindex_all or --book_uuid <uuid>."
                )
            )
            self.stdout.write(
                self.style.NOTICE(
                    "Defaulting to re-indexing all books for safety in MVP."
                )
            )
            books_to_index = list(Book.objects.all())
            if not books_to_index:
                self.stdout.write(
                    self.style.WARNING("No books found in the database to index.")
                )
                return
            self.stdout.write(
                self.style.SUCCESS(
                    f"Defaulting: Found {len(books_to_index)} books to re-index."
                )
            )

        indexed_count = 0
        failed_count = 0

        for book in books_to_index:
            self.stdout.write(f"Indexing book: {book.title} (UUID: {book.uuid})...")
            try:
                index_book(book, all_tag_ids)
                self.stdout.write(
                    self.style.SUCCESS(f"  Successfully indexed: {book.title}")
                )
                indexed_count += 1
            except Exception as e:
                logger.error(
                    f"Failed to index book {book.title} (UUID: {book.uuid}): {e}",
                    exc_info=True,
                )
                self.stderr.write(
                    self.style.ERROR(f"  Failed to index: {book.title}. Error: {e}")
                )
                failed_count += 1

        self.stdout.write(self.style.SUCCESS("\nIndexing process finished."))
        self.stdout.write(
            self.style.SUCCESS(f"Successfully indexed: {indexed_count} books.")
        )
        if failed_count > 0:
            self.stdout.write(
                self.style.ERROR(
                    f"Failed to index: {failed_count} books. Check logs for details."
                )
            )
        else:
            self.stdout.write(
                self.style.SUCCESS(
                    "All selected books processed without critical errors during indexing."
                )
            )
