from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse_lazy, reverse
from django.views.generic.edit import CreateView
from django.contrib.auth.views import LoginView
from .forms import (
    CustomUserCreationForm,
    BookUploadForm,
    BookRenameForm,
    BookQuoteSearchForm,
)
from .models import Book, Tag
from django.contrib.auth.decorators import login_required
from django.http import FileResponse, Http404
from django.conf import settings
import os
import uuid
from django.contrib import messages
from django.core.validators import FileExtensionValidator
from django.core.exceptions import ValidationError, PermissionDenied
from .utils import extract_text_from_file
from .ml_models.tagger import get_tags_for_texts
from .indexing import index_book
from .chroma_utils import get_book_quotes_collection
import mimetypes
import re
import logging

logger = logging.getLogger(__name__)


class CustomLoginView(LoginView):
    def dispatch(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            return redirect(reverse_lazy("library:book_list"))
        return super().dispatch(request, *args, **kwargs)


class RegisterView(CreateView):
    form_class = CustomUserCreationForm
    success_url = reverse_lazy("login")
    template_name = "registration/register.html"


@login_required
def book_list(request):
    user_books = Book.objects.filter(user=request.user)
    all_tags = Tag.objects.all().order_by("id")
    selected_tag_ids = request.GET.getlist("tags")

    if selected_tag_ids:
        for tag_id in selected_tag_ids:
            user_books = user_books.filter(tags__id=tag_id)

    user_books = user_books.order_by("-uploaded_at")

    return render(
        request,
        "library/book_list.html",
        {
            "books": user_books,
            "all_tags": all_tags,
            "selected_tag_ids": selected_tag_ids,
        },
    )


@login_required
def upload_book(request):
    if request.method == "POST":
        form = BookUploadForm(request.POST)
        uploaded_files = request.FILES.getlist("book_files")

        if form.is_valid():
            successful_uploads = []
            failed_uploads = []
            validate_extension = FileExtensionValidator(
                allowed_extensions=["pdf", "epub"]
            )
            all_tag_ids_for_indexing = list(Tag.objects.values_list("id", flat=True))

            if not uploaded_files:
                messages.error(request, "No files were selected for upload.")
                return render(request, "library/book_upload.html", {"form": form})

            for uploaded_file in uploaded_files:
                original_filename = uploaded_file.name
                file_base, file_extension_with_dot = os.path.splitext(original_filename)
                book_instance = None

                try:
                    validate_extension(uploaded_file)
                except ValidationError:
                    failed_uploads.append(
                        f"{original_filename} (Unsupported file type. Only PDF and EPUB allowed.)"
                    )
                    continue

                book_title = file_base
                unique_filename = f"{uuid.uuid4()}{file_extension_with_dot.lower()}"

                user_media_dir = os.path.join(
                    settings.MEDIA_ROOT, f"user_{request.user.id}"
                )
                if not os.path.exists(user_media_dir):
                    os.makedirs(user_media_dir)

                file_path_on_server = os.path.join(user_media_dir, unique_filename)
                relative_file_path = os.path.join(
                    f"user_{request.user.id}", unique_filename
                )
                extracted_book_text = None
                predicted_tag_ids = ["O"]

                try:
                    with open(file_path_on_server, "wb+") as destination:
                        for chunk in uploaded_file.chunks():
                            destination.write(chunk)

                    extracted_book_text = extract_text_from_file(
                        file_path_on_server, file_extension_with_dot.lower()
                    )
                    current_tags_for_book = ["O"]
                    if extracted_book_text:
                        tag_id_lists = get_tags_for_texts([extracted_book_text])
                        if tag_id_lists and tag_id_lists[0]:
                            current_tags_for_book = tag_id_lists[0]
                        else:
                            messages.info(
                                request,
                                f"No specific tags identified for '{original_filename}'. Marked as 'Other'.",
                            )
                    else:
                        messages.warning(
                            request,
                            f"Could not extract text from '{original_filename}'. Tagging skipped, marked as 'Other'.",
                        )

                    predicted_tag_ids = current_tags_for_book

                    book_instance = Book.objects.create(
                        user=request.user,
                        title=book_title,
                        original_filename=original_filename,
                        file_path=relative_file_path,
                        extracted_text=extracted_book_text,
                    )

                    tag_objects = []
                    if predicted_tag_ids:
                        for tag_id in predicted_tag_ids:
                            try:
                                tag = Tag.objects.get(id=tag_id)
                                tag_objects.append(tag)
                            except Tag.DoesNotExist:
                                messages.warning(
                                    request,
                                    f"Tag ID '{tag_id}' predicted for '{book_title}' but not found. Skipping.",
                                )
                        if not tag_objects:
                            other_tag, _ = Tag.objects.get_or_create(
                                id="O",
                                defaults={
                                    "name": "Other",
                                    "description": "Fallback tag",
                                },
                            )
                            tag_objects.append(other_tag)
                    else:
                        other_tag, _ = Tag.objects.get_or_create(
                            id="O",
                            defaults={"name": "Other", "description": "Fallback tag"},
                        )
                        tag_objects.append(other_tag)

                    book_instance.tags.set(tag_objects)

                    try:
                        logger.info(
                            f"Attempting to index book UUID {book_instance.uuid}"
                        )
                        index_book(book_instance, all_tag_ids_for_indexing)
                    except Exception as e:
                        logger.error(
                            f"Error during indexing of book UUID {book_instance.uuid} (Title: {book_instance.title}): {e}"
                        )
                        messages.warning(
                            request,
                            f"Book '{book_instance.title}' was uploaded and saved, but there was an issue adding it to the search index. Error: {e}",
                        )

                    successful_uploads.append(
                        f"'{book_title}' (was {original_filename})"
                    )

                except Exception as e:
                    failed_uploads.append(
                        f"{original_filename} (DB save or other critical error: {e})"
                    )
                    if os.path.exists(file_path_on_server):
                        try:
                            os.remove(file_path_on_server)
                        except OSError:
                            pass
                    if book_instance and book_instance.pk:
                        try:
                            book_instance.delete()
                        except Exception:
                            pass

            if successful_uploads:
                messages.success(
                    request,
                    f"Successfully created books: {', '.join(successful_uploads)}.",
                )
            if failed_uploads:
                messages.error(
                    request,
                    f"Failed to create some books: {', '.join(failed_uploads)}.",
                )

            if not successful_uploads and failed_uploads:
                return render(request, "library/book_upload.html", {"form": form})

            return redirect(reverse("library:book_list"))
        else:
            messages.error(
                request,
                "There was an error with the form submission. Please check for any errors.",
            )
            pass

    form = BookUploadForm()
    return render(request, "library/book_upload.html", {"form": form})


def home_redirect_view(request):
    if request.user.is_authenticated:
        return redirect(reverse_lazy("library:book_list"))
    else:
        return redirect(reverse_lazy("login"))


@login_required
def rename_book(request, book_uuid):
    book = get_object_or_404(Book, uuid=book_uuid)
    if book.user != request.user:
        raise PermissionDenied

    if request.method == "POST":
        form = BookRenameForm(request.POST)
        if form.is_valid():
            book.title = form.cleaned_data["title"]
            book.save()
            messages.success(request, f"Book '{book.title}' renamed successfully.")
            return redirect(reverse("library:book_list"))
    else:
        form = BookRenameForm(initial={"title": book.title})

    return render(
        request, "library/book_rename_form.html", {"form": form, "book": book}
    )


@login_required
def delete_book(request, book_uuid):
    book = get_object_or_404(Book, uuid=book_uuid)
    if book.user != request.user:
        raise PermissionDenied

    if request.method == "POST":
        book_title = book.title
        file_path_on_server = os.path.join(settings.MEDIA_ROOT, book.file_path)

        try:
            book.delete()
            if os.path.exists(file_path_on_server):
                try:
                    os.remove(file_path_on_server)
                    messages.success(
                        request, f"Book '{book_title}' and its file have been deleted."
                    )
                except OSError as e:
                    messages.warning(
                        request,
                        f"Book '{book_title}' was deleted from database, but its file could not be removed: {e}",
                    )
            else:
                messages.success(
                    request, f"Book '{book_title}' deleted (file was already missing)."
                )
        except Exception as e:
            messages.error(
                request, f"An error occurred while trying to delete '{book_title}': {e}"
            )

        return redirect(reverse("library:book_list"))

    return render(request, "library/book_confirm_delete.html", {"book": book})


@login_required
def download_book(request, book_uuid):
    book = get_object_or_404(Book, uuid=book_uuid)
    if book.user != request.user:
        raise PermissionDenied

    file_path_on_server = os.path.join(settings.MEDIA_ROOT, book.file_path)

    if os.path.exists(file_path_on_server):
        content_type, encoding = mimetypes.guess_type(book.original_filename)
        if content_type is None:
            content_type = "application/octet-stream"

        _, file_extension = os.path.splitext(book.original_filename)
        safe_title = re.sub(r"[^\w\s-]", "", book.title).strip().replace(" ", "_")
        if not safe_title:
            safe_title = "downloaded_book"
        download_filename = f"{safe_title}{file_extension}"

        try:
            response = FileResponse(
                open(file_path_on_server, "rb"),
                as_attachment=True,
                filename=download_filename,
            )
            return response
        except FileNotFoundError:
            messages.error(
                request,
                f"Error: The file for '{book.title}' was not found on the server unexpectedly.",
            )
            raise Http404("Book file not found on server path.")
        except Exception as e:
            messages.error(
                request,
                f"An error occurred while trying to prepare '{book.title}' for download: {e}",
            )
            return redirect(reverse("library:book_list"))
    else:
        messages.error(
            request, f"Error: The file for '{book.title}' does not exist on the server."
        )
        raise Http404("Book file does not exist on server path.")


@login_required
def book_quote_search_view(request):
    form = BookQuoteSearchForm(request.POST or None)
    results_for_template = []
    search_performed = False

    user_book_uuids = list(
        Book.objects.filter(user=request.user).values_list("uuid", flat=True)
    )
    user_book_uuids_str = [str(u) for u in user_book_uuids]

    if request.method == "POST":
        if form.is_valid():
            search_performed = True
            search_query_text = form.cleaned_data["query"]
            selected_tag_ids = form.cleaned_data["tags"]

            if not user_book_uuids_str:
                messages.info(request, "You have no books to search within.")
            else:
                collection = get_book_quotes_collection()

                query_conditions = [{"book_id": {"$in": user_book_uuids_str}}]

                if selected_tag_ids:
                    for tag_id in selected_tag_ids:
                        query_conditions.append({f"tag_{tag_id}": True})

                metadata_filter = (
                    {"$and": query_conditions}
                    if len(query_conditions) > 1
                    else query_conditions[0]
                )

                try:
                    chroma_results = collection.query(
                        query_texts=[search_query_text],
                        n_results=10,
                        where=metadata_filter,
                        include=["metadatas", "documents"],
                    )

                    if (
                        chroma_results
                        and chroma_results.get("ids")
                        and chroma_results["ids"][0]
                    ):
                        book_uuids_from_chroma = set()
                        temp_results = []

                        ids_list = chroma_results.get("ids", [[]])[0]
                        metadatas_list = chroma_results.get("metadatas", [[]])[0]
                        documents_list = chroma_results.get("documents", [[]])[0]
                        num_results_returned = len(ids_list)

                        if not (
                            len(metadatas_list) == num_results_returned
                            and len(documents_list) == num_results_returned
                        ):
                            logger.error(
                                f"ChromaDB returned inconsistent list lengths for query results. Query: {search_query_text}"
                            )
                            pass

                        for i in range(num_results_returned):
                            if i >= len(metadatas_list) or i >= len(documents_list):
                                logger.warning(
                                    f"Skipping result index {i} due to inconsistent list lengths from ChromaDB."
                                )
                                continue

                            meta = metadatas_list[i]
                            doc = documents_list[i]
                            book_uuid = meta.get("book_id")

                            if book_uuid and book_uuid in user_book_uuids_str:
                                book_uuids_from_chroma.add(book_uuid)
                                temp_results.append(
                                    {
                                        "text": doc,
                                        "book_uuid": book_uuid,
                                    }
                                )

                        if book_uuids_from_chroma:
                            books_with_details = Book.objects.filter(
                                user=request.user, uuid__in=list(book_uuids_from_chroma)
                            ).prefetch_related("tags")

                            book_details_map = {}
                            for book in books_with_details:
                                book_details_map[str(book.uuid)] = {
                                    "title": book.title,
                                    "tags": [tag.name for tag in book.tags.all()],
                                }

                            for res in temp_results:
                                book_detail = book_details_map.get(res["book_uuid"])
                                if book_detail:
                                    res["book_title"] = book_detail["title"]
                                    res["book_tags"] = book_detail["tags"]
                                else:
                                    res["book_title"] = "Unknown Title"
                                    res["book_tags"] = []
                                results_for_template.append(res)

                except Exception as e:
                    logger.error(
                        f"Error during ChromaDB query or processing for user {request.user.id}: {e}",
                        exc_info=True,
                    )
                    messages.error(request, f"An error occurred while searching: {e}")
        else:
            search_performed = True

    context = {
        "form": form,
        "results": results_for_template,
        "search_performed": search_performed,
    }
    return render(request, "library/book_quote_search.html", context)


@login_required
def book_detail_view(request, book_uuid):
    """Displays details for a specific book, owned by the current user."""
    book = get_object_or_404(Book, uuid=book_uuid)

    if book.user != request.user:
        raise PermissionDenied("You do not have permission to view this book.")

    context = {"book": book}
    return render(request, "library/book_detail.html", context)
