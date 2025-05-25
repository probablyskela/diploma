from django.urls import path
from .views import (
    RegisterView,
    book_list,
    upload_book,
    rename_book,
    delete_book,
    download_book,
    book_quote_search_view,
    book_detail_view,
)

app_name = "library"

urlpatterns = [
    path("register/", RegisterView.as_view(), name="register"),
    path("books/", book_list, name="book_list"),
    path("books/upload/", upload_book, name="upload_book"),
    path("books/search/", book_quote_search_view, name="book_quote_search"),
    path("books/<uuid:book_uuid>/", book_detail_view, name="book_detail"),
    path("books/<uuid:book_uuid>/rename/", rename_book, name="rename_book"),
    path("books/<uuid:book_uuid>/delete/", delete_book, name="delete_book"),
    path("books/<uuid:book_uuid>/download/", download_book, name="download_book"),
]
