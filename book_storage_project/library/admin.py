from django.contrib import admin
from .models import Book, Tag

# Register your models here.
admin.site.register(Book)
admin.site.register(Tag)
