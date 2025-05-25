import uuid
from django.db import models
from django.contrib.auth.models import User


class Tag(models.Model):
    id = models.CharField(primary_key=True, max_length=10)
    name = models.CharField(max_length=255)
    parent_tag = models.ForeignKey(
        "self", on_delete=models.SET_NULL, null=True, blank=True
    )
    description = models.TextField(blank=True, null=True)

    def __str__(self):
        return self.name


class Book(models.Model):
    uuid = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=255)
    original_filename = models.CharField(max_length=255)
    file_path = models.CharField(max_length=1024)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    tags = models.ManyToManyField(Tag, blank=True)
    extracted_text = models.TextField(null=True, blank=True)

    def __str__(self):
        return self.title
