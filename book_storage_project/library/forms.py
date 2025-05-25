from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import Tag


class CustomUserCreationForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta(UserCreationForm.Meta):
        model = User
        fields = UserCreationForm.Meta.fields + ("email",)


class BookUploadForm(forms.Form):
    pass


class BookRenameForm(forms.Form):
    title = forms.CharField(max_length=255, label="New Book Title")


class BookQuoteSearchForm(forms.Form):
    query = forms.CharField(
        label="Search Query",
        required=True,
        widget=forms.TextInput(
            attrs={
                "placeholder": "Enter keywords, phrases, or questions...",
                "class": "form-control",
            }
        ),
    )
    tags = forms.MultipleChoiceField(
        widget=forms.CheckboxSelectMultiple, required=False, label="Filter by Tags"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["tags"].choices = [
            (tag.id, f"{tag.name} ({tag.id})")
            for tag in Tag.objects.all().order_by("id")
        ]
