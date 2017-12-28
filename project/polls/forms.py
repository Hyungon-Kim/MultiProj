from django import forms

from .models import UploadFileModel,DownloadFileModel

class UploadFileForm(forms.ModelForm):
    class Meta:
        model = UploadFileModel
        fields = ('image',)

class DownloadFileForm(forms.ModelForm):
    class Meta1:
        model = DownloadFileModel
        fields = ('text',)
