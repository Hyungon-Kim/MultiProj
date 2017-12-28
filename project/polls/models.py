from django.db import models

# Create your models here.
class UploadFileModel(models.Model):
    image = models.ImageField(default='')

class DownloadFileModel(models.Model):
    text = models.TextField(default='')