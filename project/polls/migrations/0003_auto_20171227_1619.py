# -*- coding: utf-8 -*-
# Generated by Django 1.11.8 on 2017-12-27 07:19
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('polls', '0002_uploadfilemodel_text'),
    ]

    operations = [
        migrations.CreateModel(
            name='DownloadFileModel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('text', models.TextField(default='')),
            ],
        ),
        migrations.RemoveField(
            model_name='uploadfilemodel',
            name='text',
        ),
    ]