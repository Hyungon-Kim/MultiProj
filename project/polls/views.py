from django.shortcuts import render
from django.http import HttpResponse ,HttpRequest, HttpResponseRedirect
from .forms import UploadFileForm ,DownloadFileForm
from polls import demo_keyframe
import os
from rest_framework.decorators import api_view
from rest_framework.response import *


def post_list(request):
   # HttpRequest.FILES
    #result = tensor.TensorFace()

    return render(request, 'polls/post_list.html',{})

# Create your views here.

def upload(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)

        if form.is_valid():
            form.save()

            name,accu = demo_keyframe.main(str(request.FILES['image']))
            print (accu)

            return render(request, 'polls/result.html', {'name' : name, 'accu' : accu })

    elif request.method =='GET':
        form = UploadFileForm()
    return render(request, 'polls/post_list.html', {'form': form})

