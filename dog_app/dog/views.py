from django.shortcuts import render
from django.http import HttpResponseRedirect
from .forms import UploadFileForm
from .predict import predict_breed  # Implement this function to use your model

def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            image_path = handle_uploaded_file(request.FILES['file'])
            predicted_breed = predict_breed(image_path)
            predicted_breed = predicted_breed[4:]
            return render(request, 'dog/result.html', {'breed': predicted_breed})
    else:
        form = UploadFileForm()
    return render(request, 'dog/upload.html', {'form': form})

def handle_uploaded_file(file):
    with open('uploaded_image.jpg', 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)
    return 'uploaded_image.jpg'
