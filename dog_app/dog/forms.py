from django import forms

class UploadFileForm(forms.Form):
    file = forms.ImageField(widget=forms.ClearableFileInput(attrs={'id': 'uploadInput', 'accept': 'image/*', 'onchange': 'previewImage()'}))
