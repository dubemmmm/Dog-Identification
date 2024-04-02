function previewImage() {
    const input = document.getElementById('uploadInput');
    const preview = document.getElementById('imagePreview');
    
    while (preview.firstChild) {
      preview.removeChild(preview.firstChild);
    }
  
    const files = input.files;
  
    if (files.length > 0) {
      const img = document.createElement('img');
      img.src = URL.createObjectURL(files[0]);
      preview.appendChild(img);
    }
  
    preview.classList.remove('hidden');
  }
  