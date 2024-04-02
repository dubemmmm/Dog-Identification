import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from .imagenet1000 import class_mapping, class_name
import os
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 28 * 28, 500)
        self.fc2 = nn.Linear(500, 133)
        self.dropout = nn.Dropout(p=0.2)
        self.batch_norm = nn.BatchNorm1d(num_features=500)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.batch_norm(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    



# Define the image preprocessing transform
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the trained model 
model2 = models.vgg16(weights='VGG16_Weights.DEFAULT')
for param in model2.features.parameters():
    param.required_grad = False
n_inputs = model2.classifier[6].in_features
last_layer = nn.Linear(n_inputs, 133)
model2.classifier[6] = last_layer
model_path2 = '/Users/chidubemonwuchuluba/Desktop/djangostuff/dog_app/model_transfer_vgg.pth'
model2.load_state_dict(torch.load(model_path2))
model2.eval()


# Function to predict the dog breed
def predict_breed(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = test_transform(image)
    input_batch = input_tensor.unsqueeze(0)
    # Make predictions
    with torch.no_grad():
        output = model2(input_batch)
    predicted_class_index = torch.argmax(output).item()
    predicted_class_name = class_name[predicted_class_index]

    return predicted_class_name