import torch
from PIL import Image
import torchvision.transforms as transforms
from model import CNN  # make sure class name matches model.py

# Load model
model = CNN()
model.load_state_dict(torch.load("digit_cnn.pth"))
model.eval()

# Image preprocessing (VERY IMPORTANT)
transform = transforms.Compose([
    transforms.Grayscale(),          # convert to grayscale
    transforms.Resize((28, 28)),     # resize to MNIST size
    transforms.ToTensor(),           # convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # normalize
])

# Load image
img = Image.open("test.png")

# Apply transform
img = transform(img).unsqueeze(0)  # add batch dimension

# Predict
with torch.no_grad():
    output = model(img)
    _, predicted = torch.max(output, 1)

print("Predicted digit:", predicted.item())
