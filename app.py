import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from torch import device
import torchvision.models as models

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model (with no pretrained weights initially)
model = models.efficientnet_b0(weights=None)  # Corrected way to load EfficientNet without pre-trained weights
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)  # 2 classes: pothole and normal

# Load the model weights onto CPU (or GPU if available)
model.load_state_dict(torch.load('model/best_model.pth', map_location=device))

# Move model to the appropriate device (CPU or GPU)
model = model.to(device)

# Set model to evaluation mode
model.eval()


# Set up the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Prediction function
def predict_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
    
    return 'Pothole' if preds.item() == 1 else 'Normal'

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        prediction = predict_image(filepath)

        return render_template('index.html', filename=filename, prediction=prediction)

    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
