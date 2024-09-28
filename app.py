import os
from flask import Flask, render_template, request, send_file
from PIL import Image
import torch.nn as nn
import torch
import torchvision.transforms as transforms

# Define the Flask app
app = Flask(__name__)

# Directories for uploading and saving results
UPLOAD_FOLDER = 'uploads/'
RESULT_FOLDER = 'results/'

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Define the simple super-resolution model
class SimpleSRModel(nn.Module):
    def __init__(self):
        super(SimpleSRModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 3, kernel_size=9, padding=4)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x

# Load the pre-trained model
device = torch.device('cpu')  # Change to 'cuda' if using a GPU
model = SimpleSRModel()
model.load_state_dict(torch.load("sr_model_final.pth", map_location=device), strict=False)
model = model.to(device)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# Function to perform super-resolution
def super_resolve(image):
    low_res_image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        high_res_image = model(low_res_image)
    high_res_image = high_res_image.squeeze(0).cpu().clamp(0, 1)  # Remove batch dimension and clamp to [0, 1]
    return transforms.ToPILImage()(high_res_image)  # Convert tensor to PIL image

@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image colorization
@app.route('/colorize', methods=['POST'])
def colorize():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Colorize the image (assumed to be defined in colorize.py)
        output_path = os.path.join(RESULT_FOLDER, 'colorized_' + file.filename)
        colorized_image_path = colorized_image(file_path, output_path)
        
        return send_file(colorized_image_path, mimetype='image/jpeg')

# Route to handle super-resolution
@app.route('/super_resolve', methods=['POST'])
def super_resolve_image():
    if 'file' not in request.files:
        return "No file part", 400  # Return a bad request error if 'file' is missing
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400  # Return error if no file is selected
    
    # Proceed with processing the image
    image = Image.open(file).convert("RGB")
    super_resolved_image = super_resolve(image)
    
    # Save the super-resolved image to a BytesIO object for download
    output_path = os.path.join(RESULT_FOLDER, "super_resolved_image.png")
    super_resolved_image.save(output_path)

    return send_file(output_path, mimetype='image/png', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)