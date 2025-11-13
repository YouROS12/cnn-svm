# 🛠️ Tools & Libraries for Plant Disease Detection

> Comprehensive guide to software tools, frameworks, and libraries

---

## 📋 Table of Contents

- [Deep Learning Frameworks](#deep-learning-frameworks)
- [Computer Vision Libraries](#computer-vision-libraries)
- [Model Hubs & Pre-trained Models](#model-hubs--pre-trained-models)
- [Data Augmentation](#data-augmentation)
- [Visualization & Interpretation](#visualization--interpretation)
- [Experiment Tracking](#experiment-tracking)
- [Development Tools](#development-tools)
- [Deployment](#deployment)
- [Cloud Platforms](#cloud-platforms)

---

## 🧠 Deep Learning Frameworks

### PyTorch (Recommended)

**Why PyTorch**:
- Pythonic, intuitive API
- Dynamic computation graphs
- Strong research community
- Excellent for prototyping

**Installation**:
```bash
# CPU
pip install torch torchvision torchaudio

# GPU (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# GPU (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Key Modules**:
- `torch.nn` - Neural network layers
- `torch.optim` - Optimizers (Adam, SGD, AdamW)
- `torch.utils.data` - Data loading utilities
- `torchvision` - Vision models and transforms

**Resources**:
- [Official Docs](https://pytorch.org/docs/)
- [Tutorials](https://pytorch.org/tutorials/)
- [GitHub](https://github.com/pytorch/pytorch)

---

### TensorFlow / Keras

**Why TensorFlow**:
- Production-ready
- Strong mobile/edge support (TFLite)
- Keras high-level API

**Installation**:
```bash
# CPU
pip install tensorflow

# GPU (requires CUDA, cuDNN)
pip install tensorflow[and-cuda]
```

**Key Modules**:
- `tf.keras` - High-level API
- `tf.data` - Efficient data pipelines
- `tf.saved_model` - Model serialization

**Resources**:
- [Official Docs](https://www.tensorflow.org/)
- [Keras Docs](https://keras.io/)
- [GitHub](https://github.com/tensorflow/tensorflow)

---

### JAX (Advanced)

**Why JAX**:
- Functional programming paradigm
- Automatic differentiation
- JIT compilation
- GPU/TPU acceleration

**Installation**:
```bash
pip install jax jaxlib
```

**Libraries**:
- **Flax**: Neural network library
- **Optax**: Gradient processing and optimization

**Resources**:
- [Official Docs](https://jax.readthedocs.io/)
- [GitHub](https://github.com/google/jax)

---

## 👁️ Computer Vision Libraries

### OpenCV

**Features**:
- Image I/O, preprocessing
- Classical computer vision algorithms
- Real-time processing

**Installation**:
```bash
pip install opencv-python opencv-contrib-python
```

**Common Uses**:
```python
import cv2

# Read image
img = cv2.imread('plant.jpg')

# Resize
img = cv2.resize(img, (224, 224))

# Color space conversion
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Save image
cv2.imwrite('output.jpg', img)
```

**Resources**:
- [Official Docs](https://docs.opencv.org/)
- [Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

---

### Pillow (PIL)

**Features**:
- Lightweight image processing
- Compatible with PyTorch transforms

**Installation**:
```bash
pip install Pillow
```

**Common Uses**:
```python
from PIL import Image

# Load image
img = Image.open('plant.jpg')

# Resize
img = img.resize((224, 224))

# Convert to RGB
img = img.convert('RGB')

# Save
img.save('output.jpg')
```

---

### Albumentations

**Features**:
- Fast image augmentation
- 70+ transformation types
- Compatible with PyTorch, TensorFlow

**Installation**:
```bash
pip install albumentations
```

**Example**:
```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.RandomResizedCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Apply
augmented = transform(image=image)
image = augmented['image']
```

**Resources**:
- [Official Docs](https://albumentations.ai/)
- [GitHub](https://github.com/albumentations-team/albumentations)

---

## 🏗️ Model Hubs & Pre-trained Models

### Timm (PyTorch Image Models)

**Features**:
- 700+ pre-trained models
- ResNet, EfficientNet, ViT, DeiT, Swin, etc.
- Easy fine-tuning

**Installation**:
```bash
pip install timm
```

**Usage**:
```python
import timm

# List available models
models = timm.list_models('*resnet*', pretrained=True)

# Create model
model = timm.create_model('resnet50', pretrained=True, num_classes=38)

# Get model info
print(model.default_cfg)

# Feature extraction
model = timm.create_model('resnet50', pretrained=True, num_classes=0)  # Remove classifier
features = model(x)  # [B, 2048]
```

**Resources**:
- [GitHub](https://github.com/huggingface/pytorch-image-models)
- [Documentation](https://huggingface.co/docs/timm)

---

### Hugging Face Transformers

**Features**:
- Vision Transformers (ViT, DeiT, Swin)
- Foundation models (CLIP, DINOv2, SAM)
- Easy integration with PyTorch/TensorFlow

**Installation**:
```bash
pip install transformers
```

**Usage**:
```python
from transformers import AutoModel, AutoFeatureExtractor

# Load DINOv2
model = AutoModel.from_pretrained('facebook/dinov2-base')
feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/dinov2-base')

# Load CLIP
from transformers import CLIPModel, CLIPProcessor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
```

**Resources**:
- [Official Docs](https://huggingface.co/docs/transformers)
- [Model Hub](https://huggingface.co/models)
- [GitHub](https://github.com/huggingface/transformers)

---

### TorchVision Models

**Features**:
- Classic CNN architectures
- Pre-trained on ImageNet

**Usage**:
```python
from torchvision import models

# ResNet
model = models.resnet50(pretrained=True)

# EfficientNet
model = models.efficientnet_b0(pretrained=True)

# Vision Transformer
model = models.vit_b_16(pretrained=True)

# Customize classifier
model.fc = torch.nn.Linear(model.fc.in_features, 38)
```

---

## 🎨 Data Augmentation

### Albumentations (see above)

### Torchvision Transforms

**Usage**:
```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

### Kornia

**Features**:
- Differentiable image augmentation
- GPU-accelerated
- Geometric and color transforms

**Installation**:
```bash
pip install kornia
```

**Usage**:
```python
import kornia as K

# Define augmentation
aug = K.augmentation.AugmentationSequential(
    K.augmentation.RandomHorizontalFlip(p=0.5),
    K.augmentation.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
    K.augmentation.RandomRotation(degrees=15, p=0.5)
)

# Apply (on GPU)
augmented = aug(images)  # images: [B, C, H, W]
```

**Resources**:
- [Official Docs](https://kornia.readthedocs.io/)
- [GitHub](https://github.com/kornia/kornia)

---

## 📊 Visualization & Interpretation

### Matplotlib

**Usage**:
```python
import matplotlib.pyplot as plt

# Plot image
plt.imshow(image)
plt.title('Plant Disease')
plt.axis('off')
plt.savefig('output.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot training curves
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_curve.png', dpi=300)
```

---

### Seaborn

**Usage**:
```python
import seaborn as sns

# Confusion matrix heatmap
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png', dpi=300)
```

---

### Grad-CAM / Grad-CAM++

**pytorch-grad-cam**:

**Installation**:
```bash
pip install grad-cam
```

**Usage**:
```python
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

# Select target layer
target_layers = [model.layer4[-1]]

# Create GradCAM object
cam = GradCAMPlusPlus(model=model, target_layers=target_layers)

# Generate CAM
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

# Visualize
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
```

**Resources**:
- [GitHub](https://github.com/jacobgil/pytorch-grad-cam)

---

### t-SNE / UMAP

**Scikit-learn (t-SNE)**:
```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='tab10')
plt.colorbar()
plt.savefig('tsne.png', dpi=300)
```

**UMAP**:
```bash
pip install umap-learn
```

```python
import umap

reducer = umap.UMAP(n_components=2, random_state=42)
embeddings_2d = reducer.fit_transform(embeddings)
```

---

## 📈 Experiment Tracking

### Weights & Biases (Recommended)

**Features**:
- Experiment tracking
- Hyperparameter tuning
- Model versioning
- Team collaboration

**Installation**:
```bash
pip install wandb
```

**Usage**:
```python
import wandb

# Initialize
wandb.init(project='plant-disease', name='resnet50-exp1')

# Log hyperparameters
wandb.config.update({
    'learning_rate': 1e-4,
    'batch_size': 64,
    'epochs': 100
})

# Log metrics
for epoch in range(epochs):
    train_loss, val_loss, val_acc = train_epoch(...)
    wandb.log({
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'epoch': epoch
    })

# Log images
wandb.log({'confusion_matrix': wandb.Image(plt)})

# Finish
wandb.finish()
```

**Resources**:
- [Official Docs](https://docs.wandb.ai/)
- [Free for academics](https://wandb.ai/site/research)

---

### TensorBoard

**Features**:
- Built-in TensorFlow/PyTorch integration
- Visualize training curves, histograms

**Usage (PyTorch)**:
```python
from torch.utils.tensorboard import SummaryWriter

# Create writer
writer = SummaryWriter('runs/experiment1')

# Log scalars
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Loss/val', val_loss, epoch)
writer.add_scalar('Accuracy/val', val_acc, epoch)

# Log images
writer.add_image('Input', image, epoch)

# Log graph
writer.add_graph(model, input_tensor)

# Close
writer.close()
```

**View**:
```bash
tensorboard --logdir=runs
```

---

### MLflow

**Features**:
- Experiment tracking
- Model registry
- Model deployment

**Installation**:
```bash
pip install mlflow
```

**Resources**:
- [Official Docs](https://mlflow.org/)

---

## 🖥️ Development Tools

### Jupyter Notebook / JupyterLab

**Installation**:
```bash
pip install jupyter jupyterlab
```

**Launch**:
```bash
jupyter lab
```

**Extensions**:
- **jupyterlab-lsp**: Code completion, linting
- **jupytext**: Notebook version control

---

### Google Colab

**Features**:
- Free GPU/TPU access
- No setup required
- Easy sharing

**Pro Tips**:
```python
# Check GPU
!nvidia-smi

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install packages
!pip install timm albumentations wandb

# Download datasets
!gdown <google-drive-file-id>
```

**Resources**:
- [Colab](https://colab.research.google.com/)

---

### VS Code

**Extensions for ML/DL**:
- **Python**: IntelliSense, linting
- **Pylance**: Fast language server
- **Jupyter**: Notebook support
- **Remote - SSH**: SSH development
- **GitHub Copilot**: AI code completion

**Resources**:
- [VS Code](https://code.visualstudio.com/)

---

## 🚀 Deployment

### ONNX (Open Neural Network Exchange)

**Convert PyTorch to ONNX**:
```python
import torch.onnx

# Export
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output']
)
```

**Inference with ONNX Runtime**:
```bash
pip install onnxruntime
```

```python
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
outputs = session.run(None, {'input': input_array})
```

---

### TorchScript

**JIT Compilation**:
```python
# Script mode
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")

# Trace mode
traced_model = torch.jit.trace(model, example_input)
traced_model.save("model_traced.pt")

# Load
loaded_model = torch.jit.load("model_scripted.pt")
```

---

### TensorFlow Lite

**Convert to TFLite**:
```python
import tensorflow as tf

# Convert
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_dir')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

---

### FastAPI (Model Serving)

**Installation**:
```bash
pip install fastapi uvicorn python-multipart
```

**Example**:
```python
from fastapi import FastAPI, File, UploadFile
import torch
from PIL import Image
import io

app = FastAPI()

# Load model
model = torch.load('model.pth')
model.eval()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read image
    image = Image.open(io.BytesIO(await file.read()))

    # Preprocess
    image_tensor = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1).item()

    return {"prediction": class_names[pred]}

# Run: uvicorn main:app --reload
```

---

### Gradio (Interactive Demos)

**Installation**:
```bash
pip install gradio
```

**Example**:
```python
import gradio as gr

def classify_image(image):
    # Preprocess and predict
    prediction = model(transform(image))
    return class_names[prediction]

# Create interface
interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Plant Disease Classifier"
)

# Launch
interface.launch()
```

**Resources**:
- [Gradio](https://gradio.app/)

---

## ☁️ Cloud Platforms

### Google Cloud Platform (GCP)

**Services**:
- **Vertex AI**: Managed ML platform
- **Compute Engine**: VMs with GPUs
- **Cloud Storage**: Dataset storage

**GPU Options**:
- NVIDIA T4, V100, A100
- Free tier: $300 credits

**Resources**:
- [GCP Pricing](https://cloud.google.com/pricing)

---

### Amazon Web Services (AWS)

**Services**:
- **SageMaker**: Managed ML platform
- **EC2**: VMs with GPUs
- **S3**: Dataset storage

**GPU Instances**:
- p3 (V100), p4 (A100), g5 (A10G)

**Resources**:
- [AWS Pricing](https://aws.amazon.com/pricing/)

---

### Microsoft Azure

**Services**:
- **Azure Machine Learning**: Managed platform
- **Virtual Machines**: GPUs
- **Blob Storage**: Data storage

**Resources**:
- [Azure Pricing](https://azure.microsoft.com/en-us/pricing/)

---

### Paperspace Gradient

**Features**:
- Jupyter notebooks with GPU
- Pay-as-you-go pricing
- Easy setup

**Resources**:
- [Paperspace](https://www.paperspace.com/)

---

### Lambda Labs

**Features**:
- Dedicated GPU instances
- Cost-effective (A100 $1.10/hr)
- Pre-configured ML environments

**Resources**:
- [Lambda Labs](https://lambdalabs.com/)

---

## 📦 Additional Useful Libraries

### Data Manipulation

```bash
# NumPy - numerical computing
pip install numpy

# Pandas - data manipulation
pip install pandas

# Scikit-learn - ML utilities, metrics
pip install scikit-learn
```

### Metrics & Evaluation

```python
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score
)
```

### Progress Bars

```bash
pip install tqdm
```

```python
from tqdm import tqdm

for epoch in tqdm(range(epochs), desc="Training"):
    for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
        # training code
```

---

## 🎯 Recommended Tech Stack

### For Beginners
- **Framework**: PyTorch
- **Models**: Timm (pre-trained models)
- **Data**: Albumentations (augmentation)
- **Tracking**: Weights & Biases
- **IDE**: VS Code or Google Colab

### For Research
- **Framework**: PyTorch
- **Models**: Timm + Hugging Face Transformers
- **Visualization**: Grad-CAM, t-SNE
- **Tracking**: Weights & Biases
- **Deployment**: ONNX or TorchScript

### For Production
- **Framework**: PyTorch or TensorFlow
- **Serving**: FastAPI or TF Serving
- **Deployment**: ONNX Runtime or TFLite
- **Monitoring**: Prometheus + Grafana
- **Cloud**: AWS SageMaker or GCP Vertex AI

---

## 📚 Installation Script

**Create a comprehensive environment**:

```bash
# Create conda environment
conda create -n plant-disease python=3.10
conda activate plant-disease

# Install PyTorch (GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install essential libraries
pip install timm transformers albumentations opencv-python Pillow
pip install matplotlib seaborn scikit-learn pandas numpy
pip install tqdm wandb grad-cam
pip install jupyter jupyterlab

# Install deployment tools (optional)
pip install fastapi uvicorn gradio onnxruntime

# Install for development
pip install black flake8 pylint pytest
```

---

**This toolkit provides everything you need for plant disease detection research and deployment!** 🛠️🚀🌱
