# 🚀 Google Colab Quick Setup Guide

This guide will help you run the Contrastive SVM experiment on Google Colab in **under 5 minutes**.

## Step 1: Open Notebook in Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File → Upload Notebook**
3. Upload `contrastive_svm_plant_disease.ipynb`

OR

1. Click this badge (add to README):
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/cnn-svm/blob/main/contrastive_svm_plant_disease.ipynb)

## Step 2: Enable GPU

1. Click **Runtime → Change runtime type**
2. Select **GPU** under Hardware accelerator
3. Choose **T4** or **V100** (if available)
4. Click **Save**

## Step 3: Upload Your Dataset

### Option A: From Google Drive

```python
# Add this cell at the beginning of the notebook
from google.colab import drive
drive.mount('/content/drive')

# Update config to point to your dataset
config.DATA_ROOT = '/content/drive/MyDrive/plantwildV2'
```

### Option B: Upload Directly

```python
# Create upload cell
from google.colab import files
import zipfile

# Upload zip file
uploaded = files.upload()

# Extract
zip_filename = list(uploaded.keys())[0]
with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
    zip_ref.extractall('./plantwildV2')

# Update config
config.DATA_ROOT = './plantwildV2'
```

### Option C: Download from URL

```python
# Add this cell
!wget https://your-dataset-url.com/plantwildv2.zip
!unzip plantwildv2.zip -d ./plantwildV2
!rm plantwildv2.zip

# Update config
config.DATA_ROOT = './plantwildV2'
```

## Step 4: Install Dependencies

The notebook automatically installs all dependencies in the first cell. Just run:

```python
# This cell is already in the notebook
!pip install -q torch torchvision torchaudio
!pip install -q scikit-learn matplotlib seaborn
!pip install -q tensorboard pillow tqdm
```

## Step 5: Run Experiment

Simply run all cells:
1. Click **Runtime → Run all**
2. Wait for training to complete (~2-3 hours on T4 GPU)
3. Results will be saved in `./results/`

## Step 6: Download Results

```python
# Add this cell at the end
from google.colab import files
import shutil

# Create zip of results
shutil.make_archive('results', 'zip', './results')
files.download('results.zip')

# Download trained models
shutil.make_archive('checkpoints', 'zip', './checkpoints')
files.download('checkpoints.zip')
```

## ⚙️ Configuration for Colab

### For Faster Experiments (Testing)

```python
class Config:
    # Reduce epochs for quick testing
    CONTRASTIVE_EPOCHS = 50  # instead of 200
    FINETUNE_EPOCHS = 20     # instead of 50

    # Smaller batch size if running out of memory
    BATCH_SIZE = 64          # instead of 128

    # Skip few-shot for quick results
    ENABLE_FEW_SHOT = False
```

### For Full Experiments (Publication)

```python
class Config:
    # Keep default values
    CONTRASTIVE_EPOCHS = 200
    FINETUNE_EPOCHS = 50
    BATCH_SIZE = 128
    ENABLE_FEW_SHOT = True
```

## 📊 Expected Runtime

| Configuration | GPU | Time |
|--------------|-----|------|
| Quick Test (50 epochs) | T4 | ~45 min |
| Full Run (200 epochs) | T4 | ~2-3 hours |
| Full Run (200 epochs) | V100 | ~1-1.5 hours |
| Full Run (200 epochs) | A100 | ~30-45 min |

## 💡 Tips for Colab

### 1. Prevent Disconnection

```python
# Add this to keep Colab alive
import time
from IPython.display import Javascript

def keep_alive():
    while True:
        display(Javascript('document.querySelector("colab-connect-button").click()'))
        time.sleep(60)

# Run in background
import threading
threading.Thread(target=keep_alive).start()
```

### 2. Save Checkpoints Regularly

```python
# The notebook auto-saves best model, but you can also:
# Sync to Google Drive every N epochs
if epoch % 20 == 0:
    !cp -r ./checkpoints /content/drive/MyDrive/
```

### 3. Monitor GPU Usage

```python
# Add this cell to check GPU utilization
!nvidia-smi
```

### 4. Use Tensorboard

```python
# Load tensorboard extension
%load_ext tensorboard
%tensorboard --logdir ./logs
```

## 🐛 Common Issues

### Issue 1: "Runtime disconnected"

**Solution**:
- Colab has 12-hour limit for free tier
- Use Colab Pro for longer runtimes
- Save checkpoints regularly to Google Drive

### Issue 2: "Out of memory"

**Solution**:
```python
# Reduce batch size
config.BATCH_SIZE = 32

# Reduce image size
config.IMG_SIZE = 128

# Clear GPU cache
import torch
torch.cuda.empty_cache()
```

### Issue 3: "Dataset not found"

**Solution**:
```python
# Check if path exists
import os
print(os.path.exists(config.DATA_ROOT))
print(os.listdir(config.DATA_ROOT))
```

### Issue 4: "Training too slow"

**Solution**:
- Make sure GPU is enabled
- Reduce number of workers: `config.NUM_WORKERS = 2`
- Use smaller model: `base_model='resnet18'`

## 📱 Using Colab on Mobile

1. Install Google Colab app (iOS/Android)
2. Open notebook
3. Start training
4. Close app - training continues!
5. Check back later for results

## 🎓 Free Colab Limits

| Resource | Free Tier | Colab Pro | Colab Pro+ |
|----------|-----------|-----------|------------|
| GPU | T4 | T4/V100 | A100 |
| Runtime | 12 hours | 24 hours | 24 hours |
| RAM | 12 GB | 25 GB | 50 GB |
| Storage | 15 GB | 100 GB | 200 GB |

## 💰 Cost Estimate

- **Free Tier**: $0 (sufficient for testing and small experiments)
- **Colab Pro**: $9.99/month (recommended for full experiments)
- **Colab Pro+**: $49.99/month (for multiple runs and large datasets)

## ✅ Checklist Before Running

- [ ] GPU is enabled (Runtime → Change runtime type)
- [ ] Dataset is uploaded and path is correct
- [ ] Config values are set appropriately
- [ ] Google Drive is mounted (if saving there)
- [ ] Sufficient storage space available (~5GB)

## 🎯 Expected Results

After completion, you should have:

```
results/
├── training_loss.png          # Contrastive training curve
├── tsne_features.png          # Feature visualization
├── confusion_matrix_svm.png   # Classification results
├── methods_comparison.png     # Bar chart
└── results.json               # Numerical results

checkpoints/
├── best_contrastive.pth       # Trained encoder
├── encoder_final.pth          # Final encoder weights
├── svm_final.pkl              # Trained SVM
└── config.json                # Configuration used
```

## 📞 Need Help?

If you encounter issues:

1. Check the [Troubleshooting section](README_CONTRASTIVE_SVM.md#troubleshooting)
2. Open an issue on GitHub
3. Contact: your.email@university.edu

---

**Happy Experimenting! 🚀🌱**
