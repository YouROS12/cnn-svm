# 📊 Plant Disease Detection Datasets

## 🌟 Essential Public Datasets

### 1. **PlantVillage**
- **Size**: 54,303 images
- **Classes**: 38 diseases + healthy across 14 crop species
- **Format**: RGB images, 256x256 pixels
- **License**: CC0 (Public Domain)
- **Download**: https://github.com/spMohanty/PlantVillage-Dataset
- **Kaggle**: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

**Crops**: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

**Best For**: Benchmarking, transfer learning, few-shot learning

---

### 2. **Plant Pathology 2020**
- **Size**: 3,651 images
- **Classes**: 4 (healthy, multiple diseases, rust, scab)
- **Format**: High-resolution RGB images
- **License**: Open (Kaggle Competition)
- **Download**: https://www.kaggle.com/c/plant-pathology-2020-fgvc7

**Crop**: Apple leaves

**Best For**: Fine-grained recognition, imbalanced learning, real-world conditions

---

### 3. **PlantDoc**
- **Size**: 2,598 images
- **Classes**: 27 diseases + healthy across 13 plants
- **Format**: RGB images with bounding boxes
- **License**: CC BY 4.0
- **Download**: https://github.com/pratikkayal/PlantDoc-Dataset

**Best For**: Object detection, in-the-wild evaluation, bounding box annotations

---

### 4. **Crop Diseases Dataset** (Kaggle)
- **Size**: 8,000+ images
- **Classes**: 4 main crops, multiple diseases
- **Format**: Organized by crop and disease
- **Download**: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

**Crops**: Tomato, Potato, Pepper, Corn

**Best For**: Multi-crop studies

---

### 5. **Indian Plant Disease Dataset**
- **Size**: 7,000+ images
- **Classes**: 20+ diseases
- **Format**: RGB images, various resolutions
- **License**: Research use
- **Download**: https://data.mendeley.com/datasets/tywbtsjrjv/1

**Best For**: Geographic diversity

---

## 🔬 Specialized Datasets

### Hyperspectral & Multi-Modal

**6. Hyperspectral Plant Disease**
- **Size**: 1,000+ samples
- **Format**: Hyperspectral cubes (400-1000nm, 100-200 bands)
- **Access**: Various university repositories
- **Note**: Often requires partnership/request

**7. Thermal Imaging Datasets**
- Limited public availability
- Check university agricultural research stations

---

## 🌍 Regional Datasets

### Africa
- **Cassava Leaf Disease Dataset**: https://www.kaggle.com/c/cassava-leaf-disease-classification
- **Maize Disease Dataset**: Various African research institutions

### Asia
- **Rice Disease Dataset**: Multiple sources in China, India, Japan
- **Tea Leaf Disease**: Available from agricultural universities

### Europe
- **Grapevine Disease Dataset**: European viticulture research
- **Wheat Disease Dataset**: CIMMYT, European labs

---

## 📋 Dataset Comparison

| Dataset | Size | Classes | Quality | In-Wild | Multi-Modal | Best Use Case |
|---------|------|---------|---------|---------|-------------|---------------|
| **PlantVillage** | 54K | 38 | High | No | RGB only | Benchmarking, Few-shot |
| **Plant Pathology** | 3.6K | 4 | Very High | Yes | RGB only | Real-world eval |
| **PlantDoc** | 2.6K | 27 | High | Yes | RGB + bbox | Detection tasks |
| **Crop Diseases** | 8K | 15+ | Medium | No | RGB only | Multi-crop |
| **Cassava** | 21K | 5 | High | Yes | RGB only | Production system |

---

## 🛠️ How to Use Multiple Datasets

### Strategy 1: Benchmark Across Datasets
Train on one, test on all others (measure generalization)

### Strategy 2: Combined Training
Pool datasets, train on mixture (increase diversity)

### Strategy 3: Meta-Learning
Use each dataset as a separate task (learn to learn)

### Strategy 4: Transfer Learning
Pretrain on large dataset, fine-tune on target

---

## 📥 Download Scripts

### PlantVillage
```bash
# Via Kaggle API
kaggle datasets download -d abdallahalidev/plantvillage-dataset
unzip plantvillage-dataset.zip -d ./data/plantvillage
```

### Plant Pathology 2020
```bash
kaggle competitions download -c plant-pathology-2020-fgvc7
unzip plant-pathology-2020-fgvc7.zip -d ./data/plant-pathology
```

---

## 💡 Dataset Best Practices

### Splitting
- **Standard**: 70% train, 15% val, 15% test
- **Few-shot**: Separate validation set for episodic sampling
- **Cross-dataset**: Never mix datasets in train/test

### Augmentation
- Use domain-specific augmentations (rotations for plants)
- Avoid unrealistic transforms (extreme color shifts)
- Test augmentation strength as hyperparameter

### Preprocessing
- Resize consistently (224x224 standard)
- Normalize using ImageNet stats
- Handle class imbalance (oversampling, focal loss)

---

## 🌱 Create Your Own Dataset

### Equipment Needed
- **Basic**: Smartphone (12MP+ camera)
- **Better**: DSLR camera with macro lens
- **Best**: Controlled lighting setup, multiple angles

### Collection Protocol
1. Multiple angles per sample (top, side, close-up)
2. Consistent lighting conditions
3. Include healthy samples
4. Metadata: Date, location, growth stage
5. Expert validation of labels

### Annotation Tools
- **LabelImg**: Bounding boxes
- **Labelme**: Polygons, segmentation
- **CVAT**: Multi-format annotations
- **Roboflow**: End-to-end platform

---

## 📊 Dataset Statistics Tools

```python
# Analyze dataset distribution
import matplotlib.pyplot as plt
from collections import Counter
import os

def analyze_dataset(root_dir):
    class_counts = Counter()
    for class_dir in os.listdir(root_dir):
        path = os.path.join(root_dir, class_dir)
        if os.path.isdir(path):
            count = len(os.listdir(path))
            class_counts[class_dir] = count

    # Plot distribution
    plt.figure(figsize=(12, 6))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Number of Images')
    plt.title('Dataset Class Distribution')
    plt.tight_layout()
    plt.savefig('dataset_distribution.png')

    # Print statistics
    print(f"Total classes: {len(class_counts)}")
    print(f"Total images: {sum(class_counts.values())}")
    print(f"Min: {min(class_counts.values())}")
    print(f"Max: {max(class_counts.values())}")
    print(f"Mean: {sum(class_counts.values()) / len(class_counts):.1f}")

analyze_dataset('./data/plantvillage')
```

---

## 🔍 Dataset Quality Checks

### Check for Issues
```python
from PIL import Image
import os

def check_dataset_quality(root_dir):
    issues = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(('.jpg', '.png')):
                path = os.path.join(root, file)
                try:
                    img = Image.open(path)
                    # Check size
                    if img.size[0] < 50 or img.size[1] < 50:
                        issues.append(f"Too small: {path}")
                    # Check format
                    if img.mode not in ['RGB', 'L']:
                        issues.append(f"Wrong format: {path}")
                except Exception as e:
                    issues.append(f"Corrupted: {path}")

    print(f"Found {len(issues)} issues")
    return issues

issues = check_dataset_quality('./data/plantvillage')
```

---

## 📚 Citation Information

When using datasets, always cite the original papers:

**PlantVillage:**
```bibtex
@article{hughes2015open,
  title={An open access repository of images on plant health to enable the development of mobile disease diagnostics},
  author={Hughes, David P and Salath{\'e}, Marcel},
  journal={arXiv preprint arXiv:1511.08060},
  year={2015}
}
```

**Plant Pathology 2020:**
```bibtex
@misc{plant-pathology-2020,
  title={Plant Pathology 2020 - FGVC7},
  author={Ranjita Thapa and others},
  year={2020},
  howpublished={Kaggle}
}
```

---

## 🆘 Need Help?

- **Missing dataset?** Check [Papers With Code](https://paperswithcode.com/datasets?task=plant-disease-detection)
- **Access issues?** Contact dataset authors directly
- **Custom dataset?** Use data collection services or hire annotators

---

**Updated**: November 2024
