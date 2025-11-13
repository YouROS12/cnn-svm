# 🚀 Quick Start Guide

Get up and running with Contrastive SVM for plant disease detection in minutes!

## 📦 What You Got

This implementation includes:

```
cnn-svm/
├── contrastive_svm_plant_disease.ipynb  ⭐ Main Jupyter notebook (START HERE!)
├── train_contrastive_svm.py             🐍 Standalone training script
├── inference.py                         🔮 Inference on new images
├── requirements_contrastive.txt         📋 Dependencies
├── README_CONTRASTIVE_SVM.md           📖 Full documentation
├── COLAB_SETUP.md                      ☁️  Google Colab guide
├── PAPER_TEMPLATE.md                   📄 Publication template
└── plantwildV2/                        📁 Your dataset (you provide)
```

## 🎯 Choose Your Path

### Path 1: Jupyter Notebook (Recommended for First Time) ⭐

**Best for**: Interactive exploration, visualization, learning

1. **Open the notebook:**
   ```bash
   jupyter notebook contrastive_svm_plant_disease.ipynb
   ```

2. **Update dataset path** in Cell 3:
   ```python
   config.DATA_ROOT = './plantwildV2'  # Change this!
   ```

3. **Run all cells**: Runtime → Run all

4. **Check results** in `./results/` folder

**Time**: ~2-3 hours on GPU

---

### Path 2: Google Colab (No Local Setup Needed) ☁️

**Best for**: No GPU, quick testing, collaboration

1. **Upload notebook** to Google Colab
2. **Enable GPU**: Runtime → Change runtime type → GPU
3. **Upload dataset** (see COLAB_SETUP.md for details)
4. **Run all cells**

See [COLAB_SETUP.md](COLAB_SETUP.md) for detailed instructions.

---

### Path 3: Command Line Script (Production) 🐍

**Best for**: Automated runs, hyperparameter sweeps, reproducibility

```bash
# Install dependencies
pip install -r requirements_contrastive.txt

# Train model
python train_contrastive_svm.py \
    --data_root ./plantwildV2 \
    --epochs 200 \
    --batch_size 128 \
    --svm_c 1.0

# Make predictions
python inference.py \
    --image path/to/test_image.jpg \
    --checkpoint ./checkpoints
```

---

## 📊 Expected Timeline

| Stage | Time (GPU) | Time (CPU) |
|-------|-----------|------------|
| Setup | 5 min | 5 min |
| Contrastive pretraining | 2-3 hours | 24+ hours ⚠️ |
| Feature extraction | 5 min | 15 min |
| SVM training | 2 min | 5 min |
| Evaluation | 5 min | 10 min |
| **Total** | **~3 hours** | **~24 hours** |

💡 **Tip**: Use GPU for contrastive pretraining! CPU training is impractically slow.

---

## ✅ Pre-flight Checklist

Before running, make sure you have:

- [ ] **Dataset** in correct format (see below)
- [ ] **GPU** available (strongly recommended)
- [ ] **16GB+ RAM** (8GB minimum)
- [ ] **10GB+ disk space** for checkpoints and results
- [ ] **Python 3.8+** installed
- [ ] **PyTorch with CUDA** (if using GPU)

---

## 📂 Dataset Format

Your PlantWildV2 dataset should look like this:

```
plantwildV2/
├── healthy/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
├── bacterial_spot/
│   ├── img_001.jpg
│   └── ...
├── early_blight/
│   └── ...
└── [more disease classes]/
```

**Requirements:**
- One folder per class
- Folder name = class name
- Images: .jpg, .jpeg, .png, or .bmp
- Minimum: 100 images per class
- Recommended: 500+ images per class

**Don't have a dataset?** Use public datasets:
- PlantVillage: https://github.com/spMohanty/PlantVillage-Dataset
- Plant Pathology 2020: https://www.kaggle.com/c/plant-pathology-2020-fgvc7

---

## 🎛️ Quick Configuration

### For Testing (Fast)

```python
config.CONTRASTIVE_EPOCHS = 50      # Quick test
config.BATCH_SIZE = 64              # Lower memory
config.ENABLE_FEW_SHOT = False      # Skip few-shot
```

**Time**: ~45 minutes

### For Publication (Full)

```python
config.CONTRASTIVE_EPOCHS = 200     # Full training
config.BATCH_SIZE = 128             # Optimal
config.ENABLE_FEW_SHOT = True       # Include few-shot
```

**Time**: ~3 hours

---

## 📈 Understanding Your Results

After running, you'll get:

### 1. Numerical Results (`results/results.json`)

```json
{
    "svm_accuracy": 92.45,
    "linear_probe_accuracy": 89.23,
    "softmax_accuracy": 91.87,
    "few_shot_results": {
        "1-shot": 65.34,
        "5-shot": 82.17,
        "10-shot": 87.92
    }
}
```

**What this means:**
- ✅ SVM outperforms linear probe
- ✅ Competitive with softmax
- ✅ Strong few-shot performance

### 2. Visualizations (`results/*.png`)

- **training_loss.png**: Shows if model converged
- **tsne_features.png**: Feature space visualization (should see clear clusters)
- **confusion_matrix_svm.png**: Per-class performance
- **methods_comparison.png**: Bar chart comparing methods

### 3. Trained Models (`checkpoints/`)

- **encoder_final.pth**: Trained encoder (reusable!)
- **svm_final.pkl**: Trained SVM classifier
- **best_contrastive.pth**: Best checkpoint during training

---

## 🎯 What's a "Good" Result?

### Accuracy Benchmarks

| Dataset Size | Expected Accuracy | Excellent Result |
|-------------|------------------|------------------|
| Small (<1K images) | 80-85% | >90% |
| Medium (1K-10K) | 85-92% | >95% |
| Large (>10K) | 90-95% | >97% |

### Few-Shot Performance

| K-shot | Good | Excellent |
|--------|------|-----------|
| 1-shot | >50% | >70% |
| 5-shot | >70% | >85% |
| 10-shot | >80% | >90% |

---

## 🔧 Troubleshooting

### "Out of memory" Error

```python
# Solution 1: Reduce batch size
config.BATCH_SIZE = 64  # or 32

# Solution 2: Smaller image size
config.IMG_SIZE = 128

# Solution 3: Use ResNet18
encoder = ContrastiveEncoder(base_model='resnet18')
```

### Training is Too Slow

```python
# Solution 1: Reduce epochs (for testing)
config.CONTRASTIVE_EPOCHS = 50

# Solution 2: Check GPU is being used
print(torch.cuda.is_available())  # Should be True

# Solution 3: Reduce num_workers
config.NUM_WORKERS = 2
```

### Dataset Not Found

```bash
# Check path exists
ls plantwildV2/

# Should show class folders
# If not, check config.DATA_ROOT
```

### Poor Results

**If accuracy < 70%:**
- Check dataset quality (mislabeled images?)
- Increase training epochs
- Verify data augmentation is working
- Try different SVM C values

**If SVM worse than softmax:**
- This is normal for very large datasets
- SVM shines in few-shot scenarios
- Check if features are well-separated (t-SNE)

---

## 🚀 Next Steps After First Run

### 1. Optimize Hyperparameters

```bash
# Try different SVM C values
python train_contrastive_svm.py --svm_c 0.1
python train_contrastive_svm.py --svm_c 10.0

# Try different temperatures
python train_contrastive_svm.py --temperature 0.1
python train_contrastive_svm.py --temperature 1.0
```

### 2. Test on New Images

```bash
# Single image
python inference.py --image test.jpg --checkpoint ./checkpoints

# Batch prediction
python inference.py --batch_dir ./test_images --checkpoint ./checkpoints
```

### 3. Analyze Results

- Look at confusion matrix - which classes are confused?
- Check t-SNE - are clusters well-separated?
- Examine failure cases - what patterns emerge?

### 4. Improve Performance

- Collect more data for confused classes
- Adjust augmentation strategies
- Try ensemble of multiple runs
- Experiment with RBF kernel: `--svm_kernel rbf`

### 5. Prepare for Publication

- Run with 3-5 different random seeds
- Compare against more baselines
- Test on multiple datasets
- See PAPER_TEMPLATE.md for guidance

---

## 📚 Learn More

- **Full Documentation**: [README_CONTRASTIVE_SVM.md](README_CONTRASTIVE_SVM.md)
- **Colab Guide**: [COLAB_SETUP.md](COLAB_SETUP.md)
- **Paper Template**: [PAPER_TEMPLATE.md](PAPER_TEMPLATE.md)
- **Original CNN-SVM Paper**: [Agarap, 2017](https://arxiv.org/abs/1712.03541)
- **SimCLR Paper**: [Chen et al., 2020](https://arxiv.org/abs/2002.05709)

---

## 🆘 Getting Help

If you're stuck:

1. **Check troubleshooting section** above
2. **Read error messages** carefully
3. **Verify dataset format** is correct
4. **Test with smaller settings** first
5. **Open GitHub issue** with:
   - Error message
   - Your configuration
   - Dataset statistics
   - Hardware specs

---

## 🎉 Success Indicators

You'll know it's working when:

- ✅ Training loss decreases smoothly
- ✅ t-SNE shows clear class clusters
- ✅ SVM accuracy > 85% on your dataset
- ✅ Few-shot accuracy improves with more shots
- ✅ Confusion matrix shows diagonal pattern

---

## 💡 Pro Tips

1. **Start small**: Test with 50 epochs before full run
2. **Save checkpoints**: Training can be interrupted
3. **Use tensorboard**: Monitor training in real-time
4. **Version control**: Track different experiments
5. **Document everything**: Note what works and what doesn't

---

## 🎯 Your Journey

```
Day 1: Setup and first run (3-4 hours)
├── Install dependencies
├── Prepare dataset
├── Run notebook with reduced epochs
└── Verify everything works

Day 2: Full training (4-6 hours)
├── Run full 200 epochs
├── Analyze results
├── Test inference
└── Tune hyperparameters

Day 3: Optimization (variable)
├── Try different configurations
├── Run few-shot experiments
├── Compare with baselines
└── Prepare visualizations

Week 2-4: Publication prep
├── Run multiple seeds
├── Test on more datasets
├── Write paper
└── Prepare submission
```

---

**Ready? Let's go! 🚀**

```bash
# Start here:
jupyter notebook contrastive_svm_plant_disease.ipynb
```

**Questions?** Open an issue or contact [your.email@domain.com]

---

*Happy training! May your margins be maximal and your losses be minimal* 😄🌱🤖
