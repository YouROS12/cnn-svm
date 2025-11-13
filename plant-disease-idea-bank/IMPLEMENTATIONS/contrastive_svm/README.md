# Contrastive SVM Experiment

**Margin-Aware Contrastive Learning with SVM for Plant Disease Detection**

This folder contains a complete implementation of contrastive learning combined with SVM classification for plant disease detection using the PlantWildV2 dataset.

## 📁 Contents

```
contrastive_svm_experiment/
├── README.md                              ← You are here
├── contrastive_svm_plant_disease.ipynb   ⭐ Main notebook (START HERE!)
├── train_contrastive_svm.py              🐍 Training script
├── inference.py                          🔮 Inference script
├── requirements_contrastive.txt          📋 Dependencies
├── README_CONTRASTIVE_SVM.md            📖 Full documentation
├── QUICKSTART.md                         🚀 Quick start guide
├── COLAB_SETUP.md                       ☁️  Google Colab setup
└── PAPER_TEMPLATE.md                    📄 Publication template
```

## 🚀 Quick Start

### Step 1: Install Dependencies
```bash
cd contrastive_svm_experiment
pip install -r requirements_contrastive.txt
```

### Step 2: Prepare Your Dataset
Make sure your PlantWildV2 dataset is in this structure:
```
../plantwildV2/          (or any path you choose)
├── class1/
│   ├── img1.jpg
│   └── ...
├── class2/
│   └── ...
```

### Step 3: Run Experiment

**Option A: Jupyter Notebook (Recommended)**
```bash
jupyter notebook contrastive_svm_plant_disease.ipynb
```

**Option B: Command Line**
```bash
python train_contrastive_svm.py \
    --data_root ../plantwildV2 \
    --epochs 200 \
    --batch_size 128
```

**Option C: Google Colab**
- Upload `contrastive_svm_plant_disease.ipynb` to Colab
- Follow instructions in `COLAB_SETUP.md`

## 📖 Documentation

- **New to this?** → Start with `QUICKSTART.md`
- **Need details?** → Read `README_CONTRASTIVE_SVM.md`
- **Using Colab?** → Check `COLAB_SETUP.md`
- **Writing paper?** → Use `PAPER_TEMPLATE.md`

## 🎯 What This Experiment Does

1. **Contrastive Pretraining** (SimCLR)
   - Learns robust feature representations
   - Uses strong data augmentation
   - NT-Xent loss optimization

2. **SVM Classification**
   - Maximum margin classification
   - Linear or RBF kernel support
   - Better few-shot performance

3. **Baseline Comparisons**
   - Linear probe
   - Softmax fine-tuning
   - Supervised training

4. **Few-Shot Evaluation**
   - 1, 5, 10, 20-shot scenarios
   - Tests data efficiency

## 📊 Expected Output

After running, you'll get:
```
results/
├── training_loss.png          # Training curves
├── tsne_features.png          # Feature visualization
├── confusion_matrix_svm.png   # Classification results
├── methods_comparison.png     # Performance comparison
└── results.json               # Numerical results

checkpoints/
├── encoder_final.pth          # Trained encoder
├── svm_final.pkl             # Trained SVM
└── best_contrastive.pth      # Best checkpoint
```

## 🔬 Research Contribution

This experiment implements a novel approach combining:
- **Self-supervised learning** for robust representations
- **Maximum margin classification** for better generalization
- **Few-shot learning** for data-scarce scenarios

**Target for publication in:** IEEE TPAMI, Pattern Recognition, IEEE TNNLS

## ⏱️ Runtime Estimates

| Hardware | Training Time | Inference Time |
|----------|--------------|----------------|
| Tesla T4 | ~2-3 hours | ~1 sec/image |
| V100 | ~1-1.5 hours | ~0.5 sec/image |
| A100 | ~30-45 min | ~0.3 sec/image |
| CPU | ~24+ hours ⚠️ | ~5 sec/image |

## 🎓 Citation

If you use this code in your research:

```bibtex
@article{contrastive_svm_2025,
  title={Margin-Aware Contrastive Learning with SVM for Plant Disease Detection},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

## 📧 Support

- Read `QUICKSTART.md` for troubleshooting
- Check `README_CONTRASTIVE_SVM.md` for detailed docs
- Open GitHub issue for bugs

## 🔗 Related Files

This experiment is part of the larger CNN-SVM project:
- Original work: See `../README.md`
- TensorFlow implementation: See `../model/`
- PyTorch CNN: See `../pt_cnn_svm/`

---

**Ready to start?** Open `QUICKSTART.md` or launch the notebook! 🚀
