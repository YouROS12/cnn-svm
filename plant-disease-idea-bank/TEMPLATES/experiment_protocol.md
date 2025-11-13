# 🔬 Experimental Protocol Template

> Use this template to design rigorous, reproducible experiments for Q1-level research

---

## 📋 Experiment Overview

**Title**: [Descriptive experiment name]

**Objective**: [What are you trying to prove or discover?]

**Hypothesis**: [Clear, testable hypothesis]

**Research Questions**:
1. [Question 1]
2. [Question 2]
3. [Question 3]

**Expected Outcomes**: [What results do you expect?]

---

## 🎯 Experimental Design

### 1. Datasets

| Dataset | Size | Classes | Train/Val/Test Split | Purpose |
|---------|------|---------|---------------------|---------|
| Dataset 1 | X images | Y classes | 70/15/15 | Main evaluation |
| Dataset 2 | X images | Y classes | 70/15/15 | Cross-dataset validation |
| Dataset 3 | X images | Y classes | 70/15/15 | Generalization test |

**Justification for Dataset Selection**: [Why these datasets?]

**Data Preprocessing**:
- [ ] Resizing: [dimensions]
- [ ] Normalization: [method]
- [ ] Augmentation: [techniques]
- [ ] Class balancing: [method if needed]

### 2. Baselines

Comparison methods (minimum 5):

1. **Method 1**: [Name and brief description]
   - Paper: [Citation]
   - Implementation: [Source]
   - Hyperparameters: [Specify]

2. **Method 2**: [Name and brief description]
   - Paper: [Citation]
   - Implementation: [Source]
   - Hyperparameters: [Specify]

3. **Method 3**: [Name and brief description]
   - Paper: [Citation]
   - Implementation: [Source]
   - Hyperparameters: [Specify]

4. **Method 4**: [Name and brief description]
   - Paper: [Citation]
   - Implementation: [Source]
   - Hyperparameters: [Specify]

5. **Method 5**: [Name and brief description]
   - Paper: [Citation]
   - Implementation: [Source]
   - Hyperparameters: [Specify]

### 3. Evaluation Metrics

**Primary Metrics**:
- [ ] Accuracy
- [ ] F1-Score (macro, weighted)
- [ ] Precision
- [ ] Recall

**Secondary Metrics**:
- [ ] AUC-ROC
- [ ] Confusion Matrix
- [ ] Per-class accuracy
- [ ] Inference time
- [ ] Model parameters

**Justification**: [Why these metrics matter for your problem]

### 4. Experimental Conditions

**Hardware**:
- GPU: [Model]
- RAM: [Amount]
- Storage: [Amount]

**Software**:
- Framework: [PyTorch, TensorFlow, etc.]
- Version: [X.X.X]
- CUDA: [Version]
- Python: [Version]

**Random Seeds**: [5, 42, 123, 456, 789]
- **Rationale**: Run all experiments with 5 different seeds for statistical significance

---

## 🧪 Experiment Series

### Experiment 1: Baseline Performance

**Goal**: Establish baseline performance on all datasets

**Setup**:
- Train baseline models on Dataset 1
- Evaluate on test sets of all 3 datasets
- Record mean ± std across 5 seeds

**Success Criteria**: [What constitutes success?]

**Timeline**: Week 1-2

---

### Experiment 2: Main Method Evaluation

**Goal**: Evaluate proposed method vs baselines

**Setup**:
- Train proposed method on Dataset 1
- Compare with all 5 baselines
- Report statistical significance (p-values)

**Success Criteria**: [What constitutes success?]

**Timeline**: Week 3-4

---

### Experiment 3: Cross-Dataset Evaluation

**Goal**: Test generalization across different datasets

**Setup**:
- Train on Dataset 1, test on Dataset 2 and 3
- Train on Dataset 2, test on Dataset 1 and 3
- Compare cross-dataset performance

**Success Criteria**: [What constitutes success?]

**Timeline**: Week 5

---

### Experiment 4: Few-Shot Learning

**Goal**: Evaluate data efficiency

**Setup**:
- K-shot evaluation: K = 1, 3, 5, 10, 20
- 10 episodes per K value
- Report mean ± std

**Success Criteria**: [What constitutes success?]

**Timeline**: Week 6

---

### Experiment 5: Ablation Studies

**Goal**: Understand contribution of each component

**Ablation Table**:

| Variant | Component A | Component B | Component C | Accuracy | Δ from Full |
|---------|------------|-------------|-------------|----------|-------------|
| Full Model | ✓ | ✓ | ✓ | XX.XX% | - |
| w/o Component A | ✗ | ✓ | ✓ | XX.XX% | -X.XX% |
| w/o Component B | ✓ | ✗ | ✓ | XX.XX% | -X.XX% |
| w/o Component C | ✓ | ✓ | ✗ | XX.XX% | -X.XX% |

**Success Criteria**: [What constitutes success?]

**Timeline**: Week 7

---

### Experiment 6: Hyperparameter Sensitivity

**Goal**: Show robustness to hyperparameter choices

**Parameters to Vary**:
- Learning rate: [0.0001, 0.001, 0.01]
- Batch size: [32, 64, 128]
- Architecture depth: [18, 34, 50]

**Success Criteria**: [What constitutes success?]

**Timeline**: Week 8

---

## 📊 Results Collection

### Results Tables Template

**Table 1: Main Results**

| Method | Dataset 1 | Dataset 2 | Dataset 3 | Avg | Params |
|--------|-----------|-----------|-----------|-----|--------|
| Baseline 1 | XX.XX ± X.XX | XX.XX ± X.XX | XX.XX ± X.XX | XX.XX | XXM |
| Baseline 2 | XX.XX ± X.XX | XX.XX ± X.XX | XX.XX ± X.XX | XX.XX | XXM |
| Baseline 3 | XX.XX ± X.XX | XX.XX ± X.XX | XX.XX ± X.XX | XX.XX | XXM |
| Baseline 4 | XX.XX ± X.XX | XX.XX ± X.XX | XX.XX ± X.XX | XX.XX | XXM |
| Baseline 5 | XX.XX ± X.XX | XX.XX ± X.XX | XX.XX ± X.XX | XX.XX | XXM |
| **Ours** | **XX.XX ± X.XX** | **XX.XX ± X.XX** | **XX.XX ± X.XX** | **XX.XX** | **XXM** |

**Table 2: Few-Shot Results**

| Method | 1-shot | 3-shot | 5-shot | 10-shot | 20-shot |
|--------|--------|--------|--------|---------|---------|
| Baseline | XX.XX ± X.XX | XX.XX ± X.XX | XX.XX ± X.XX | XX.XX ± X.XX | XX.XX ± X.XX |
| **Ours** | **XX.XX ± X.XX** | **XX.XX ± X.XX** | **XX.XX ± X.XX** | **XX.XX ± X.XX** | **XX.XX ± X.XX** |

---

## 📈 Statistical Analysis

### Significance Testing

**Paired t-test** (comparing two methods):
```python
from scipy.stats import ttest_rel

# Results from 5 seeds
ours = [acc1, acc2, acc3, acc4, acc5]
baseline = [acc1, acc2, acc3, acc4, acc5]

t_stat, p_value = ttest_rel(ours, baseline)
print(f"p-value: {p_value:.4f}")
```

**Interpretation**:
- p < 0.001: *** (highly significant)
- p < 0.01: ** (very significant)
- p < 0.05: * (significant)
- p ≥ 0.05: not significant

### Effect Size

**Cohen's d**:
```python
import numpy as np

def cohens_d(x1, x2):
    nx1, nx2 = len(x1), len(x2)
    dof = nx1 + nx2 - 2
    return (np.mean(x1) - np.mean(x2)) / np.sqrt(((nx1-1)*np.std(x1, ddof=1)**2 + (nx2-1)*np.std(x2, ddof=1)**2) / dof)

d = cohens_d(ours, baseline)
print(f"Cohen's d: {d:.3f}")
```

**Interpretation**:
- |d| < 0.2: small effect
- 0.2 ≤ |d| < 0.5: medium effect
- |d| ≥ 0.5: large effect

---

## 📸 Visualization Requirements

### Figure 1: Performance Comparison
- Bar chart with error bars
- All methods on all datasets
- 300 DPI, vector format (PDF/SVG)

### Figure 2: Few-Shot Learning Curves
- Line plot with error bands
- X-axis: K (shots), Y-axis: Accuracy
- Multiple methods compared

### Figure 3: Confusion Matrix
- Heatmap for best model
- Per-class breakdown

### Figure 4: t-SNE Visualization
- Feature space visualization
- Compare learned representations

### Figure 5: Ablation Study
- Bar chart showing component contributions

### Figure 6-8: [Additional figures as needed]

**Figure Guidelines**:
- [ ] 300 DPI minimum
- [ ] Vector formats (PDF, SVG) preferred
- [ ] Colorblind-friendly palette
- [ ] Clear legends
- [ ] Properly labeled axes with units
- [ ] Self-contained captions

---

## 🗂️ Data Management

### Directory Structure

```
experiment_name/
├── data/
│   ├── raw/              # Original datasets
│   ├── processed/        # Preprocessed data
│   └── splits/           # Train/val/test splits
├── models/
│   ├── checkpoints/      # Model checkpoints
│   └── best/             # Best models per seed
├── results/
│   ├── logs/             # Training logs
│   ├── metrics/          # Evaluation metrics (CSV)
│   ├── figures/          # Generated figures
│   └── tables/           # Results tables
├── code/
│   ├── models/           # Model architectures
│   ├── data/             # Data loaders
│   ├── train.py          # Training script
│   ├── eval.py           # Evaluation script
│   └── utils.py          # Utility functions
└── README.md             # Experiment documentation
```

### File Naming Convention

- Models: `{model_name}_seed{seed}_epoch{epoch}.pth`
- Results: `{experiment_name}_seed{seed}_results.csv`
- Figures: `{figure_name}_300dpi.pdf`

---

## ✅ Reproducibility Checklist

### Before Running Experiments
- [ ] All random seeds fixed (Python, NumPy, PyTorch/TensorFlow)
- [ ] Deterministic algorithms enabled
- [ ] Environment fully specified (requirements.txt, environment.yml)
- [ ] Data preprocessing documented
- [ ] Hyperparameters logged

### During Experiments
- [ ] Log all hyperparameters
- [ ] Save model checkpoints
- [ ] Record training curves
- [ ] Monitor resource usage
- [ ] Version control code

### After Experiments
- [ ] Save all raw results
- [ ] Compute statistics (mean, std, CI)
- [ ] Generate all figures
- [ ] Run statistical tests
- [ ] Document any issues or anomalies

---

## 📝 Experiment Log Template

### Run Log: [Date]

**Seed**: [X]

**Configuration**:
```yaml
model: [architecture]
learning_rate: [value]
batch_size: [value]
epochs: [value]
optimizer: [type]
```

**Training Progress**:
- Epoch 10: Train Loss = X.XX, Val Acc = XX.XX%
- Epoch 20: Train Loss = X.XX, Val Acc = XX.XX%
- ...
- Epoch N: Train Loss = X.XX, Val Acc = XX.XX%

**Final Results**:
- Test Accuracy: XX.XX%
- Test F1-Score: XX.XX
- Inference Time: X.XX ms/image

**Issues/Notes**: [Any observations]

---

## 🚨 Common Pitfalls to Avoid

1. **Data Leakage**: Ensure strict separation of train/val/test
2. **Cherry-picking**: Report all runs, not just best seed
3. **Overfitting**: Monitor validation performance
4. **Unfair Comparison**: Use same preprocessing for all methods
5. **Missing Statistics**: Always report mean ± std
6. **P-hacking**: Don't run tests until you get p < 0.05
7. **Incomplete Ablations**: Test one component at a time

---

## 📅 Timeline Template

**Week 1-2**: Setup and baseline experiments
**Week 3-4**: Main method implementation and evaluation
**Week 5**: Cross-dataset evaluation
**Week 6**: Few-shot learning experiments
**Week 7**: Ablation studies
**Week 8**: Hyperparameter sensitivity
**Week 9**: Additional experiments (if needed)
**Week 10**: Results analysis and figure generation

---

## 📞 Questions to Answer Before Starting

1. [ ] Have I clearly defined the research question?
2. [ ] Are my datasets appropriate and sufficient?
3. [ ] Have I identified all necessary baselines?
4. [ ] Are my evaluation metrics aligned with my goals?
5. [ ] Do I have the computational resources needed?
6. [ ] Is my experimental design sound and unbiased?
7. [ ] Can someone else reproduce my experiments?

---

**Good luck with your experiments! Rigorous experimental design is the foundation of Q1-level research.** 🔬📊🎯
