# 🔬 Experimental Design Best Practices

> How to design rigorous, reproducible experiments for Q1-level research

---

## 🎯 Core Principles

### 1. Reproducibility

**Every experiment must be exactly reproducible by another researcher.**

**Requirements**:
- Fix all random seeds (Python, NumPy, PyTorch/TensorFlow)
- Document all hyperparameters
- Specify software versions
- Provide data splits
- Share code and model weights

**Example (PyTorch)**:
```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

---

### 2. Statistical Rigor

**Never report results from a single run.**

**Minimum requirements for Q1**:
- 3-5 random seeds
- Report mean ± std (standard deviation)
- Statistical significance tests (t-test, ANOVA)
- Effect sizes (Cohen's d)

**Example**:
```
Method A: 92.3 ± 0.4%
Method B: 89.7 ± 0.6%
p-value: 0.003 (t-test, paired)
Effect size (Cohen's d): 0.85 (large)
```

---

### 3. Fair Comparison

**All methods must be compared under identical conditions.**

**Ensure**:
- Same data splits for all methods
- Same preprocessing
- Same evaluation metrics
- Same hardware (or normalize for hardware differences)
- Hyperparameters tuned equally (same budget for all)

---

## 📊 Experimental Components

### Component 1: Datasets

**Minimum for Q1**: 2-3 datasets
**Gold standard**: 3+ datasets with different characteristics

**Selection criteria**:
1. **Primary dataset**: Large, standard benchmark (e.g., PlantVillage)
2. **Cross-domain dataset**: Different imaging conditions (e.g., PlantDoc - real-world)
3. **Specialized dataset**: Different crop/task (e.g., IP102 - pests + diseases)

**Dataset documentation**:
```markdown
## Datasets

### PlantVillage
- **Size**: 54,305 images
- **Classes**: 38 (14 crop species, various diseases + healthy)
- **Source**: [Citation + link]
- **Characteristics**: Laboratory conditions, uniform background, controlled lighting
- **Splits**: 70% train (38,014), 15% val (8,146), 15% test (8,145)
- **Preprocessing**: Resize to 224×224, normalize with ImageNet statistics
```

**Data splits**:
- **Fixed splits**: Use official splits if available
- **Random splits**: Document seed used
- **Stratified**: Ensure class balance across splits
- **Share splits**: Provide split files (train.txt, val.txt, test.txt)

---

### Component 2: Baselines

**Minimum for Q1**: 5-6 baseline methods
**Categories**:
1. **Simple baseline**: CNN from scratch or traditional ML
2. **Transfer learning baseline**: ImageNet-pretrained model
3. **Domain-specific baseline**: Method designed for agriculture
4. **Recent SOTA**: State-of-the-art from last 1-2 years
5. **Ablation baseline**: Your method without key component
6. **Upper bound** (optional): Oracle or ideal performance

**Example baseline table**:
```markdown
| Baseline | Type | Source | Hyperparameters |
|----------|------|--------|-----------------|
| ResNet-50 scratch | Simple | - | lr=1e-3, bs=64, 200 epochs |
| ResNet-50 ImageNet | Transfer | torchvision | lr=1e-4, bs=64, 100 epochs |
| SimCLR | Self-supervised | [Citation] | Official implementation |
| ViT-PlantDisease | Recent SOTA | [Citation] | Reproduced from paper |
| Ours w/o component X | Ablation | - | Same as ours |
| Ours (full) | Proposed | - | lr=1e-4, bs=64, 100 epochs |
```

**Hyperparameter tuning**:
- Tune on validation set (NOT test set)
- Use same tuning budget for all methods
- Document search space and selected values
- Consider: grid search, random search, or Bayesian optimization

---

### Component 3: Evaluation Metrics

**Primary metrics** (report for all experiments):
- **Accuracy**: Overall classification accuracy
- **F1-score (macro)**: Handles class imbalance
- **Precision**: Minimize false positives
- **Recall**: Minimize false negatives

**Secondary metrics** (report when relevant):
- **Per-class accuracy**: Identify which classes are hard
- **Confusion matrix**: Visualize misclassifications
- **AUC-ROC**: Threshold-independent performance
- **Inference time**: Milliseconds per image
- **Model parameters**: Number of trainable parameters
- **FLOPs**: Computational complexity

**Example**:
```python
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# After predictions
accuracy = accuracy_score(y_true, y_pred)
f1_macro = f1_score(y_true, y_pred, average='macro')
f1_weighted = f1_score(y_true, y_pred, average='weighted')
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 (macro): {f1_macro:.4f}")
print(f"F1 (weighted): {f1_weighted:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
```

---

### Component 4: Statistical Tests

**When comparing two methods**:

**Paired t-test** (most common):
```python
from scipy.stats import ttest_rel

# Results from 5 seeds
method_a = [92.1, 92.5, 91.8, 92.3, 92.0]
method_b = [89.5, 90.1, 89.8, 89.7, 89.9]

t_stat, p_value = ttest_rel(method_a, method_b)
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("Difference is statistically significant")
else:
    print("Difference is NOT statistically significant")
```

**When comparing multiple methods**:

**ANOVA** + post-hoc tests:
```python
from scipy.stats import f_oneway
from scipy.stats import ttest_ind
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Results from 5 seeds for 3 methods
method_a = [92.1, 92.5, 91.8, 92.3, 92.0]
method_b = [89.5, 90.1, 89.8, 89.7, 89.9]
method_c = [91.0, 91.5, 90.8, 91.2, 91.1]

# ANOVA
f_stat, p_value = f_oneway(method_a, method_b, method_c)
print(f"ANOVA p-value: {p_value:.4f}")

# Post-hoc pairwise comparison (if ANOVA significant)
if p_value < 0.05:
    # Tukey's HSD
    data = np.concatenate([method_a, method_b, method_c])
    labels = ['A']*5 + ['B']*5 + ['C']*5
    tukey = pairwise_tukeyhsd(data, labels)
    print(tukey)
```

**Effect size (Cohen's d)**:
```python
import numpy as np

def cohens_d(x1, x2):
    """Calculate Cohen's d for two samples."""
    nx1, nx2 = len(x1), len(x2)
    dof = nx1 + nx2 - 2
    return (np.mean(x1) - np.mean(x2)) / np.sqrt(
        ((nx1-1)*np.std(x1, ddof=1)**2 + (nx2-1)*np.std(x2, ddof=1)**2) / dof
    )

d = cohens_d(method_a, method_b)
print(f"Cohen's d: {d:.3f}")

# Interpretation:
# |d| < 0.2: small effect
# 0.2 ≤ |d| < 0.5: medium effect
# |d| ≥ 0.5: large effect
```

---

### Component 5: Ablation Studies

**Purpose**: Understand contribution of each component

**Design**:
1. **Full model**: Your complete method
2. **Ablations**: Remove one component at a time
3. **Report**: Δ (delta) from full model

**Example ablation table**:
```markdown
| Configuration | Accuracy | Δ from Full |
|---------------|----------|-------------|
| Full Model | 92.3 ± 0.4 | - |
| w/o Contrastive Loss | 89.7 ± 0.5 | -2.6 |
| w/o Data Augmentation | 90.2 ± 0.6 | -2.1 |
| w/o Fine-tuning | 88.5 ± 0.7 | -3.8 |
| w/o LoRA Adapters | 91.8 ± 0.5 | -0.5 |
```

**Interpretation**:
- Largest Δ = most important component (Fine-tuning: -3.8)
- Smallest Δ = least important (LoRA Adapters: -0.5)
- Guides future work (which components to improve)

---

### Component 6: Cross-Dataset Evaluation

**Purpose**: Test generalization beyond training distribution

**Setup**:
- Train on Dataset A
- Test on Dataset B (no fine-tuning)
- Measure performance drop

**Example**:
```markdown
| Train → Test | Method A | Method B | Ours |
|--------------|----------|----------|------|
| PV → PV (in-domain) | 92.5 | 94.1 | 96.3 |
| PV → PD (cross-domain) | 68.4 | 71.2 | 84.3 |
| **Drop** | **-24.1** | **-22.9** | **-12.0** |
```

**Interpretation**: Ours has smallest domain shift drop (-12.0 vs. -24.1)

---

### Component 7: Few-Shot Evaluation

**Purpose**: Test data efficiency (important for plant disease detection)

**Setup**:
- K-shot: Use only K labeled examples per class
- Typical K values: 1, 3, 5, 10, 20
- Multiple episodes (10+) to reduce variance

**Protocol**:
```python
def few_shot_evaluation(model, dataset, K=5, num_episodes=10):
    """
    Evaluate model in K-shot setting.

    Args:
        model: Pre-trained model
        dataset: Full dataset
        K: Number of examples per class
        num_episodes: Number of evaluation episodes

    Returns:
        mean_accuracy, std_accuracy
    """
    accuracies = []

    for episode in range(num_episodes):
        # Sample K examples per class for support set
        support_set = sample_k_shot(dataset, K=K, seed=episode)

        # Fine-tune model on support set (few epochs)
        model_finetuned = finetune(model, support_set, epochs=20)

        # Evaluate on test set
        acc = evaluate(model_finetuned, dataset.test_set)
        accuracies.append(acc)

    return np.mean(accuracies), np.std(accuracies)
```

**Example results table**:
```markdown
| Method | 1-shot | 3-shot | 5-shot | 10-shot | 20-shot | Full |
|--------|--------|--------|--------|---------|---------|------|
| Baseline | 66.2 ± 2.4 | 75.8 ± 1.9 | 81.4 ± 1.6 | 86.3 ± 1.2 | 89.7 ± 0.9 | 92.5 |
| Ours | **78.4 ± 1.7** | **86.1 ± 1.3** | **91.2 ± 1.0** | **93.7 ± 0.7** | **95.3 ± 0.5** | **96.3** |
| Improvement | +12.2 | +10.3 | +9.8 | +7.4 | +5.6 | +3.8 |
```

---

## 📋 Experiment Execution Checklist

### Before Running Experiments

**Environment**:
- [ ] All packages installed (requirements.txt)
- [ ] Correct PyTorch/TensorFlow version
- [ ] GPU verified (run test script)
- [ ] Sufficient disk space for checkpoints

**Data**:
- [ ] All datasets downloaded
- [ ] Data integrity verified (checksums)
- [ ] Splits created and documented
- [ ] Preprocessing pipeline tested

**Code**:
- [ ] Code reviewed (no obvious bugs)
- [ ] Logging implemented (Weights & Biases, TensorBoard)
- [ ] Checkpointing implemented (save best model)
- [ ] Random seeds fixed

---

### During Experiments

**Monitoring**:
- [ ] Training curves look reasonable (no divergence)
- [ ] Validation accuracy improving
- [ ] No NaN losses
- [ ] GPU utilization high (>80%)

**Logging**:
- [ ] All hyperparameters logged
- [ ] Train/val metrics logged every epoch
- [ ] Best model checkpoint saved
- [ ] Experiment name/ID tracked

**Issues**:
- [ ] Document any anomalies
- [ ] Save error messages
- [ ] Note training time

---

### After Experiments

**Results**:
- [ ] All models trained (5 seeds × N methods)
- [ ] Metrics computed for all runs
- [ ] Statistical tests performed
- [ ] Results tables created

**Artifacts**:
- [ ] Best model checkpoints saved
- [ ] Training logs archived
- [ ] Figures generated (300 DPI)
- [ ] Code tagged/versioned (git tag)

**Documentation**:
- [ ] Experiment report written
- [ ] Reproduce instructions documented
- [ ] Unexpected findings noted

---

## 🎨 Visualization Best Practices

### Figure 1: Performance Comparison (Bar Chart)

**Good practices**:
- Include error bars (± std)
- Use distinct colors
- Add significance markers (*, **, ***)
- Label axes clearly
- Legend inside or outside plot

**Example code**:
```python
import matplotlib.pyplot as plt
import numpy as np

methods = ['Baseline 1', 'Baseline 2', 'Ours']
accuracies = [89.7, 91.2, 96.3]
stds = [0.6, 0.5, 0.4]

fig, ax = plt.subplots(figsize=(8, 6))
x = np.arange(len(methods))
bars = ax.bar(x, accuracies, yerr=stds, capsize=5,
              color=['#1f77b4', '#ff7f0e', '#2ca02c'])

# Add value labels on bars
for i, (acc, std) in enumerate(zip(accuracies, stds)):
    ax.text(i, acc + std + 1, f'{acc:.1f}±{std:.1f}',
            ha='center', va='bottom')

ax.set_xlabel('Method', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Performance Comparison on PlantVillage', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylim([0, 100])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('comparison.pdf', dpi=300, bbox_inches='tight')
plt.show()
```

---

### Figure 2: Learning Curves (Line Plot)

```python
import matplotlib.pyplot as plt

epochs = np.arange(1, 101)
train_loss = ... # Your training loss
val_loss = ... # Your validation loss
val_acc = ... # Your validation accuracy

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Loss curves
ax1.plot(epochs, train_loss, label='Train Loss', linewidth=2)
ax1.plot(epochs, val_loss, label='Val Loss', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Training Curves', fontsize=14)
ax1.legend()
ax1.grid(alpha=0.3)

# Accuracy curve
ax2.plot(epochs, val_acc, label='Val Accuracy', linewidth=2, color='green')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('Validation Accuracy', fontsize=14)
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('learning_curves.pdf', dpi=300, bbox_inches='tight')
```

---

### Figure 3: Confusion Matrix (Heatmap)

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Normalize by row (true labels)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Recall'})
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_title('Confusion Matrix (Normalized)', fontsize=14)

plt.tight_layout()
plt.savefig('confusion_matrix.pdf', dpi=300, bbox_inches='tight')
```

---

### Figure 4: t-SNE Visualization

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Extract features from model
features = model.extract_features(dataloader)  # Shape: [N, feature_dim]
labels = ... # True labels [N]

# t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
features_2d = tsne.fit_transform(features)

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1],
                     c=labels, cmap='tab10', s=20, alpha=0.6)
ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
ax.set_title('Feature Space Visualization (t-SNE)', fontsize=14)

# Add legend
legend = ax.legend(*scatter.legend_elements(num=len(class_names)),
                   title="Classes", loc="best")
ax.add_artist(legend)

plt.tight_layout()
plt.savefig('tsne.pdf', dpi=300, bbox_inches='tight')
```

---

## 🚨 Common Pitfalls

### Pitfall 1: Data Leakage

**Problem**: Information from test set leaks into training

**Examples**:
- ❌ Normalizing entire dataset before split
- ❌ Augmenting test set
- ❌ Hyperparameter tuning on test set
- ❌ Using test samples in contrastive pretraining

**Solution**:
- ✅ Split FIRST, then normalize (fit on train only)
- ✅ Augment only training set
- ✅ Tune on validation set, evaluate once on test
- ✅ Strictly separate train/val/test

---

### Pitfall 2: Unfair Comparison

**Problem**: Baselines not properly tuned or implemented

**Examples**:
- ❌ Using baseline with default hyperparameters (not tuned)
- ❌ Different preprocessing for different methods
- ❌ Different train/test splits
- ❌ Running baseline with fewer epochs

**Solution**:
- ✅ Tune all methods equally (same budget)
- ✅ Identical preprocessing for all
- ✅ Same data splits for all
- ✅ Train all methods until convergence

---

### Pitfall 3: P-Hacking

**Problem**: Running many experiments until getting p < 0.05

**Examples**:
- ❌ Trying 20 methods, only reporting the one with p < 0.05
- ❌ Re-running with different seeds until significant
- ❌ Choosing metrics that favor your method

**Solution**:
- ✅ Pre-register experiments (decide what to test before running)
- ✅ Report all results (even negative)
- ✅ Use consistent metrics across all experiments

---

### Pitfall 4: Cherry-Picking Results

**Problem**: Only reporting best seed or dataset

**Examples**:
- ❌ Running 10 seeds, reporting best 5
- ❌ Only showing dataset where method works best
- ❌ Hiding failure cases

**Solution**:
- ✅ Report all seeds (mean ± std)
- ✅ Report all datasets
- ✅ Discuss failure cases honestly

---

## ✅ Final Checklist

**Before claiming your experiments are complete**:

**Comprehensiveness**:
- [ ] 2-3 datasets evaluated
- [ ] 5+ baselines compared
- [ ] 3-5 random seeds per experiment
- [ ] Statistical significance tested
- [ ] Ablation studies performed

**Reproducibility**:
- [ ] All random seeds documented
- [ ] All hyperparameters logged
- [ ] Software versions recorded
- [ ] Data splits provided
- [ ] Code shared (or will be upon publication)

**Visualization**:
- [ ] 8+ figures created (300 DPI)
- [ ] 4+ tables formatted
- [ ] All figures have captions
- [ ] All tables have captions

**Statistical Rigor**:
- [ ] Mean ± std reported for all results
- [ ] Statistical tests performed (t-test, ANOVA)
- [ ] Effect sizes computed
- [ ] Significance marked in tables (*, **, ***)

**Quality**:
- [ ] Training curves look reasonable (no divergence)
- [ ] Results are consistent across seeds (low std)
- [ ] Improvements are meaningful (not just 0.1%)
- [ ] Failure cases analyzed

---

**If you checked 18+ boxes, your experiments are Q1-ready!** 🎉

---

## 📚 Additional Resources

- **TEMPLATES/experiment_protocol.md**: Detailed experiment planning template
- **TEMPLATES/Q1_submission_checklist.md**: Pre-submission checklist
- **DOCS/publication_guide.md**: How to write and publish

---

**Remember: Good experimental design is the foundation of impactful research. Take the time to do it right!** 🔬📊✅
