"""
Template for Plant Disease Detection Research Code

This template provides a reusable structure for implementing plant disease
detection models. Customize the components as needed for your specific method.

Author: [Your Name]
Date: [Date]
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import logging
from tqdm import tqdm
import random
import json


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """
    Configuration class for experiment settings.
    Modify these parameters for your specific experiment.
    """
    # Data
    data_dir: str = "/path/to/dataset"
    image_size: int = 224
    num_classes: int = 38

    # Training
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5

    # Model
    model_name: str = "resnet50"  # or "vit_base", "dinov2", etc.
    pretrained: bool = True
    dropout: float = 0.2

    # Optimization
    optimizer: str = "adamw"  # "adam", "sgd", "adamw"
    scheduler: str = "cosine"  # "cosine", "step", "plateau"
    label_smoothing: float = 0.1

    # Regularization
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0

    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    seed: int = 42

    # Paths
    checkpoint_dir: str = "./checkpoints"
    results_dir: str = "./results"
    log_dir: str = "./logs"

    # Logging
    log_interval: int = 10
    save_interval: int = 5

    # Few-shot settings (if applicable)
    few_shot_k: List[int] = [1, 3, 5, 10, 20]
    num_episodes: int = 10


# ============================================================================
# Utility Functions
# ============================================================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_file: str = "experiment.log"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer,
                   epoch: int, best_acc: float, filepath: str):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
    }, filepath)


def load_checkpoint(model: nn.Module, optimizer: optim.Optimizer,
                   filepath: str) -> Tuple[int, float]:
    """Load model checkpoint."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['best_acc']


# ============================================================================
# Dataset
# ============================================================================

class PlantDiseaseDataset(Dataset):
    """
    Custom dataset for plant disease images.

    Expected directory structure:
    data_dir/
        train/
            class1/
                image1.jpg
                image2.jpg
            class2/
                ...
        val/
            ...
        test/
            ...
    """

    def __init__(self, root_dir: str, split: str = "train",
                 transform: Optional[transforms.Compose] = None):
        """
        Args:
            root_dir: Root directory of dataset
            split: "train", "val", or "test"
            transform: Torchvision transforms to apply
        """
        self.root_dir = Path(root_dir) / split
        self.transform = transform

        # Get all image paths and labels
        self.samples = []
        self.class_to_idx = {}

        for idx, class_dir in enumerate(sorted(self.root_dir.iterdir())):
            if class_dir.is_dir():
                self.class_to_idx[class_dir.name] = idx
                for img_path in class_dir.glob("*.jpg"):  # Adjust extension if needed
                    self.samples.append((str(img_path), idx))

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Load image
        from PIL import Image
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(config: Config) -> Dict[str, transforms.Compose]:
    """Get data augmentation transforms for train/val/test."""

    # Normalization statistics (ImageNet)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                              saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.Resize(int(config.image_size * 1.14)),  # Slightly larger
        transforms.CenterCrop(config.image_size),
        transforms.ToTensor(),
        normalize,
    ])

    return {
        'train': train_transform,
        'val': test_transform,
        'test': test_transform
    }


def get_dataloaders(config: Config) -> Dict[str, DataLoader]:
    """Create dataloaders for train/val/test."""
    transforms_dict = get_transforms(config)

    datasets = {
        split: PlantDiseaseDataset(
            root_dir=config.data_dir,
            split=split,
            transform=transforms_dict[split]
        )
        for split in ['train', 'val', 'test']
    }

    dataloaders = {
        'train': DataLoader(
            datasets['train'],
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True
        ),
        'val': DataLoader(
            datasets['val'],
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        ),
        'test': DataLoader(
            datasets['test'],
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
    }

    return dataloaders


# ============================================================================
# Model
# ============================================================================

class PlantDiseaseClassifier(nn.Module):
    """
    Plant disease classification model.

    Can use various backbones (ResNet, ViT, DINOv2, etc.)
    with a classification head.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Load backbone
        if config.model_name == "resnet50":
            self.backbone = models.resnet50(pretrained=config.pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove classification head

        elif config.model_name == "resnet101":
            self.backbone = models.resnet101(pretrained=config.pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif config.model_name == "vit_base":
            self.backbone = models.vit_b_16(pretrained=config.pretrained)
            feature_dim = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Identity()

        else:
            raise ValueError(f"Unknown model: {config.model_name}")

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(feature_dim, config.num_classes)
        )

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input images [B, 3, H, W]
            return_features: If True, return features instead of logits

        Returns:
            logits [B, num_classes] or features [B, feature_dim]
        """
        features = self.backbone(x)

        if return_features:
            return features

        logits = self.classifier(features)
        return logits


# ============================================================================
# Training
# ============================================================================

def train_epoch(model: nn.Module, dataloader: DataLoader,
                criterion: nn.Module, optimizer: optim.Optimizer,
                device: str, epoch: int, logger) -> float:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrics
        running_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Compute epoch metrics
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)

    logger.info(f"Epoch {epoch} - Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

    return epoch_loss


def validate(model: nn.Module, dataloader: DataLoader,
            criterion: nn.Module, device: str, epoch: int, logger) -> Tuple[float, float]:
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            logits = model(images)
            loss = criterion(logits, labels)

            # Metrics
            running_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute epoch metrics
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)

    logger.info(f"Epoch {epoch} - Val Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

    return epoch_loss, epoch_acc


def train(config: Config):
    """Main training loop."""
    # Setup
    set_seed(config.seed)
    logger = setup_logging(Path(config.log_dir) / "train.log")
    device = torch.device(config.device)

    # Create directories
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.results_dir).mkdir(parents=True, exist_ok=True)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)

    # Data
    dataloaders = get_dataloaders(config)

    # Model
    model = PlantDiseaseClassifier(config).to(device)
    logger.info(f"Model: {config.model_name}, Params: {sum(p.numel() for p in model.parameters()):,}")

    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    # Optimizer
    if config.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate,
                               weight_decay=config.weight_decay)
    elif config.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate,
                               weight_decay=config.weight_decay)
    elif config.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate,
                             momentum=0.9, weight_decay=config.weight_decay)

    # Scheduler
    if config.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.num_epochs
        )
    elif config.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1
        )
    elif config.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=5, factor=0.5
        )

    # Training loop
    best_acc = 0.0
    train_losses, val_losses, val_accs = [], [], []

    for epoch in range(1, config.num_epochs + 1):
        # Train
        train_loss = train_epoch(model, dataloaders['train'], criterion,
                                 optimizer, device, epoch, logger)

        # Validate
        val_loss, val_acc = validate(model, dataloaders['val'], criterion,
                                     device, epoch, logger)

        # Step scheduler
        if config.scheduler == "plateau":
            scheduler.step(val_acc)
        else:
            scheduler.step()

        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Save checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, best_acc,
                Path(config.checkpoint_dir) / f"best_model_seed{config.seed}.pth"
            )
            logger.info(f"New best model saved! Acc: {best_acc:.4f}")

        if epoch % config.save_interval == 0:
            save_checkpoint(
                model, optimizer, epoch, best_acc,
                Path(config.checkpoint_dir) / f"checkpoint_epoch{epoch}.pth"
            )

    # Save training history
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'val_acc': val_accs,
        'best_acc': best_acc
    }

    with open(Path(config.results_dir) / f"history_seed{config.seed}.json", 'w') as f:
        json.dump(history, f, indent=4)

    logger.info(f"Training complete! Best Val Acc: {best_acc:.4f}")

    return model, history


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(model: nn.Module, dataloader: DataLoader,
            device: str, logger) -> Dict[str, float]:
    """Comprehensive evaluation of model."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_macro': f1_score(all_labels, all_preds, average='macro'),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted'),
        'precision': precision_score(all_labels, all_preds, average='weighted'),
        'recall': recall_score(all_labels, all_preds, average='weighted'),
    }

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Log results
    logger.info("Evaluation Results:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    return metrics, cm, all_probs


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str],
                         save_path: str):
    """Plot and save confusion matrix."""
    import seaborn as sns

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    """Main entry point."""
    # Configuration
    config = Config()

    # Train
    print("Starting training...")
    model, history = train(config)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    logger = setup_logging(Path(config.log_dir) / "eval.log")
    device = torch.device(config.device)

    dataloaders = get_dataloaders(config)
    metrics, cm, probs = evaluate(model, dataloaders['test'], device, logger)

    # Save results
    with open(Path(config.results_dir) / f"test_metrics_seed{config.seed}.json", 'w') as f:
        json.dump(metrics, f, indent=4)

    np.save(Path(config.results_dir) / f"confusion_matrix_seed{config.seed}.npy", cm)

    print("\nExperiment complete!")
    print(f"Best Val Acc: {history['best_acc']:.4f}")
    print(f"Test Acc: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
