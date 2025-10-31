#!/usr/bin/env python3
"""
Standalone training script for Contrastive Learning + SVM

Usage:
    python train_contrastive_svm.py --data_root ./plantwildV2 --epochs 200
"""

import argparse
import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


# Set random seeds
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


set_seed(42)


# ==================== Data Loading ====================

class ContrastiveAugmentation:
    """Strong augmentation for contrastive learning"""

    def __init__(self, img_size=224, strength=0.5):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.8*strength, 0.8*strength, 0.8*strength, 0.2*strength)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(23, sigma=(0.1, 2.0))], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)


class StandardAugmentation:
    """Standard augmentation for supervised training"""

    def __init__(self, img_size=224, is_train=True):
        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __call__(self, x):
        return self.transform(x)


class PlantDataset(Dataset):
    """Dataset loader for plant disease images"""

    def __init__(self, root_dir, transform=None, mode='contrastive'):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.mode = mode

        self.samples = []
        self.class_to_idx = {}
        self.classes = []

        for idx, class_dir in enumerate(sorted(self.root_dir.iterdir())):
            if class_dir.is_dir():
                class_name = class_dir.name
                self.classes.append(class_name)
                self.class_to_idx[class_name] = idx

                for img_path in class_dir.glob('*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        self.samples.append((str(img_path), idx))

        print(f"Found {len(self.samples)} images from {len(self.classes)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            if self.mode == 'contrastive':
                view1, view2 = self.transform(image)
                return view1, view2, label
            else:
                image = self.transform(image)
                return image, label

        return image, label


# ==================== Models ====================

class ProjectionHead(nn.Module):
    """Projection head for contrastive learning"""

    def __init__(self, input_dim=512, hidden_dim=512, output_dim=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.projection(x)


class ContrastiveEncoder(nn.Module):
    """Encoder network for contrastive learning"""

    def __init__(self, base_model='resnet50', projection_dim=128, pretrained=True):
        super().__init__()

        if base_model == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            self.embedding_dim = resnet.fc.in_features
            self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        elif base_model == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            self.embedding_dim = resnet.fc.in_features
            self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        else:
            raise ValueError(f"Unknown base model: {base_model}")

        self.projection_head = ProjectionHead(
            input_dim=self.embedding_dim,
            hidden_dim=self.embedding_dim,
            output_dim=projection_dim
        )

    def forward(self, x, return_embedding=False):
        h = self.encoder(x)
        h = torch.flatten(h, 1)

        if return_embedding:
            return h

        z = self.projection_head(h)
        return z

    def get_embedding(self, x):
        with torch.no_grad():
            return self.forward(x, return_embedding=True)


class SVMClassifier:
    """SVM classifier wrapper"""

    def __init__(self, C=1.0, kernel='linear', max_iter=1000):
        if kernel == 'linear':
            self.svm = LinearSVC(C=C, max_iter=max_iter, dual=True)
        else:
            self.svm = SVC(C=C, kernel=kernel, max_iter=max_iter)

        self.scaler = StandardScaler()

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.svm.fit(X_scaled, y)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.svm.predict(X_scaled)

    def score(self, X, y):
        X_scaled = self.scaler.transform(X)
        return self.svm.score(X_scaled, y)


# ==================== Loss Functions ====================

class NTXentLoss(nn.Module):
    """NT-Xent loss for contrastive learning"""

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]

        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)

        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1),
            representations.unsqueeze(0),
            dim=2
        )

        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)

        positives = torch.cat([
            torch.diag(similarity_matrix, batch_size),
            torch.diag(similarity_matrix, -batch_size)
        ], dim=0).reshape(2 * batch_size, 1)

        negatives = similarity_matrix[~mask].reshape(2 * batch_size, -1)

        logits = torch.cat([positives, negatives], dim=1) / self.temperature

        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z_i.device)

        loss = F.cross_entropy(logits, labels)

        return loss


# ==================== Training Functions ====================

def train_contrastive_epoch(model, dataloader, optimizer, criterion, device):
    """Train one epoch of contrastive learning"""
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc='Training')
    for view1, view2, _ in pbar:
        view1, view2 = view1.to(device), view2.to(device)

        z1 = model(view1)
        z2 = model(view2)

        loss = criterion(z1, z2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(dataloader)


def extract_features(model, dataloader, device):
    """Extract features from pretrained encoder"""
    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Extracting features'):
            images = images.to(device)
            features = model.get_embedding(images)
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())

    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    return features, labels


# ==================== Main Training Pipeline ====================

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    print("\n" + "="*70)
    print("CONTRASTIVE LEARNING + SVM FOR PLANT DISEASE DETECTION")
    print("="*70)

    # Load datasets
    print(f"\n[1/4] Loading dataset from {args.data_root}...")

    contrastive_dataset = PlantDataset(
        root_dir=args.data_root,
        transform=ContrastiveAugmentation(args.img_size),
        mode='contrastive'
    )

    supervised_dataset = PlantDataset(
        root_dir=args.data_root,
        transform=StandardAugmentation(args.img_size, is_train=True),
        mode='supervised'
    )

    supervised_dataset_test = PlantDataset(
        root_dir=args.data_root,
        transform=StandardAugmentation(args.img_size, is_train=False),
        mode='supervised'
    )

    num_classes = len(contrastive_dataset.classes)
    classes = contrastive_dataset.classes

    # Split datasets
    train_size = int(0.7 * len(supervised_dataset))
    val_size = len(supervised_dataset) - train_size
    train_dataset, _ = random_split(supervised_dataset, [train_size, val_size])
    _, test_dataset = random_split(supervised_dataset_test, [train_size, val_size])

    # Create dataloaders
    contrastive_loader = DataLoader(
        contrastive_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    print(f"  ✓ Classes: {num_classes}")
    print(f"  ✓ Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # Contrastive pretraining
    print(f"\n[2/4] Contrastive pretraining ({args.epochs} epochs)...")

    encoder = ContrastiveEncoder(
        base_model=args.backbone,
        projection_dim=args.projection_dim,
        pretrained=True
    ).to(device)

    criterion = NTXentLoss(temperature=args.temperature).to(device)
    optimizer = optim.Adam(
        encoder.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float('inf')

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        loss = train_contrastive_epoch(encoder, contrastive_loader, optimizer, criterion, device)
        scheduler.step()

        print(f"Loss: {loss:.4f}")

        if loss < best_loss:
            best_loss = loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': encoder.state_dict(),
                'loss': loss,
            }, os.path.join(args.checkpoint_dir, 'best_contrastive.pth'))
            print("✓ Saved best model")

    # Extract features
    print("\n[3/4] Extracting features...")
    X_train, y_train = extract_features(encoder, train_loader, device)
    X_test, y_test = extract_features(encoder, test_loader, device)

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # Train SVM
    print(f"\n[4/4] Training SVM (C={args.svm_c}, kernel={args.svm_kernel})...")
    svm = SVMClassifier(C=args.svm_c, kernel=args.svm_kernel, max_iter=args.svm_max_iter)
    svm.fit(X_train, y_train)

    train_acc = svm.score(X_train, y_train)
    test_acc = svm.score(X_test, y_test)

    print(f"\nResults:")
    print(f"  Train Accuracy: {train_acc*100:.2f}%")
    print(f"  Test Accuracy: {test_acc*100:.2f}%")

    # Classification report
    y_pred = svm.predict(X_test)
    print("\n" + "="*70)
    print("Classification Report")
    print("="*70)
    print(classification_report(y_test, y_pred, target_names=classes))

    # Save results
    results = {
        'test_accuracy': float(test_acc * 100),
        'train_accuracy': float(train_acc * 100),
        'num_classes': num_classes,
        'classes': classes,
        'args': vars(args)
    }

    with open(os.path.join(args.results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    # Save models
    torch.save(encoder.state_dict(), os.path.join(args.checkpoint_dir, 'encoder_final.pth'))

    import pickle
    with open(os.path.join(args.checkpoint_dir, 'svm_final.pkl'), 'wb') as f:
        pickle.dump(svm, f)

    print(f"\n✓ Results saved to {args.results_dir}")
    print(f"✓ Models saved to {args.checkpoint_dir}")
    print("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Contrastive SVM for Plant Disease Detection')

    # Data
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')

    # Model
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet18', 'resnet50'], help='Backbone model')
    parser.add_argument('--projection_dim', type=int, default=128, help='Projection dimension')

    # Training - Contrastive
    parser.add_argument('--epochs', type=int, default=200, help='Number of contrastive epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay')
    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature for NT-Xent loss')

    # SVM
    parser.add_argument('--svm_c', type=float, default=1.0, help='SVM penalty parameter')
    parser.add_argument('--svm_kernel', type=str, default='linear', choices=['linear', 'rbf'], help='SVM kernel')
    parser.add_argument('--svm_max_iter', type=int, default=1000, help='SVM max iterations')

    # Output
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--results_dir', type=str, default='./results', help='Results directory')

    args = parser.parse_args()

    main(args)
