#!/usr/bin/env python3
"""
Inference script for Contrastive SVM model

Usage:
    python inference.py --image path/to/image.jpg --checkpoint ./checkpoints
"""

import argparse
import os
import json
import pickle
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np


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

    def __init__(self, base_model='resnet50', projection_dim=128, pretrained=False):
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


def load_models(checkpoint_dir, device):
    """Load encoder and SVM from checkpoint"""

    # Load config
    config_path = os.path.join(checkpoint_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        img_size = config['IMG_SIZE']
        classes = config['classes']
    else:
        print("Warning: config.json not found, using defaults")
        img_size = 224
        classes = None

    # Load encoder
    encoder = ContrastiveEncoder(base_model='resnet50', projection_dim=128, pretrained=False)

    encoder_path = os.path.join(checkpoint_dir, 'encoder_final.pth')
    if os.path.exists(encoder_path):
        checkpoint = torch.load(encoder_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            encoder.load_state_dict(checkpoint['model_state_dict'])
        else:
            encoder.load_state_dict(checkpoint)
        print(f"✓ Loaded encoder from {encoder_path}")
    else:
        raise FileNotFoundError(f"Encoder not found at {encoder_path}")

    encoder = encoder.to(device)
    encoder.eval()

    # Load SVM
    svm_path = os.path.join(checkpoint_dir, 'svm_final.pkl')
    if os.path.exists(svm_path):
        with open(svm_path, 'rb') as f:
            svm = pickle.load(f)
        print(f"✓ Loaded SVM from {svm_path}")
    else:
        raise FileNotFoundError(f"SVM not found at {svm_path}")

    # Load classes from results if not in config
    if classes is None:
        results_path = os.path.join(os.path.dirname(checkpoint_dir), 'results', 'results.json')
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results = json.load(f)
                classes = results.get('classes', [f"Class_{i}" for i in range(10)])
        else:
            print("Warning: Could not load class names")
            classes = [f"Class_{i}" for i in range(10)]

    return encoder, svm, classes, img_size


def preprocess_image(image_path, img_size=224):
    """Preprocess image for inference"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    return image, image_tensor


def predict(encoder, svm, image_tensor, classes, device):
    """Make prediction on image"""
    image_tensor = image_tensor.to(device)

    # Extract features
    with torch.no_grad():
        features = encoder.get_embedding(image_tensor)
        features = features.cpu().numpy()

    # Predict with SVM
    prediction = svm.predict(features)[0]
    predicted_class = classes[prediction]

    return predicted_class, prediction


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load models
    print("Loading models...")
    encoder, svm, classes, img_size = load_models(args.checkpoint, device)
    print(f"✓ Loaded {len(classes)} classes: {classes}\n")

    # Process single image or batch
    if args.image:
        # Single image prediction
        print(f"Processing image: {args.image}")
        image, image_tensor = preprocess_image(args.image, img_size)
        predicted_class, pred_idx = predict(encoder, svm, image_tensor, classes, device)

        print(f"\n{'='*50}")
        print(f"Prediction: {predicted_class}")
        print(f"Class Index: {pred_idx}")
        print(f"{'='*50}")

        # Save result if requested
        if args.save_result:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 6))
            plt.imshow(image)
            plt.axis('off')
            plt.title(f'Predicted: {predicted_class}', fontsize=16, fontweight='bold')
            plt.tight_layout()

            output_path = args.image.replace(os.path.splitext(args.image)[1], '_prediction.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\n✓ Saved result to {output_path}")
            plt.close()

    elif args.batch_dir:
        # Batch prediction
        print(f"Processing images from: {args.batch_dir}")

        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(list(Path(args.batch_dir).glob(ext)))

        if len(image_files) == 0:
            print(f"No images found in {args.batch_dir}")
            return

        print(f"Found {len(image_files)} images\n")

        results = []
        for img_path in image_files:
            try:
                _, image_tensor = preprocess_image(str(img_path), img_size)
                predicted_class, pred_idx = predict(encoder, svm, image_tensor, classes, device)

                results.append({
                    'filename': img_path.name,
                    'prediction': predicted_class,
                    'class_idx': int(pred_idx)
                })

                print(f"✓ {img_path.name}: {predicted_class}")

            except Exception as e:
                print(f"✗ Error processing {img_path.name}: {e}")

        # Save batch results
        if args.save_result:
            output_path = os.path.join(args.batch_dir, 'predictions.json')
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"\n✓ Saved batch results to {output_path}")

    else:
        print("Error: Please provide either --image or --batch_dir")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference with Contrastive SVM model')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', type=str, help='Path to single image')
    group.add_argument('--batch_dir', type=str, help='Path to directory with images')

    parser.add_argument('--checkpoint', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--save_result', action='store_true', help='Save prediction result')

    args = parser.parse_args()

    # Import Path if needed
    if args.batch_dir:
        from pathlib import Path

    main(args)
