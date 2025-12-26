#!/usr/bin/env python3
"""
Deepfake Detection Baseline
============================

極簡版訓練 + 推論腳本，使用 CLIP ViT-B/32 作為 backbone。

使用方式:
    # 訓練
    python baseline.py --mode train
    
    # 推論
    python baseline.py --mode inference
    
    # 訓練 + 推論
    python baseline.py --mode both

特點:
    - 單一檔案，約 300 行
    - CLIP ViT-B/32 凍結 backbone，只訓練分類頭
    - 輸出 Kaggle 格式 (filename, real/fake)
    - 支援 threshold 調整和 50/50 強制分布
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm

# ============================================================================
# 配置
# ============================================================================

CONFIG = {
    # 資料
    'data_path': './dataset',
    'image_size': 224,
    'val_split': 0.15,
    
    # 訓練
    'batch_size': 128,
    'epochs': 30,
    'lr': 1e-3,
    'weight_decay': 0.01,
    'patience': 8,
    
    # 輸出
    'output_dir': './outputs/baseline',
    'seed': 42,
}

# CLIP 預處理參數
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# 標籤映射（按字母順序：fake=0, real=1）
LABEL_MAP = {0: 'fake', 1: 'real'}


# ============================================================================
# 模型
# ============================================================================

class CLIPClassifier(nn.Module):
    """
    CLIP ViT-B/32 + 線性分類頭
    凍結 CLIP backbone，只訓練分類頭
    """
    
    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        
        # 載入 CLIP
        import open_clip
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='openai'
        )
        
        # 凍結 CLIP
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.eval()
        
        # 獲取特徵維度
        self.embed_dim = self.clip_model.visual.output_dim  # 512
        
        # 分類頭
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, num_classes)
        )
        
        print(f"✅ CLIP ViT-B/32 loaded (frozen)")
        print(f"   Embed dim: {self.embed_dim}")
        print(f"   Trainable params: {sum(p.numel() for p in self.classifier.parameters()):,}")
    
    def forward(self, x):
        with torch.no_grad():
            features = self.clip_model.encode_image(x).float()
        return self.classifier(features)


# ============================================================================
# 資料
# ============================================================================

def get_transforms():
    """獲取訓練和測試用的 transforms"""
    train_transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])
    
    return train_transform, test_transform


def get_dataloaders():
    """載入訓練和驗證資料"""
    train_transform, _ = get_transforms()
    
    # 使用 ImageFolder 自動讀取 train/fake 和 train/real
    train_dir = os.path.join(CONFIG['data_path'], 'train')
    full_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    
    # 確認類別順序
    print(f"📁 Classes: {full_dataset.classes}")  # 應該是 ['fake', 'real']
    print(f"📊 Total samples: {len(full_dataset)}")
    
    # 分割訓練/驗證
    val_size = int(len(full_dataset) * CONFIG['val_split'])
    train_size = len(full_dataset) - val_size
    
    torch.manual_seed(CONFIG['seed'])
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"   Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    
    return train_loader, val_loader


class TestDataset(torch.utils.data.Dataset):
    """測試資料集"""
    
    def __init__(self, test_dir, transform):
        self.test_dir = Path(test_dir)
        self.transform = transform
        
        valid_ext = {'.jpg', '.jpeg', '.png'}
        self.images = sorted(
            [f for f in self.test_dir.iterdir() if f.suffix.lower() in valid_ext],
            key=lambda x: int(x.stem) if x.stem.isdigit() else x.stem
        )
        print(f"📁 Test images: {len(self.images)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, img_path.stem


# ============================================================================
# 訓練
# ============================================================================

def train():
    """訓練模型"""
    print("\n" + "=" * 60)
    print("🚀 Training Baseline Model")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 資料
    train_loader, val_loader = get_dataloaders()
    
    # 模型
    model = CLIPClassifier(num_classes=2, dropout=0.3).to(device)
    
    # 優化器（只優化分類頭）
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=CONFIG['lr'],
        weight_decay=CONFIG['weight_decay']
    )
    
    # 損失函數
    criterion = nn.CrossEntropyLoss()
    
    # 訓練
    best_val_acc = 0.0
    patience_counter = 0
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    for epoch in range(CONFIG['epochs']):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # 儲存最佳模型
            save_path = os.path.join(CONFIG['output_dir'], 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, save_path)
            print(f"💾 Saved best model (val_acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                break
    
    print(f"\n✅ Training complete! Best Val Acc: {best_val_acc:.2f}%")
    return best_val_acc


# ============================================================================
# 推論
# ============================================================================

@torch.no_grad()
def inference(use_median_threshold: bool = True):
    """推論並生成提交檔案"""
    print("\n" + "=" * 60)
    print("🔮 Running Inference")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 載入模型
    model = CLIPClassifier(num_classes=2).to(device)
    checkpoint_path = os.path.join(CONFIG['output_dir'], 'best_model.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Model not found: {checkpoint_path}")
        print("   Please run training first: python baseline.py --mode train")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Loaded model (val_acc: {checkpoint['val_acc']:.2f}%)")
    
    # 資料
    _, test_transform = get_transforms()
    test_dir = os.path.join(CONFIG['data_path'], 'test')
    test_dataset = TestDataset(test_dir, test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    
    # 推論
    all_filenames = []
    all_probs = []
    
    for images, filenames in tqdm(test_loader, desc="Inference"):
        images = images.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        
        # 取 fake 的機率 (index 0)
        fake_probs = probs[:, 0].cpu().numpy()
        
        all_filenames.extend(filenames)
        all_probs.extend(fake_probs)
    
    all_probs = np.array(all_probs)
    
    # 決定 threshold
    if use_median_threshold:
        threshold = np.median(all_probs)
        print(f"📊 Using median threshold: {threshold:.4f} (forces 50/50 split)")
    else:
        threshold = 0.5
        print(f"📊 Using threshold: {threshold}")
    
    # 生成標籤
    labels = ['fake' if p > threshold else 'real' for p in all_probs]
    
    # 統計
    fake_count = labels.count('fake')
    real_count = labels.count('real')
    total = len(labels)
    
    print(f"\n📈 Distribution:")
    print(f"   Fake: {fake_count} ({fake_count/total*100:.1f}%)")
    print(f"   Real: {real_count} ({real_count/total*100:.1f}%)")
    
    # 儲存
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Kaggle 格式
    submission_path = os.path.join(CONFIG['output_dir'], 'submission.csv')
    with open(submission_path, 'w') as f:
        f.write("filename,label\n")
        for fname, label in zip(all_filenames, labels):
            f.write(f"{fname},{label}\n")
    print(f"\n✅ Saved: {submission_path}")
    
    # 機率版本（方便調整 threshold）
    probs_path = os.path.join(CONFIG['output_dir'], 'predictions_probs.csv')
    with open(probs_path, 'w') as f:
        f.write("filename,fake_prob\n")
        for fname, prob in zip(all_filenames, all_probs):
            f.write(f"{fname},{prob:.6f}\n")
    print(f"✅ Saved: {probs_path}")
    
    print("\n✅ Inference complete!")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Deepfake Detection Baseline')
    parser.add_argument('--mode', type=str, default='both',
                        choices=['train', 'inference', 'both'],
                        help='train, inference, or both')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Custom threshold (default: use median for 50/50)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔬 Deepfake Detection Baseline")
    print("   Model: CLIP ViT-B/32 (frozen) + Linear Classifier")
    print("=" * 60)
    
    if args.mode in ['train', 'both']:
        train()
    
    if args.mode in ['inference', 'both']:
        use_median = args.threshold is None
        inference(use_median_threshold=use_median)


if __name__ == '__main__':
    main()
