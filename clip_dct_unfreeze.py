#!/usr/bin/env python3
"""
CLIP + DCT + Unfreeze Top Layers
================================

基於隊友的高分方法：clip-model-unfreezed 3 layer-DCT (86.07%)

特點：
1. CLIP ViT-B/32 作為 backbone
2. 解凍最後 3 層 transformer blocks
3. 加入 DCT 頻域特徵
4. 分層學習率（backbone 用較小 lr）

使用方式:
    python clip_dct_unfreeze.py --mode train
    python clip_dct_unfreeze.py --mode inference
    python clip_dct_unfreeze.py --mode both
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
    'batch_size': 64,          # 因為要訓練更多參數，batch 小一點
    'epochs': 30,
    'lr': 5e-5,                # 主學習率
    'backbone_lr_scale': 0.1,  # backbone 用 1/10 學習率
    'weight_decay': 0.01,
    'patience': 10,
    'warmup_epochs': 2,
    
    # 模型
    'unfreeze_layers': 3,      # 解凍最後 N 層
    'dct_dim': 128,
    'fusion_dim': 512,
    'dropout': 0.3,
    
    # 輸出
    'output_dir': './outputs/clip_dct_unfreeze',
    'seed': 42,
}

# CLIP 預處理參數
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

LABEL_MAP = {0: 'fake', 1: 'real'}


# ============================================================================
# DCT 特徵提取
# ============================================================================

class DCTFeatureExtractor(nn.Module):
    """DCT 頻域特徵提取器"""
    
    def __init__(self, output_dim: int = 128):
        super().__init__()
        self.output_dim = output_dim
        
        # 將 DCT 係數映射到特徵
        # 假設輸入是 224x224，DCT 後取 low frequency 區域
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, output_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        # 簡化版：直接用 conv 提取頻率相關特徵
        # 真正的 DCT 需要 torch_dct，這裡用 learnable 替代
        features = self.conv(x)
        features = self.fc(features)
        return features


# ============================================================================
# 模型
# ============================================================================

class CLIPDCTUnfreeze(nn.Module):
    """
    CLIP + DCT + Unfreeze Top Layers
    
    解凍 CLIP 最後 N 層，加入 DCT 特徵
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        unfreeze_layers: int = 3,
        dct_dim: int = 128,
        fusion_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.unfreeze_layers = unfreeze_layers
        
        # 載入 CLIP
        import open_clip
        print(f"Loading CLIP ViT-B/32...")
        self.clip, _, _ = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='openai'
        )
        self.encoder = self.clip.visual
        self.embed_dim = self.encoder.output_dim  # 512
        
        # 設定凍結策略
        self._setup_freeze()
        
        # DCT 特徵
        self.dct = DCTFeatureExtractor(output_dim=dct_dim)
        
        # 融合層
        total_dim = self.embed_dim + dct_dim  # 512 + 128 = 640
        self.fusion = nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Linear(total_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # 分類頭
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_classes),
        )
        
        # 刪除 text encoder
        del self.clip.transformer
        del self.clip.token_embedding
        del self.clip.ln_final
        if hasattr(self.clip, 'text_projection'):
            del self.clip.text_projection
        
        self._print_info()
    
    def _setup_freeze(self):
        """凍結除了最後 N 層以外的所有層"""
        # 先凍結所有
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # 獲取 transformer blocks
        if hasattr(self.encoder, 'transformer') and hasattr(self.encoder.transformer, 'resblocks'):
            blocks = self.encoder.transformer.resblocks
            total_blocks = len(blocks)
            
            # 解凍最後 N 層
            unfreeze_start = total_blocks - self.unfreeze_layers
            for i in range(unfreeze_start, total_blocks):
                for param in blocks[i].parameters():
                    param.requires_grad = True
            
            print(f"🔓 Unfreezing last {self.unfreeze_layers} / {total_blocks} transformer blocks")
        
        # 解凍 ln_post（最後的 LayerNorm）
        if hasattr(self.encoder, 'ln_post'):
            for param in self.encoder.ln_post.parameters():
                param.requires_grad = True
        
        # 解凍 proj（投影層）
        if hasattr(self.encoder, 'proj') and self.encoder.proj is not None:
            self.encoder.proj.requires_grad = True
    
    def _print_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"\n📊 Model Statistics:")
        print(f"   Total params: {total_params:,}")
        print(f"   Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        print(f"   Frozen: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
    
    def get_param_groups(self, base_lr: float, backbone_lr_scale: float = 0.1):
        """分層學習率"""
        param_groups = []
        
        # Backbone（解凍的層）
        backbone_params = [p for p in self.encoder.parameters() if p.requires_grad]
        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': base_lr * backbone_lr_scale,
                'name': 'clip_backbone'
            })
        
        # DCT
        param_groups.append({
            'params': list(self.dct.parameters()),
            'lr': base_lr,
            'name': 'dct'
        })
        
        # Fusion + Classifier
        param_groups.append({
            'params': list(self.fusion.parameters()) + list(self.classifier.parameters()),
            'lr': base_lr,
            'name': 'head'
        })
        
        return param_groups
    
    def forward(self, x):
        # CLIP features
        clip_features = self.encoder(x)
        if clip_features.dim() > 2:
            clip_features = clip_features.mean(dim=1)
        
        # DCT features
        dct_features = self.dct(x)
        
        # 融合
        combined = torch.cat([clip_features, dct_features], dim=1)
        fused = self.fusion(combined)
        
        # 分類
        logits = self.classifier(fused)
        return logits


# ============================================================================
# 資料
# ============================================================================

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
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
    train_transform, test_transform = get_transforms()
    
    train_dir = os.path.join(CONFIG['data_path'], 'train')
    full_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    
    print(f"📁 Classes: {full_dataset.classes}")
    print(f"📊 Total samples: {len(full_dataset)}")
    
    val_size = int(len(full_dataset) * CONFIG['val_split'])
    train_size = len(full_dataset) - val_size
    
    torch.manual_seed(CONFIG['seed'])
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # 驗證集用 test transform
    val_dataset.dataset = datasets.ImageFolder(train_dir, transform=test_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'] * 2,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    
    print(f"📊 Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    return train_loader, val_loader


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, test_dir, transform):
        self.test_dir = Path(test_dir)
        self.transform = transform
        
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        self.image_files = sorted(
            [f for f in self.test_dir.iterdir() if f.suffix.lower() in valid_extensions],
            key=lambda x: int(x.stem) if x.stem.isdigit() else x.stem
        )
        print(f"📁 Found {len(self.image_files)} test images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, img_path.stem


# ============================================================================
# 訓練
# ============================================================================

def train(model, train_loader, val_loader, device):
    print("\n" + "=" * 60)
    print("Training CLIP + DCT + Unfreeze")
    print("=" * 60)
    
    # 分層學習率
    param_groups = model.get_param_groups(
        base_lr=CONFIG['lr'],
        backbone_lr_scale=CONFIG['backbone_lr_scale']
    )
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=CONFIG['weight_decay'])
    
    # Warmup + Cosine scheduler
    total_steps = len(train_loader) * CONFIG['epochs']
    warmup_steps = len(train_loader) * CONFIG['warmup_epochs']
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(CONFIG['epochs']):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_acc = 100. * train_correct / train_total
        
        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}: Train {train_acc:.2f}% | Val {val_acc:.2f}% | LR {current_lr:.2e}")
        
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            os.makedirs(CONFIG['output_dir'], exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'epoch': epoch,
                'config': CONFIG,
            }, os.path.join(CONFIG['output_dir'], 'best_model.pth'))
            print(f"  ✅ Saved best model (val_acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                print(f"\n⏹ Early stopping at epoch {epoch+1}")
                break
    
    print(f"\n🏆 Best Validation Accuracy: {best_val_acc:.2f}%")
    return best_val_acc


# ============================================================================
# TTA Transforms
# ============================================================================

def get_tta_transforms():
    """獲取 TTA 用的多個 transforms"""
    base_size = CONFIG['image_size']
    
    tta_transforms = [
        # 1. 原始圖像
        transforms.Compose([
            transforms.Resize((base_size, base_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]),
        # 2. 水平翻轉
        transforms.Compose([
            transforms.Resize((base_size, base_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]),
        # 3. 輕微放大裁切
        transforms.Compose([
            transforms.Resize((int(base_size * 1.1), int(base_size * 1.1))),
            transforms.CenterCrop(base_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]),
        # 4. 輕微放大裁切 + 水平翻轉
        transforms.Compose([
            transforms.Resize((int(base_size * 1.1), int(base_size * 1.1))),
            transforms.CenterCrop(base_size),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]),
    ]
    
    return tta_transforms


# ============================================================================
# 推論
# ============================================================================

@torch.no_grad()
def inference(model, device, use_tta: bool = False, threshold: float = None):
    """
    推論函數
    
    Args:
        model: 訓練好的模型
        device: 運算設備
        use_tta: 是否使用 Test-Time Augmentation
        threshold: 分類閾值，None 則使用 median（強制 50/50）
    """
    print("\n" + "=" * 60)
    print(f"Inference {'with TTA' if use_tta else '(no TTA)'}")
    print("=" * 60)
    
    test_dir = os.path.join(CONFIG['data_path'], 'test')
    model.eval()
    
    import pandas as pd
    
    if use_tta:
        # TTA: 多個 transform 的預測平均
        tta_transforms = get_tta_transforms()
        print(f"📊 Using {len(tta_transforms)} TTA transforms")
        
        # 收集所有 filenames（只需要第一次）
        _, test_transform = get_transforms()
        first_dataset = TestDataset(test_dir, test_transform)
        all_filenames = [first_dataset.image_files[i].stem for i in range(len(first_dataset))]
        
        # 累積預測
        accumulated_probs = {fname: 0.0 for fname in all_filenames}
        
        for t_idx, tta_transform in enumerate(tta_transforms):
            dataset = TestDataset(test_dir, tta_transform)
            loader = DataLoader(
                dataset,
                batch_size=CONFIG['batch_size'] * 2,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
            
            for images, filenames in tqdm(loader, desc=f"TTA {t_idx+1}/{len(tta_transforms)}"):
                images = images.to(device)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                
                probs = F.softmax(outputs, dim=1)
                fake_probs = probs[:, 0].cpu().numpy()  # fake = class 0
                
                for fname, prob in zip(filenames, fake_probs):
                    accumulated_probs[fname] += prob
        
        # 平均
        all_probs = [accumulated_probs[fname] / len(tta_transforms) for fname in all_filenames]
        
    else:
        # 無 TTA：標準推論
        _, test_transform = get_transforms()
        dataset = TestDataset(test_dir, test_transform)
        loader = DataLoader(
            dataset,
            batch_size=CONFIG['batch_size'] * 2,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        
        all_filenames = []
        all_probs = []
        
        for images, filenames in tqdm(loader, desc="Inference"):
            images = images.to(device)
            
            with torch.amp.autocast('cuda'):
                outputs = model(images)
            
            probs = F.softmax(outputs, dim=1)
            fake_probs = probs[:, 0].cpu().numpy()  # fake = class 0
            
            all_filenames.extend(filenames)
            all_probs.extend(fake_probs)
    
    all_probs = np.array(all_probs)
    
    # 儲存機率（方便調 threshold）
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    prob_df = pd.DataFrame({
        'filename': all_filenames,
        'fake_prob': all_probs
    })
    prob_path = os.path.join(CONFIG['output_dir'], 'predictions_probs.csv')
    prob_df.to_csv(prob_path, index=False)
    print(f"✅ Probabilities saved to {prob_path}")
    
    # 統計機率分布
    print(f"\n📊 Probability Statistics:")
    print(f"   Mean: {all_probs.mean():.4f}")
    print(f"   Std: {all_probs.std():.4f}")
    print(f"   Median: {np.median(all_probs):.4f}")
    print(f"   Min: {all_probs.min():.4f}, Max: {all_probs.max():.4f}")
    
    # 決定 threshold
    if threshold is None:
        # 使用 median（強制 50/50 分布）
        threshold = np.median(all_probs)
        print(f"\n📊 Using median threshold: {threshold:.4f} (forces 50/50 split)")
    else:
        print(f"\n📊 Using specified threshold: {threshold:.4f}")
    
    # 生成 Kaggle 格式
    labels = ['fake' if p > threshold else 'real' for p in all_probs]
    
    submission_df = pd.DataFrame({
        'filename': all_filenames,
        'label': labels
    })
    
    # 根據是否使用 TTA 命名
    suffix = '_tta' if use_tta else ''
    submission_path = os.path.join(CONFIG['output_dir'], f'submission{suffix}.csv')
    submission_df.to_csv(submission_path, index=False)
    
    fake_count = labels.count('fake')
    real_count = labels.count('real')
    print(f"\n📊 Prediction Distribution:")
    print(f"   Fake: {fake_count} ({fake_count/len(labels)*100:.1f}%)")
    print(f"   Real: {real_count} ({real_count/len(labels)*100:.1f}%)")
    print(f"\n✅ Submission saved to {submission_path}")
    
    # 也生成不同 threshold 的版本
    print(f"\n📊 Generating submissions with different thresholds...")
    for t in [0.3, 0.4, 0.5, 0.6]:
        t_labels = ['fake' if p > t else 'real' for p in all_probs]
        t_df = pd.DataFrame({'filename': all_filenames, 'label': t_labels})
        t_path = os.path.join(CONFIG['output_dir'], f'submission{suffix}_t{int(t*10)}.csv')
        t_df.to_csv(t_path, index=False)
        t_fake = t_labels.count('fake')
        print(f"   t={t}: Fake={t_fake} ({t_fake/len(t_labels)*100:.1f}%) -> {t_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='CLIP + DCT + Unfreeze Training & Inference')
    
    # 模式
    parser.add_argument('--mode', type=str, default='both', 
                        choices=['train', 'inference', 'both'],
                        help='運行模式：train, inference, 或 both')
    
    # 模型設定
    parser.add_argument('--unfreeze-layers', type=int, default=None,
                        help='解凍最後 N 層（預設：3）')
    
    # 推論設定
    parser.add_argument('--tta', action='store_true',
                        help='啟用 Test-Time Augmentation')
    parser.add_argument('--threshold', type=float, default=None,
                        help='分類閾值（預設：使用 median 強制 50/50）')
    parser.add_argument('--model-path', type=str, default=None,
                        help='指定模型權重路徑（預設：使用 output_dir/best_model.pth）')
    
    args = parser.parse_args()
    
    if args.unfreeze_layers:
        CONFIG['unfreeze_layers'] = args.unfreeze_layers
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥 Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # 創建模型
    model = CLIPDCTUnfreeze(
        num_classes=2,
        unfreeze_layers=CONFIG['unfreeze_layers'],
        dct_dim=CONFIG['dct_dim'],
        fusion_dim=CONFIG['fusion_dim'],
        dropout=CONFIG['dropout'],
    ).to(device)
    
    if args.mode in ['train', 'both']:
        train_loader, val_loader = get_dataloaders()
        train(model, train_loader, val_loader, device)
    
    if args.mode in ['inference', 'both']:
        # 載入模型
        model_path = args.model_path or os.path.join(CONFIG['output_dir'], 'best_model.pth')
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            val_acc = checkpoint.get('val_acc', 'N/A')
            print(f"✅ Loaded model from {model_path}")
            if val_acc != 'N/A':
                print(f"   Val accuracy: {val_acc:.2f}%")
        else:
            print(f"⚠️ Model not found at {model_path}, using current weights")
        
        inference(model, device, use_tta=args.tta, threshold=args.threshold)


if __name__ == '__main__':
    main()
