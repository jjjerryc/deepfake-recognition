#!/usr/bin/env python3
"""
簡潔的訓練腳本 v4.0

透過 --config 指定模型配置，自動載入對應的模型和訓練參數。

使用方式:
    # 訓練 DINOv2
    python -m src.train_v2 --config configs/dino_vitl14.yaml
    
    # 訓練並指定種子
    python -m src.train_v2 --config configs/dino_vitl14.yaml --seed 100
    
    # 列出可用配置
    python -m src.train_v2 --list
"""

import argparse
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from .utils.config import load_config, list_configs, get_model_name
from .models.factory import create_model_from_config


# ============================================================================
# 資料集
# ============================================================================

class DeepfakeDataset(Dataset):
    """Deepfake 檢測數據集"""
    
    def __init__(self, data_dir: str, transform=None):
        self.transform = transform
        self.samples = []
        
        # 支援兩種目錄結構: fake/real 或 Fake/Real
        for label_name, label in [('fake', 0), ('Fake', 0), ('real', 1), ('Real', 1)]:
            label_dir = os.path.join(data_dir, label_name)
            if os.path.exists(label_dir):
                for f in os.listdir(label_dir):
                    if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                        self.samples.append((os.path.join(label_dir, f), label))
        
        random.shuffle(self.samples)
        print(f"📊 Loaded {len(self.samples)} samples from {data_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # 返回黑圖
            return torch.zeros(3, 224, 224), label


# ============================================================================
# 工具函數
# ============================================================================

def set_seed(seed: int):
    """設置隨機種子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms(config: Dict[str, Any], is_train: bool = True):
    """根據配置獲取資料增強"""
    size = config['data']['image_size']
    
    # 獲取 normalize 參數
    preprocess = config.get('preprocessing', {})
    mean = preprocess.get('mean', [0.485, 0.456, 0.406])
    std = preprocess.get('std', [0.229, 0.224, 0.225])
    
    if not is_train:
        return transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    
    # 訓練增強
    aug_config = config.get('augmentation', {})
    aug_mode = aug_config.get('mode', 'std')
    
    if aug_mode == 'hard':
        # Hard 增強：強力模糊
        aug_params = aug_config.get('hard', {})
        blur_cfg = aug_params.get('gaussian_blur', {})
        
        transform_list = [
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.RandomHorizontalFlip(p=aug_params.get('horizontal_flip', 0.5)),
            transforms.RandomApply([
                transforms.GaussianBlur(
                    kernel_size=blur_cfg.get('kernel_size', 5),
                    sigma=tuple(blur_cfg.get('sigma', [0.1, 5.0]))
                )
            ], p=blur_cfg.get('probability', 0.5)),
        ]
        
        # 顏色抖動
        cj = aug_params.get('color_jitter', {})
        if cj:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=cj.get('brightness', 0.2),
                    contrast=cj.get('contrast', 0.2),
                    saturation=cj.get('saturation', 0),
                    hue=cj.get('hue', 0)
                )
            )
    else:
        # 標準增強
        aug_params = aug_config.get('train', {})
        
        transform_list = [
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.RandomHorizontalFlip(p=aug_params.get('horizontal_flip', 0.5)),
        ]
        
        # 旋轉
        rotation = aug_params.get('rotation_degrees', 0)
        if rotation > 0:
            transform_list.append(transforms.RandomRotation(rotation))
        
        # 高斯模糊
        blur_cfg = aug_params.get('gaussian_blur', {})
        if blur_cfg:
            transform_list.append(
                transforms.RandomApply([
                    transforms.GaussianBlur(
                        kernel_size=blur_cfg.get('kernel_size', 3),
                        sigma=tuple(blur_cfg.get('sigma', [0.1, 2.0]))
                    )
                ], p=blur_cfg.get('probability', 0.2))
            )
        
        # 顏色抖動
        cj = aug_params.get('color_jitter', {})
        if cj:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=cj.get('brightness', 0.2),
                    contrast=cj.get('contrast', 0.2),
                    saturation=cj.get('saturation', 0.1),
                    hue=cj.get('hue', 0)
                )
            )
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    return transforms.Compose(transform_list)


def create_model(config: Dict[str, Any], device: torch.device):
    """根據配置創建模型"""
    model_config = config['model']
    model_type = model_config.get('type', 'dino')
    
    if model_type == 'dino':
        from .models.dino_model import DINOv2Classifier
        
        model = DINOv2Classifier(
            backbone=model_config.get('backbone', 'dinov2_vitl14'),
            num_classes=model_config.get('num_classes', 2),
            unfreeze_layers=model_config.get('unfreeze_layers', 2),
            unfreeze_norm=model_config.get('unfreeze_norm', True),
            pooling=model_config.get('pooling', 'cls'),
            hidden_dim=model_config.get('head', {}).get('hidden_dim', 512),
            dropout=model_config.get('head', {}).get('dropout', 0.4),
            use_batchnorm=model_config.get('head', {}).get('use_batchnorm', True),
        )
        
    elif model_type == 'convnext':
        from .models.convnext_model import ConvNeXtClassifier
        
        model = ConvNeXtClassifier(
            backbone=model_config.get('backbone', 'convnextv2_base'),
            num_classes=model_config.get('num_classes', 2),
            pretrained=model_config.get('pretrained', True),
            freeze_backbone=model_config.get('freeze_backbone', False),
            hidden_dim=model_config.get('head', {}).get('hidden_dim', 512),
            dropout=model_config.get('head', {}).get('dropout', 0.4),
            use_batchnorm=model_config.get('head', {}).get('use_batchnorm', True),
        )
        
    elif model_type == 'clip':
        from .models.clip_model import CLIPClassifier
        
        model = CLIPClassifier(
            clip_model=model_config.get('backbone', 'ViT-B-32'),
            pretrained=model_config.get('pretrained', 'openai'),
            num_classes=model_config.get('num_classes', 2),
            freeze_encoder=model_config.get('freeze_backbone', True),
            dropout=model_config.get('head', {}).get('dropout', 0.5),
            hidden_dim=model_config.get('head', {}).get('hidden_dim', 512),
        )
        
    elif model_type == 'efficientnet_dct':
        from .models.efficientnet_dct import EfficientNetDCT
        
        model = EfficientNetDCT(
            backbone_name=model_config.get('backbone', 'efficientnet_b4'),
            num_classes=model_config.get('num_classes', 2),
            pretrained=model_config.get('pretrained', True),
            dct_dim=model_config.get('dct', {}).get('dim', 128),
            fusion_dim=model_config.get('fusion', {}).get('dim', 512),
            dropout=model_config.get('fusion', {}).get('dropout', 0.3),
        )
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)


def get_dataloaders(config: Dict[str, Any], seed: int):
    """創建資料載入器"""
    data_config = config['data']
    train_dir = os.path.join(data_config['data_path'], 'train')
    
    train_transform = get_transforms(config, is_train=True)
    val_transform = get_transforms(config, is_train=False)
    
    full_dataset = DeepfakeDataset(train_dir, transform=train_transform)
    
    # 分割訓練/驗證
    val_size = int(len(full_dataset) * data_config['val_split'])
    train_size = len(full_dataset) - val_size
    
    torch.manual_seed(seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # 驗證集使用不同的 transform
    val_dataset_with_transform = DeepfakeDataset(train_dir, transform=val_transform)
    val_indices = val_dataset.indices
    val_dataset = torch.utils.data.Subset(val_dataset_with_transform, val_indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'] * 2,
        shuffle=False,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=True,
    )
    
    print(f"📊 Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    return train_loader, val_loader


# ============================================================================
# 訓練
# ============================================================================

def train(config: Dict[str, Any], seed: int):
    """主訓練函數"""
    model_name = get_model_name(config)
    train_config = config['training']
    
    print("\n" + "=" * 60)
    print(f"🚀 Training: {model_name}")
    print(f"   Seed: {seed}")
    print("=" * 60)
    
    # 設置種子
    set_seed(seed)
    
    # 設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥  Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # 創建模型
    model = create_model(config, device)
    
    # 創建資料載入器
    train_loader, val_loader = get_dataloaders(config, seed)
    
    # 優化器
    base_lr = train_config['learning_rate']
    backbone_lr_mult = train_config.get('backbone_lr_multiplier', 0.1)
    
    if hasattr(model, 'get_param_groups'):
        param_groups = model.get_param_groups(base_lr, backbone_lr_mult)
    else:
        param_groups = [{'params': model.parameters(), 'lr': base_lr}]
    
    optimizer = optim.AdamW(
        param_groups,
        weight_decay=train_config.get('weight_decay', 0.01)
    )
    
    # 學習率調度器
    scheduler_config = config.get('scheduler', {})
    scheduler_type = scheduler_config.get('type', 'CosineAnnealingWarmRestarts')
    
    if scheduler_type == 'CosineAnnealingWarmRestarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_config.get('T_0', 10),
            T_mult=scheduler_config.get('T_mult', 2),
            eta_min=scheduler_config.get('eta_min', 1e-6)
        )
    elif scheduler_type == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=scheduler_config.get('factor', 0.5),
            patience=scheduler_config.get('patience', 3),
            min_lr=scheduler_config.get('min_lr', 1e-6)
        )
    else:
        scheduler = None
    
    # 損失函數
    label_smoothing = train_config.get('label_smoothing', 0.1)
    num_classes = config['model'].get('num_classes', 2)
    
    if num_classes == 2:
        # Binary classification
        criterion = nn.BCEWithLogitsLoss()
        is_binary = True
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        is_binary = False
    
    # AMP
    use_amp = config.get('amp', {}).get('enabled', True)
    scaler = torch.amp.GradScaler('cuda') if use_amp and device.type == 'cuda' else None
    
    # 輸出目錄
    output_dir = Path(config['output']['save_dir']) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 訓練循環
    best_val_acc = 0
    patience_counter = 0
    patience = train_config.get('patience', 10)
    
    for epoch in range(train_config['epochs']):
        # 訓練
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_config['epochs']}")
        
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    if is_binary:
                        outputs = outputs.squeeze(-1)
                        loss = criterion(outputs, labels.float())
                    else:
                        loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    train_config.get('gradient_clip', 5.0)
                )
                
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                if is_binary:
                    outputs = outputs.squeeze(-1)
                    loss = criterion(outputs, labels.float())
                else:
                    loss = criterion(outputs, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    train_config.get('gradient_clip', 5.0)
                )
                optimizer.step()
            
            train_loss += loss.item()
            
            if is_binary:
                predicted = (torch.sigmoid(outputs) > 0.5).long()
            else:
                _, predicted = outputs.max(1)
            
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_acc = 100. * train_correct / train_total
        
        # 驗證
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                if scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = model(images)
                else:
                    outputs = model(images)
                
                if is_binary:
                    outputs = outputs.squeeze(-1)
                    predicted = (torch.sigmoid(outputs) > 0.5).long()
                else:
                    _, predicted = outputs.max(1)
                
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        # 更新學習率
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: Train {train_acc:.2f}% | Val {val_acc:.2f}% | LR {current_lr:.2e}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'epoch': epoch,
                'config': config,
                'seed': seed,
            }
            torch.save(checkpoint, output_dir / 'best_model.pth')
            print(f"  ✅ Saved best model (val_acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹ Early stopping at epoch {epoch+1}")
                break
    
    print(f"\n🏆 Best Validation Accuracy: {best_val_acc:.2f}%")
    return best_val_acc


# ============================================================================
# 主程式
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Deepfake Detection Training v4.0')
    
    parser.add_argument('--config', type=str, required=False,
                        help='Path to config file (e.g., configs/dino_vitl14.yaml)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config)')
    parser.add_argument('--list', action='store_true',
                        help='List available configs')
    
    args = parser.parse_args()
    
    if args.list:
        print("\n📋 Available configs:")
        for name in list_configs():
            print(f"   - {name}")
        return
    
    if not args.config:
        parser.error("--config is required (or use --list to see available configs)")
    
    # 載入配置
    config = load_config(args.config)
    
    # 覆蓋種子
    seed = args.seed if args.seed is not None else config['training'].get('seed', 42)
    
    # 開始訓練
    train(config, seed)


if __name__ == '__main__':
    main()
