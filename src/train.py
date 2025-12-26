"""
多模型訓練腳本 v3.3

支援多種 backbone 架構的完整訓練流程：
- 動態模型載入（EfficientNet-B0~B4 等）
- 分層學習率（backbone 和 classifier 使用不同學習率）
- 凍結 backbone 選項（類似 CLIP 方法）
- Mixup/CutMix 數據增強
- ImageNet 標準化預處理
- 混合精度訓練 (AMP)
- 學習率調度和早停機制

使用方式:
    python -m src.train
    python -m src.train --config config.json
    python -m src.train --model efficientnet_b0  # 快速測試
    python -m src.train --freeze-backbone        # 凍結 backbone
"""

import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from .models import create_model, create_model_from_config, list_available_models


# ============== Mixup / CutMix ==============

def mixup_data(x, y, alpha=0.4):
    """Mixup: 混合兩個樣本的圖像和標籤"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """CutMix: 將一個樣本的區域貼到另一個樣本上"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    # 計算裁切區域
    W, H = x.size(2), x.size(3)
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    mixed_x = x.clone()
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # 調整 lambda 以反映實際區域比例
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup/CutMix 的損失計算"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class DeepfakeDataset(Dataset):
    """Deepfake 檢測數據集"""
    
    def __init__(self, data_dir: str, transform=None):
        self.transform = transform
        self.samples = []
        
        # 載入 fake 圖像 (label=0)
        fake_dir = os.path.join(data_dir, 'fake')
        if os.path.exists(fake_dir):
            for f in os.listdir(fake_dir):
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.samples.append((os.path.join(fake_dir, f), 0))
        
        # 載入 real 圖像 (label=1)
        real_dir = os.path.join(data_dir, 'real')
        if os.path.exists(real_dir):
            for f in os.listdir(real_dir):
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.samples.append((os.path.join(real_dir, f), 1))
        
        random.shuffle(self.samples)
        print(f"Loaded {len(self.samples)} samples from {data_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def set_seed(seed: int = 42):
    """設置隨機種子以確保可重現性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """載入配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_transforms(config: dict, model=None, is_train: bool = True):
    """
    獲取數據轉換
    
    使用 ImageNet 標準化參數，配合預訓練模型。
    
    Args:
        config: 配置字典
        model: 模型實例（用於獲取預處理參數）
        is_train: 是否為訓練模式
    """
    size = config['data']['image_size']
    
    # 獲取標準化參數
    preprocess_config = config.get('preprocessing', {})
    if preprocess_config.get('use_model_default', True) and model is not None:
        model_preprocess = model.get_preprocessing_config()
        mean = model_preprocess['mean']
        std = model_preprocess['std']
    else:
        normalize_cfg = preprocess_config.get('normalize', {})
        mean = normalize_cfg.get('mean', [0.485, 0.456, 0.406])
        std = normalize_cfg.get('std', [0.229, 0.224, 0.225])
    
    if is_train:
        aug_cfg = config['augmentation']['train']
        transform_list = []
        
        # 隨機裁切
        if 'random_crop_scale' in aug_cfg:
            scale = aug_cfg['random_crop_scale']
            transform_list.append(
                transforms.RandomResizedCrop(size, scale=tuple(scale), ratio=(0.9, 1.1))
            )
        else:
            transform_list.append(transforms.Resize((size, size)))
        
        # 水平翻轉
        if aug_cfg.get('horizontal_flip', 0) > 0:
            transform_list.append(transforms.RandomHorizontalFlip(aug_cfg['horizontal_flip']))
        
        # 垂直翻轉
        if aug_cfg.get('vertical_flip', 0) > 0:
            transform_list.append(transforms.RandomVerticalFlip(aug_cfg['vertical_flip']))
        
        # 旋轉
        rotation = aug_cfg.get('rotation_degrees', 0)
        if rotation > 0:
            transform_list.append(transforms.RandomRotation(rotation))
        
        # 顏色抖動
        brightness = aug_cfg.get('brightness', 0)
        contrast = aug_cfg.get('contrast', 0)
        saturation = aug_cfg.get('saturation', 0)
        hue = aug_cfg.get('hue', 0)
        if any([brightness, contrast, saturation, hue]):
            transform_list.append(
                transforms.ColorJitter(brightness, contrast, saturation, hue)
            )
        
        # 轉為 Tensor 並標準化
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=mean, std=std))
        
        # Random Erasing
        random_erasing = aug_cfg.get('random_erasing', 0)
        if random_erasing > 0:
            transform_list.append(transforms.RandomErasing(p=random_erasing))
        
    else:
        transform_list = [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    
    return transforms.Compose(transform_list)


class EarlyStopping:
    """早停機制"""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_score: float) -> bool:
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0
        return self.early_stop


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler=None,
    gradient_clip: float = 0.0,
    scheduler=None,
    step_each_batch: bool = False,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    mix_prob: float = 0.5
) -> Tuple[float, float]:
    """
    訓練一個 epoch
    
    Args:
        scheduler: 學習率調度器（用於 OneCycleLR）
        step_each_batch: 是否每個 batch 都更新學習率
        mixup_alpha: Mixup 的 alpha 參數（0 表示不使用）
        cutmix_alpha: CutMix 的 alpha 參數（0 表示不使用）
        mix_prob: 使用 Mixup/CutMix 的機率
    
    Returns:
        avg_loss, accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    use_mix = mixup_alpha > 0 or cutmix_alpha > 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # 決定是否使用 Mixup/CutMix
        do_mix = use_mix and (np.random.random() < mix_prob)
        
        if do_mix:
            # 隨機選擇 Mixup 或 CutMix
            if cutmix_alpha > 0 and (mixup_alpha == 0 or np.random.random() > 0.5):
                images, labels_a, labels_b, lam = cutmix_data(images, labels, cutmix_alpha)
            else:
                images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha)
        
        optimizer.zero_grad()
        
        # 混合精度訓練
        if scaler is not None:
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                if do_mix:
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            
            if gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            if do_mix:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
        
        # OneCycleLR 每個 batch 更新學習率
        if step_each_batch and scheduler is not None:
            scheduler.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 顯示當前學習率
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%',
            'lr': f'{current_lr:.2e}'
        })
    
    avg_loss = running_loss / total
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> Tuple[float, float]:
    """
    驗證模型
    
    Returns:
        avg_loss, accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    avg_loss = running_loss / total
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def create_optimizer(model, config: dict) -> optim.Optimizer:
    """
    創建優化器，支援分層學習率
    
    backbone 使用較小的學習率，classifier 使用較大的學習率
    """
    base_lr = config['training']['learning_rate']
    backbone_multiplier = config['training'].get('backbone_lr_multiplier', 0.1)
    weight_decay = config['training']['weight_decay']
    
    # 檢查模型是否支援分層學習率
    if hasattr(model, 'get_param_groups'):
        # 嘗試使用新的 API（DCT 模型）
        import inspect
        sig = inspect.signature(model.get_param_groups)
        params = list(sig.parameters.keys())
        
        if 'base_lr' in params:
            # DCT 模型使用 base_lr 和 backbone_lr_scale
            param_groups = model.get_param_groups(
                base_lr=base_lr,
                backbone_lr_scale=backbone_multiplier
            )
        else:
            # 標準 API
            param_groups = model.get_param_groups(
                backbone_lr=base_lr * backbone_multiplier,
                classifier_lr=base_lr
            )
        
        print(f"Using layered learning rates:")
        for pg in param_groups:
            name = pg.get('name', 'unnamed')
            lr = pg.get('lr', base_lr)
            print(f"  {name}: {lr:.2e}")
    else:
        param_groups = model.parameters()
        print(f"Using uniform learning rate: {base_lr:.2e}")
    
    optimizer = optim.AdamW(
        param_groups,
        lr=base_lr,  # 這個會被 param_groups 覆蓋
        weight_decay=weight_decay,
        betas=tuple(config['optimizer']['betas']),
        eps=config['optimizer']['eps']
    )
    
    return optimizer


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Model Training Script for Deepfake Detection'
    )
    parser.add_argument(
        '--config', type=str, default='config.json',
        help='配置文件路徑'
    )
    parser.add_argument(
        '--model', type=str, default=None,
        help=f'模型名稱（覆蓋配置文件）。可選: {list_available_models()}'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='恢復訓練的檢查點路徑'
    )
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='訓練 epochs（覆蓋配置文件）'
    )
    parser.add_argument(
        '--batch-size', type=int, default=None,
        help='Batch size（覆蓋配置文件）'
    )
    parser.add_argument(
        '--lr', type=float, default=None,
        help='學習率（覆蓋配置文件）'
    )
    parser.add_argument(
        '--freeze-backbone', action='store_true',
        help='凍結 backbone，只訓練分類器（類似 CLIP 方法）'
    )
    
    args = parser.parse_args()
    
    # 載入配置
    config = load_config(args.config)
    
    # 命令行參數覆蓋模型名稱（需要先處理，才能判斷模型類型）
    if args.model:
        config['model']['name'] = args.model
    
    # 根據模型類型選擇對應的訓練參數
    model_name = config['model']['name']
    is_clip_model = model_name.startswith('clip_')
    
    if is_clip_model:
        # 使用 CLIP 專用參數（如果存在）
        if 'training_clip' in config:
            print(f"[Info] Detected CLIP model: {model_name}")
            print(f"[Info] Using CLIP-specific training parameters")
            # 合併 CLIP 參數（CLIP 參數覆蓋預設參數）
            for key, value in config['training_clip'].items():
                if not key.startswith('_'):  # 跳過註解
                    config['training'][key] = value
        
        if 'scheduler_clip' in config:
            print(f"[Info] Using CLIP-specific scheduler parameters")
            for key, value in config['scheduler_clip'].items():
                if not key.startswith('_'):
                    config['scheduler'][key] = value
    
    # 其他命令行參數覆蓋
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.freeze_backbone:
        config['training']['freeze_backbone'] = True
    
    print("=" * 60)
    print(f"Project: {config['project']['name']} v{config['project']['version']}")
    print(f"Config: {args.config}")
    print("=" * 60)
    
    # 設置隨機種子
    set_seed(config['training']['seed'])
    
    # 設置設備
    device_name = config.get('hardware', {}).get('device', 'cuda')
    if device_name == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif device_name == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Device: Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
        print("Device: CPU")
    
    # 混合精度訓練
    use_amp = config.get('hardware', {}).get('mixed_precision', False) and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    print(f"Mixed Precision: {'Enabled' if use_amp else 'Disabled'}")
    
    # 初始化模型（先初始化，才能知道模型名稱）
    print("\n" + "-" * 40)
    print("Initializing Model...")
    model = create_model_from_config(config)
    model = model.to(device)
    
    # 模型信息
    model_name = config['model'].get('name', 'unknown')
    total_params = model.count_parameters(trainable_only=False)
    trainable_params = model.count_parameters(trainable_only=True)
    print(f"Model: {model_name}")
    print(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # 創建模型專屬輸出目錄
    base_output_dir = Path(config['output']['output_dir'])
    output_dir = base_output_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = Path(config['logging']['log_dir']) / model_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Log directory: {log_dir}")
    
    # 凍結 backbone（類似 CLIP 方法）
    freeze_backbone = config['training'].get('freeze_backbone', False)
    if freeze_backbone:
        print("🔒 Freezing backbone (CLIP-style training)")
        if hasattr(model, 'freeze_backbone'):
            model.freeze_backbone()
        elif hasattr(model, 'backbone'):
            for param in model.backbone.parameters():
                param.requires_grad = False
        else:
            print("  Warning: Model does not have a 'backbone' attribute")
    
    # 獲取數據轉換（使用模型的預處理配置）
    train_transform = get_transforms(config, model=model, is_train=True)
    val_transform = get_transforms(config, model=model, is_train=False)
    
    # 載入數據
    print("\n" + "-" * 40)
    print("Loading Data...")
    train_dir = os.path.join(config['data']['data_path'], 'train')
    full_dataset = DeepfakeDataset(train_dir, transform=train_transform)
    
    # 分割訓練集和驗證集
    val_split = config['data']['val_split']
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config['training']['seed'])
    )
    
    # 為驗證集更換 transform
    class TransformDataset:
        def __init__(self, dataset, transform):
            self.dataset = dataset
            self.transform = transform
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, idx):
            image, label = self.dataset.dataset.samples[self.dataset.indices[idx]]
            image = Image.open(image).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
    
    val_dataset_transformed = TransformDataset(val_dataset, val_transform)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Image size: {config['data']['image_size']}x{config['data']['image_size']}")
    
    # 創建數據載入器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset_transformed,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    # 損失函數
    label_smoothing = config['training'].get('label_smoothing', 0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    # 優化器（支援分層學習率）
    print("\n" + "-" * 40)
    print("Setting up Optimizer...")
    optimizer = create_optimizer(model, config)
    
    # 計算每個 epoch 的步數（用於 OneCycleLR）
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * config['training']['epochs']
    
    # 學習率調度器
    scheduler_type = config['scheduler']['type']
    step_each_batch = False  # 是否每個 batch 更新一次
    step_with_metrics = False  # 是否需要傳入 metrics 來 step
    
    if scheduler_type == 'OneCycleLR':
        # OneCycleLR：每個 batch 都要 step
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['scheduler'].get('max_lr', config['training']['learning_rate']),
            total_steps=total_steps,
            pct_start=config['scheduler'].get('pct_start', 0.1),
            anneal_strategy=config['scheduler'].get('anneal_strategy', 'cos'),
            div_factor=config['scheduler'].get('div_factor', 25),
            final_div_factor=config['scheduler'].get('final_div_factor', 1000),
        )
        step_each_batch = True
        print(f"Scheduler: {scheduler_type} (step per batch)")
        print(f"  Max LR: {config['scheduler'].get('max_lr', config['training']['learning_rate']):.2e}")
        print(f"  Total steps: {total_steps}")
    elif scheduler_type == 'ReduceLROnPlateau':
        # ReduceLROnPlateau：根據驗證指標自適應調整
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config['scheduler'].get('mode', 'max'),  # 'max' for accuracy, 'min' for loss
            factor=config['scheduler'].get('factor', 0.5),
            patience=config['scheduler'].get('patience', 3),
            min_lr=config['scheduler'].get('min_lr', 1e-6),
            threshold=config['scheduler'].get('threshold', 0.001)
        )
        step_with_metrics = True
        print(f"Scheduler: {scheduler_type}")
        print(f"  Mode: {config['scheduler'].get('mode', 'max')}")
        print(f"  Factor: {config['scheduler'].get('factor', 0.5)}")
        print(f"  Patience: {config['scheduler'].get('patience', 3)}")
        print(f"  Min LR: {config['scheduler'].get('min_lr', 1e-6):.2e}")
    elif scheduler_type == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['scheduler']['T_max'],
            eta_min=config['scheduler']['eta_min']
        )
        print(f"Scheduler: {scheduler_type}")
    elif scheduler_type == 'CosineAnnealingWarmRestarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config['scheduler']['T_0'],
            T_mult=config['scheduler'].get('T_mult', 1),
            eta_min=config['scheduler']['eta_min']
        )
        print(f"Scheduler: {scheduler_type}")
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        print(f"Scheduler: StepLR (default)")
    
    # 早停機制
    early_stopping = EarlyStopping(
        patience=config['training']['patience'],
        min_delta=config['training']['min_delta']
    )
    
    # 梯度裁剪
    gradient_clip = config['training'].get('gradient_clip', 0.0)
    
    # 恢復訓練
    start_epoch = 0
    best_val_acc = 0.0
    
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0)
        print(f"Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.2f}%")
    
    # 訓練記錄
    history = {
        'model_name': model_name,
        'config': {
            'batch_size': config['training']['batch_size'],
            'learning_rate': config['training']['learning_rate'],
            'image_size': config['data']['image_size'],
        },
        'epochs': []
    }
    
    # 日誌文件路徑（實時寫入）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"training_log_{model_name}_{timestamp}.txt"
    history_file = log_dir / f"history_{model_name}_{timestamp}.json"
    
    def write_log(message: str, also_print: bool = True):
        """寫入日誌文件並可選打印"""
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
        if also_print:
            print(message)
    
    def save_history():
        """保存訓練歷史到 JSON"""
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    
    # 開始訓練
    header = "\n" + "=" * 60 + "\n"
    header += "Starting Training\n"
    header += f"  Model: {model_name}\n"
    header += f"  Epochs: {config['training']['epochs']}\n"
    header += f"  Batch size: {config['training']['batch_size']}\n"
    header += f"  Learning rate: {config['training']['learning_rate']}\n"
    header += f"  Label smoothing: {label_smoothing}\n"
    header += f"  Scheduler: {scheduler_type}\n"
    # Mixup/CutMix 配置
    aug_config = config.get('augmentation', {}).get('train', {})
    mixup_alpha = aug_config.get('mixup_alpha', 0.0)
    cutmix_alpha = aug_config.get('cutmix_alpha', 0.0)
    mix_prob = aug_config.get('mix_prob', 0.5)
    
    if mixup_alpha > 0 or cutmix_alpha > 0:
        print(f"Mixup/CutMix: mixup_alpha={mixup_alpha}, cutmix_alpha={cutmix_alpha}, prob={mix_prob}")
    
    header += "=" * 60 + "\n"
    write_log(header)
    
    start_time = time.time()
    
    for epoch in range(start_epoch, config['training']['epochs']):
        epoch_start = time.time()
        
        # 訓練（傳入 scheduler 用於 OneCycleLR，傳入 mixup/cutmix 參數）
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch + 1, scaler=scaler, gradient_clip=gradient_clip,
            scheduler=scheduler, step_each_batch=step_each_batch,
            mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha, mix_prob=mix_prob
        )
        
        # 驗證
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch + 1
        )
        
        # 更新學習率（根據 scheduler 類型）
        current_lr = optimizer.param_groups[0]['lr']
        if step_with_metrics:
            # ReduceLROnPlateau 需要傳入 metric
            scheduler.step(val_acc)
        elif not step_each_batch:
            # 其他 epoch-based scheduler
            scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # 記錄本 epoch 結果
        epoch_result = {
            'epoch': epoch + 1,
            'train_loss': round(train_loss, 4),
            'train_acc': round(train_acc, 2),
            'val_loss': round(val_loss, 4),
            'val_acc': round(val_acc, 2),
            'lr': current_lr,
            'time': round(epoch_time, 1)
        }
        history['epochs'].append(epoch_result)
        
        # 構建日誌行
        is_best = val_acc > best_val_acc
        log_line = (f"Epoch {epoch + 1:3d}/{config['training']['epochs']} | "
                    f"Train: {train_acc:.2f}% ({train_loss:.4f}) | "
                    f"Val: {val_acc:.2f}% ({val_loss:.4f}) | "
                    f"LR: {current_lr:.2e} | "
                    f"Time: {epoch_time:.1f}s")
        
        if is_best:
            log_line += " | ★ Best!"
        
        # 寫入日誌並打印
        write_log(log_line)
        
        # 即時保存訓練歷史（每個 epoch 都存）
        save_history()
        
        # 保存最佳模型
        if is_best:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_name': model_name,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'config': config
            }, output_dir / 'best_model.pth')
        
        # 定期保存檢查點
        checkpoint_interval = config['logging'].get('checkpoint_interval', 5)
        if (epoch + 1) % checkpoint_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_name': model_name,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'config': config
            }, output_dir / 'latest_checkpoint.pth')
        
        # 早停檢查
        if early_stopping(val_acc):
            write_log(f"\n⚠ Early stopping triggered at epoch {epoch + 1}")
            break
    
    # 最終摘要
    history['summary'] = {
        'best_val_acc': round(best_val_acc, 2),
        'total_epochs': len(history['epochs']),
        'total_time_minutes': round((time.time() - start_time) / 60, 2)
    }
    save_history()
    
    # 訓練完成
    total_time = time.time() - start_time
    
    summary = "\n" + "=" * 60 + "\n"
    summary += "Training Complete!\n"
    summary += f"  Model: {model_name}\n"
    summary += f"  Total Time: {total_time / 60:.2f} minutes\n"
    summary += f"  Best Validation Accuracy: {best_val_acc:.2f}%\n"
    summary += f"  Best model: {output_dir / 'best_model.pth'}\n"
    summary += f"  Training log: {log_file}\n"
    summary += f"  History JSON: {history_file}\n"
    summary += "=" * 60
    write_log(summary)


if __name__ == "__main__":
    main()
