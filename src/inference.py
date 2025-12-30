#!/usr/bin/env python3
"""
推論腳本 v4.0

單模型推論，支援 TTA 和多種閾值輸出。

使用方式:
    # 基本推論
    python -m src.inference_v2 --config configs/dino_vitl14.yaml
    
    # 啟用 TTA
    python -m src.inference_v2 --config configs/dino_vitl14.yaml --tta
    
    # 指定 checkpoint
    python -m src.inference_v2 --config configs/dino_vitl14.yaml --checkpoint outputs/dino_vitl14/best_model.pth
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from .utils.config import load_config, get_model_name
from .train import create_model, get_transforms


# ============================================================================
# 資料集
# ============================================================================

class TestDataset(Dataset):
    """測試資料集"""
    
    def __init__(self, test_dir: str, transform=None):
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
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, img_path.stem
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(3, 224, 224), img_path.stem


# ============================================================================
# TTA
# ============================================================================

def get_tta_transforms(config: Dict[str, Any]) -> List:
    """獲取 TTA transforms"""
    size = config['data']['image_size']
    
    preprocess = config.get('preprocessing', {})
    mean = preprocess.get('mean', [0.485, 0.456, 0.406])
    std = preprocess.get('std', [0.229, 0.224, 0.225])
    
    tta_transforms = [
        # 1. 原始圖像
        transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]),
        # 2. 水平翻轉
        transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]),
        # 3. 輕微放大裁切
        transforms.Compose([
            transforms.Resize((int(size * 1.1), int(size * 1.1)), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]),
        # 4. 輕微放大裁切 + 水平翻轉
        transforms.Compose([
            transforms.Resize((int(size * 1.1), int(size * 1.1)), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(size),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]),
    ]
    
    return tta_transforms


# ============================================================================
# 推論
# ============================================================================

@torch.no_grad()
def inference(
    config: Dict[str, Any],
    checkpoint_path: Optional[str] = None,
    use_tta: bool = False,
    thresholds: List[float] = [0.3, 0.4, 0.5, 0.6],
):
    """
    執行推論
    
    Args:
        config: 配置字典
        checkpoint_path: 模型權重路徑
        use_tta: 是否使用 TTA
        thresholds: 要生成的閾值列表
    """
    model_name = get_model_name(config)
    
    print("\n" + "=" * 60)
    print(f"🔍 Inference: {model_name}")
    print(f"   TTA: {'Enabled' if use_tta else 'Disabled'}")
    print("=" * 60)
    
    # 設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥  Device: {device}")
    
    # 創建模型
    model = create_model(config, device)
    
    # 載入權重
    if checkpoint_path is None:
        output_dir = Path(config['output']['save_dir']) / model_name
        checkpoint_path = output_dir / 'best_model.pth'
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        val_acc = checkpoint.get('val_acc', 'N/A')
        print(f"✅ Loaded model from {checkpoint_path}")
        if val_acc != 'N/A':
            print(f"   Val accuracy: {val_acc:.2f}%")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model.eval()
    
    # 測試目錄
    test_dir = os.path.join(config['data']['data_path'], 'test')
    
    # 是否為 binary classification
    num_classes = config['model'].get('num_classes', 2)
    is_binary = (num_classes == 2)
    
    # 收集預測
    all_filenames = []
    all_probs = []
    
    if use_tta:
        # TTA 推論
        tta_transforms = get_tta_transforms(config)
        print(f"📊 Using {len(tta_transforms)} TTA transforms")
        
        # 先收集 filenames
        base_transform = get_transforms(config, is_train=False)
        first_dataset = TestDataset(test_dir, base_transform)
        all_filenames = [first_dataset.image_files[i].stem for i in range(len(first_dataset))]
        
        # 累積預測
        accumulated_probs = {fname: 0.0 for fname in all_filenames}
        
        for t_idx, tta_transform in enumerate(tta_transforms):
            dataset = TestDataset(test_dir, tta_transform)
            loader = DataLoader(
                dataset,
                batch_size=config['training']['batch_size'] * 2,
                shuffle=False,
                num_workers=config['data'].get('num_workers', 4),
                pin_memory=True,
            )
            
            for images, filenames in tqdm(loader, desc=f"TTA {t_idx+1}/{len(tta_transforms)}"):
                images = images.to(device)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                
                if is_binary:
                    # Binary: sigmoid 得到 fake 機率
                    probs = torch.sigmoid(outputs.squeeze(-1)).cpu().numpy()
                else:
                    # Multi-class: softmax 取 fake (class 0) 機率
                    probs = F.softmax(outputs, dim=1)[:, 0].cpu().numpy()
                
                for fname, prob in zip(filenames, probs):
                    accumulated_probs[fname] += prob
        
        # 平均
        all_probs = np.array([accumulated_probs[fname] / len(tta_transforms) for fname in all_filenames])
        
    else:
        # 標準推論
        transform = get_transforms(config, is_train=False)
        dataset = TestDataset(test_dir, transform)
        loader = DataLoader(
            dataset,
            batch_size=config['training']['batch_size'] * 2,
            shuffle=False,
            num_workers=config['data'].get('num_workers', 4),
            pin_memory=True,
        )
        
        for images, filenames in tqdm(loader, desc="Inference"):
            images = images.to(device)
            
            with torch.amp.autocast('cuda'):
                outputs = model(images)
            
            if is_binary:
                probs = torch.sigmoid(outputs.squeeze(-1)).cpu().numpy()
            else:
                probs = F.softmax(outputs, dim=1)[:, 0].cpu().numpy()
            
            all_filenames.extend(filenames)
            all_probs.extend(probs)
        
        all_probs = np.array(all_probs)
    
    # 輸出目錄
    output_dir = Path(config['output']['save_dir']) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存機率 (按數字排序)
    suffix = '_tta' if use_tta else ''
    prob_df = pd.DataFrame({
        'filename': all_filenames,
        'prob': all_probs
    })
    # 按數字排序 (1, 2, 3...) 而非字串排序 (1, 10, 100...)
    prob_df['_sort_key'] = prob_df['filename'].astype(int)
    prob_df = prob_df.sort_values('_sort_key')
    prob_df = prob_df.drop(columns=['_sort_key'])

    # 更新排序後的值
    all_filenames = prob_df['filename'].tolist()
    all_probs = prob_df['prob'].values

    prob_path = output_dir / f'predictions_probs{suffix}.csv'
    prob_df.to_csv(prob_path, index=False)
    print(f"\n✅ Probabilities saved to {prob_path}")
    
    # 統計
    print(f"\n📊 Probability Statistics:")
    print(f"   Mean: {all_probs.mean():.4f}")
    print(f"   Std: {all_probs.std():.4f}")
    print(f"   Median: {np.median(all_probs):.4f}")
    print(f"   Min: {all_probs.min():.4f}, Max: {all_probs.max():.4f}")
    
    # 生成不同閾值的 submission
    print(f"\n📊 Generating submissions with different thresholds...")
    
    for t in thresholds:
        # 注意：模型訓練時 fake=0, real=1，所以高機率是 real
        labels = ['real' if p > t else 'fake' for p in all_probs]
        t_df = pd.DataFrame({'filename': all_filenames, 'label': labels})
        t_path = output_dir / f'submission{suffix}_t{int(t*10)}.csv'
        t_df.to_csv(t_path, index=False)
        fake_count = labels.count('fake')
        print(f"   t={t}: Fake={fake_count} ({fake_count/len(labels)*100:.1f}%) -> {t_path}")
    
    # 使用 median 作為預設 submission
    median_threshold = np.median(all_probs)
    # 注意：模型訓練時 fake=0, real=1，所以高機率是 real
    labels = ['real' if p > median_threshold else 'fake' for p in all_probs]
    submission_df = pd.DataFrame({'filename': all_filenames, 'label': labels})
    submission_path = output_dir / f'submission{suffix}.csv'
    submission_df.to_csv(submission_path, index=False)
    
    fake_count = labels.count('fake')
    print(f"\n📊 Default submission (median threshold {median_threshold:.4f}):")
    print(f"   Fake: {fake_count} ({fake_count/len(labels)*100:.1f}%)")
    print(f"   Real: {len(labels)-fake_count} ({(len(labels)-fake_count)/len(labels)*100:.1f}%)")
    print(f"   Saved to: {submission_path}")
    
    return all_filenames, all_probs


# ============================================================================
# 主程式
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Deepfake Detection Inference v4.0')
    
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint (default: auto-detect)')
    parser.add_argument('--tta', action='store_true',
                        help='Enable Test-Time Augmentation')
    parser.add_argument('--thresholds', type=float, nargs='+', 
                        default=[0.3, 0.4, 0.5, 0.6],
                        help='Thresholds to generate submissions for')
    
    args = parser.parse_args()
    
    # 載入配置
    config = load_config(args.config)
    
    # 執行推論
    inference(
        config=config,
        checkpoint_path=args.checkpoint,
        use_tta=args.tta,
        thresholds=args.thresholds,
    )


if __name__ == '__main__':
    main()
