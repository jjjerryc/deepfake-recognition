"""
多模型推論腳本 v3.0

對測試集進行預測並生成符合競賽格式的 submission.csv 文件。

功能特點：
- 自動載入訓練時使用的模型架構
- Test-Time Augmentation (TTA) 可選
- 混合精度推論
- ImageNet 標準化預處理

使用方式:
    python -m src.inference
    python -m src.inference --config config.json
    python -m src.inference --tta  # 啟用 TTA
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .models import create_model, create_model_from_config


class TestDataset(Dataset):
    """
    測試集資料載入器
    
    測試集沒有子資料夾結構，直接讀取目錄下的所有圖片。
    """
    
    def __init__(self, test_dir: str, transform=None):
        self.test_dir = Path(test_dir)
        self.transform = transform
        
        # 獲取所有圖片文件
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        self.image_files = sorted(
            [f for f in self.test_dir.iterdir() if f.suffix.lower() in valid_extensions],
            key=lambda x: int(x.stem) if x.stem.isdigit() else x.stem
        )
        
        print(f"Found {len(self.image_files)} test images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # 返回圖像和文件名（不含擴展名）
        filename = img_path.stem
        
        return image, filename


def load_config(config_path: str) -> dict:
    """載入配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_transform(image_size: int, mean: List[float], std: List[float]):
    """獲取推論用的數據轉換"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_tta_transforms(image_size: int, mean: List[float], std: List[float]):
    """獲取 TTA 用的多個數據轉換"""
    base_transform = [
        transforms.Resize((image_size, image_size)),
    ]
    
    tta_list = [
        # 原始圖像
        transforms.Compose(base_transform + [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]),
        # 水平翻轉
        transforms.Compose(base_transform + [
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]),
    ]
    
    return tta_list


@torch.no_grad()
def inference_single(model, loader, device, use_amp: bool = False):
    """標準推論 (無 TTA)"""
    model.eval()
    predictions = []
    
    for images, filenames in tqdm(loader, desc="Inference"):
        images = images.to(device)
        
        if use_amp:
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
        else:
            outputs = model(images)
        
        probs = F.softmax(outputs, dim=1)
        _, predicted = torch.max(probs, 1)
        
        for filename, pred in zip(filenames, predicted.cpu().numpy()):
            predictions.append((filename, pred))
    
    return predictions


@torch.no_grad()
def inference_tta(model, test_dir, tta_transforms, batch_size, num_workers, device, use_amp: bool = False):
    """TTA 推論 - 對每個圖像應用多個增強並平均結果"""
    model.eval()
    
    # 第一次遍歷獲取所有文件名
    first_dataset = TestDataset(test_dir, transform=tta_transforms[0])
    all_filenames = [first_dataset.image_files[i].stem for i in range(len(first_dataset))]
    
    # 累積所有 TTA 變換的預測概率
    accumulated_probs = {}
    for filename in all_filenames:
        accumulated_probs[filename] = None
    
    for t_idx, transform in enumerate(tta_transforms):
        dataset = TestDataset(test_dir, transform=transform)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        for images, filenames in tqdm(loader, desc=f"TTA {t_idx+1}/{len(tta_transforms)}"):
            images = images.to(device)
            
            if use_amp:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(images)
            else:
                outputs = model(images)
            
            probs = F.softmax(outputs, dim=1)
            
            for filename, prob in zip(filenames, probs.cpu()):
                if accumulated_probs[filename] is None:
                    accumulated_probs[filename] = prob
                else:
                    accumulated_probs[filename] = accumulated_probs[filename] + prob
    
    # 平均並生成最終預測
    predictions = []
    for filename in all_filenames:
        avg_prob = accumulated_probs[filename] / len(tta_transforms)
        pred = avg_prob.argmax().item()
        predictions.append((filename, pred))
    
    return predictions


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Model Inference Script for Deepfake Detection'
    )
    parser.add_argument(
        '--config', type=str, default='config.json',
        help='配置文件路徑'
    )
    parser.add_argument(
        '--tta', action='store_true',
        help='啟用 Test-Time Augmentation'
    )
    parser.add_argument(
        '--model-path', type=str, default=None,
        help='模型檢查點路徑 (覆蓋配置文件)'
    )
    parser.add_argument(
        '--model', type=str, default=None,
        help='模型名稱（用於自動定位檢查點）'
    )
    
    args = parser.parse_args()
    
    # 載入配置
    config = load_config(args.config)
    
    # 如果指定了模型名稱，覆蓋配置
    if args.model:
        config['model']['name'] = args.model
    
    print("=" * 60)
    print(f"Project: {config['project']['name']} v{config['project']['version']}")
    print(f"Config: {args.config}")
    print("=" * 60)
    
    # 設置設備
    device_name = config.get('hardware', {}).get('device', 'cuda')
    if device_name == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Device: {torch.cuda.get_device_name(0)}")
    elif device_name == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Device: Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
        print("Device: CPU")
    
    # 混合精度推論
    use_amp = config.get('hardware', {}).get('mixed_precision', False) and device.type == 'cuda'
    print(f"Mixed Precision: {'Enabled' if use_amp else 'Disabled'}")
    
    # 確定模型名稱
    model_name = config['model'].get('name', 'efficientnet_b4')
    
    # 確定模型路徑（支援 {model_name} 佔位符）
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = config['inference']['model_path']
        # 替換佔位符
        model_path = model_path.replace('{model_name}', model_name)
    
    print(f"\nLoading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # 從檢查點獲取模型配置（優先使用檢查點中的配置）
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        saved_model_name = saved_config['model'].get('name', model_name)
        # 如果命令行沒有指定模型，使用檢查點中的模型名稱
        if not args.model:
            model_name = saved_model_name
    elif 'model_name' in checkpoint:
        if not args.model:
            model_name = checkpoint['model_name']
    
    print(f"Model: {model_name}")
    
    # 創建模型
    # 使用保存的配置，或當前配置
    model_config = saved_config if 'config' in checkpoint else config
    model = create_model_from_config(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # 獲取預處理參數
    preprocess_config = model.get_preprocessing_config()
    mean = preprocess_config['mean']
    std = preprocess_config['std']
    image_size = config['data']['image_size']
    
    print(f"Image size: {image_size}x{image_size}")
    print(f"Normalization: mean={mean}, std={std}")
    
    # 設置測試目錄和輸出（使用模型專屬目錄）
    test_dir = os.path.join(config['data']['data_path'], 'test')
    base_output_dir = Path(config['inference']['output_dir'])
    output_dir = base_output_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = config['inference']['output_name']
    
    print(f"Output directory: {output_dir}")
    
    batch_size = config['inference']['batch_size']
    num_workers = config['data']['num_workers']
    
    # 推論
    print(f"\nStarting inference...")
    print(f"TTA: {'Enabled' if args.tta else 'Disabled'}")
    
    if args.tta:
        tta_transforms = get_tta_transforms(image_size, mean, std)
        predictions = inference_tta(
            model, test_dir, tta_transforms,
            batch_size, num_workers, device, use_amp
        )
    else:
        transform = get_transform(image_size, mean, std)
        dataset = TestDataset(test_dir, transform=transform)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        predictions = inference_single(model, loader, device, use_amp)
    
    # 生成 submission 文件
    output_path = output_dir / output_name
    with open(output_path, 'w') as f:
        f.write("id,label\n")
        for filename, pred in predictions:
            f.write(f"{filename},{pred}\n")
    
    print(f"\n" + "=" * 60)
    print("Inference Complete!")
    print(f"  Model: {model_name}")
    print(f"  Predictions: {len(predictions)}")
    print(f"  Output: {output_path}")
    
    # 統計預測分布
    fake_count = sum(1 for _, pred in predictions if pred == 0)
    real_count = sum(1 for _, pred in predictions if pred == 1)
    print(f"  Distribution: Fake={fake_count}, Real={real_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
