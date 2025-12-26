#!/usr/bin/env python3
"""
Ensemble 推論腳本 - v2.0
========================

從多個訓練好的模型進行集成推論，支援 config.json 配置。

核心特性:
1. 從 config.json 讀取 ensemble 配置
2. 支援不同模型的預處理（EfficientNet vs CLIP）
3. 多種集成策略 (average, weighted_average, vote, max)
4. 自動權重計算（根據驗證準確率）
5. 保存個別模型預測結果

使用方式:
    # 使用 config.json（推薦）
    python -m src.ensemble_inference
    python -m src.ensemble_inference --config config.json
    
    # 覆蓋配置
    python -m src.ensemble_inference --strategy average
    python -m src.ensemble_inference --auto-weight
    
    # 傳統方式（指定 checkpoints）
    python -m src.ensemble_inference --checkpoints model1.pth model2.pth
    python -m src.ensemble_inference --checkpoints model1.pth model2.pth --weights 0.6 0.4
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np

from .models import create_model, ModelEnsemble


# ============================================================================
# 預處理配置
# ============================================================================

PREPROCESSING_CONFIGS = {
    'efficientnet': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
    },
    'clip': {
        'mean': [0.48145466, 0.4578275, 0.40821073],
        'std': [0.26862954, 0.26130258, 0.27577711],
    },
}


def get_preprocessing_type(model_name: str) -> str:
    """根據模型名稱決定預處理類型"""
    if model_name.startswith('clip_'):
        return 'clip'
    return 'efficientnet'


# ============================================================================
# 多預處理資料集
# ============================================================================

class MultiPreprocessTestDataset(Dataset):
    """
    支援多模型預處理的測試集
    為每個模型提供正確的預處理圖像
    """
    
    def __init__(
        self,
        test_dir: str,
        model_names: List[str],
        image_size: int = 224,
    ):
        self.test_dir = Path(test_dir)
        self.model_names = model_names
        self.image_size = image_size
        
        # 收集圖像
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        self.image_files = sorted(
            [f for f in self.test_dir.iterdir() if f.suffix.lower() in valid_extensions],
            key=lambda x: int(x.stem) if x.stem.isdigit() else x.stem
        )
        print(f"Found {len(self.image_files)} test images")
        
        # 建立預處理
        self.transforms = self._build_transforms()
    
    def _build_transforms(self) -> Dict[str, transforms.Compose]:
        """為每個模型建立對應的預處理"""
        model_transforms = {}
        
        for model_name in self.model_names:
            preprocess_type = get_preprocessing_type(model_name)
            config = PREPROCESSING_CONFIGS[preprocess_type]
            
            transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=config['mean'], std=config['std'])
            ])
            
            model_transforms[model_name] = transform
        
        return model_transforms
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx) -> Tuple[Dict[str, torch.Tensor], str]:
        """
        Returns:
            images: Dict[model_name, preprocessed_tensor]
            filename: 圖像文件名（不含副檔名）
        """
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        images = {}
        for model_name, transform in self.transforms.items():
            images[model_name] = transform(image)
        
        return images, img_path.stem


class SinglePreprocessTestDataset(Dataset):
    """單一預處理的測試集（向後兼容）"""
    
    def __init__(self, test_dir: str, transform=None):
        self.test_dir = Path(test_dir)
        self.transform = transform
        
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
        
        return image, img_path.stem


# ============================================================================
# 模型載入
# ============================================================================

def load_model_from_checkpoint(
    checkpoint_path: str,
    device: str = 'cuda'
) -> Tuple[torch.nn.Module, str, Dict]:
    """
    從 checkpoint 載入模型
    
    Returns:
        model: 載入的模型
        model_name: 模型名稱
        info: 額外資訊（包含 val_acc 等）
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 獲取模型名稱和配置
    model_name = checkpoint.get('model_name', 'efficientnet_b4')
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})
    
    # 創建模型
    model = create_model(
        model_name=model_name,
        num_classes=model_config.get('num_classes', 2),
        pretrained=False,
        dropout=model_config.get('dropout', 0.3),
        drop_path_rate=model_config.get('drop_path_rate', 0.2),
    )
    
    # 載入權重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # 收集額外資訊
    info = {
        'best_val_acc': checkpoint.get('best_val_acc', None),
        'epoch': checkpoint.get('epoch', None),
        'model_name': model_name,
    }
    
    return model, model_name, info


# ============================================================================
# Ensemble 推論引擎
# ============================================================================

class EnsembleInferenceEngine:
    """
    支援多預處理的集成推論引擎
    """
    
    def __init__(
        self,
        models: Dict[str, nn.Module],
        weights: Dict[str, float],
        strategy: str = 'weighted_average',
        temperature: float = 1.0,
        device: str = 'cuda',
    ):
        self.models = models
        self.weights = weights
        self.strategy = strategy
        self.temperature = temperature
        self.device = device
        
        # 正規化權重
        total = sum(self.weights.values())
        self.normalized_weights = {k: v / total for k, v in self.weights.items()}
        
        self.model_names = list(models.keys())
    
    @torch.no_grad()
    def predict_batch(
        self,
        batch_images: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        對一個 batch 進行預測
        
        Args:
            batch_images: Dict[model_name, (B, C, H, W)]
            
        Returns:
            ensemble_probs: (B, num_classes)
            individual_probs: Dict[model_name, (B, num_classes)]
        """
        individual_probs = {}
        
        for name, model in self.models.items():
            images = batch_images[name].to(self.device)
            
            with torch.cuda.amp.autocast():
                logits = model(images)
            
            probs = F.softmax(logits / self.temperature, dim=-1)
            individual_probs[name] = probs
        
        # 集成
        ensemble_probs = self._ensemble(individual_probs)
        
        return ensemble_probs, individual_probs
    
    def _ensemble(self, individual_probs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """根據策略集成預測"""
        probs_list = [individual_probs[name] for name in self.model_names]
        stacked = torch.stack(probs_list, dim=0)  # (num_models, B, C)
        
        if self.strategy == 'average':
            return stacked.mean(dim=0)
        
        elif self.strategy == 'weighted_average':
            weights = torch.tensor(
                [self.normalized_weights[name] for name in self.model_names],
                device=self.device
            ).view(-1, 1, 1)
            return (stacked * weights).sum(dim=0)
        
        elif self.strategy == 'vote':
            # Soft voting
            return stacked.mean(dim=0)
        
        elif self.strategy == 'max':
            # 選擇最有信心的模型
            max_probs, _ = stacked.max(dim=-1)  # (num_models, B)
            best_idx = max_probs.argmax(dim=0)  # (B,)
            B = stacked.shape[1]
            result = torch.zeros_like(stacked[0])
            for b in range(B):
                result[b] = stacked[best_idx[b], b]
            return result
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")


def run_ensemble_inference(
    engine: EnsembleInferenceEngine,
    loader: DataLoader,
    save_individual: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    執行集成推論
    
    Returns:
        ensemble_df: 集成預測 DataFrame
        individual_dfs: Dict[model_name, DataFrame]
    """
    all_filenames = []
    all_ensemble_probs = []
    all_individual_probs = {name: [] for name in engine.model_names}
    
    for batch_images, filenames in tqdm(loader, desc="Ensemble Inference"):
        ensemble_probs, individual_probs = engine.predict_batch(batch_images)
        
        all_filenames.extend(filenames)
        all_ensemble_probs.append(ensemble_probs.cpu())
        
        if save_individual:
            for name in engine.model_names:
                all_individual_probs[name].append(individual_probs[name].cpu())
    
    # 合併結果
    ensemble_probs = torch.cat(all_ensemble_probs, dim=0).numpy()
    # 注意：類別順序為 [fake=0, real=1]，所以取 [:, 0] 是 fake 機率
    fake_probs = ensemble_probs[:, 0]  # fake 類別機率 (index 0)
    
    ensemble_df = pd.DataFrame({
        'id': [int(f) if f.isdigit() else f for f in all_filenames],
        'label': fake_probs
    })
    
    individual_dfs = {}
    if save_individual:
        for name in engine.model_names:
            probs = torch.cat(all_individual_probs[name], dim=0).numpy()
            individual_dfs[name] = pd.DataFrame({
                'id': [int(f) if f.isdigit() else f for f in all_filenames],
                'label': probs[:, 0]  # fake 類別機率 (index 0)
            })
    
    return ensemble_df, individual_dfs


# ============================================================================
# Config 載入器
# ============================================================================

def load_ensemble_from_config(
    config: Dict,
    device: str = 'cuda',
    auto_weight: bool = False,
) -> Tuple[EnsembleInferenceEngine, List[str]]:
    """
    從 config.json 載入集成模型
    
    Returns:
        engine: 集成推論引擎
        model_names: 模型名稱列表
    """
    ensemble_config = config['ensemble']
    model_configs = ensemble_config['models']
    
    models = {}
    weights = {}
    val_accs = {}
    
    print("\n" + "=" * 60)
    print("Loading Ensemble Models")
    print("=" * 60)
    
    for cfg in model_configs:
        name = cfg['name']
        checkpoint_path = cfg['checkpoint']
        weight = cfg.get('weight', 1.0)
        
        if not os.path.exists(checkpoint_path):
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
            continue
        
        try:
            model, actual_name, info = load_model_from_checkpoint(checkpoint_path, device)
            models[name] = model
            weights[name] = weight
            
            if info['best_val_acc']:
                val_accs[name] = info['best_val_acc']
                print(f"✅ {name}: val_acc={info['best_val_acc']:.4f}, weight={weight:.3f}")
            else:
                print(f"✅ {name}: weight={weight:.3f}")
                
        except Exception as e:
            print(f"❌ Failed to load {name}: {e}")
            continue
    
    if not models:
        raise RuntimeError("No models loaded successfully!")
    
    # 自動計算權重
    if auto_weight and val_accs:
        print("\n📊 Auto-calculating weights from validation accuracy...")
        total_acc = sum(val_accs.values())
        for name in weights:
            if name in val_accs:
                weights[name] = val_accs[name] / total_acc
                print(f"   {name}: {weights[name]:.4f}")
    
    # 創建引擎
    engine = EnsembleInferenceEngine(
        models=models,
        weights=weights,
        strategy=ensemble_config.get('strategy', 'weighted_average'),
        temperature=ensemble_config.get('temperature', 1.0),
        device=device,
    )
    
    print(f"\nEnsemble Strategy: {engine.strategy}")
    print(f"Temperature: {engine.temperature}")
    
    return engine, list(models.keys())


# ============================================================================
# Legacy 支援
# ============================================================================

def create_ensemble_from_checkpoints(
    checkpoint_paths: List[str],
    weights: Optional[List[float]] = None,
    strategy: str = 'weighted_average',
    device: str = 'cuda'
) -> ModelEnsemble:
    """從 checkpoint 列表創建集成模型（向後兼容）"""
    models = []
    
    for path in checkpoint_paths:
        model, _, _ = load_model_from_checkpoint(path, device)
        models.append(model)
    
    if weights is None:
        weights = [1.0] * len(models)
    
    return ModelEnsemble(models=models, weights=weights, strategy=strategy)


def get_transform(image_size: int = 224, model_name: str = 'efficientnet'):
    """獲取標準化轉換"""
    preprocess_type = get_preprocessing_type(model_name)
    config = PREPROCESSING_CONFIGS[preprocess_type]
    
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['mean'], std=config['std']),
    ])


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Ensemble Inference v2.0')
    
    # Config 來源
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to config.json')
    
    # 傳統方式（向後兼容）
    parser.add_argument('--checkpoints', nargs='+', type=str,
                        help='Checkpoint file paths (legacy mode)')
    parser.add_argument('--weights', nargs='+', type=float, default=None,
                        help='Model weights (legacy mode)')
    
    # 覆蓋配置
    parser.add_argument('--strategy', type=str, default=None,
                        choices=['average', 'weighted_average', 'vote', 'max'],
                        help='Ensemble strategy')
    parser.add_argument('--auto-weight', action='store_true',
                        help='Auto-calculate weights from validation accuracy')
    
    # 資料設定
    parser.add_argument('--test-dir', type=str, default=None,
                        help='Test data directory')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Data loading workers')
    
    # 輸出設定
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--no-individual', action='store_true',
                        help='Do not save individual model predictions')
    parser.add_argument('--kaggle-format', action='store_true',
                        help='Output in Kaggle format (filename,real/fake) instead of probabilities')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for fake classification (default: 0.5)')
    
    # 硬體設定
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device')
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    print("=" * 60)
    print("Ensemble Inference v2.0")
    print("=" * 60)
    print(f"Device: {device}")
    
    # Legacy mode: 使用 --checkpoints
    if args.checkpoints:
        print("\n[Legacy Mode] Using --checkpoints")
        
        ensemble = create_ensemble_from_checkpoints(
            checkpoint_paths=args.checkpoints,
            weights=args.weights,
            strategy=args.strategy or 'weighted_average',
            device=device
        )
        
        test_dir = args.test_dir or './dataset/test'
        output_dir = args.output_dir or './outputs'
        
        transform = get_transform(224, 'efficientnet')
        dataset = SinglePreprocessTestDataset(test_dir, transform=transform)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True
        )
        
        predictions = {}
        with torch.no_grad():
            for images, filenames in tqdm(loader, desc="Inference"):
                images = images.to(device)
                with torch.cuda.amp.autocast():
                    probs = ensemble.predict_proba(images)
                preds = probs[:, 1].cpu().numpy()
                for f, p in zip(filenames, preds):
                    predictions[f] = p
        
        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame([
            {'id': int(k) if k.isdigit() else k, 'label': v}
            for k, v in predictions.items()
        ]).sort_values('id')
        df.to_csv(os.path.join(output_dir, 'ensemble_submission.csv'), index=False)
        print(f"\n✅ Saved to {output_dir}/ensemble_submission.csv")
        return
    
    # Config mode: 使用 config.json
    print(f"\n[Config Mode] Loading from {args.config}")
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # 覆蓋配置
    if args.strategy:
        config['ensemble']['strategy'] = args.strategy
    
    # 載入模型
    engine, model_names = load_ensemble_from_config(
        config, device, auto_weight=args.auto_weight
    )
    
    # 設定路徑
    test_dir = args.test_dir or os.path.join(config['data']['data_path'], 'test')
    output_dir = args.output_dir or os.path.join(config['inference']['output_dir'], 'ensemble')
    
    # 資料集
    def collate_fn(batch):
        images_dict = {}
        filenames = []
        for images, fname in batch:
            for model_name, img in images.items():
                if model_name not in images_dict:
                    images_dict[model_name] = []
                images_dict[model_name].append(img)
            filenames.append(fname)
        for model_name in images_dict:
            images_dict[model_name] = torch.stack(images_dict[model_name])
        return images_dict, filenames
    
    dataset = MultiPreprocessTestDataset(
        test_dir=test_dir,
        model_names=model_names,
        image_size=config['data']['image_size'],
    )
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    # 推論
    ensemble_df, individual_dfs = run_ensemble_inference(
        engine, loader, save_individual=not args.no_individual
    )
    
    # 保存結果
    os.makedirs(output_dir, exist_ok=True)
    
    # 統計（基於機率）
    fake_probs = ensemble_df['label'].values
    threshold = args.threshold
    
    print("\n" + "=" * 60)
    print("Prediction Statistics")
    print("=" * 60)
    print(f"Total samples: {len(fake_probs)}")
    print(f"Threshold: {threshold}")
    print(f"Predicted fake (>{threshold}): {(fake_probs > threshold).sum()} ({(fake_probs > threshold).mean()*100:.1f}%)")
    print(f"Predicted real (≤{threshold}): {(fake_probs <= threshold).sum()} ({(fake_probs <= threshold).mean()*100:.1f}%)")
    print(f"Mean probability: {fake_probs.mean():.4f}")
    print(f"Std probability: {fake_probs.std():.4f}")
    
    # 根據格式選項保存
    output_name = config['ensemble'].get('output', {}).get('output_name', 'ensemble_submission.csv')
    
    if args.kaggle_format:
        # 轉換成 Kaggle 格式: filename, real/fake
        kaggle_df = ensemble_df.copy()
        kaggle_df.columns = ['filename', 'label']
        kaggle_df['label'] = kaggle_df['label'].apply(lambda x: 'fake' if x > threshold else 'real')
        
        kaggle_path = os.path.join(output_dir, output_name.replace('.csv', '_kaggle.csv'))
        kaggle_df.to_csv(kaggle_path, index=False)
        print(f"\n✅ Kaggle submission saved: {kaggle_path}")
    
    # 總是保存機率版本（方便後續調整 threshold）
    prob_path = os.path.join(output_dir, output_name.replace('.csv', '_probs.csv'))
    ensemble_df.to_csv(prob_path, index=False)
    print(f"✅ Probability submission saved: {prob_path}")
    
    if not args.no_individual:
        ind_dir = os.path.join(output_dir, 'individual')
        os.makedirs(ind_dir, exist_ok=True)
        for name, df in individual_dfs.items():
            df.to_csv(os.path.join(ind_dir, f'{name}_submission.csv'), index=False)
        print(f"✅ Individual predictions saved: {ind_dir}/")
    
    print("\n✅ Ensemble inference complete!")


if __name__ == '__main__':
    main()

