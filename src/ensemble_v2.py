#!/usr/bin/env python3
"""
集成推論腳本 v4.0

支援多種集成策略：平均、加權平均、階層式集成。

使用方式:
    # 使用 ensemble.yaml 配置
    python -m src.ensemble_v2
    
    # 指定策略
    python -m src.ensemble_v2 --strategy hierarchical
    
    # 只使用部分模型
    python -m src.ensemble_v2 --models dino_vitl14 dino_vitl14_deep convnext_base
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils.config import load_config


# ============================================================================
# 集成策略
# ============================================================================

def ensemble_average(prob_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """簡單平均"""
    probs = list(prob_dict.values())
    return np.mean(probs, axis=0)


def ensemble_weighted_average(
    prob_dict: Dict[str, np.ndarray], 
    weights: Dict[str, float]
) -> np.ndarray:
    """加權平均"""
    result = np.zeros_like(list(prob_dict.values())[0])
    total_weight = 0
    
    for name, prob in prob_dict.items():
        weight = weights.get(name, 1.0)
        result += prob * weight
        total_weight += weight
    
    return result / total_weight


def ensemble_hierarchical(
    prob_dict: Dict[str, np.ndarray],
    groups: List[Dict],
) -> np.ndarray:
    """
    階層式集成
    
    先在各陣營內部平均，再用權重合併陣營
    """
    result = np.zeros_like(list(prob_dict.values())[0])
    total_weight = 0
    
    for group in groups:
        group_name = group['name']
        group_weight = group['weight']
        group_models = group['models']
        
        # 計算陣營內部平均
        group_probs = []
        for model_name in group_models:
            if model_name in prob_dict:
                group_probs.append(prob_dict[model_name])
            else:
                print(f"⚠️ Model {model_name} not found in predictions, skipping")
        
        if group_probs:
            group_avg = np.mean(group_probs, axis=0)
            result += group_avg * group_weight
            total_weight += group_weight
            print(f"   {group_name}: {len(group_probs)} models, weight {group_weight}")
    
    return result / total_weight if total_weight > 0 else result


def ensemble_vote(prob_dict: Dict[str, np.ndarray], threshold: float = 0.5) -> np.ndarray:
    """Soft voting (基於多數決)"""
    probs = list(prob_dict.values())
    votes = np.array([p > threshold for p in probs])
    return votes.mean(axis=0)


# ============================================================================
# 主要函數
# ============================================================================

def load_predictions(output_dir: str, model_names: List[str], use_tta: bool = True) -> Dict[str, np.ndarray]:
    """載入各模型的預測"""
    prob_dict = {}
    filenames = None
    
    suffix = '_tta' if use_tta else ''
    
    for model_name in model_names:
        prob_path = Path(output_dir) / model_name / f'predictions_probs{suffix}.csv'
        
        if not prob_path.exists():
            # 嘗試不帶 TTA 的版本
            prob_path = Path(output_dir) / model_name / 'predictions_probs.csv'
        
        if prob_path.exists():
            df = pd.read_csv(prob_path)
            df['filename'] = df['filename'].astype(str)
            df = df.sort_values('filename')
            
            if filenames is None:
                filenames = df['filename'].tolist()
            
            prob_dict[model_name] = df['prob'].values
            print(f"✅ Loaded {model_name}: {len(df)} predictions")
        else:
            print(f"⚠️ Not found: {prob_path}")
    
    return prob_dict, filenames


def run_ensemble(
    config_path: str = "configs/ensemble.yaml",
    strategy: Optional[str] = None,
    model_names: Optional[List[str]] = None,
    use_tta: bool = True,
    thresholds: List[float] = [0.3, 0.4, 0.5, 0.6],
):
    """
    執行集成
    
    Args:
        config_path: 集成配置路徑
        strategy: 集成策略 (覆蓋配置)
        model_names: 要集成的模型列表 (覆蓋配置)
        use_tta: 是否使用 TTA 預測
        thresholds: 要生成的閾值列表
    """
    # 載入配置
    config = load_config(config_path)
    ensemble_config = config.get('ensemble', {})
    
    # 確定策略
    if strategy is None:
        strategy = ensemble_config.get('strategy', 'average')
    
    print("\n" + "=" * 60)
    print(f"🔗 Ensemble with strategy: {strategy}")
    print("=" * 60)
    
    # 確定要使用的模型
    if model_names is None:
        if strategy == 'hierarchical':
            groups = ensemble_config.get('hierarchical', {}).get('groups', [])
            model_names = []
            for group in groups:
                model_names.extend(group['models'])
        elif strategy == 'weighted_average':
            models_config = ensemble_config.get('weighted_average', {}).get('models', [])
            model_names = [m['name'] for m in models_config]
        else:
            # 從輸出目錄自動偵測
            output_dir = config.get('output', {}).get('save_dir', './outputs')
            model_names = [d.name for d in Path(output_dir).iterdir() 
                          if d.is_dir() and (d / 'predictions_probs.csv').exists()]
    
    print(f"📊 Models to ensemble: {model_names}")
    
    # 載入預測
    output_dir = config.get('output', {}).get('save_dir', './outputs')
    prob_dict, filenames = load_predictions(output_dir, model_names, use_tta)
    
    if not prob_dict:
        print("❌ No predictions found!")
        return
    
    print(f"\n📊 Successfully loaded {len(prob_dict)} models")
    
    # 執行集成
    if strategy == 'average':
        final_probs = ensemble_average(prob_dict)
        
    elif strategy == 'weighted_average':
        models_config = ensemble_config.get('weighted_average', {}).get('models', [])
        weights = {m['name']: m['weight'] for m in models_config}
        final_probs = ensemble_weighted_average(prob_dict, weights)
        
    elif strategy == 'hierarchical':
        groups = ensemble_config.get('hierarchical', {}).get('groups', [])
        print("\n📊 Hierarchical ensemble:")
        final_probs = ensemble_hierarchical(prob_dict, groups)
        
    elif strategy == 'vote':
        final_probs = ensemble_vote(prob_dict)
        
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # 輸出目錄
    ensemble_output_dir = Path(output_dir) / 'ensemble'
    ensemble_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存機率
    suffix = '_tta' if use_tta else ''
    prob_df = pd.DataFrame({
        'filename': filenames,
        'prob': final_probs
    })
    prob_path = ensemble_output_dir / f'ensemble_probs{suffix}.csv'
    prob_df.to_csv(prob_path, index=False)
    print(f"\n✅ Ensemble probabilities saved to {prob_path}")
    
    # 統計
    print(f"\n📊 Ensemble Probability Statistics:")
    print(f"   Mean: {final_probs.mean():.4f}")
    print(f"   Std: {final_probs.std():.4f}")
    print(f"   Median: {np.median(final_probs):.4f}")
    
    # 生成不同閾值的 submission
    print(f"\n📊 Generating submissions...")
    
    for t in thresholds:
        labels = ['fake' if p > t else 'real' for p in final_probs]
        t_df = pd.DataFrame({'filename': filenames, 'label': labels})
        t_path = ensemble_output_dir / f'ensemble_submission{suffix}_t{int(t*10)}.csv'
        t_df.to_csv(t_path, index=False)
        fake_count = labels.count('fake')
        print(f"   t={t}: Fake={fake_count} ({fake_count/len(labels)*100:.1f}%) -> {t_path}")
    
    # 使用 median 作為預設
    median_threshold = np.median(final_probs)
    labels = ['fake' if p > median_threshold else 'real' for p in final_probs]
    submission_df = pd.DataFrame({'filename': filenames, 'label': labels})
    submission_path = ensemble_output_dir / f'ensemble_submission{suffix}.csv'
    submission_df.to_csv(submission_path, index=False)
    
    fake_count = labels.count('fake')
    print(f"\n🏆 Final ensemble submission (median threshold {median_threshold:.4f}):")
    print(f"   Fake: {fake_count} ({fake_count/len(labels)*100:.1f}%)")
    print(f"   Real: {len(labels)-fake_count} ({(len(labels)-fake_count)/len(labels)*100:.1f}%)")
    print(f"   Saved to: {submission_path}")
    
    # 計算模型相關性
    if len(prob_dict) > 1:
        print("\n📊 Model Correlation Matrix:")
        df_corr = pd.DataFrame(prob_dict)
        corr_matrix = df_corr.corr()
        print(corr_matrix.to_string())
        
        # 保存相關性
        corr_path = ensemble_output_dir / 'model_correlation.csv'
        corr_matrix.to_csv(corr_path)
    
    return filenames, final_probs


# ============================================================================
# 主程式
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Deepfake Detection Ensemble v4.0')
    
    parser.add_argument('--config', type=str, default='configs/ensemble.yaml',
                        help='Path to ensemble config')
    parser.add_argument('--strategy', type=str, default=None,
                        choices=['average', 'weighted_average', 'hierarchical', 'vote'],
                        help='Ensemble strategy (overrides config)')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Models to ensemble (overrides config)')
    parser.add_argument('--no-tta', action='store_true',
                        help='Use non-TTA predictions')
    parser.add_argument('--thresholds', type=float, nargs='+',
                        default=[0.3, 0.4, 0.5, 0.6],
                        help='Thresholds to generate submissions')
    
    args = parser.parse_args()
    
    run_ensemble(
        config_path=args.config,
        strategy=args.strategy,
        model_names=args.models,
        use_tta=not args.no_tta,
        thresholds=args.thresholds,
    )


if __name__ == '__main__':
    main()
