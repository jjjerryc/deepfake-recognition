"""
配置載入器

支援 YAML 配置檔案，自動處理繼承關係。

使用方式:
    from src.utils.config import load_config
    
    config = load_config("configs/dino_vitl14.yaml")
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import copy

import yaml


def deep_merge(base: Dict, override: Dict) -> Dict:
    """
    深度合併兩個字典
    
    Args:
        base: 基礎字典
        override: 覆蓋字典
        
    Returns:
        合併後的新字典
    """
    result = copy.deepcopy(base)
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    
    return result


def load_config(config_path: Union[str, Path], base_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    載入 YAML 配置檔案
    
    支援 _base_ 繼承機制：
    - 如果配置中有 _base_ 欄位，會先載入基礎配置
    - 然後用當前配置覆蓋基礎配置
    
    Args:
        config_path: 配置檔案路徑
        base_dir: 基礎目錄（用於解析相對路徑）
        
    Returns:
        配置字典
    """
    config_path = Path(config_path)
    
    if base_dir is None:
        base_dir = config_path.parent
    else:
        base_dir = Path(base_dir)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        config = {}
    
    # 處理繼承
    if '_base_' in config:
        base_path = base_dir / config['_base_']
        base_config = load_config(base_path, base_dir)
        
        # 移除 _base_ 欄位
        del config['_base_']
        
        # 合併配置
        config = deep_merge(base_config, config)
    
    return config


def save_config(config: Dict[str, Any], config_path: Union[str, Path]):
    """
    保存配置到 YAML 檔案
    
    Args:
        config: 配置字典
        config_path: 輸出路徑
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def get_model_name(config: Dict[str, Any]) -> str:
    """從配置中獲取模型名稱"""
    return config.get('model', {}).get('name', 'unknown')


def list_configs(config_dir: str = "configs") -> list:
    """列出所有可用的配置檔案"""
    config_dir = Path(config_dir)
    configs = []
    
    for f in config_dir.glob("*.yaml"):
        if f.name != "base.yaml" and f.name != "ensemble.yaml":
            configs.append(f.stem)
    
    return sorted(configs)


def print_config(config: Dict[str, Any], indent: int = 0):
    """美觀地打印配置"""
    prefix = "  " * indent
    
    for key, value in config.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_config(value, indent + 1)
        else:
            print(f"{prefix}{key}: {value}")
