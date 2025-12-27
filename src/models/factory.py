"""
模型工廠

提供統一的模型創建介面，支援通過配置文件動態載入不同架構。

使用方式:
    # 方法 1: 直接創建
    model = create_model('efficientnet_b4', num_classes=2, pretrained=True)
    
    # 方法 2: 從配置創建
    model = create_model_from_config(config)
    
    # 列出可用模型
    print(list_available_models())
"""

from typing import Dict, Any, Optional, List, Type

import torch.nn as nn

from .base import BaseModel
from .efficientnet import EfficientNetModel, EFFICIENTNET_CONFIGS
from .efficientnet_dct import EfficientNetDCT, EfficientNetDCTAttention

# 嘗試載入 CLIP 模型（可選）
try:
    from .clip_model import CLIPClassifier, CLIPWithDCT, CLIP_CONFIGS
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    CLIP_CONFIGS = {}

# 嘗試載入 DINOv2 模型（可選）
try:
    from .dino_model import DINOv2Classifier, DINO_CONFIGS
    DINO_AVAILABLE = True
except ImportError:
    DINO_AVAILABLE = False
    DINO_CONFIGS = {}

# 嘗試載入 ConvNeXt 模型（可選）
try:
    from .convnext_model import ConvNeXtClassifier, CONVNEXT_CONFIGS
    CONVNEXT_AVAILABLE = True
except ImportError:
    CONVNEXT_AVAILABLE = False
    CONVNEXT_CONFIGS = {}


# 模型註冊表
MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {}

# 模型配置
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {}


def register_model(name: str, model_class: Type[BaseModel], config: Dict[str, Any] = None):
    """
    註冊模型到工廠
    
    Args:
        name: 模型名稱
        model_class: 模型類別
        config: 模型配置
    """
    MODEL_REGISTRY[name] = model_class
    if config:
        MODEL_CONFIGS[name] = config


# 註冊 EfficientNet 系列
for variant in ['b0', 'b1', 'b2', 'b3', 'b4']:
    model_name = f'efficientnet_{variant}'
    register_model(
        name=model_name,
        model_class=EfficientNetModel,
        config=EFFICIENTNET_CONFIGS.get(model_name, {})
    )


# 註冊 EfficientNet + DCT 系列
for variant in ['b0', 'b1', 'b2', 'b3', 'b4']:
    # 基本 DCT 融合模型
    model_name = f'efficientnet_{variant}_dct'
    register_model(
        name=model_name,
        model_class=EfficientNetDCT,
        config={
            'backbone_name': f'efficientnet_{variant}',
            'dct_dim': 128,
            'fusion_dim': 512,
        }
    )
    
    # 帶注意力的 DCT 模型
    model_name_attn = f'efficientnet_{variant}_dct_attn'
    register_model(
        name=model_name_attn,
        model_class=EfficientNetDCTAttention,
        config={
            'backbone_name': f'efficientnet_{variant}',
            'dct_dim': 256,
            'num_heads': 8,
        }
    )


# 註冊 CLIP 系列模型
if CLIP_AVAILABLE:
    # CLIP Classifier（凍結 encoder）
    for clip_name, clip_cfg in CLIP_CONFIGS.items():
        register_model(
            name=clip_name,
            model_class=CLIPClassifier,
            config=clip_cfg
        )
    
    # CLIP + DCT 融合模型
    register_model(
        name='clip_vit_b32_dct',
        model_class=CLIPWithDCT,
        config={
            'clip_model': 'ViT-B-32',
            'pretrained': 'openai',
            'dct_dim': 128,
            'fusion_dim': 512,
        }
    )
    register_model(
        name='clip_vit_b16_dct',
        model_class=CLIPWithDCT,
        config={
            'clip_model': 'ViT-B-16',
            'pretrained': 'openai',
            'dct_dim': 128,
            'fusion_dim': 512,
        }
    )


# 註冊 DINOv2 系列模型
if DINO_AVAILABLE:
    for dino_name, dino_cfg in DINO_CONFIGS.items():
        register_model(
            name=dino_name,
            model_class=DINOv2Classifier,
            config=dino_cfg
        )


# 註冊 ConvNeXt 系列模型
if CONVNEXT_AVAILABLE:
    for convnext_name, convnext_cfg in CONVNEXT_CONFIGS.items():
        register_model(
            name=convnext_name,
            model_class=ConvNeXtClassifier,
            config=convnext_cfg
        )


def list_available_models() -> List[str]:
    """
    列出所有可用的模型
    
    Returns:
        模型名稱列表
    """
    return list(MODEL_REGISTRY.keys())


def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    獲取模型配置
    
    Args:
        model_name: 模型名稱
        
    Returns:
        模型配置字典
    """
    if model_name not in MODEL_CONFIGS:
        return {}
    return MODEL_CONFIGS[model_name].copy()


def create_model(
    model_name: str,
    num_classes: int = 2,
    pretrained: bool = True,
    **kwargs
) -> BaseModel:
    """
    創建模型實例
    
    Args:
        model_name: 模型名稱（如 'efficientnet_b4'）
        num_classes: 輸出類別數
        pretrained: 是否使用預訓練權重
        **kwargs: 傳遞給模型的其他參數
        
    Returns:
        BaseModel 實例
        
    Raises:
        ValueError: 如果模型名稱不存在
    """
    if model_name not in MODEL_REGISTRY:
        available = list_available_models()
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {available}"
        )
    
    model_class = MODEL_REGISTRY[model_name]
    
    # 對於 EfficientNet，需要傳入 model_name
    if model_class == EfficientNetModel:
        return model_class(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )
    
    # 對於 EfficientNet + DCT，需要傳入 backbone_name
    if model_class in (EfficientNetDCT, EfficientNetDCTAttention):
        # 從模型名稱提取 backbone 名稱
        # efficientnet_b4_dct -> efficientnet_b4
        parts = model_name.split('_')
        backbone_name = '_'.join(parts[:2])  # efficientnet_b4
        return model_class(
            backbone_name=backbone_name,
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )
    
    # 對於 CLIP 模型，過濾掉不支援的參數
    if CLIP_AVAILABLE and model_class == CLIPClassifier:
        clip_cfg = CLIP_CONFIGS.get(model_name, {})
        # CLIP 只支援這些參數
        clip_kwargs = {k: v for k, v in kwargs.items() 
                       if k in ('dropout', 'hidden_dim', 'freeze_encoder')}
        return model_class(
            clip_model=clip_cfg.get('model_name', 'ViT-B-32'),
            pretrained=clip_cfg.get('pretrained', 'openai'),
            num_classes=num_classes,
            freeze_encoder=True,  # 預設凍結
            **clip_kwargs
        )
    
    # 對於 CLIP + DCT 模型，過濾掉不支援的參數
    if CLIP_AVAILABLE and model_class == CLIPWithDCT:
        model_cfg = MODEL_CONFIGS.get(model_name, {})
        # CLIP+DCT 支援的參數
        clip_dct_kwargs = {k: v for k, v in kwargs.items() 
                          if k in ('dropout', 'dct_dim', 'fusion_dim', 'freeze_encoder')}
        return model_class(
            clip_model=model_cfg.get('clip_model', 'ViT-B-32'),
            pretrained=model_cfg.get('pretrained', 'openai'),
            num_classes=num_classes,
            dct_dim=clip_dct_kwargs.pop('dct_dim', model_cfg.get('dct_dim', 128)),
            fusion_dim=clip_dct_kwargs.pop('fusion_dim', model_cfg.get('fusion_dim', 512)),
            freeze_encoder=True,
            **clip_dct_kwargs
        )
    
    # 其他模型直接創建
    return model_class(
        num_classes=num_classes,
        pretrained=pretrained,
        **kwargs
    )


def create_model_from_config(config: Dict[str, Any]) -> BaseModel:
    """
    從配置字典創建模型
    
    配置格式:
    {
        "model": {
            "name": "efficientnet_b4",
            "num_classes": 2,
            "pretrained": true,
            "dropout": 0.3,
            "drop_path_rate": 0.2
        }
    }
    
    Args:
        config: 配置字典
        
    Returns:
        BaseModel 實例
    """
    model_config = config.get('model', {})
    
    model_name = model_config.get('name', 'efficientnet_b4')
    num_classes = model_config.get('num_classes', 2)
    pretrained = model_config.get('pretrained', True)
    
    # 提取其他參數
    extra_kwargs = {}
    for key in ['dropout', 'drop_path_rate']:
        if key in model_config:
            extra_kwargs[key] = model_config[key]
    
    return create_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        **extra_kwargs
    )


def get_model_info(model_name: str) -> str:
    """
    獲取模型的詳細信息
    
    Args:
        model_name: 模型名稱
        
    Returns:
        格式化的模型信息字符串
    """
    if model_name not in MODEL_REGISTRY:
        return f"Model '{model_name}' not found."
    
    config = get_model_config(model_name)
    
    info_lines = [
        f"Model: {model_name}",
        f"  Class: {MODEL_REGISTRY[model_name].__name__}",
    ]
    
    if config:
        info_lines.append("  Config:")
        for key, value in config.items():
            info_lines.append(f"    {key}: {value}")
    
    return "\n".join(info_lines)


def print_available_models():
    """打印所有可用模型的信息"""
    print("=" * 60)
    print("Available Models")
    print("=" * 60)
    
    for model_name in list_available_models():
        print(get_model_info(model_name))
        print("-" * 40)
