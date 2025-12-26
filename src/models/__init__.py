"""
模型模組

提供統一的模型介面和工廠函數，支援多種 backbone 架構。

可用模型:
- EfficientNet-B0~B4: 輕量到中等規模的 CNN，適合 deepfake 檢測
- EfficientNet-B0~B4-DCT: 結合頻域特徵的 CNN，更好地檢測 GAN 偽影
- EfficientNet-B0~B4-DCT-Attn: 帶交叉注意力的頻域融合模型
- CLIP-ViT-B32/B16/L14: CLIP 視覺編碼器 + 分類頭，更好的泛化能力
- CLIP-ViT-B32/B16-DCT: CLIP + DCT 頻域融合

使用方式:
    from src.models import create_model, list_available_models
    
    # 列出可用模型
    print(list_available_models())
    
    # 創建模型
    model = create_model('efficientnet_b4', num_classes=2, pretrained=True)
    
    # 創建 CLIP 模型
    model = create_model('clip_vit_b32', num_classes=2)
"""

from .factory import create_model, list_available_models, get_model_config, create_model_from_config
from .base import BaseModel
from .dct import DCTFeatureExtractor, MultiScaleDCT
from .efficientnet_dct import EfficientNetDCT, EfficientNetDCTAttention
from .ensemble import ModelEnsemble, EnsembleFromCheckpoints, create_ensemble

# 嘗試導入 CLIP 模型
try:
    from .clip_model import CLIPClassifier, CLIPWithDCT
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

__all__ = [
    'create_model',
    'create_model_from_config',
    'list_available_models', 
    'get_model_config',
    'BaseModel',
    'DCTFeatureExtractor',
    'MultiScaleDCT',
    'EfficientNetDCT',
    'EfficientNetDCTAttention',
    'ModelEnsemble',
    'EnsembleFromCheckpoints',
    'create_ensemble',
]

# 如果 CLIP 可用，加入 export
if CLIP_AVAILABLE:
    __all__.extend(['CLIPClassifier', 'CLIPWithDCT'])
