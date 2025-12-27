"""
模型模組

提供統一的模型介面和工廠函數，支援多種 backbone 架構。

可用模型:
- EfficientNet-B0~B4: 輕量到中等規模的 CNN，適合 deepfake 檢測
- EfficientNet-B0~B4-DCT: 結合頻域特徵的 CNN，更好地檢測 GAN 偽影
- EfficientNet-B0~B4-DCT-Attn: 帶交叉注意力的頻域融合模型
- CLIP-ViT-B32/B16/L14: CLIP 視覺編碼器 + 分類頭，更好的泛化能力
- CLIP-ViT-B32/B16-DCT: CLIP + DCT 頻域融合
- DINOv2-ViT-S/B/L/G: 自監督視覺模型，對底層視覺結構敏感
- ConvNeXt-V2: 現代化 CNN，擅長紋理特徵

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

# 嘗試導入 DINOv2 模型
try:
    from .dino_model import DINOv2Classifier, DINO_CONFIGS
    DINO_AVAILABLE = True
except ImportError:
    DINO_AVAILABLE = False

# 嘗試導入 ConvNeXt 模型
try:
    from .convnext_model import ConvNeXtClassifier, CONVNEXT_CONFIGS
    CONVNEXT_AVAILABLE = True
except ImportError:
    CONVNEXT_AVAILABLE = False

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

# 如果 DINOv2 可用，加入 export
if DINO_AVAILABLE:
    __all__.extend(['DINOv2Classifier', 'DINO_CONFIGS'])

# 如果 ConvNeXt 可用，加入 export
if CONVNEXT_AVAILABLE:
    __all__.extend(['ConvNeXtClassifier', 'CONVNEXT_CONFIGS'])

