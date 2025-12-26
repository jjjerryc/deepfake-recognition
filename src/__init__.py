"""
Deepfake Detection - Multi-Model Framework v3.3

支援多種 backbone 架構的 Deepfake 檢測框架。

模塊:
    - models: 模型工廠和各種 backbone 實現 (EfficientNet, CLIP, DCT)
    - train: 訓練腳本 (支援 Mixup/CutMix, ReduceLROnPlateau)
    - inference: 單模型推論腳本
    - ensemble_inference: 集成推論腳本 (config.json 控制)
    - convert_submission: 格式轉換 (機率 → real/fake)
"""

from .models import create_model, list_available_models, get_model_config

__version__ = "3.3.0"
__all__ = [
    # Model Factory
    "create_model",
    "list_available_models",
    "get_model_config",
]