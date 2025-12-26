"""
EfficientNet 系列模型

使用 timm 庫實現 EfficientNet-B0 到 B4，支援 ImageNet 預訓練權重。

EfficientNet 特點：
- Compound Scaling：同時優化深度、寬度和解析度
- MBConv 結構：結合 depthwise separable convolution 和 squeeze-and-excitation
- 對 deepfake 偽影敏感：保留了高頻細節特徵

模型規格：
| Model | Params | Top-1 Acc | Input Size |
|-------|--------|-----------|------------|
| B0    | 5.3M   | 77.1%     | 224        |
| B1    | 7.8M   | 79.1%     | 240        |
| B2    | 9.2M   | 80.1%     | 260        |
| B3    | 12M    | 81.6%     | 300        |
| B4    | 19M    | 82.9%     | 380        |
"""

from typing import Dict, Any, Optional

import torch
import torch.nn as nn

try:
    import timm
    from timm.data import resolve_data_config
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

from .base import BaseModel


# EfficientNet 變體的預設配置
EFFICIENTNET_CONFIGS = {
    'efficientnet_b0': {
        'input_size': 224,
        'params': '5.3M',
        'description': '輕量級，適合快速實驗',
    },
    'efficientnet_b1': {
        'input_size': 240,
        'params': '7.8M',
        'description': '平衡效能和速度',
    },
    'efficientnet_b2': {
        'input_size': 260,
        'params': '9.2M',
        'description': '中等規模',
    },
    'efficientnet_b3': {
        'input_size': 300,
        'params': '12M',
        'description': '較大容量',
    },
    'efficientnet_b4': {
        'input_size': 380,
        'params': '19M',
        'description': 'Deepfake 檢測推薦選擇',
    },
}


class EfficientNetModel(BaseModel):
    """
    EfficientNet 模型封裝
    
    基於 timm 庫實現，支援 B0-B4 變體和預訓練權重。
    
    Args:
        model_name: 模型名稱，如 'efficientnet_b0', 'efficientnet_b4'
        num_classes: 輸出類別數
        pretrained: 是否使用 ImageNet 預訓練權重
        dropout: 分類頭的 dropout 比率
        drop_path_rate: DropPath 比率（正則化）
    """
    
    def __init__(
        self,
        model_name: str = 'efficientnet_b4',
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
        drop_path_rate: float = 0.2,
    ):
        super().__init__(num_classes=num_classes)
        
        if not TIMM_AVAILABLE:
            raise ImportError(
                "timm is required for EfficientNet models. "
                "Install it with: pip install timm"
            )
        
        self._model_name = model_name
        self.pretrained = pretrained
        
        # 創建 timm 模型
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
            drop_path_rate=drop_path_rate,
        )
        
        # 獲取模型配置
        self.data_config = resolve_data_config({}, model=self.model)
        
        # 保存原始分類頭，用於分層學習率
        # timm 的 EfficientNet 使用 'classifier' 作為最後的全連接層
        self._classifier = self.model.classifier
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向傳播"""
        return self.model(x)
    
    def get_classifier(self) -> nn.Module:
        """獲取分類頭"""
        return self.model.classifier
    
    def get_backbone(self) -> nn.Module:
        """
        獲取 backbone（除了分類頭以外的所有層）
        
        timm 的 EfficientNet 結構:
        - conv_stem: 初始卷積
        - bn1: 初始 BN
        - blocks: MBConv blocks
        - conv_head: 最終卷積
        - bn2: 最終 BN
        - global_pool: 全局池化
        - classifier: 分類頭
        """
        # 創建一個包含除 classifier 外所有模組的容器
        class BackboneWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.conv_stem = model.conv_stem
                self.bn1 = model.bn1
                self.blocks = model.blocks
                self.conv_head = model.conv_head
                self.bn2 = model.bn2
                
            def forward(self, x):
                x = self.conv_stem(x)
                x = self.bn1(x)
                x = self.blocks(x)
                x = self.conv_head(x)
                x = self.bn2(x)
                return x
        
        return BackboneWrapper(self.model)
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """獲取 timm 模型的預處理配置"""
        return {
            'mean': list(self.data_config['mean']),
            'std': list(self.data_config['std']),
            'input_size': self.data_config['input_size'][-1],  # H 或 W
            'interpolation': self.data_config['interpolation'],
        }
    
    def get_feature_dim(self) -> int:
        """獲取特徵維度（分類頭輸入維度）"""
        return self.model.num_features
    
    def reset_classifier(self, num_classes: int, dropout: float = 0.3):
        """
        重置分類頭
        
        用於遷移學習時更換輸出類別數
        """
        self.num_classes = num_classes
        self.model.reset_classifier(num_classes=num_classes)
        self._classifier = self.model.classifier


def create_efficientnet(
    variant: str = 'b4',
    num_classes: int = 2,
    pretrained: bool = True,
    **kwargs
) -> EfficientNetModel:
    """
    創建 EfficientNet 模型的便捷函數
    
    Args:
        variant: 模型變體 ('b0', 'b1', 'b2', 'b3', 'b4')
        num_classes: 輸出類別數
        pretrained: 是否使用預訓練權重
        **kwargs: 傳遞給 EfficientNetModel 的其他參數
        
    Returns:
        EfficientNetModel 實例
    """
    model_name = f'efficientnet_{variant}'
    
    if model_name not in EFFICIENTNET_CONFIGS:
        available = list(EFFICIENTNET_CONFIGS.keys())
        raise ValueError(
            f"Unknown EfficientNet variant: {variant}. "
            f"Available: {available}"
        )
    
    return EfficientNetModel(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        **kwargs
    )
