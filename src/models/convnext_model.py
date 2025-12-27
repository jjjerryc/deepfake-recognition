"""
ConvNeXt V2 Deepfake Detection Model

使用 ConvNeXt V2 作為 backbone，
CNN 架構擅長捕捉紋理特徵。

ConvNeXt 的優勢：
1. 純 CNN 架構，對局部紋理敏感
2. 高效的計算效率
3. 強大的歸納偏置 (局部性、平移不變性)
"""

from typing import Dict, Any, Optional

import torch
import torch.nn as nn

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

from .base import BaseModel


class ConvNeXtClassifier(BaseModel):
    """
    ConvNeXt V2 + Classification Head
    
    支援多種 ConvNeXt 變體
    """
    
    def __init__(
        self,
        backbone: str = "convnextv2_base",
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        hidden_dim: int = 512,
        dropout: float = 0.4,
        use_batchnorm: bool = True,
    ):
        super().__init__(num_classes=num_classes)
        
        if not TIMM_AVAILABLE:
            raise ImportError("Please install timm: pip install timm")
        
        self.backbone_name = backbone
        self.freeze_backbone_flag = freeze_backbone
        
        # 載入 ConvNeXt
        print(f"Loading {backbone}...")
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0  # 移除原本的分類頭
        )
        
        # 獲取 embedding 維度
        self.embed_dim = self._get_embed_dim(backbone)
        
        # 凍結策略
        if freeze_backbone:
            self._freeze_backbone()
        
        # 分類頭
        if use_batchnorm:
            self.classifier = nn.Sequential(
                nn.Linear(self.embed_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes if num_classes > 2 else 1),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.embed_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes if num_classes > 2 else 1),
            )
        
        self._print_info()
    
    def _get_embed_dim(self, backbone: str) -> int:
        """根據 backbone 名稱獲取 embedding 維度"""
        embed_dims = {
            'convnext_tiny': 768,
            'convnext_small': 768,
            'convnext_base': 1024,
            'convnext_large': 1536,
            'convnextv2_tiny': 768,
            'convnextv2_small': 768,
            'convnextv2_base': 1024,
            'convnextv2_large': 1536,
            'convnextv2_huge': 2816,
        }
        
        if backbone in embed_dims:
            return embed_dims[backbone]
        
        # 動態推斷
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = self.backbone(dummy)
            return out.shape[-1]
    
    def _freeze_backbone(self):
        """凍結 backbone"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("🔒 Backbone frozen")
    
    def _print_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"\n📊 ConvNeXt Model Statistics:")
        print(f"   Backbone: {self.backbone_name}")
        print(f"   Total params: {total_params:,}")
        print(f"   Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        print(f"   Frozen: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)
    
    def get_classifier(self) -> nn.Module:
        return self.classifier
    
    def get_backbone(self) -> nn.Module:
        return self.backbone
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        return {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'input_size': 224,
            'interpolation': 'bicubic',
        }


# ConvNeXt 模型配置
CONVNEXT_CONFIGS = {
    'convnext_tiny': {
        'backbone': 'convnext_tiny',
        'embed_dim': 768,
    },
    'convnext_small': {
        'backbone': 'convnext_small',
        'embed_dim': 768,
    },
    'convnext_base': {
        'backbone': 'convnext_base',
        'embed_dim': 1024,
    },
    'convnextv2_tiny': {
        'backbone': 'convnextv2_tiny',
        'embed_dim': 768,
    },
    'convnextv2_base': {
        'backbone': 'convnextv2_base',
        'embed_dim': 1024,
    },
    'convnextv2_large': {
        'backbone': 'convnextv2_large',
        'embed_dim': 1536,
    },
}
