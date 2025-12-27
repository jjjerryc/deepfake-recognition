"""
DINOv2 Deepfake Detection Model

使用 DINOv2 ViT-Large/14 作為 backbone，
支援差異化解凍策略和多種池化方式。

DINOv2 的優勢：
1. 自監督學習，對底層視覺結構敏感
2. 能捕捉細微的紋理不一致和邊緣偽影
3. 比 CLIP 更適合像素級別的偽造檢測
"""

from typing import Dict, Any, Optional

import torch
import torch.nn as nn

from .base import BaseModel


class DINOv2Classifier(BaseModel):
    """
    DINOv2 Visual Encoder + Classification Head
    
    支援：
    - 差異化解凍策略 (解凍最後 N 層)
    - 多種池化方式 (cls token / patch mean)
    - 分層學習率
    """
    
    def __init__(
        self,
        backbone: str = "dinov2_vitl14",
        num_classes: int = 2,
        unfreeze_layers: int = 2,
        unfreeze_norm: bool = True,
        pooling: str = "cls",  # "cls" or "avg"
        hidden_dim: int = 512,
        dropout: float = 0.4,
        use_batchnorm: bool = True,
    ):
        super().__init__(num_classes=num_classes)
        
        self.backbone_name = backbone
        self.unfreeze_layers = unfreeze_layers
        self.pooling = pooling
        
        # 載入 DINOv2
        print(f"Loading {backbone}...")
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2', 
            backbone,
            pretrained=True
        )
        
        # 獲取 embedding 維度
        if backbone == "dinov2_vitl14":
            self.embed_dim = 1024
        elif backbone == "dinov2_vitb14":
            self.embed_dim = 768
        elif backbone == "dinov2_vits14":
            self.embed_dim = 384
        elif backbone == "dinov2_vitg14":
            self.embed_dim = 1536
        else:
            # 動態推斷
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                out = self.backbone(dummy)
                self.embed_dim = out.shape[-1]
        
        # 設定凍結策略
        self._setup_freeze(unfreeze_norm)
        
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
    
    def _setup_freeze(self, unfreeze_norm: bool = True):
        """凍結除了最後 N 層以外的所有層"""
        # 先凍結所有
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # 獲取 transformer blocks
        if hasattr(self.backbone, 'blocks'):
            blocks = self.backbone.blocks
            total_blocks = len(blocks)
            
            if self.unfreeze_layers > 0:
                # 解凍最後 N 層 Blocks
                unfreeze_start = total_blocks - self.unfreeze_layers
                for i in range(unfreeze_start, total_blocks):
                    for param in blocks[i].parameters():
                        param.requires_grad = True
                
                print(f"🔓 Unfreezing last {self.unfreeze_layers}/{total_blocks} transformer blocks")
        
        # 解凍 Norm 層
        if unfreeze_norm and hasattr(self.backbone, 'norm'):
            for param in self.backbone.norm.parameters():
                param.requires_grad = True
    
    def _print_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"\n📊 DINOv2 Model Statistics:")
        print(f"   Backbone: {self.backbone_name}")
        print(f"   Pooling: {self.pooling}")
        print(f"   Total params: {total_params:,}")
        print(f"   Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        print(f"   Frozen: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pooling == "avg":
            # 使用 patch tokens 的平均值
            features_dict = self.backbone.forward_features(x)
            features = features_dict['x_norm_patchtokens'].mean(dim=1)
        else:
            # 使用 CLS token (預設)
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
    
    def get_param_groups(self, base_lr: float, backbone_lr_multiplier: float = 0.1) -> list:
        """分層學習率"""
        param_groups = []
        
        # Backbone (解凍的層)
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': base_lr * backbone_lr_multiplier,
                'name': 'backbone'
            })
        
        # Classifier
        param_groups.append({
            'params': list(self.classifier.parameters()),
            'lr': base_lr,
            'name': 'classifier'
        })
        
        return param_groups


# DINOv2 模型配置
DINO_CONFIGS = {
    'dino_vits14': {
        'backbone': 'dinov2_vits14',
        'embed_dim': 384,
    },
    'dino_vitb14': {
        'backbone': 'dinov2_vitb14',
        'embed_dim': 768,
    },
    'dino_vitl14': {
        'backbone': 'dinov2_vitl14',
        'embed_dim': 1024,
    },
    'dino_vitg14': {
        'backbone': 'dinov2_vitg14',
        'embed_dim': 1536,
    },
}
