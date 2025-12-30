"""
EfficientNet + DCT 融合模型

結合 CNN 空間特徵和 DCT 頻域特徵的雙流架構：
- 空間流：EfficientNet 提取視覺語義特徵
- 頻域流：DCT 提取頻譜分佈特徵
- 融合層：結合兩種特徵進行分類

這種設計可以同時捕捉：
1. 視覺上的不自然（臉部扭曲、邊界問題）
2. 頻域上的偽影（GAN fingerprint、壓縮痕跡）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import timm

from .base import BaseModel
from .dct import DCTFeatureExtractor, MultiScaleDCT


class EfficientNetDCT(BaseModel):
    """
    EfficientNet + DCT 雙流融合模型
    
    架構：
    ┌─────────────┐     ┌─────────────┐
    │   Input     │     │   Input     │
    │   Image     │     │   Image     │
    └──────┬──────┘     └──────┬──────┘
           │                   │
           ▼                   ▼
    ┌─────────────┐     ┌─────────────┐
    │ EfficientNet│     │  DCT        │
    │  Backbone   │     │  Features   │
    └──────┬──────┘     └──────┬──────┘
           │                   │
           ▼                   ▼
    ┌─────────────┐     ┌─────────────┐
    │ Spatial     │     │ Frequency   │
    │ Features    │     │ Features    │
    └──────┬──────┘     └──────┬──────┘
           │                   │
           └────────┬──────────┘
                    │
                    ▼
             ┌─────────────┐
             │   Fusion    │
             │   Module    │
             └──────┬──────┘
                    │
                    ▼
             ┌─────────────┐
             │ Classifier  │
             └─────────────┘
    """
    
    def __init__(
        self,
        backbone_name: str = 'efficientnet_b4',
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
        drop_path_rate: float = 0.2,
        dct_dim: int = 128,
        fusion_dim: int = 512,
        use_multiscale_dct: bool = False,
        freeze_backbone_bn: bool = False,
    ):
        """
        Args:
            backbone_name: EfficientNet 變體名稱
            num_classes: 分類類別數
            pretrained: 是否使用預訓練權重
            dropout: Dropout 比例
            drop_path_rate: DropPath 比例（Stochastic Depth）
            dct_dim: DCT 特徵維度
            fusion_dim: 融合層維度
            use_multiscale_dct: 是否使用多尺度 DCT
            freeze_backbone_bn: 是否凍結 backbone 的 BatchNorm
        """
        super().__init__(num_classes=num_classes)
        
        self._model_name = f'{backbone_name}_dct'
        self.backbone_name = backbone_name
        self.use_multiscale_dct = use_multiscale_dct
        self.freeze_backbone_bn = freeze_backbone_bn
        
        # ========== 空間流：EfficientNet ==========
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # 移除分類頭
            drop_rate=dropout,
            drop_path_rate=drop_path_rate,
        )
        
        # 獲取 backbone 輸出維度
        self.spatial_dim = self.backbone.num_features
        
        # ========== 頻域流：DCT ==========
        if use_multiscale_dct:
            self.dct_extractor = MultiScaleDCT(
                scales=(56, 112, 224),
                output_dim=dct_dim
            )
        else:
            self.dct_extractor = DCTFeatureExtractor(
                size=224,
                output_dim=dct_dim
            )
        
        self.dct_dim = dct_dim
        
        # ========== 融合模組 ==========
        combined_dim = self.spatial_dim + dct_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.LayerNorm(fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        
        # ========== 分類頭 (二元分類輸出 1，多類別輸出 num_classes) ==========
        output_dim = 1 if num_classes == 2 else num_classes
        self.classifier = nn.Linear(fusion_dim // 2, output_dim)
        
        # 初始化分類頭
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def train(self, mode: bool = True):
        """覆寫 train 方法，可選凍結 BN"""
        super().train(mode)
        if self.freeze_backbone_bn and mode:
            for module in self.backbone.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self
    
    def forward(
        self, 
        x: torch.Tensor,
        return_features: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向傳播
        
        Args:
            x: (B, C, H, W) 輸入圖像
            return_features: 是否返回中間特徵（用於分析）
            
        Returns:
            logits: (B, num_classes) 分類 logits
            features: (可選) 包含中間特徵的字典
        """
        # 空間特徵
        spatial_feat = self.backbone(x)  # (B, spatial_dim)
        
        # 頻域特徵
        dct_feat = self.dct_extractor(x)  # (B, dct_dim)
        
        # 融合
        combined = torch.cat([spatial_feat, dct_feat], dim=-1)  # (B, combined_dim)
        fused = self.fusion(combined)  # (B, fusion_dim // 2)
        
        # 分類
        logits = self.classifier(fused)
        
        if return_features:
            features = {
                'spatial': spatial_feat,
                'dct': dct_feat,
                'fused': fused,
            }
            return logits, features
        
        return logits
    
    def get_classifier(self) -> nn.Module:
        """返回分類器部分（用於分層學習率）"""
        return nn.ModuleList([self.fusion, self.classifier])
    
    def get_backbone(self) -> nn.Module:
        """返回 backbone 部分"""
        return self.backbone
    
    def get_dct_module(self) -> nn.Module:
        """返回 DCT 模組"""
        return self.dct_extractor
    
    def get_param_groups(self, base_lr: float, backbone_lr_scale: float = 0.1) -> List[Dict]:
        """
        獲取分層學習率的參數組
        
        三層學習率策略：
        - backbone: base_lr * backbone_lr_scale
        - dct: base_lr * 0.5（新模組，稍微小一點的學習率）
        - fusion + classifier: base_lr
        """
        backbone_params = list(self.backbone.parameters())
        dct_params = list(self.dct_extractor.parameters())
        fusion_params = list(self.fusion.parameters()) + list(self.classifier.parameters())
        
        param_groups = [
            {
                'params': backbone_params,
                'lr': base_lr * backbone_lr_scale,
                'name': 'backbone'
            },
            {
                'params': dct_params,
                'lr': base_lr * 0.5,  # DCT 模組使用中等學習率
                'name': 'dct'
            },
            {
                'params': fusion_params,
                'lr': base_lr,
                'name': 'fusion_classifier'
            },
        ]
        
        return param_groups
    
    def freeze_backbone(self, freeze: bool = True):
        """凍結/解凍 backbone"""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze


class EfficientNetDCTAttention(BaseModel):
    """
    帶注意力機制的 EfficientNet + DCT 模型
    
    使用交叉注意力讓空間特徵和頻域特徵相互增強。
    """
    
    def __init__(
        self,
        backbone_name: str = 'efficientnet_b4',
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
        drop_path_rate: float = 0.2,
        dct_dim: int = 256,
        num_heads: int = 8,
    ):
        super().__init__(num_classes=num_classes)
        
        self._model_name = f'{backbone_name}_dct_attn'
        
        # 空間流
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=dropout,
            drop_path_rate=drop_path_rate,
        )
        self.spatial_dim = self.backbone.num_features
        
        # 頻域流
        self.dct_extractor = DCTFeatureExtractor(size=224, output_dim=dct_dim)
        
        # 投影到相同維度
        hidden_dim = max(self.spatial_dim, dct_dim)
        self.spatial_proj = nn.Linear(self.spatial_dim, hidden_dim)
        self.dct_proj = nn.Linear(dct_dim, hidden_dim)
        
        # 交叉注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # 融合
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # 分類器
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 提取特徵
        spatial_feat = self.backbone(x)
        dct_feat = self.dct_extractor(x)
        
        # 投影
        spatial_proj = self.spatial_proj(spatial_feat).unsqueeze(1)  # (B, 1, D)
        dct_proj = self.dct_proj(dct_feat).unsqueeze(1)  # (B, 1, D)
        
        # 交叉注意力：空間特徵關注頻域特徵
        attended, _ = self.cross_attention(
            query=spatial_proj,
            key=dct_proj,
            value=dct_proj
        )
        
        # 融合
        combined = torch.cat([spatial_proj.squeeze(1), attended.squeeze(1)], dim=-1)
        fused = self.fusion(combined)
        
        return self.classifier(fused)
    
    def get_classifier(self) -> nn.Module:
        return nn.ModuleList([self.fusion, self.classifier])
    
    def get_backbone(self) -> nn.Module:
        return self.backbone


if __name__ == '__main__':
    # 測試模型
    print("Testing EfficientNet + DCT models...")
    
    # 基本融合模型
    model = EfficientNetDCT(
        backbone_name='efficientnet_b0',
        num_classes=2,
        pretrained=False,  # 測試時不下載
        dct_dim=128,
    )
    
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"EfficientNetDCT output: {y.shape}")
    print(f"Parameters: {model.count_parameters():,}")
    
    # 測試返回特徵
    y, features = model(x, return_features=True)
    print(f"Spatial features: {features['spatial'].shape}")
    print(f"DCT features: {features['dct'].shape}")
    print(f"Fused features: {features['fused'].shape}")
    
    # 注意力模型
    model_attn = EfficientNetDCTAttention(
        backbone_name='efficientnet_b0',
        num_classes=2,
        pretrained=False,
    )
    y_attn = model_attn(x)
    print(f"\nEfficientNetDCTAttention output: {y_attn.shape}")
    print(f"Parameters: {model_attn.count_parameters():,}")
