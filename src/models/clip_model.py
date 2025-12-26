"""
CLIP-based Deepfake Detection Model

使用 OpenCLIP 的 visual encoder 作為 backbone
凍結 encoder 只訓練分類頭，達到更好的泛化能力。

CLIP 的優勢：
1. 在大量圖像-文字配對數據上預訓練，學到更通用的視覺特徵
2. 對 "什麼是自然圖像" 有深刻理解
3. 更容易泛化到未見過的生成模型
"""

from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import open_clip
except ImportError:
    raise ImportError("Please install open_clip_torch: pip install open_clip_torch")

from .base import BaseModel


# 支援的 CLIP 模型配置
CLIP_CONFIGS = {
    'clip_vit_b32': {
        'model_name': 'ViT-B-32',
        'pretrained': 'openai',
        'embed_dim': 512,
    },
    'clip_vit_b16': {
        'model_name': 'ViT-B-16',
        'pretrained': 'openai',
        'embed_dim': 512,
    },
    'clip_vit_l14': {
        'model_name': 'ViT-L-14',
        'pretrained': 'openai',
        'embed_dim': 768,
    },
    'clip_vit_b16_laion': {
        'model_name': 'ViT-B-16',
        'pretrained': 'laion2b_s34b_b88k',
        'embed_dim': 512,
    },
    'clip_convnext_base': {
        'model_name': 'convnext_base_w',
        'pretrained': 'laion2b_s13b_b82k',
        'embed_dim': 640,
    },
}


class CLIPClassifier(BaseModel):
    """
    CLIP Visual Encoder + Classification Head
    
    凍結 CLIP encoder，只訓練分類頭
    """
    
    def __init__(
        self,
        clip_model: str = 'ViT-B-32',
        pretrained: str = 'openai',
        num_classes: int = 2,
        dropout: float = 0.5,
        freeze_encoder: bool = True,
        hidden_dim: int = 512,
    ):
        super().__init__(num_classes=num_classes)
        
        self.clip_model_name = clip_model
        self.pretrained = pretrained
        self.freeze_encoder = freeze_encoder
        
        # 載入 CLIP 模型
        print(f"Loading CLIP: {clip_model} ({pretrained})")
        self.clip, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_model, pretrained=pretrained
        )
        
        # 只保留 visual encoder
        self.encoder = self.clip.visual
        
        # 獲取特徵維度
        # 嘗試不同的方式獲取 embed_dim
        if hasattr(self.encoder, 'output_dim'):
            self.embed_dim = self.encoder.output_dim
        elif hasattr(self.encoder, 'embed_dim'):
            self.embed_dim = self.encoder.embed_dim
        elif hasattr(self.clip, 'visual') and hasattr(self.clip.visual, 'output_dim'):
            self.embed_dim = self.clip.visual.output_dim
        else:
            # 動態推斷：做一次 forward pass
            with torch.no_grad():
                dummy_input = torch.zeros(1, 3, 224, 224)
                dummy_output = self.encoder(dummy_input)
                if dummy_output.dim() > 2:
                    dummy_output = dummy_output.mean(dim=1)
                self.embed_dim = dummy_output.shape[-1]
            
        print(f"CLIP embed_dim: {self.embed_dim}")
        
        # 凍結 encoder
        if freeze_encoder:
            self.freeze_backbone()
        
        # 分類頭
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self._model_name = f"CLIP_{clip_model.replace('-', '_')}"
        
        # 刪除不需要的 text encoder 以節省記憶體
        del self.clip.transformer
        del self.clip.token_embedding
        del self.clip.ln_final
        if hasattr(self.clip, 'text_projection'):
            del self.clip.text_projection
        
    def freeze_backbone(self):
        """凍結 CLIP encoder"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("🔒 CLIP encoder frozen")
        
    def unfreeze_backbone(self):
        """解凍 CLIP encoder"""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("🔓 CLIP encoder unfrozen")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            x: [B, 3, H, W] 輸入圖像
            
        Returns:
            [B, num_classes] logits
        """
        # CLIP visual encoder
        with torch.set_grad_enabled(not self.freeze_encoder):
            features = self.encoder(x)
        
        # 確保是 2D tensor [B, embed_dim]
        if features.dim() > 2:
            features = features.mean(dim=1)  # 如果是 [B, N, D]，取平均
        
        # 分類
        logits = self.classifier(features)
        
        return logits
    
    def get_classifier(self) -> nn.Module:
        return self.classifier
    
    def get_backbone(self) -> nn.Module:
        return self.encoder
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """CLIP 使用自己的預處理參數"""
        # OpenCLIP 的標準化參數
        return {
            'mean': [0.48145466, 0.4578275, 0.40821073],
            'std': [0.26862954, 0.26130258, 0.27577711],
            'input_size': 224,
            'interpolation': 'bicubic',
        }
    
    def get_param_groups(
        self, 
        base_lr: float = 1e-4,
        backbone_lr_scale: float = 0.0  # 預設凍結 backbone
    ) -> List[Dict[str, Any]]:
        """
        獲取參數組，支援分層學習率
        """
        param_groups = []
        
        # Encoder 參數（如果沒凍結）
        if not self.freeze_encoder:
            encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
            if encoder_params:
                param_groups.append({
                    'params': encoder_params,
                    'lr': base_lr * backbone_lr_scale,
                    'name': 'clip_encoder'
                })
        
        # Classifier 參數
        classifier_params = list(self.classifier.parameters())
        param_groups.append({
            'params': classifier_params,
            'lr': base_lr,
            'name': 'classifier'
        })
        
        return param_groups


class CLIPWithDCT(BaseModel):
    """
    CLIP Visual Encoder + DCT Frequency Features
    
    結合 CLIP 的語意特徵和 DCT 的頻率特徵
    """
    
    def __init__(
        self,
        clip_model: str = 'ViT-B-32',
        pretrained: str = 'openai',
        num_classes: int = 2,
        dropout: float = 0.5,
        freeze_encoder: bool = True,
        dct_dim: int = 128,
        fusion_dim: int = 512,
    ):
        super().__init__(num_classes=num_classes)
        
        from .dct import DCTFeatureExtractor
        
        self.clip_model_name = clip_model
        self.freeze_encoder = freeze_encoder
        
        # 載入 CLIP
        print(f"Loading CLIP: {clip_model} ({pretrained})")
        self.clip, _, _ = open_clip.create_model_and_transforms(
            clip_model, pretrained=pretrained
        )
        self.encoder = self.clip.visual
        
        # 獲取特徵維度
        if hasattr(self.encoder, 'output_dim'):
            self.embed_dim = self.encoder.output_dim
        elif hasattr(self.encoder, 'embed_dim'):
            self.embed_dim = self.encoder.embed_dim
        elif hasattr(self.clip, 'visual') and hasattr(self.clip.visual, 'output_dim'):
            self.embed_dim = self.clip.visual.output_dim
        else:
            # 動態推斷
            with torch.no_grad():
                dummy_input = torch.zeros(1, 3, 224, 224)
                dummy_output = self.encoder(dummy_input)
                if dummy_output.dim() > 2:
                    dummy_output = dummy_output.mean(dim=1)
                self.embed_dim = dummy_output.shape[-1]
            
        print(f"CLIP embed_dim: {self.embed_dim}")
        
        # 凍結 encoder
        if freeze_encoder:
            self.freeze_backbone()
        
        # DCT 特徵提取器
        self.dct_extractor = DCTFeatureExtractor(output_dim=dct_dim)
        
        # 融合層
        total_dim = self.embed_dim + dct_dim
        self.fusion = nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Linear(total_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # 分類頭
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_classes)
        )
        
        self._model_name = f"CLIP_{clip_model.replace('-', '_')}_DCT"
        
        # 刪除 text encoder
        del self.clip.transformer
        del self.clip.token_embedding
        del self.clip.ln_final
        if hasattr(self.clip, 'text_projection'):
            del self.clip.text_projection
    
    def freeze_backbone(self):
        """凍結 CLIP encoder"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("🔒 CLIP encoder frozen")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CLIP features
        with torch.set_grad_enabled(not self.freeze_encoder):
            clip_features = self.encoder(x)
        
        if clip_features.dim() > 2:
            clip_features = clip_features.mean(dim=1)
        
        # DCT features
        dct_features = self.dct_extractor(x)
        
        # 融合
        combined = torch.cat([clip_features, dct_features], dim=1)
        fused = self.fusion(combined)
        
        # 分類
        logits = self.classifier(fused)
        
        return logits
    
    def get_classifier(self) -> nn.Module:
        return self.classifier
    
    def get_backbone(self) -> nn.Module:
        return self.encoder
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        return {
            'mean': [0.48145466, 0.4578275, 0.40821073],
            'std': [0.26862954, 0.26130258, 0.27577711],
            'input_size': 224,
            'interpolation': 'bicubic',
        }
    
    def get_param_groups(
        self, 
        base_lr: float = 1e-4,
        backbone_lr_scale: float = 0.0
    ) -> List[Dict[str, Any]]:
        param_groups = []
        
        # Encoder 參數
        if not self.freeze_encoder:
            encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
            if encoder_params:
                param_groups.append({
                    'params': encoder_params,
                    'lr': base_lr * backbone_lr_scale,
                    'name': 'clip_encoder'
                })
        
        # DCT 參數
        param_groups.append({
            'params': list(self.dct_extractor.parameters()),
            'lr': base_lr * 0.5,
            'name': 'dct'
        })
        
        # Fusion + Classifier
        param_groups.append({
            'params': list(self.fusion.parameters()) + list(self.classifier.parameters()),
            'lr': base_lr,
            'name': 'fusion_classifier'
        })
        
        return param_groups
