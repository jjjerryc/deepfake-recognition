"""
Base Model 抽象類別

定義所有模型必須實現的介面，確保訓練腳本可以統一處理不同架構。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """
    所有 Deepfake 檢測模型的基類
    
    子類必須實現:
    - forward(): 前向傳播
    - get_classifier(): 獲取分類頭（用於單獨調整學習率）
    - get_backbone(): 獲取特徵提取器
    
    可選覆寫:
    - get_preprocessing_config(): 返回模型需要的預處理參數
    """
    
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self._model_name = self.__class__.__name__
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            x: 輸入張量，形狀 [B, C, H, W]
            
        Returns:
            輸出 logits，形狀 [B, num_classes]
        """
        pass
    
    @abstractmethod
    def get_classifier(self) -> nn.Module:
        """
        獲取分類頭模組
        
        用於設定不同的學習率（backbone 用較小學習率，classifier 用較大學習率）
        
        Returns:
            分類頭模組
        """
        pass
    
    @abstractmethod
    def get_backbone(self) -> nn.Module:
        """
        獲取特徵提取器（backbone）
        
        Returns:
            backbone 模組
        """
        pass
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """
        獲取模型需要的預處理配置
        
        Returns:
            包含 mean, std, input_size 等預處理參數的字典
        """
        # 預設使用 ImageNet 標準化
        return {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'input_size': 224,
            'interpolation': 'bilinear',
        }
    
    def freeze_backbone(self):
        """凍結 backbone 參數（用於 linear probing）"""
        for param in self.get_backbone().parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """解凍 backbone 參數"""
        for param in self.get_backbone().parameters():
            param.requires_grad = True
    
    def get_param_groups(self, backbone_lr: float, classifier_lr: float) -> list:
        """
        獲取分層學習率的參數組
        
        Args:
            backbone_lr: backbone 的學習率
            classifier_lr: classifier 的學習率
            
        Returns:
            可傳給 optimizer 的參數組列表
        """
        return [
            {'params': self.get_backbone().parameters(), 'lr': backbone_lr},
            {'params': self.get_classifier().parameters(), 'lr': classifier_lr},
        ]
    
    def count_parameters(self, trainable_only: bool = True) -> int:
        """
        計算模型參數量
        
        Args:
            trainable_only: 是否只計算可訓練參數
            
        Returns:
            參數數量
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    @property
    def model_name(self) -> str:
        """模型名稱"""
        return self._model_name
    
    def __repr__(self) -> str:
        total_params = self.count_parameters(trainable_only=False)
        trainable_params = self.count_parameters(trainable_only=True)
        return (
            f"{self._model_name}(\n"
            f"  num_classes={self.num_classes},\n"
            f"  total_params={total_params:,},\n"
            f"  trainable_params={trainable_params:,}\n"
            f")"
        )
