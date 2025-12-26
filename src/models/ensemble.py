"""
模型集成（Ensemble）框架

支援多種集成策略：
1. 平均融合（Averaging）：對多個模型的預測取平均
2. 加權平均（Weighted Averaging）：根據驗證集表現加權
3. 投票（Voting）：多數投票決定最終類別
4. 堆疊（Stacking）：用 meta-learner 學習如何結合

使用方式:
    # 創建集成模型
    ensemble = ModelEnsemble(
        model_configs=[
            {'name': 'efficientnet_b4', 'weight': 0.4},
            {'name': 'efficientnet_b4_dct', 'weight': 0.6},
        ],
        strategy='weighted_average'
    )
    
    # 推論
    predictions = ensemble.predict(images)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

from .factory import create_model
from .base import BaseModel


class ModelEnsemble(nn.Module):
    """
    模型集成類別
    
    支援多種集成策略，可以從 checkpoint 載入多個模型並進行聯合推論。
    """
    
    def __init__(
        self,
        models: List[BaseModel] = None,
        weights: List[float] = None,
        strategy: str = 'average',
        temperature: float = 1.0,
    ):
        """
        Args:
            models: 已初始化的模型列表
            weights: 模型權重（用於加權平均）
            strategy: 集成策略 ['average', 'weighted_average', 'vote', 'max']
            temperature: softmax 溫度參數
        """
        super().__init__()
        
        self.strategy = strategy
        self.temperature = temperature
        
        # 儲存模型
        if models:
            self.models = nn.ModuleList(models)
            self.num_models = len(models)
        else:
            self.models = nn.ModuleList()
            self.num_models = 0
        
        # 權重
        if weights:
            assert len(weights) == self.num_models, "權重數量必須與模型數量相同"
            weights_tensor = torch.tensor(weights, dtype=torch.float32)
            weights_tensor = weights_tensor / weights_tensor.sum()  # 正規化
            self.register_buffer('weights', weights_tensor)
        else:
            self.weights = None
    
    def add_model(self, model: BaseModel, weight: float = 1.0):
        """
        添加模型到集成
        
        Args:
            model: 模型實例
            weight: 模型權重
        """
        self.models.append(model)
        self.num_models += 1
        
        # 更新權重
        if self.weights is None:
            self.weights = torch.ones(self.num_models)
        else:
            new_weights = torch.zeros(self.num_models)
            new_weights[:-1] = self.weights
            new_weights[-1] = weight
            self.weights = new_weights
        
        # 正規化
        self.weights = self.weights / self.weights.sum()
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        集成推論
        
        Args:
            x: (B, C, H, W) 輸入圖像
            
        Returns:
            (B, num_classes) 集成後的預測 logits
        """
        if self.num_models == 0:
            raise RuntimeError("集成中沒有模型！請先添加模型。")
        
        # 收集所有模型的預測
        all_logits = []
        for model in self.models:
            model.eval()
            logits = model(x)
            all_logits.append(logits)
        
        # 堆疊: (num_models, B, num_classes)
        stacked_logits = torch.stack(all_logits, dim=0)
        
        # 根據策略融合
        if self.strategy == 'average':
            # 簡單平均
            return stacked_logits.mean(dim=0)
        
        elif self.strategy == 'weighted_average':
            # 加權平均
            weights = self.weights.view(-1, 1, 1)  # (num_models, 1, 1)
            return (stacked_logits * weights).sum(dim=0)
        
        elif self.strategy == 'vote':
            # 投票（基於 argmax）
            # 先轉換為機率
            probs = F.softmax(stacked_logits / self.temperature, dim=-1)
            # 投票：取每個模型的預測類別，然後統計
            votes = probs.argmax(dim=-1)  # (num_models, B)
            # 返回平均機率作為置信度（用於軟投票）
            return probs.mean(dim=0)
        
        elif self.strategy == 'max':
            # 取最大值（最自信的預測）
            probs = F.softmax(stacked_logits / self.temperature, dim=-1)
            max_confidence, _ = probs.max(dim=-1, keepdim=True)  # (num_models, B, 1)
            best_model_idx = max_confidence.squeeze(-1).argmax(dim=0)  # (B,)
            
            # 選擇最自信模型的預測
            B = x.shape[0]
            result = torch.zeros_like(stacked_logits[0])
            for b in range(B):
                result[b] = stacked_logits[best_model_idx[b], b]
            return result
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        預測機率分佈
        
        Args:
            x: 輸入圖像
            
        Returns:
            (B, num_classes) 機率分佈
        """
        logits = self.forward(x)
        return F.softmax(logits / self.temperature, dim=-1)
    
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        預測類別
        
        Args:
            x: 輸入圖像
            
        Returns:
            (B,) 預測的類別索引
        """
        logits = self.forward(x)
        return logits.argmax(dim=-1)
    
    def get_individual_predictions(
        self, 
        x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        獲取每個模型的個別預測（用於分析）
        
        Args:
            x: 輸入圖像
            
        Returns:
            logits_list: 每個模型的 logits
            probs_list: 每個模型的機率
        """
        logits_list = []
        probs_list = []
        
        with torch.no_grad():
            for model in self.models:
                model.eval()
                logits = model(x)
                probs = F.softmax(logits / self.temperature, dim=-1)
                logits_list.append(logits)
                probs_list.append(probs)
        
        return logits_list, probs_list


class EnsembleFromCheckpoints:
    """
    從 checkpoint 文件載入集成模型的工具類別
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
    
    def load_ensemble(
        self,
        checkpoint_configs: List[Dict],
        strategy: str = 'weighted_average',
    ) -> ModelEnsemble:
        """
        從 checkpoint 配置載入集成模型
        
        Args:
            checkpoint_configs: checkpoint 配置列表，每個包含:
                - path: checkpoint 路徑
                - weight: 模型權重（可選）
            strategy: 集成策略
            
        Returns:
            ModelEnsemble 實例
            
        Example:
            loader = EnsembleFromCheckpoints()
            ensemble = loader.load_ensemble([
                {'path': 'outputs/efficientnet_b4/best.pth', 'weight': 0.4},
                {'path': 'outputs/efficientnet_b4_dct/best.pth', 'weight': 0.6},
            ])
        """
        models = []
        weights = []
        
        for config in checkpoint_configs:
            path = config['path']
            weight = config.get('weight', 1.0)
            
            # 載入 checkpoint
            checkpoint = torch.load(path, map_location=self.device)
            
            # 獲取模型配置
            model_name = checkpoint.get('model_name', 'efficientnet_b4')
            model_config = checkpoint.get('config', {}).get('model', {})
            
            # 創建模型
            model = create_model(
                model_name=model_name,
                num_classes=model_config.get('num_classes', 2),
                pretrained=False,  # 從 checkpoint 載入權重
                dropout=model_config.get('dropout', 0.3),
            )
            
            # 載入權重
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            
            model = model.to(self.device)
            model.eval()
            
            models.append(model)
            weights.append(weight)
            
            print(f"Loaded {model_name} from {path} (weight: {weight})")
        
        ensemble = ModelEnsemble(
            models=models,
            weights=weights,
            strategy=strategy,
        )
        
        return ensemble


def create_ensemble(
    model_configs: List[Dict],
    strategy: str = 'weighted_average',
    pretrained: bool = True,
) -> ModelEnsemble:
    """
    從配置創建集成模型（不載入 checkpoint，用於訓練新模型）
    
    Args:
        model_configs: 模型配置列表
        strategy: 集成策略
        pretrained: 是否使用預訓練權重
        
    Returns:
        ModelEnsemble 實例
        
    Example:
        ensemble = create_ensemble([
            {'name': 'efficientnet_b4', 'weight': 0.5},
            {'name': 'efficientnet_b4_dct', 'weight': 0.5},
        ])
    """
    models = []
    weights = []
    
    for config in model_configs:
        model = create_model(
            model_name=config['name'],
            num_classes=config.get('num_classes', 2),
            pretrained=pretrained,
            dropout=config.get('dropout', 0.3),
        )
        models.append(model)
        weights.append(config.get('weight', 1.0))
    
    return ModelEnsemble(
        models=models,
        weights=weights,
        strategy=strategy,
    )


if __name__ == '__main__':
    # 測試集成模型
    print("Testing Ensemble framework...")
    
    # 創建測試模型
    from .factory import create_model
    
    model1 = create_model('efficientnet_b0', pretrained=False)
    model2 = create_model('efficientnet_b0', pretrained=False)
    
    # 創建集成
    ensemble = ModelEnsemble(
        models=[model1, model2],
        weights=[0.6, 0.4],
        strategy='weighted_average'
    )
    
    # 測試推論
    x = torch.randn(4, 3, 224, 224)
    
    print(f"Number of models: {ensemble.num_models}")
    print(f"Weights: {ensemble.weights}")
    
    output = ensemble(x)
    print(f"Ensemble output shape: {output.shape}")
    
    probs = ensemble.predict_proba(x)
    print(f"Probabilities shape: {probs.shape}")
    print(f"Probabilities sum: {probs.sum(dim=-1)}")  # 應該都是 1
    
    preds = ensemble.predict(x)
    print(f"Predictions: {preds}")
    
    # 測試不同策略
    for strategy in ['average', 'weighted_average', 'vote', 'max']:
        ensemble.strategy = strategy
        out = ensemble(x)
        print(f"Strategy '{strategy}': output shape = {out.shape}")
