"""
DCT (Discrete Cosine Transform) 頻域特徵模組

Deepfake 圖像在頻域中往往會留下特定的偽影痕跡：
- GAN 生成圖像在高頻區域有特定的 pattern
- 壓縮和重採樣會留下可檢測的頻率特徵
- 不同生成器會產生不同的頻譜簽名

此模組提供 DCT 特徵提取功能，可與 CNN backbone 結合使用。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


def create_dct_matrix(n: int) -> torch.Tensor:
    """
    創建 DCT-II 變換矩陣
    
    Args:
        n: 矩陣大小
        
    Returns:
        (n, n) DCT 變換矩陣
    """
    dct_matrix = torch.zeros(n, n)
    for k in range(n):
        for i in range(n):
            if k == 0:
                dct_matrix[k, i] = 1.0 / np.sqrt(n)
            else:
                dct_matrix[k, i] = np.sqrt(2.0 / n) * np.cos(
                    np.pi * k * (2 * i + 1) / (2 * n)
                )
    return dct_matrix


class DCT2D(nn.Module):
    """
    2D DCT (Discrete Cosine Transform) 層
    
    對圖像進行 2D DCT 變換，提取頻域特徵。
    使用預計算的 DCT 矩陣，通過矩陣乘法實現高效變換。
    """
    
    def __init__(self, size: int = 224):
        """
        Args:
            size: 輸入圖像的大小（假設為正方形）
        """
        super().__init__()
        self.size = size
        
        # 預計算 DCT 矩陣（不需要梯度）
        dct_matrix = create_dct_matrix(size)
        self.register_buffer('dct_matrix', dct_matrix)
        self.register_buffer('dct_matrix_t', dct_matrix.t())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        對輸入進行 2D DCT 變換
        
        Args:
            x: (B, C, H, W) 輸入張量
            
        Returns:
            (B, C, H, W) DCT 係數
        """
        # DCT: Y = D @ X @ D^T
        # 對每個通道獨立進行 DCT
        return torch.matmul(
            torch.matmul(self.dct_matrix, x),
            self.dct_matrix_t
        )


class DCTFeatureExtractor(nn.Module):
    """
    DCT 頻域特徵提取器
    
    提取多種頻域統計特徵：
    1. 低頻、中頻、高頻能量分佈
    2. 頻譜統計量（均值、標準差、峰度等）
    3. 特定頻率區域的特徵
    
    這些特徵對於檢測 GAN 生成圖像特別有效。
    """
    
    def __init__(
        self, 
        size: int = 224,
        num_freq_bands: int = 3,
        output_dim: int = 128,
        use_log_scale: bool = True
    ):
        """
        Args:
            size: 輸入圖像大小
            num_freq_bands: 頻帶數量（低/中/高頻）
            output_dim: 輸出特徵維度
            use_log_scale: 是否使用對數尺度（更好地處理能量差異）
        """
        super().__init__()
        self.size = size
        self.num_freq_bands = num_freq_bands
        self.use_log_scale = use_log_scale
        
        # DCT 變換層
        self.dct = DCT2D(size)
        
        # 創建頻帶遮罩
        self._create_frequency_masks()
        
        # 每個頻帶的統計特徵數：均值、標準差、最大值、能量（4個）
        # 3個通道 × 3個頻帶 × 4個統計量 = 36 維
        # 加上全局頻譜特徵
        stats_dim = 3 * num_freq_bands * 4 + 12  # 額外的全局特徵
        
        # 特徵處理 MLP
        self.feature_mlp = nn.Sequential(
            nn.Linear(stats_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim),
        )
        
    def _create_frequency_masks(self):
        """創建頻帶遮罩，用於分離低/中/高頻"""
        h, w = self.size, self.size
        
        # 創建距離矩陣（到左上角的距離，即 DC 分量）
        y_coords = torch.arange(h).float().view(-1, 1).expand(h, w)
        x_coords = torch.arange(w).float().view(1, -1).expand(h, w)
        distance = torch.sqrt(y_coords ** 2 + x_coords ** 2)
        max_dist = np.sqrt(h ** 2 + w ** 2)
        
        # 正規化距離
        normalized_dist = distance / max_dist
        
        # 創建頻帶遮罩
        # 低頻：0-15% 的頻率範圍
        # 中頻：15-50% 的頻率範圍  
        # 高頻：50-100% 的頻率範圍
        masks = []
        band_edges = [0, 0.15, 0.50, 1.0]
        
        for i in range(self.num_freq_bands):
            mask = ((normalized_dist >= band_edges[i]) & 
                   (normalized_dist < band_edges[i + 1])).float()
            masks.append(mask)
        
        # 註冊為 buffer（不參與訓練）
        self.register_buffer('freq_masks', torch.stack(masks))  # (num_bands, H, W)
    
    def extract_band_statistics(
        self, 
        dct_coeffs: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        提取單個頻帶的統計特徵
        
        Args:
            dct_coeffs: (B, C, H, W) DCT 係數
            mask: (H, W) 頻帶遮罩
            
        Returns:
            (B, C, 4) 統計特徵：均值、標準差、最大值、能量
        """
        B, C, H, W = dct_coeffs.shape
        
        # 應用遮罩
        masked = dct_coeffs * mask.unsqueeze(0).unsqueeze(0)
        
        # 計算非零元素數量
        num_elements = mask.sum().clamp(min=1)
        
        # 展平空間維度
        masked_flat = masked.view(B, C, -1)  # (B, C, H*W)
        
        # 使用絕對值（DCT 係數可能為負）
        if self.use_log_scale:
            # 對數尺度，加小常數避免 log(0)
            abs_masked = torch.log1p(masked_flat.abs())
        else:
            abs_masked = masked_flat.abs()
        
        # 統計量
        mean = abs_masked.sum(dim=-1) / num_elements
        std = ((abs_masked - mean.unsqueeze(-1)) ** 2).sum(dim=-1) / num_elements
        std = std.sqrt()
        max_val = abs_masked.max(dim=-1)[0]
        energy = (masked_flat ** 2).sum(dim=-1) / num_elements
        
        if self.use_log_scale:
            energy = torch.log1p(energy)
        
        return torch.stack([mean, std, max_val, energy], dim=-1)  # (B, C, 4)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取 DCT 頻域特徵
        
        Args:
            x: (B, C, H, W) 輸入圖像，應該已經過標準化
            
        Returns:
            (B, output_dim) 頻域特徵向量
        """
        B = x.shape[0]
        
        # 確保輸入尺寸正確
        if x.shape[2] != self.size or x.shape[3] != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode='bilinear', align_corners=False)
        
        # 計算 DCT
        dct_coeffs = self.dct(x)  # (B, C, H, W)
        
        # 提取各頻帶統計量
        band_features = []
        for i in range(self.num_freq_bands):
            stats = self.extract_band_statistics(dct_coeffs, self.freq_masks[i])
            band_features.append(stats.view(B, -1))  # (B, C*4)
        
        # 全局頻譜特徵
        dct_flat = dct_coeffs.view(B, 3, -1)
        global_mean = dct_flat.mean(dim=-1)  # (B, 3)
        global_std = dct_flat.std(dim=-1)    # (B, 3)
        
        # DC 分量（左上角）
        dc_component = dct_coeffs[:, :, 0, 0]  # (B, 3)
        
        # 高頻能量比例
        high_freq_energy = (dct_coeffs * self.freq_masks[-1]).pow(2).sum(dim=[2, 3])
        total_energy = dct_coeffs.pow(2).sum(dim=[2, 3]).clamp(min=1e-8)
        high_freq_ratio = high_freq_energy / total_energy  # (B, 3)
        
        # 組合所有特徵
        all_features = torch.cat(
            band_features + [global_mean, global_std, dc_component, high_freq_ratio],
            dim=-1
        )  # (B, 3*num_bands*4 + 12)
        
        # MLP 處理
        output = self.feature_mlp(all_features)
        
        return output


class MultiScaleDCT(nn.Module):
    """
    多尺度 DCT 特徵提取器
    
    在多個尺度上提取 DCT 特徵，捕捉不同粒度的頻域信息。
    """
    
    def __init__(
        self,
        scales: Tuple[int, ...] = (56, 112, 224),
        output_dim: int = 256,
    ):
        """
        Args:
            scales: 要分析的圖像尺度
            output_dim: 最終輸出維度
        """
        super().__init__()
        self.scales = scales
        
        # 每個尺度的 DCT 提取器
        per_scale_dim = output_dim // len(scales)
        self.dct_extractors = nn.ModuleList([
            DCTFeatureExtractor(size=s, output_dim=per_scale_dim)
            for s in scales
        ])
        
        # 融合層
        self.fusion = nn.Sequential(
            nn.Linear(per_scale_dim * len(scales), output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        多尺度 DCT 特徵提取
        
        Args:
            x: (B, C, H, W) 輸入圖像
            
        Returns:
            (B, output_dim) 多尺度頻域特徵
        """
        scale_features = []
        
        for scale, extractor in zip(self.scales, self.dct_extractors):
            # 縮放到目標尺度
            if x.shape[2] != scale or x.shape[3] != scale:
                scaled = F.interpolate(x, size=(scale, scale), mode='bilinear', align_corners=False)
            else:
                scaled = x
            
            # 提取特徵
            feat = extractor(scaled)
            scale_features.append(feat)
        
        # 融合
        combined = torch.cat(scale_features, dim=-1)
        return self.fusion(combined)


if __name__ == '__main__':
    # 測試 DCT 模組
    print("Testing DCT modules...")
    
    # 測試基本 DCT
    dct = DCT2D(size=224)
    x = torch.randn(2, 3, 224, 224)
    y = dct(x)
    print(f"DCT2D: {x.shape} -> {y.shape}")
    
    # 測試特徵提取器
    extractor = DCTFeatureExtractor(size=224, output_dim=128)
    feat = extractor(x)
    print(f"DCTFeatureExtractor: {x.shape} -> {feat.shape}")
    
    # 測試多尺度
    ms_dct = MultiScaleDCT(scales=(56, 112, 224), output_dim=256)
    ms_feat = ms_dct(x)
    print(f"MultiScaleDCT: {x.shape} -> {ms_feat.shape}")
    
    # 參數量
    print(f"\nDCTFeatureExtractor params: {sum(p.numel() for p in extractor.parameters()):,}")
    print(f"MultiScaleDCT params: {sum(p.numel() for p in ms_dct.parameters()):,}")
