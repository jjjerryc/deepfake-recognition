# Deepfake Detection v4.0

基於 **DINOv2 + CLIP + EfficientNet + ConvNeXt** 的 Deepfake 圖像檢測框架。

## ✨ v4.0 新功能

- 🎯 **DINOv2 支援**：自監督視覺模型，對底層視覺結構敏感
- 🏗 **模組化配置**：每個模型一個 YAML 配置檔
- 🔗 **階層式集成**：支援陣營分組的集成策略
- 📁 **清晰的腳本分離**：train / inference / ensemble 獨立

## 快速開始

### 1. 列出可用配置

```bash
python -m src.train_v2 --list
```

### 2. 訓練模型

```bash
# 訓練 DINOv2 (推薦)
python -m src.train_v2 --config configs/dino_vitl14.yaml

# 訓練 DINOv2 深層解凍版本
python -m src.train_v2 --config configs/dino_vitl14_deep.yaml

# 訓練 ConvNeXt
python -m src.train_v2 --config configs/convnext_base.yaml

# 訓練 CLIP + DCT
python -m src.train_v2 --config configs/clip_vit_b32_dct.yaml

# 使用不同種子訓練 (用於集成)
python -m src.train_v2 --config configs/dino_vitl14.yaml --seed 100
python -m src.train_v2 --config configs/dino_vitl14.yaml --seed 200
```

### 3. 推論

```bash
# 基本推論
python -m src.inference_v2 --config configs/dino_vitl14.yaml

# 啟用 TTA (推薦)
python -m src.inference_v2 --config configs/dino_vitl14.yaml --tta

# 指定 checkpoint
python -m src.inference_v2 --config configs/dino_vitl14.yaml --checkpoint outputs/dino_vitl14/best_model.pth
```

### 4. 集成

```bash
# 使用 ensemble.yaml 配置
python -m src.ensemble_v2

# 指定策略
python -m src.ensemble_v2 --strategy hierarchical

# 只使用特定模型
python -m src.ensemble_v2 --models dino_vitl14 dino_vitl14_deep convnext_base
```

## 配置系統

所有配置檔案在 `configs/` 目錄：

```
configs/
├── base.yaml                 # 基礎配置 (被其他配置繼承)
├── dino_vitl14.yaml          # DINOv2 ViT-L/14 基準版
├── dino_vitl14_deep.yaml     # DINOv2 深層解凍 (5 層)
├── dino_vitl14_blur.yaml     # DINOv2 強增強版
├── dino_vitl14_avg.yaml      # DINOv2 全局池化版
├── convnext_base.yaml        # ConvNeXt V2 Base
├── clip_vit_b32.yaml         # CLIP ViT-B/32 凍結版
├── clip_vit_b32_unfreeze.yaml# CLIP ViT-B/32 解凍版
├── clip_vit_b32_dct.yaml     # CLIP + DCT 融合
├── efficientnet_b4_dct.yaml  # EfficientNet-B4 + DCT
└── ensemble.yaml             # 集成配置
```

### 配置繼承

配置檔案支援 `_base_` 繼承：

```yaml
# dino_vitl14_deep.yaml
_base_: "dino_vitl14.yaml"

model:
  name: "dino_vitl14_deep"
  unfreeze_layers: 5          # 覆蓋基準配置

training:
  backbone_lr_multiplier: 0.05
```

## 模型說明

### DINOv2 (推薦)

| 配置 | 特點 | 適用場景 |
|------|------|----------|
| `dino_vitl14` | 解凍 2 層，基準版 | 通用 |
| `dino_vitl14_deep` | 解凍 5 層，更深 | 訓練數據多時 |
| `dino_vitl14_blur` | 強力模糊增強 | 抗噪專家 |
| `dino_vitl14_avg` | 使用 patch 平均池化 | 全局視野 |

### ConvNeXt

純 CNN 架構，擅長捕捉紋理特徵，與 ViT 互補。

### CLIP

語義理解專家，跨域泛化能力強。

## 集成策略

在 `configs/ensemble.yaml` 中配置：

```yaml
ensemble:
  strategy: "hierarchical"
  
  hierarchical:
    groups:
      - name: "dino_camp"
        weight: 0.7
        models:
          - "dino_vitl14"
          - "dino_vitl14_deep"
          - "dino_vitl14_blur"
      
      - name: "cnn_camp"
        weight: 0.3
        models:
          - "convnext_base"
```

## 專案結構

```
deepfake-recognition/
├── configs/                  # 模型配置檔案
│   ├── base.yaml
│   ├── dino_vitl14.yaml
│   └── ...
├── src/
│   ├── models/
│   │   ├── dino_model.py     # DINOv2 模型
│   │   ├── convnext_model.py # ConvNeXt 模型
│   │   ├── clip_model.py     # CLIP 模型
│   │   └── ...
│   ├── utils/
│   │   └── config.py         # 配置載入器
│   ├── train_v2.py           # 訓練腳本
│   ├── inference_v2.py       # 推論腳本
│   └── ensemble_v2.py        # 集成腳本
└── outputs/                  # 模型輸出
    ├── dino_vitl14/
    │   ├── best_model.pth
    │   └── predictions_probs_tta.csv
    └── ensemble/
        └── ensemble_submission_tta.csv
```

## 完整工作流程

```bash
# 1. 訓練多個差異化模型
python -m src.train_v2 --config configs/dino_vitl14.yaml
python -m src.train_v2 --config configs/dino_vitl14_deep.yaml
python -m src.train_v2 --config configs/dino_vitl14_blur.yaml
python -m src.train_v2 --config configs/convnext_base.yaml

# 2. 對每個模型執行 TTA 推論
python -m src.inference_v2 --config configs/dino_vitl14.yaml --tta
python -m src.inference_v2 --config configs/dino_vitl14_deep.yaml --tta
python -m src.inference_v2 --config configs/dino_vitl14_blur.yaml --tta
python -m src.inference_v2 --config configs/convnext_base.yaml --tta

# 3. 執行階層式集成
python -m src.ensemble_v2 --strategy hierarchical

# 4. 提交
# outputs/ensemble/ensemble_submission_tta.csv
```

## 環境需求

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (GPU 訓練)
- 推薦：RTX 3090/4090 或 V100 32GB

## 安裝

```bash
# 使用 uv
uv sync

# 或使用 pip
pip install torch torchvision timm open_clip_torch pyyaml tqdm pandas numpy pillow
```

## Changelog

### v4.0.0

- ✨ 新增 DINOv2 模型支援
- ✨ 新增 ConvNeXt V2 模型支援
- ✨ 新的 YAML 配置系統（每個模型一個配置）
- ✨ 階層式集成策略
- ✨ 獨立的 train / inference / ensemble 腳本
- 🔧 配置繼承機制
- 🔧 TTA 支援多種變換

### v3.3.0

- CLIP 模型支援
- config.json 配置
- Ensemble Pipeline
