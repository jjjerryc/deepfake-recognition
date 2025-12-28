# Deepfake Detection v4.0

基於 **DINOv2 + CLIP + ConvNeXt + EfficientNet** 的 Deepfake 圖像檢測框架，採用分層集成策略。

## 專案結構

```
deepfake-recognition/
├── configs/                    # YAML 配置檔案
│   ├── base.yaml              # 基礎配置（共用預設值）
│   ├── dino_vitl14.yaml       # DINOv2 ViT-L/14
│   ├── dino_vitl14_deep.yaml  # DINOv2 深度解凍版
│   ├── dino_vitl14_blur.yaml  # DINOv2 + Hard 增強
│   ├── clip_vitb32.yaml       # CLIP ViT-B/32
│   ├── clip_vitb32_dct.yaml   # CLIP + DCT
│   ├── convnext_base.yaml     # ConvNeXt V2 Base
│   ├── efficientnet_b4.yaml   # EfficientNet-B4
│   ├── efficientnet_b4_dct.yaml
│   └── ensemble.yaml          # 集成配置
├── src/
│   ├── models/
│   │   ├── base.py            # 模型基類
│   │   ├── dino_model.py      # DINOv2 分類器
│   │   ├── clip_model.py      # CLIP 分類器
│   │   ├── convnext_model.py  # ConvNeXt 分類器
│   │   ├── efficientnet.py    # EfficientNet
│   │   ├── efficientnet_dct.py
│   │   ├── dct.py             # DCT 頻域特徵
│   │   ├── ensemble.py        # 集成框架
│   │   └── factory.py         # 模型工廠
│   ├── utils/
│   │   └── config.py          # YAML 配置載入器
│   ├── train.py               # 訓練腳本
│   ├── inference.py           # 推論腳本
│   └── ensemble.py            # 集成推論腳本
├── outputs/                   # 輸出目錄
├── pyproject.toml
└── README.md
```

## 可用模型

### DINOv2 系列
| 配置檔案 | 模型 | Embedding | 說明 |
|---------|------|-----------|------|
| `dino_vitl14.yaml` | ViT-L/14 | 1024d | 基礎版本，解凍 2 層 |
| `dino_vitl14_deep.yaml` | ViT-L/14 | 1024d | 深度解凍 5 層 |
| `dino_vitl14_blur.yaml` | ViT-L/14 | 1024d | Hard 增強模式 |

### CLIP 系列
| 配置檔案 | 模型 | Embedding | 說明 |
|---------|------|-----------|------|
| `clip_vitb32.yaml` | ViT-B/32 | 512d | 速度快 |
| `clip_vitb32_dct.yaml` | ViT-B/32 + DCT | 512d | 頻域融合 |

### CNN 系列
| 配置檔案 | 模型 | 說明 |
|---------|------|------|
| `convnext_base.yaml` | ConvNeXt V2 Base | 紋理專家 |
| `efficientnet_b4.yaml` | EfficientNet-B4 | 輕量 CNN |
| `efficientnet_b4_dct.yaml` | EfficientNet-B4 + DCT | 頻域融合 |

## 快速開始

### 1. 安裝依賴

```bash
uv sync
```

### 2. 訓練模型

```bash
# 使用 YAML 配置訓練
python -m src.train --config configs/dino_vitl14.yaml

# DINOv2 + Hard 增強
python -m src.train --config configs/dino_vitl14_blur.yaml

# CLIP + DCT
python -m src.train --config configs/clip_vitb32_dct.yaml

# 恢復訓練
python -m src.train --config configs/dino_vitl14.yaml --resume
```

### 3. 單模型推論

```bash
# 推論
python -m src.inference --config configs/dino_vitl14.yaml

# 啟用 TTA (Test-Time Augmentation)
python -m src.inference --config configs/dino_vitl14.yaml --tta

# 多閾值輸出
python -m src.inference --config configs/dino_vitl14.yaml --thresholds 0.3 0.4 0.5
```

### 4. 集成推論

```bash
# 使用集成配置
python -m src.ensemble --config configs/ensemble.yaml

# 指定策略
python -m src.ensemble --config configs/ensemble.yaml --strategy hierarchical
```

## 配置系統

### 繼承機制

```yaml
# configs/dino_vitl14_blur.yaml
_base_: base.yaml

model:
  type: dino
  backbone: dinov2_vitl14

augmentation:
  mode: hard  # 覆蓋為 hard 增強
```

### 增強選項

```yaml
augmentation:
  mode: std  # "std" 或 "hard"
  train:
    horizontal_flip: 0.5
    vertical_flip: 0.0
    rotation_degrees: 10
    random_crop_scale: [0.9, 1.0]
    random_erasing: 0.0
    color_jitter:
      brightness: 0.2
      contrast: 0.2
      saturation: 0.1
      hue: 0.0
    gaussian_blur:
      kernel_size: 3
      sigma: [0.1, 2.0]
      probability: 0.2
  
  hard:  # Hard 模式 (強力模糊)
    horizontal_flip: 0.5
    gaussian_blur:
      kernel_size: 5
      sigma: [0.1, 5.0]
      probability: 0.5
```

## 集成策略

| 策略 | 說明 |
|------|------|
| `average` | 簡單平均所有模型機率 |
| `weighted_average` | 加權平均 |
| `vote` | 多數投票 |

### 分層集成配置範例

```yaml
# configs/ensemble.yaml
ensemble:
  strategy: hierarchical
  camps:
    dino:
      weight: 0.7
      models:
        - dino_vitl14
        - dino_vitl14_blur
    cnn:
      weight: 0.3
      models:
        - convnext_base
        - efficientnet_b4_dct
```

## License

MIT
