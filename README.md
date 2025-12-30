# NYCU DL final project Deepfake Detection

基於 **DINOv2 + CLIP + EfficientNet** 的 Deepfake 圖像檢測框架，採用多模型集成策略。

## 專案結構

```
deepfake-recognition/
├── configs/                    
│   ├── base.yaml              
│   ├── dino_vitl14.yaml       
│   ├── dino_vitl14_deep.yaml  
│   ├── dino_vitl14_blur.yaml 
│   ├── clip_vitb32.yaml      
│   ├── clip_vitb32_dct.yaml   
│   ├── convnext_base.yaml     
│   ├── efficientnet_b4.yaml   
│   ├── efficientnet_b4_dct.yaml
│   └── ensemble.yaml          
├── src/
│   ├── models/
│   │   ├── base.py            
│   │   ├── dino_model.py     
│   │   ├── clip_model.py  
│   │   ├── convnext_model.py
│   │   ├── efficientnet.py
│   │   ├── efficientnet_dct.py
│   │   ├── dct.py      
│   │   ├── ensemble.py 
│   │   └── factory.py   
│   ├── utils/
│   │   └── config.py     
│   ├── train.py         
│   ├── inference.py    
│   └── ensemble.py 
├── outputs/     
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

## 最終CSV選用模型

| 配置檔案 | 模型 | Embedding | 說明 |
|---------|------|-----------|------|
| `dino_vitl14.yaml` | ViT-L/14 | 1024d | 基礎版本，解凍 2 層 |
| `clip_vitb32_dct.yaml` | ViT-B/32 + DCT | 512d | 頻域融合 |
| `efficientnet_b4_dct.yaml` | EfficientNet-B4 + DCT | / | 頻域融合 |

## 快速開始

### 1. 安裝依賴

```bash
uv sync
```

### 2. 訓練模型

```bash
# dino_vitl14
python -m src.train --config configs/dino_vitl14.yaml

# CLIP + DCT
python -m src.train --config configs/clip_vitb32_dct.yaml

# EfficientNet + DCT
python -m src.train --config configs/efficientnet_b4_dct.yaml

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
python -m src.ensemble --config configs/ensemble.yaml --weighted_average
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

最終使用 `average` 策略

## License

MIT
