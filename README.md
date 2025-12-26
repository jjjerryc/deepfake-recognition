# Deepfake Detection v3.3

基於 **EfficientNet + DCT + CLIP** 的 Deepfake 圖像檢測框架，專為跨生成器泛化設計。

## ✨ 特色功能

- 🎯 **雙流架構**：CNN 空間特徵 + DCT 頻域特徵融合
- 🤖 **CLIP 支援**：利用 CLIP 視覺編碼器提升跨域泛化能力
- 🔧 **22 種模型**：EfficientNet B0-B4、DCT 變體、CLIP 變體
- 📊 **Ensemble Pipeline**：支援多種集成策略，透過 config.json 控制
- ⚡ **高效訓練**：混合精度 (AMP)、ReduceLROnPlateau、Mixup/CutMix
- 🔄 **自動預處理**：EfficientNet/CLIP 自動選擇正確的 normalize
- 📝 **完整日誌**：每個 epoch 實時保存到文件

## 專案結構

```
final/
├── config.json              # 訓練與 ensemble 配置
├── pyproject.toml           # 依賴管理
├── README.md
├── dataset/
│   ├── train/
│   │   ├── fake/           # 假圖 (15,000)
│   │   └── real/           # 真圖 (15,000)
│   ├── test/               # 測試集 (14,000)
│   └── sample_submission.csv
├── src/
│   ├── models/
│   │   ├── base.py             # 模型基類
│   │   ├── efficientnet.py     # EfficientNet 封裝
│   │   ├── dct.py              # DCT 頻域特徵模組
│   │   ├── efficientnet_dct.py # EfficientNet + DCT 融合
│   │   ├── clip_model.py       # CLIP 視覺編碼器模型
│   │   ├── ensemble.py         # 模型集成框架
│   │   └── factory.py          # 模型工廠 (22 種模型)
│   ├── train.py                # 訓練腳本 (v3.3)
│   ├── inference.py            # 單模型推論
│   ├── ensemble_inference.py   # 集成推論 (v2.0)
│   └── convert_submission.py   # 格式轉換 (機率 → real/fake)
└── outputs/
    ├── {model_name}/           # 各模型獨立目錄
    │   ├── best_model.pth
    │   └── latest_checkpoint.pth
    ├── ensemble/               # Ensemble 輸出
    │   ├── ensemble_submission.csv
    │   └── individual/
    └── logs/
```

## 可用模型 (22 種)

### EfficientNet 系列
| 模型名稱 | 參數量 | 說明 |
|---------|--------|------|
| `efficientnet_b0` ~ `b4` | 4M~18M | 基礎 CNN |
| `efficientnet_b0_dct` ~ `b4_dct` | 5M~19M | + DCT 頻域特徵 |
| `efficientnet_b0_dct_attn` ~ `b4_dct_attn` | ~35M | + 交叉注意力 |

### CLIP 系列 (推薦用於跨域泛化)
| 模型名稱 | 參數量 | 可訓練參數 | 說明 |
|---------|--------|-----------|------|
| `clip_vit_b32` | 88M | 304K | ViT-B/32，速度快 |
| `clip_vit_b16` | 86M | 304K | ViT-B/16，更細緻 |
| `clip_vit_l14` | 304M | 525K | ViT-L/14，最強 |
| `clip_convnext_base` | 88M | 394K | ConvNeXt 架構 |
| `clip_vit_b32_dct` | 88M | 547K | CLIP + DCT 混合 |
| `clip_vit_b16_dct` | 87M | 547K | CLIP + DCT 混合 |

## 快速開始

### 1. 安裝依賴

```bash
uv sync
```

### 2. 訓練模型

```bash
# EfficientNet + DCT（頻域特徵）
python -m src.train --model efficientnet_b4_dct

# CLIP 模型（凍結 backbone，僅訓練分類頭）
python -m src.train --model clip_vit_b32 --freeze-backbone

# CLIP + DCT 混合
python -m src.train --model clip_vit_b32_dct --freeze-backbone

# 恢復訓練
python -m src.train --model efficientnet_b4_dct --resume
```

### 3. 單模型推論

```bash
# 自動載入對應模型的 checkpoint
python -m src.inference --model efficientnet_b4_dct

# 啟用 TTA
python -m src.inference --model clip_vit_b32 --tta
```

### 4. Ensemble 推論

```bash
# 使用 config.json 的 ensemble 配置
python -m src.ensemble_inference

# 自動根據驗證準確率計算權重
python -m src.ensemble_inference --auto-weight

# 指定策略
python -m src.ensemble_inference --strategy average
```

### 5. 格式轉換

```bash
# 將機率輸出轉換為 Kaggle 提交格式 (real/fake)
python -m src.convert_submission -i outputs/ensemble/ensemble_submission.csv -o final_submission.csv

# 調整閾值
python -m src.convert_submission -i outputs/submission.csv -o final.csv -t 0.6

# 分析預測分佈
python -m src.convert_submission -i outputs/submission.csv --analyze
```

## 配置說明 (`config.json`)

### 主要區塊

```json
{
    "model": {
        "name": "efficientnet_b0_dct",
        "dropout": 0.5,
        "dct_dim": 128
    },
    "training": {
        "batch_size": 64,
        "learning_rate": 1e-4,
        "epochs": 50,
        "label_smoothing": 0.1
    },
    "scheduler": {
        "type": "ReduceLROnPlateau",
        "factor": 0.5,
        "patience": 3
    },
    "augmentation": {
        "train": {
            "mixup_alpha": 0.1,
            "cutmix_alpha": 0.0,
            "mix_prob": 0.2
        }
    },
    "ensemble": {
        "strategy": "weighted_average",
        "models": [
            {"name": "efficientnet_b4_dct", "weight": 0.3},
            {"name": "clip_vit_b32", "weight": 0.25},
            {"name": "clip_vit_b16", "weight": 0.25},
            {"name": "clip_vit_b32_dct", "weight": 0.2}
        ]
    }
}
```

### Ensemble 策略

| 策略 | 說明 | 適用場景 |
|------|------|----------|
| `average` | 簡單平均 | 模型性能相近 |
| `weighted_average` | 加權平均 | 模型性能差異大 |
| `vote` | Soft Voting | 模型數量多 |
| `max` | 選擇最有信心的 | 模型專長不同 |

## 技術原理

### DCT 頻域特徵

Deepfake 圖像在頻域中會留下特定偽影：

1. **GAN Fingerprint**：生成器會產生特定的頻譜模式
2. **壓縮痕跡**：重採樣和壓縮會改變高頻分佈
3. **邊界效應**：合成圖像的拼接邊界在頻域更明顯

### CLIP 視覺編碼器

CLIP 模型在大規模圖像-文字配對上預訓練，具有：

1. **強大的語義理解**：理解圖像的高層次語義
2. **跨域泛化**：在不同生成器間泛化能力強
3. **凍結策略**：只訓練分類頭，避免過擬合

### 雙流融合架構

```
┌─────────────┐     ┌─────────────┐
│   Input     │     │   Input     │
└──────┬──────┘     └──────┬──────┘
       │                   │
       ▼                   ▼
┌─────────────┐     ┌─────────────┐
│ EfficientNet│     │     DCT     │
│  (空間特徵)  │     │  (頻域特徵)  │
└──────┬──────┘     └──────┬──────┘
       │                   │
       └────────┬──────────┘
                │
                ▼
         ┌─────────────┐
         │   Fusion    │
         │   + MLP     │
         └──────┬──────┘
                │
                ▼
         ┌─────────────┐
         │  Classifier │
         └─────────────┘
```

## 命令行參數

### train.py

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--config` | 配置文件路徑 | config.json |
| `--model` | 模型名稱 | 從 config 讀取 |
| `--epochs` | 訓練 epochs | 從 config 讀取 |
| `--batch-size` | Batch size | 從 config 讀取 |
| `--lr` | 學習率 | 從 config 讀取 |
| `--resume` | 恢復訓練 | False |
| `--freeze-backbone` | 凍結 backbone | False (CLIP 建議開啟) |

### inference.py

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--config` | 配置文件路徑 | config.json |
| `--model` | 模型名稱 | 從 config 讀取 |
| `--checkpoint` | 指定 checkpoint | 自動偵測 |
| `--tta` | 啟用 TTA | False |

### ensemble_inference.py

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--config` | 配置文件路徑 | config.json |
| `--strategy` | 集成策略 | weighted_average |
| `--auto-weight` | 自動計算權重 | False |
| `--no-individual` | 不保存個別預測 | False |

### convert_submission.py

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `-i, --input` | 輸入 CSV | 必填 |
| `-o, --output` | 輸出 CSV | 自動生成 |
| `-t, --threshold` | 分類閾值 | 0.5 |
| `--analyze` | 只分析不轉換 | False |

## 完整工作流程

```bash
# 1. 訓練多個模型
python -m src.train --model efficientnet_b4_dct
python -m src.train --model clip_vit_b32 --freeze-backbone
python -m src.train --model clip_vit_b16 --freeze-backbone

# 2. 執行 ensemble 推論
python -m src.ensemble_inference --auto-weight

# 3. 轉換為提交格式
python -m src.convert_submission \
    -i outputs/ensemble/ensemble_submission.csv \
    -o final_submission.csv

# 4. 提交到 Kaggle
kaggle competitions submit -c deepfake-detection -f final_submission.csv -m "Ensemble v3.3"
```

## 環境需求

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (GPU 訓練)
- 推薦：V100 32GB / A100

## Changelog

### v3.3.0
- ✨ 新增 CLIP 模型支援 (7 種變體)
- ✨ 新增 config.json 控制的 Ensemble Pipeline
- ✨ 新增格式轉換腳本 (機率 → real/fake)
- ✨ 模型獨立輸出目錄 (`outputs/{model_name}/`)
- ✨ 自動預處理（EfficientNet vs CLIP normalize）
- 🔧 切換到 ReduceLROnPlateau 調度器
- 🔧 新增 Mixup/CutMix 資料增強
- 🔧 新增 `--freeze-backbone` 訓練選項

### v3.2.0
- 初始 EfficientNet + DCT 架構
- OneCycleLR 調度器
- 基礎 Ensemble 框架

## License

MIT
