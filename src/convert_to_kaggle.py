#!/usr/bin/env python3
"""
快速轉換機率 CSV 到 Kaggle 格式

使用方式:
    # 使用 threshold=0.5
    python -m src.convert_to_kaggle -i probs.csv -o submission.csv
    
    # 自動使用中位數（強制 50/50 分布）
    python -m src.convert_to_kaggle -i probs.csv -o submission.csv --median
    
    # 指定 threshold
    python -m src.convert_to_kaggle -i probs.csv -o submission.csv -t 0.4
"""

import argparse
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Convert probability CSV to Kaggle format')
    parser.add_argument('-i', '--input', required=True, help='Input probability CSV file')
    parser.add_argument('-o', '--output', required=True, help='Output Kaggle format CSV file')
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help='Threshold for fake (default: 0.5)')
    parser.add_argument('--median', action='store_true', help='Use median as threshold (forces 50/50 split)')
    parser.add_argument('--invert', action='store_true', help='Invert probabilities (if labels are reversed)')
    
    args = parser.parse_args()
    
    # 讀取
    df = pd.read_csv(args.input)
    
    # 確定機率欄位名稱
    prob_col = 'label' if 'label' in df.columns else df.columns[1]
    id_col = 'id' if 'id' in df.columns else df.columns[0]
    
    probs = df[prob_col].values.astype(float)
    
    # 反轉機率（如果需要）
    if args.invert:
        probs = 1 - probs
        print(f"✅ Inverted probabilities")
    
    # 決定 threshold
    if args.median:
        threshold = np.median(probs)
        print(f"📊 Using median threshold: {threshold:.4f}")
    else:
        threshold = args.threshold
        print(f"📊 Using threshold: {threshold}")
    
    # 轉換成標籤
    # 注意：prob > threshold → fake, prob <= threshold → real
    labels = ['fake' if p > threshold else 'real' for p in probs]
    
    # 統計
    fake_count = labels.count('fake')
    real_count = labels.count('real')
    total = len(labels)
    
    print(f"\n📈 Distribution:")
    print(f"   Fake: {fake_count} ({fake_count/total*100:.1f}%)")
    print(f"   Real: {real_count} ({real_count/total*100:.1f}%)")
    
    # 建立輸出 DataFrame
    output_df = pd.DataFrame({
        'filename': df[id_col].values,
        'label': labels
    })
    
    # 儲存
    output_df.to_csv(args.output, index=False)
    print(f"\n✅ Saved to: {args.output}")
    
    # 顯示前幾行
    print(f"\n📋 Preview:")
    print(output_df.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
