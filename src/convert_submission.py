#!/usr/bin/env python3
"""
Submission 格式轉換腳本
=======================

將模型輸出的機率/數值預測轉換為 Kaggle 提交格式 (filename, label: real/fake)

支援的輸入格式：
1. 機率格式: id,label (label 是 0~1 的浮點數，表示 fake 的機率)
2. 數值格式: id,label (label 是 0 或 1，0=real, 1=fake)

輸出格式：
filename,label (label 是 "real" 或 "fake")

使用方式:
    # 基本轉換（預設閾值 0.5）
    python -m src.convert_submission -i outputs/ensemble/ensemble_submission.csv -o submission.csv
    
    # 自定義閾值
    python -m src.convert_submission -i outputs/submission.csv -o submission.csv --threshold 0.6
    
    # 批量轉換
    python -m src.convert_submission -i outputs/ensemble/individual/*.csv -o outputs/converted/ --batch
    
    # 查看統計資訊
    python -m src.convert_submission -i outputs/submission.csv -o submission.csv --stats
"""

import argparse
import os
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np


def convert_prediction(
    value: float,
    threshold: float = 0.5,
    invert: bool = False
) -> str:
    """
    將數值預測轉換為 real/fake 標籤
    
    Args:
        value: 預測值（機率或 0/1）
        threshold: 分類閾值
        invert: 是否反轉（如果模型輸出 0=fake, 1=real）
        
    Returns:
        "real" 或 "fake"
    """
    is_fake = value > threshold
    if invert:
        is_fake = not is_fake
    return "fake" if is_fake else "real"


def convert_submission(
    input_path: str,
    output_path: str,
    threshold: float = 0.5,
    invert: bool = False,
    show_stats: bool = True,
) -> pd.DataFrame:
    """
    轉換 submission 檔案格式
    
    Args:
        input_path: 輸入檔案路徑
        output_path: 輸出檔案路徑
        threshold: 分類閾值（預設 0.5）
        invert: 是否反轉標籤
        show_stats: 是否顯示統計資訊
        
    Returns:
        轉換後的 DataFrame
    """
    # 讀取輸入
    df = pd.read_csv(input_path)
    
    # 檢查欄位名稱
    if 'id' in df.columns:
        id_col = 'id'
    elif 'filename' in df.columns:
        id_col = 'filename'
    else:
        # 假設第一欄是 ID
        id_col = df.columns[0]
    
    if 'label' in df.columns:
        label_col = 'label'
    elif 'prediction' in df.columns:
        label_col = 'prediction'
    elif 'prob' in df.columns:
        label_col = 'prob'
    else:
        # 假設第二欄是 label
        label_col = df.columns[1]
    
    print(f"Input file: {input_path}")
    print(f"  ID column: {id_col}")
    print(f"  Label column: {label_col}")
    print(f"  Total samples: {len(df)}")
    
    # 檢查標籤類型
    sample_value = df[label_col].iloc[0]
    
    if isinstance(sample_value, str):
        if sample_value.lower() in ['real', 'fake']:
            print("  Format: Already in real/fake format!")
            # 已經是正確格式，只需要重新命名欄位
            result_df = pd.DataFrame({
                'filename': df[id_col].astype(int) if df[id_col].dtype != object else df[id_col],
                'label': df[label_col].str.lower()
            })
        else:
            # 嘗試轉換為數值
            df[label_col] = pd.to_numeric(df[label_col], errors='coerce')
            sample_value = df[label_col].iloc[0]
    
    if not isinstance(sample_value, str):
        # 數值格式，需要轉換
        values = df[label_col].values
        
        # 判斷是機率還是 0/1
        if np.all((values == 0) | (values == 1)):
            print(f"  Format: Binary (0/1)")
        else:
            print(f"  Format: Probability (range: {values.min():.4f} ~ {values.max():.4f})")
        
        print(f"  Threshold: {threshold}")
        
        # 轉換
        labels = [convert_prediction(v, threshold, invert) for v in values]
        
        result_df = pd.DataFrame({
            'filename': df[id_col].astype(int) if str(df[id_col].dtype).startswith(('int', 'float')) else df[id_col],
            'label': labels
        })
    
    # 排序
    try:
        result_df = result_df.sort_values('filename', key=lambda x: pd.to_numeric(x, errors='coerce'))
    except:
        result_df = result_df.sort_values('filename')
    
    # 統計
    if show_stats:
        fake_count = (result_df['label'] == 'fake').sum()
        real_count = (result_df['label'] == 'real').sum()
        print(f"\n  Statistics:")
        print(f"    Real: {real_count} ({100*real_count/len(result_df):.1f}%)")
        print(f"    Fake: {fake_count} ({100*fake_count/len(result_df):.1f}%)")
    
    # 保存
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved to: {output_path}")
    
    return result_df


def batch_convert(
    input_pattern: str,
    output_dir: str,
    threshold: float = 0.5,
    invert: bool = False,
):
    """批量轉換多個檔案"""
    from glob import glob
    
    input_files = glob(input_pattern)
    
    if not input_files:
        print(f"No files found matching: {input_pattern}")
        return
    
    print(f"Found {len(input_files)} files to convert")
    os.makedirs(output_dir, exist_ok=True)
    
    for input_path in input_files:
        filename = Path(input_path).name
        output_path = os.path.join(output_dir, f"converted_{filename}")
        
        print(f"\n{'='*60}")
        convert_submission(input_path, output_path, threshold, invert, show_stats=True)
    
    print(f"\n{'='*60}")
    print(f"✅ Batch conversion complete! Output dir: {output_dir}")


def analyze_predictions(input_path: str):
    """分析預測結果的分佈"""
    df = pd.read_csv(input_path)
    
    label_col = 'label' if 'label' in df.columns else df.columns[1]
    values = df[label_col]
    
    print(f"\n{'='*60}")
    print(f"Prediction Analysis: {input_path}")
    print(f"{'='*60}")
    
    if values.dtype == object:
        # 字串格式
        print(f"Format: Text labels")
        print(f"Distribution:")
        print(values.value_counts())
    else:
        # 數值格式
        print(f"Format: Numeric")
        print(f"  Count: {len(values)}")
        print(f"  Mean: {values.mean():.4f}")
        print(f"  Std: {values.std():.4f}")
        print(f"  Min: {values.min():.4f}")
        print(f"  Max: {values.max():.4f}")
        print(f"\nDistribution by threshold:")
        for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
            fake = (values > t).sum()
            real = len(values) - fake
            print(f"  Threshold {t}: Real={real}, Fake={fake} ({100*fake/len(values):.1f}% fake)")


def main():
    parser = argparse.ArgumentParser(
        description='Convert submission format (probability → real/fake)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic conversion
    python -m src.convert_submission -i outputs/submission.csv -o final_submission.csv
    
    # Custom threshold (more conservative, predict fake only if prob > 0.6)
    python -m src.convert_submission -i outputs/submission.csv -o final_submission.csv -t 0.6
    
    # Analyze prediction distribution
    python -m src.convert_submission -i outputs/submission.csv --analyze
    
    # Batch convert all individual predictions
    python -m src.convert_submission -i "outputs/ensemble/individual/*.csv" -o outputs/converted/ --batch
"""
    )
    
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Input CSV file path (or pattern for batch mode)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output CSV file path (or directory for batch mode)')
    parser.add_argument('-t', '--threshold', type=float, default=0.5,
                        help='Classification threshold (default: 0.5)')
    parser.add_argument('--invert', action='store_true',
                        help='Invert labels (if model outputs 0=fake, 1=real)')
    parser.add_argument('--batch', action='store_true',
                        help='Batch mode: convert multiple files')
    parser.add_argument('--analyze', action='store_true',
                        help='Only analyze predictions, do not convert')
    parser.add_argument('--no-stats', action='store_true',
                        help='Do not show statistics')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Submission Format Converter")
    print("=" * 60)
    
    if args.analyze:
        analyze_predictions(args.input)
        return
    
    if args.output is None:
        # 自動生成輸出路徑
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"final_{input_path.name}")
    
    if args.batch:
        batch_convert(args.input, args.output, args.threshold, args.invert)
    else:
        convert_submission(
            args.input,
            args.output,
            args.threshold,
            args.invert,
            show_stats=not args.no_stats
        )


if __name__ == '__main__':
    main()
