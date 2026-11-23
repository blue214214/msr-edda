"""
src/data/hit_loader.py
HIT (哈工大 HouLei) 航发中介轴承故障数据集加载器

来源: Hou L, et al. "Inter-Shaft Bearing Fault Diagnosis Based on 
Aero-Engine System: A Benchmarking Dataset Study". JDMD, 2023.

原始仓库: https://github.com/HouLeiHIT/HIT-dataset
Google Drive: https://drive.google.com/drive/folders/1Km1Go4ilB_bI033SBJ7eJ0uCzbqEqbgt

文件 (由 Git LFS 跟踪, ~2.87 GB):
    data1.npy ~ data5.npy   5 个 numpy 文件

数据特点 (待确认, 来自论文):
    - 6 通道振动信号 (转子 + 机壳传感器)
    - 故障类型: healthy, inner race fault, outer race fault, rolling element fault
    - 转速: 1200 ~ 6000 r/min (变转速)
    - 全寿命周期数据
    - 采样频率: 需确认

使用流程:
    1. git lfs pull  下载 5 个 npy 文件
    2. inspect_hit_data()  先看 shape/dtype/标签格式
    3. build_hit_task()     按实验需求构建迁移任务
"""
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional

from .datasets import BearingDataset, slice_signal


# ============================================================
# 数据探查 (首次使用必跑)
# ============================================================
def inspect_hit_data(data_dir: str) -> Dict[int, dict]:
    """
    加载每个 npy 文件并打印基本信息
    
    Returns:
        {idx: {"shape": ..., "dtype": ..., "min": ..., "max": ..., "mean": ...}}
    """
    data_dir = Path(data_dir)
    info = {}

    for i in range(1, 6):
        fpath = data_dir / f"data{i}.npy"
        if not fpath.exists():
            print(f"[HIT] ❌ {fpath.name} 不存在 (LFS 未下载?)")
            continue

        arr = np.load(str(fpath), mmap_mode='r')
        info[i] = {
            "shape": arr.shape,
            "dtype": str(arr.dtype),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
        }
        print(f"[HIT] data{i}.npy: shape={arr.shape}, dtype={arr.dtype}, "
              f"range=[{arr.min():.3f}, {arr.max():.3f}], mean={arr.mean():.3f}")

    return info


# ============================================================
# 加载单个文件为窗口
# ============================================================
def load_hit_npy(filepath: str, target_channel: int = 0) -> np.ndarray:
    """
    加载 HIT npy → 单通道一维信号
    
    Args:
        filepath:  .npy 文件路径
        target_channel: 如果多通道, 取第几通道 (默认 0)
    
    Returns:
        (L,) 一维 numpy 数组
    """
    arr = np.load(filepath)

    if arr.ndim == 1:
        return arr.astype(np.float32)
    elif arr.ndim == 2:
        # (N, channels) → 取一列
        return arr[:, target_channel].astype(np.float32)
    else:
        # 更高维 → flatten 或取第一个 slice
        print(f"[HIT] ⚠ 数组维度 {arr.ndim} (非 1D/2D), 尝试 flatten")
        return arr.flatten().astype(np.float32)


# ============================================================
# 构建迁移任务 (骨架, 等数据落地后完善)
# ============================================================
def build_hit_task(data_dir: str, src_idx: int, tgt_idx: int,
                   src_label: int, tgt_label: int,
                   window: int = 2048, stride: int = 1024,
                   target_channel: int = 0,
                   adapt_ratio: float = 0.7) -> dict:
    """
    HIT 跨域迁移任务
    
    假设 data1~5 分别对应不同工况/数据集:
      src_idx: 源域文件编号 (1-5)
      tgt_idx: 目标域文件编号 (1-5)
    
    src_label/tgt_label: 对应故障类型标签
    
    完整版本等 inspect_hit_data() 确认数据结构后完善
    """
    data_dir = Path(data_dir)
    src_file = data_dir / f"data{src_idx}.npy"
    tgt_file = data_dir / f"data{tgt_idx}.npy"

    if not src_file.exists() or not tgt_file.exists():
        print("[HIT] ❌ 文件不存在, 请先 git lfs pull")
        return None

    print(f"[HIT] 构建迁移任务: data{src_idx}(label={src_label}) → data{tgt_idx}(label={tgt_label})")

    src_signal = load_hit_npy(str(src_file), target_channel)
    tgt_signal = load_hit_npy(str(tgt_file), target_channel)

    # 切片
    src_windows = slice_signal(src_signal, window, stride)
    tgt_windows = slice_signal(tgt_signal, window, stride)

    # 归纳式划分
    src_split = int(len(src_windows) * adapt_ratio)
    tgt_split = int(len(tgt_windows) * adapt_ratio)

    src_train = BearingDataset(src_windows[:src_split],
                               np.full(src_split, src_label, dtype=np.int64))
    tgt_adapt = BearingDataset(tgt_windows[:tgt_split], None)
    tgt_test  = BearingDataset(tgt_windows[tgt_split:],
                               np.full(len(tgt_windows) - tgt_split, tgt_label, dtype=np.int64))

    print(f"  源域 train: {len(src_train)} 窗口, 标签 {src_label}")
    print(f"  目标域 adapt: {len(tgt_adapt)} 窗口")
    print(f"  目标域 test:  {len(tgt_test)} 窗口, 标签 {tgt_label}")

    return {
        "src_train": src_train,
        "tgt_adapt": tgt_adapt,
        "tgt_test": tgt_test,
        "info": {
            "src_file": f"data{src_idx}.npy",
            "tgt_file": f"data{tgt_idx}.npy",
            "src_label": src_label,
            "tgt_label": tgt_label,
            "src_signal_len": len(src_signal),
            "tgt_signal_len": len(tgt_signal),
        }
    }


# ============================================================
# 快速使用示例 (在数据落地后执行)
# ============================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python -m src.data.hit_loader inspect  <data_dir>")
        print("      python -m src.data.hit_loader task     <data_dir> <src_idx> <tgt_idx>")
        sys.exit(1)

    mode = sys.argv[1]
    data_dir = sys.argv[2]

    if mode == "inspect":
        inspect_hit_data(data_dir)
    elif mode == "task":
        src_idx = int(sys.argv[3])
        tgt_idx = int(sys.argv[4])
        build_hit_task(data_dir, src_idx, tgt_idx,
                       src_label=0, tgt_label=1)
