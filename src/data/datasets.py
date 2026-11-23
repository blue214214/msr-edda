"""
src/data/datasets.py
信号数据读取、窗口切片、归纳式划分 (adapt_ratio 协议)

归纳式协议 (UDA 标准):
    源域文件 ── 前 70% ──► source-train (有标签)
             └─ 后 30% ──► source-test  (只报告精度)
    目标域文件 ─ 前 70% ─► target-adapt (无标签, 参与训练)
              └ 后 30% ─► target-test  (有标签, 只评测)
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional


# ============================================================
# 信号切片
# ============================================================
def slice_signal(signal: np.ndarray, window: int = 2048,
                 stride: int = 1024) -> np.ndarray:
    """
    把一维信号切成窗口
    
    Args:
        signal: (L,) 原始振动信号
        window: 窗口长度
        stride: 步长
    Returns:
        (N, window) 窗口矩阵
    """
    L = len(signal)
    if L < window:
        # 信号太短时零填充
        pad = np.zeros(window - L)
        signal = np.concatenate([signal, pad])
        L = len(signal)

    starts = list(range(0, L - window + 1, stride))
    # 最后一个窗口确保覆盖尾部
    if starts[-1] + window < L:
        starts.append(L - window)

    return np.stack([signal[s:s + window] for s in starts])


def zscore_per_window(windows: np.ndarray) -> np.ndarray:
    """逐窗 z-score (与样本顺序无关, 抵抗幅值漂移)"""
    mean = windows.mean(axis=1, keepdims=True)
    std = windows.std(axis=1, keepdims=True).clip(min=1e-8)
    return (windows - mean) / std


# ============================================================
# 基础 Dataset
# ============================================================
class BearingDataset(Dataset):
    """
    轴承故障数据集: 接受 (N, 2048) 窗口数组 + 标签
    
    Args:
        windows: (N, window_length) 窗口数据
        labels:  (N,) 标签 (target domain adapt 阶段可为 None)
        normalize: 是否做逐窗 z-score
    """
    def __init__(self, windows: np.ndarray, labels: Optional[np.ndarray] = None,
                 normalize: bool = True):
        self.windows = windows.astype(np.float32)
        self.labels = labels.astype(np.int64) if labels is not None else None
        if normalize:
            self.windows = zscore_per_window(self.windows)
        # 增加通道维: (N, 1, window_length)
        self.windows = self.windows[:, np.newaxis, :]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.windows[idx])
        if self.labels is not None:
            return x, torch.tensor(self.labels[idx], dtype=torch.long)
        else:
            return x, torch.tensor(-1, dtype=torch.long)  # 伪标签 -1


# ============================================================
# 归纳式划分 + 加载
# ============================================================
def split_inductive(file_signals: dict, adapt_ratio: float = 0.7,
                    window: int = 2048, stride: int = 1024,
                    normalize: bool = True) -> dict:
    """
    按文件级归纳式划分, 再切窗口
    
    Args:
        file_signals: {condition_label: {fault_label: signal_array}}
            或简化为: {fault_label: signal_array} (同一 condition 下)
        adapt_ratio: 前 70% adapt, 后 30% test
        window, stride: 切片参数
    
    Returns:
        dict: {
            'adapt': BearingDataset,   # 无标签 (目标域) 或 有标签 (源域)
            'test':  BearingDataset,   # 有标签, 只评测
            'has_labels': bool
        }
    """
    adapt_windows, adapt_labels = [], []
    test_windows, test_labels = [], []
    has_labels = True

    for fault_label, signal in file_signals.items():
        if signal is None:
            continue
        L = len(signal)
        split = int(L * adapt_ratio)

        # 在 70%/30% 边界丢弃一个窗口避免重叠泄漏
        # adapt: [0, split - window//2], test: [split + window//2, L]
        split_adapt = split - stride // 2
        split_test = split + stride // 2

        if split_adapt > window:
            adapt_win = slice_signal(signal[:split_adapt], window, stride)
            adapt_windows.append(adapt_win)
            adapt_labels.extend([fault_label] * len(adapt_win))

        if split_test < L - window:
            test_win = slice_signal(signal[split_test:], window, stride)
            test_windows.append(test_win)
            test_labels.extend([fault_label] * len(test_win))

    adapt_windows = np.concatenate(adapt_windows, axis=0) if adapt_windows else np.zeros((0, window))
    test_windows  = np.concatenate(test_windows,  axis=0) if test_windows  else np.zeros((0, window))
    adapt_labels  = np.array(adapt_labels) if adapt_labels else None
    test_labels   = np.array(test_labels)  if test_labels  else None

    return {
        "adapt": BearingDataset(adapt_windows, adapt_labels, normalize),
        "test":  BearingDataset(test_windows,  test_labels,  normalize),
        "has_labels": has_labels,
    }


# ============================================================
# 配对 Sampler: 源域监督 loader + 目标域无监督 loader (1:1)
# ============================================================
def make_paired_loaders(src_dataset, tgt_dataset, batch_size: int = 64,
                        num_workers: int = 0, seed: int = 0) -> tuple:
    """
    创建 1:1 配对的源域/目标域 DataLoader
    
    - 源域 loader 用 cycle() 无限迭代
    - 目标域 loader 正常迭代
    - zip 后每次取一对 batch
    
    Returns:
        src_loader: 无限迭代的源域 loader
        tgt_loader: 正常迭代的目标域 loader
    """
    g = torch.Generator()
    g.manual_seed(seed)

    src_loader = DataLoader(
        src_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True, generator=g,
    )
    tgt_loader = DataLoader(
        tgt_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True, generator=g,
    )
    return src_loader, tgt_loader
