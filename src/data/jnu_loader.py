"""
src/data/jnu_loader.py
JNU (江南大学) 轴承数据集加载器

文件命名规则:
    n600_3_2.csv   → Normal, 600 r/min
    ib800_2.csv    → Inner Ball fault, 800 r/min
    ob1000_2.csv   → Outer Ball fault, 1000 r/min
    tb600_2.csv    → Ball fault (roller), 600 r/min

标签映射:
    NA (n) → 0
    IR (ib) → 1
    OR (ob) → 2
    BF (tb) → 3

采样率: 50 kHz
每个文件: ~150万点 (≈30秒)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict

from .datasets import BearingDataset, slice_signal, split_inductive


# ============================================================
# 文件命名解析
# ============================================================
# 故障类型前缀 → 标签
FAULT_MAP = {
    "n":  0,   # Normal
    "ib": 1,   # Inner Ball / Inner Race
    "ob": 2,   # Outer Ball / Outer Race
    "tb": 3,   # Ball / Roller fault
}

# 转速映射 (后缀中的第一个数字)
SPEED_MAP = {
    600: 0,    # 工况 0
    800: 1,    # 工况 1
    1000: 2,   # 工况 2
}


def parse_filename(filename: str) -> Tuple[int, int]:
    """
    解析 JNU 文件名 → (fault_label, speed_condition)
    
    示例:
        n600_3_2.csv  → (0, 0)
        ib800_2.csv   → (1, 1)
        ob1000_2.csv  → (2, 2)
        tb600_2.csv   → (3, 0)
    """
    stem = Path(filename).stem          # 去掉 .csv
    # 故障类型是前缀 (n, ib, ob, tb)
    for prefix in ["ib", "ob", "tb", "n"]:   # 长的先匹配
        if stem.startswith(prefix):
            fault = FAULT_MAP[prefix]
            rest = stem[len(prefix):]     # 剩下 "600_3_2" 或 "800_2"
            speed = int(rest.split("_")[0])
            cond = SPEED_MAP[speed]
            return fault, cond
    raise ValueError(f"无法解析文件名: {filename}")


# ============================================================
# 加载单个文件
# ============================================================
def load_jnu_csv(filepath: str) -> np.ndarray:
    """
    读取 JNU 的 CSV 文件 → 一维 numpy 数组
    
    CSV 格式: 单列, 无表头 (或表头是第一个采样值)
    """
    df = pd.read_csv(filepath, header=None)
    signal = df.iloc[:, 0].values.astype(np.float32)
    return signal


# ============================================================
# 按工况组织数据
# ============================================================
def organize_jnu_by_condition(data_dir: str) -> Dict[int, Dict[int, np.ndarray]]:
    """
    扫描 data_dir 下所有 CSV, 按 (工况 → 故障类型) 组织
    
    Returns:
        {
            0: {0: signal_array, 1: signal_array, 2: ..., 3: ...},  # 600 rpm
            1: {0: ..., 1: ..., 2: ..., 3: ...},                     # 800 rpm
            2: {0: ..., 1: ..., 2: ..., 3: ...},                     # 1000 rpm
        }
    """
    data_dir = Path(data_dir)
    result: Dict[int, Dict[int, np.ndarray]] = {0: {}, 1: {}, 2: {}}

    csv_files = sorted(data_dir.glob("*.csv"))
    print(f"[JNU] 发现 {len(csv_files)} 个 CSV 文件")

    for csv_file in csv_files:
        fault, cond = parse_filename(csv_file.name)
        signal = load_jnu_csv(str(csv_file))
        result[cond][fault] = signal
        print(f"  {csv_file.name:20s} → 标签={fault} 工况={cond} 长度={len(signal):,}")

    # 完整性检查
    for cond in [0, 1, 2]:
        missing = [f for f in range(4) if f not in result[cond]]
        if missing:
            print(f"  ⚠ 工况 {cond} 缺少故障类型: {missing}")

    return result


# ============================================================
# 构建跨工况迁移任务
# ============================================================
def build_transfer_task(data_dir: str, src_cond: int, tgt_cond: int,
                        window: int = 2048, stride: int = 1024,
                        adapt_ratio: float = 0.7,
                        normalize: bool = True) -> dict:
    """
    构建一个跨工况迁移任务: src_cond → tgt_cond
    
    Args:
        data_dir:    JNU 数据集根目录 (CSV 文件所在目录)
        src_cond:    源域工况 (0/1/2 对应 600/800/1000 rpm)
        tgt_cond:    目标域工况
        window:      窗口长度
        stride:      步长
        adapt_ratio: 归纳式划分比例
        normalize:   是否逐窗 z-score
    
    Returns:
        {
            "src_train": BearingDataset,   # 源域训练 (有标签, 70%)
            "src_test":  BearingDataset,   # 源域测试 (有标签, 30%)
            "tgt_adapt": BearingDataset,   # 目标域 adapt (无标签, 70%)
            "tgt_test":  BearingDataset,   # 目标域测试 (有标签, 30%)
        }
    """
    organized = organize_jnu_by_condition(data_dir)

    print(f"\n[JNU] 构建迁移任务: 工况 {src_cond} → 工况 {tgt_cond}")
    print(f"       采样率 50 kHz | 窗口 {window} | 步长 {stride} | 归纳式 {adapt_ratio}")

    def _make_split(signals_by_fault: dict, has_labels: bool) -> BearingDataset:
        """把一个工况下的所有故障信号切片并拼接"""
        all_windows, all_labels = [], []
        for fault_label, signal in signals_by_fault.items():
            win = slice_signal(signal, window, stride)
            # 归纳式: 前 adapt_ratio 作为 adapt/train, 后 (1-adapt_ratio) 作为 test
            L = len(signal)
            split_point = int(L * adapt_ratio)
            win_total = len(win)

            # 用文件级划分索引 (不是窗口级), 避免泄漏
            # adapt 窗口: 信号前 70% 切的窗
            split_win = int(win_total * adapt_ratio)

            all_windows.append(win[:split_win] if has_labels else win[:split_win])
            all_labels.extend([fault_label] * split_win)

        all_windows = np.concatenate(all_windows, axis=0)
        return BearingDataset(all_windows, np.array(all_labels), normalize)

    # 源域: 带标签, 训练用前 70%
    src_train = _make_split(organized[src_cond], has_labels=True)

    # 目标域 adapt: 无标签, 取前 70%
    tgt_adapt = _make_split(organized[tgt_cond], has_labels=False)

    # 目标域 test: 有标签, 取后 30%
    def _make_test(signals_by_fault: dict) -> BearingDataset:
        all_windows, all_labels = [], []
        for fault_label, signal in signals_by_fault.items():
            win = slice_signal(signal, window, stride)
            win_total = len(win)
            split_win = int(win_total * adapt_ratio)
            test_win = win[split_win:]
            if len(test_win) > 0:
                all_windows.append(test_win)
                all_labels.extend([fault_label] * len(test_win))
        if all_windows:
            all_windows = np.concatenate(all_windows, axis=0)
            return BearingDataset(all_windows, np.array(all_labels), normalize)
        return None

    src_test = _make_test(organized[src_cond])
    tgt_test = _make_test(organized[tgt_cond])

    print(f"  源域 train: {len(src_train)} 窗口")
    if src_test:
        print(f"  源域 test:  {len(src_test)} 窗口")
    print(f"  目标域 adapt: {len(tgt_adapt)} 窗口")
    if tgt_test:
        print(f"  目标域 test:  {len(tgt_test)} 窗口")

    return {
        "src_train": src_train,
        "src_test":  src_test,
        "tgt_adapt": tgt_adapt,
        "tgt_test":  tgt_test,
    }


# ============================================================
# 6 个迁移任务 (与 MCDDFA 原文一致)
# ============================================================
JNU_TASKS = [
    {"src": 0, "tgt": 1, "name": "J0→J1"},   # 600→800
    {"src": 0, "tgt": 2, "name": "J0→J2"},   # 600→1000
    {"src": 1, "tgt": 0, "name": "J1→J0"},   # 800→600
    {"src": 1, "tgt": 2, "name": "J1→J2"},   # 800→1000
    {"src": 2, "tgt": 0, "name": "J2→J0"},   # 1000→600
    {"src": 2, "tgt": 1, "name": "J2→J1"},   # 1000→800
]
