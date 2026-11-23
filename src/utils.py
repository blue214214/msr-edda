"""
src/utils.py
通用工具：随机种子、配置加载、日志、checkpoint、冻结/解冻
"""
import os
import random
import yaml
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
import csv


# ============================================================
# 随机种子
# ============================================================
def set_seed(seed: int, deterministic: bool = True):
    """四路设种子，确保可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


# ============================================================
# 配置加载
# ============================================================
def load_config(path: str) -> dict:
    """加载 YAML 配置"""
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg


def merge_configs(*cfgs: dict) -> dict:
    """深层合并多个配置字典，后者覆盖前者"""
    result = {}
    for cfg in cfgs:
        result = _deep_merge(result, cfg)
    return result


def _deep_merge(a: dict, b: dict) -> dict:
    out = a.copy()
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# ============================================================
# Run 目录管理
# ============================================================
def make_run_dir(dataset: str, task: str, method: str, seed: int,
                 timestamp: str = None, root: str = "runs") -> Path:
    """创建 run 目录：runs/{dataset}_{task}_{method}_s{seed}_{timestamp}/"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{dataset}_{task}_{method}_s{seed}_{timestamp}"
    run_dir = Path(root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config_snapshot(cfg: dict, run_dir: Path):
    """把实际使用的配置快照存入 run 目录"""
    with open(run_dir / "config.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)


# ============================================================
# 冻结 / 解冻
# ============================================================
def freeze(module: torch.nn.Module, eval_mode: bool = True):
    """冻结参数，可选切 eval 模式（对 Dropout / BN 重要）"""
    for p in module.parameters():
        p.requires_grad = False
    if eval_mode:
        module.eval()


def unfreeze(module: torch.nn.Module, train_mode: bool = True):
    """解冻参数，可选切 train 模式"""
    for p in module.parameters():
        p.requires_grad = True
    if train_mode:
        module.train()


# ============================================================
# Checkpoint
# ============================================================
def save_checkpoint(run_dir: Path, model, optimizers: dict, epoch: int,
                    metrics: dict, filename: str = "best.pt"):
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizers": {name: opt.state_dict() for name, opt in optimizers.items()},
        "metrics": metrics,
    }
    torch.save(ckpt, run_dir / filename)


def load_checkpoint(path: str, model, optimizers: dict = None, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizers is not None:
        for name, opt in optimizers.items():
            if name in ckpt["optimizers"]:
                opt.load_state_dict(ckpt["optimizers"][name])
    return ckpt["epoch"], ckpt["metrics"]


# ============================================================
# CSV 日志
# ============================================================
class CSVLogger:
    """按 epoch 一行写 CSV，便于后续画图和汇总"""
    def __init__(self, path: Path, fieldnames: list):
        self.path = path
        self.fieldnames = fieldnames
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    def log(self, row: dict):
        with open(self.path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
