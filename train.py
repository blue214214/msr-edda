"""
train.py
训练入口: --config --task --seed --method

示例:
    # JNU 数据集, J0→J1 迁移任务, seed=0
    python train.py --config configs/jnu.yaml --src 0 --tgt 1 --seed 0

    # 只用源域做 sanity check (先跑这个!)
    python train.py --config configs/jnu.yaml --src 0 --tgt 1 --seed 0 --mode source_only

    # 合成数据 (不需要数据集)
    python train.py --config configs/base.yaml --seed 0 --mode synthetic
"""
import sys
import argparse
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.utils import load_config, merge_configs, set_seed, make_run_dir, save_config_snapshot
from src.models.msr_edda import MSREDDA
from src.engine import Trainer
from src.data.datasets import BearingDataset, make_paired_loaders


# ============================================================
# 数据加载: 支持 JNU / 合成
# ============================================================
def load_data(cfg: dict, src_cond: int, tgt_cond: int, seed: int,
              mode: str = "full") -> tuple:
    """
    Returns:
        (src_loader, tgt_loader, test_loader, info_dict)
    """
    dataset_name = cfg.get("data", {}).get("name", "synthetic")
    window = cfg["data"]["window"]
    stride = cfg["data"]["stride"]
    batch_size = cfg["train"]["batch_size"]

    if dataset_name == "jnu":
        from src.data.jnu_loader import build_transfer_task

        data_dir = cfg["data"]["root"]
        task_data = build_transfer_task(
            data_dir, src_cond, tgt_cond,
            window=window, stride=stride,
            adapt_ratio=cfg["data"]["adapt_ratio"],
            normalize=(cfg["data"]["norm"] == "per_window"),
        )

        src_dataset = task_data["src_train"]
        tgt_dataset = task_data["tgt_adapt"]
        test_dataset = task_data["tgt_test"]

        info = {
            "src_cond": src_cond,
            "tgt_cond": tgt_cond,
            "src_train_size": len(src_dataset),
            "tgt_adapt_size": len(tgt_dataset),
            "tgt_test_size": len(test_dataset) if test_dataset else 0,
        }

    elif dataset_name == "synthetic":
        # 合成数据 fallback
        N = 1024
        K = cfg["model"]["num_classes"]
        np.random.seed(seed)

        src_windows = np.random.randn(N, window).astype(np.float32)
        src_labels = np.random.randint(0, K, N)
        tgt_windows = np.random.randn(N, window).astype(np.float32) + 0.5
        tgt_labels = np.random.randint(0, K, N)

        src_dataset = BearingDataset(src_windows, src_labels, normalize=True)
        tgt_dataset = BearingDataset(tgt_windows, None, normalize=True)
        test_dataset = BearingDataset(tgt_windows[:256], tgt_labels[:256], normalize=True)

        info = {"mode": "synthetic", "N": N}
    else:
        raise ValueError(f"未知数据集: {dataset_name}")

    # ── 构建 DataLoader ──
    if mode == "source_only":
        # 只跑源域监督 (sanity check)
        src_loader = torch.utils.data.DataLoader(
            src_dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, drop_last=True,
        )
        tgt_loader = None
    else:
        src_loader, tgt_loader = make_paired_loaders(
            src_dataset, tgt_dataset,
            batch_size=batch_size, seed=seed,
        )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
    )

    return src_loader, tgt_loader, test_loader, info


# ============================================================
# main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="MSR-EDDA Training")
    parser.add_argument("--config", type=str, default="configs/base.yaml",
                        help="配置文件路径")
    parser.add_argument("--src", type=int, default=0,
                        help="源域工况 (JNU: 0/1/2 对应 600/800/1000 rpm)")
    parser.add_argument("--tgt", type=int, default=1,
                        help="目标域工况")
    parser.add_argument("--seed", type=int, default=0,
                        help="随机种子")
    parser.add_argument("--device", type=str, default="cuda",
                        help="训练设备 (cuda / cpu)")
    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "source_only", "synthetic"],
                        help="full=完整三步, source_only=仅源域监督, synthetic=合成数据")
    args = parser.parse_args()

    # ── 加载配置 ──
    cfg = merge_configs(load_config("configs/base.yaml"), load_config(args.config))
    cfg["seed"] = args.seed

    # 如果指定 synthetic 模式, 覆盖数据集名
    if args.mode == "synthetic":
        cfg["data"]["name"] = "synthetic"

    # ── 设种子 ──
    set_seed(args.seed, cfg["train"]["deterministic"])
    device = args.device if torch.cuda.is_available() else "cpu"

    # ── 创建 run 目录 ──
    dataset_name = cfg.get("data", {}).get("name", "unknown")
    task_name = f"J{args.src}→J{args.tgt}" if dataset_name == "jnu" else "default"
    run_dir = make_run_dir(dataset_name, task_name, "EDDA", args.seed)
    save_config_snapshot(cfg, run_dir)

    print("=" * 60)
    print(f"  MSR-EDDA 训练启动")
    print(f"  数据集: {dataset_name}")
    print(f"  任务:   {task_name}")
    print(f"  模式:   {args.mode}")
    print(f"  种子:   {args.seed}")
    print(f"  设备:   {device}")
    print(f"  Run:    {run_dir}")
    print("=" * 60)

    # ── 加载数据 ──
    src_loader, tgt_loader, test_loader, info = load_data(
        cfg, args.src, args.tgt, args.seed, args.mode,
    )
    print(f"\n数据信息: {info}\n")

    # ── 创建模型 ──
    model = MSREDDA(
        num_classes=cfg["model"]["num_classes"],
        feature_dim=cfg["model"]["feature_dim"],
        head_hidden=cfg["model"]["head_hidden"],
        dropout=cfg["model"]["dropout"],
    )

    # ── 创建 Trainer ──
    trainer = Trainer(model, cfg, device)

    # ── 跑训练 ──
    result = trainer.fit(src_loader, tgt_loader, test_loader, run_dir)
    print(f"\n{'='*60}")
    print(f"  训练完成!")
    print(f"  Best epoch: {result['best_epoch']}")
    print(f"  Best val Acc: {result['best_metrics'].get('Acc', 'N/A')}")
    print(f"  Run dir: {run_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
