"""
src/engine.py
三步对抗训练引擎

训练伪代码 (每个 iteration):
    Step 1: 全部解冻, 计算 EDL + MMD + CDAN + CORAL + d_e_agree
    Step 2: 冻结 G, 只更新 C1/C2, 最大化目标域 d_e
    Step 3: 冻结 C1/C2 (保留反向图), 只更新 G, 最小化 d_e + unc_anchor + balance
"""
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple, Optional

from .losses import (
    edl_loss, evidence_discrepancy, mmd_loss, coral_loss,
    cdan_loss, unc_anchor_loss, balance_loss,
)
from .metrics import compute_all_metrics
from .utils import freeze, unfreeze, CSVLogger, save_checkpoint, count_parameters
from .models.heads import conditional_multilinear


class Trainer:
    """
    EDDA 三步训练器
    
    使用三个独立优化器 (opt_G, opt_C, opt_D), 覆盖互不相交的参数集合
    """
    def __init__(self, model, cfg: dict, device: str = "cuda"):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.K = model.K

        loss_cfg = cfg["loss"]
        train_cfg = cfg["train"]

        # 损失权重
        self.lam_mmd  = loss_cfg["lam_mmd"]
        self.lam_cdan = loss_cfg["lam_cdan"]
        self.lam_coral = loss_cfg["lam_coral"]
        self.lam_adv  = loss_cfg["lam_adv"]
        self.lam_agree = loss_cfg["lam_agree"]
        self.lam_unc  = loss_cfg["lam_unc"]
        self.lam_bal  = loss_cfg["lam_bal"]
        self.unc_margin = loss_cfg["unc_margin"]
        self.kl_max  = loss_cfg["kl_max"]
        self.kl_warmup_epochs = loss_cfg["kl_warmup_epochs"]

        # 优化器: 三组互不相交的参数
        self.opt_G = torch.optim.SGD(
            model.G.parameters(),
            lr=train_cfg["lr"],
            momentum=train_cfg["momentum"],
            weight_decay=train_cfg["weight_decay"],
        )
        self.opt_C = torch.optim.SGD(
            list(model.C1.parameters()) + list(model.C2.parameters()),
            lr=train_cfg["lr"],
            momentum=train_cfg["momentum"],
            weight_decay=train_cfg["weight_decay"],
        )
        self.opt_D = torch.optim.SGD(
            model.D.parameters(),
            lr=train_cfg["lr"],
            momentum=train_cfg["momentum"],
            weight_decay=train_cfg["weight_decay"],
        )

        # 学习率调度
        self.sched_G = torch.optim.lr_scheduler.StepLR(self.opt_G, step_size=50, gamma=0.5)
        self.sched_C = torch.optim.lr_scheduler.StepLR(self.opt_C, step_size=50, gamma=0.5)
        self.sched_D = torch.optim.lr_scheduler.StepLR(self.opt_D, step_size=50, gamma=0.5)

        self.grad_clip = train_cfg["grad_clip"]
        self.epochs = train_cfg["epochs"]
        self.seeds = train_cfg["seeds"]

        # 记录源域参考不确定性 (Step 1 结束后 detach)
        self._ref_uncertainty = None

        print(f"[Trainer] Model params: {count_parameters(model):,}")

    # ── KL 升温 ──
    def _kl_coeff(self, epoch: int) -> float:
        """前 kl_warmup_epochs 线性升温到 kl_max"""
        if epoch < self.kl_warmup_epochs:
            return self.kl_max * (epoch + 1) / self.kl_warmup_epochs
        return self.kl_max

    # ── GRL 系数调度 (DANN) ──
    def _grl_lambda(self, epoch: int) -> float:
        """前 10 epoch 线性升温到 1.0"""
        warmup = self.cfg.get("grl", {}).get("warmup_epochs", 10)
        lam_max = self.cfg.get("grl", {}).get("lambda_max", 1.0)
        if epoch < warmup:
            return lam_max * (epoch + 1) / warmup
        return lam_max

    # ── 单个 epoch 的三步训练 ──
    def train_epoch(self, src_loader, tgt_loader, epoch: int) -> dict:
        """
        Returns:
            dict: 各项损失与监控指标
        """
        self.model.train()
        kl_coeff = self._kl_coeff(epoch)
        grl_lambda = self._grl_lambda(epoch)

        # 用 cycle() 让源域无限迭代, 目标域迭代完本轮就停
        src_iter = iter(src_loader)
        tgt_iter = iter(tgt_loader)
        total_batches = len(tgt_loader)

        metrics_sum = {
            "loss_total": 0,
            "loss1_edl": 0, "loss1_mmd": 0, "loss1_cdan": 0,
            "loss1_coral": 0, "loss1_agree": 0,
            "loss2_edl": 0, "loss2_adv": 0,
            "loss3_disc": 0, "loss3_unc": 0, "loss3_bal": 0,
            "d_e": 0,
            "mean_S_src": 0, "mean_S_tgt": 0,
            "mean_u_src": 0, "mean_u_tgt": 0,
            "acc_src_head1": 0, "acc_src_head2": 0,
        }
        n_batches = 0

        pbar = tqdm(range(total_batches), desc=f"Epoch {epoch}", leave=False)
        for batch_idx in pbar:
            # ── 取一对 batch ──
            try:
                xs, ys = next(src_iter)
            except StopIteration:
                src_iter = iter(src_loader)
                xs, ys = next(src_iter)
            try:
                xt, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                xt, _ = next(tgt_iter)

            xs, ys = xs.to(self.device), ys.to(self.device)
            xt = xt.to(self.device)

            # ================================================================
            # Step 1: 全部解冻, 源域监督 + 全局对齐
            # ================================================================
            unfreeze(self.model.G)
            unfreeze(self.model.C1)
            unfreeze(self.model.C2)
            unfreeze(self.model.D)

            fs = self.model.G(xs)
            ft = self.model.G(xt)
            h1s = self.model.C1(fs)
            h2s = self.model.C2(fs)
            h1t = self.model.C1(ft)
            h2t = self.model.C2(ft)

            loss_edl  = edl_loss(h1s["alpha"], ys, kl_coeff) + edl_loss(h2s["alpha"], ys, kl_coeff)
            loss_mmd  = mmd_loss(fs, ft)
            loss_cdan = cdan_loss(
                self.model.D, self.model.grl,
                fs, h1s["p"], ft, h1t["p"],
                grl_lambda, conditional_multilinear,
            )
            loss_coral = coral_loss(h1s["logit"], h1t["logit"]) + coral_loss(h2s["logit"], h2t["logit"])
            d_e_batch, _ = evidence_discrepancy(h1t["o"], h2t["o"], self.K)
            loss_agree = d_e_batch  # Step 1 里希望目标域双头一致

            loss1 = (
                loss_edl
                + self.lam_mmd  * loss_mmd
                + self.lam_cdan * loss_cdan
                + self.lam_coral * loss_coral
                + self.lam_agree * loss_agree
            )

            self.opt_G.zero_grad()
            self.opt_C.zero_grad()
            self.opt_D.zero_grad()
            loss1.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.opt_G.step()
            self.opt_C.step()
            self.opt_D.step()

            # 记录 Step 1 结束时的源域平均不确定性 (detach 后作为 Step 3 参考)
            with torch.no_grad():
                self._ref_uncertainty = h1s["u"].mean().detach()

            # ================================================================
            # Step 2: 冻结主干, 最大化目标域证据分歧 (暴露边界样本)
            # ================================================================
            freeze(self.model.G)           # G 不需要训练 + eval 模式
            unfreeze(self.model.C1)
            unfreeze(self.model.C2)

            with torch.no_grad():           # 主干不需要梯度
                fs = self.model.G(xs)
                ft = self.model.G(xt)

            h1s = self.model.C1(fs)
            h2s = self.model.C2(fs)
            h1t = self.model.C1(ft)
            h2t = self.model.C2(ft)

            loss_edl_s2  = edl_loss(h1s["alpha"], ys, kl_coeff) + edl_loss(h2s["alpha"], ys, kl_coeff)
            d_e_batch2, _ = evidence_discrepancy(h1t["o"], h2t["o"], self.K)
            loss_coral_s2 = coral_loss(h1s["logit"], h1t["logit"]) + coral_loss(h2s["logit"], h2t["logit"])

            # 最大化分歧 = 最小化负分歧
            loss2 = (
                loss_edl_s2
                - self.lam_adv * d_e_batch2
                + self.lam_coral * loss_coral_s2
            )

            self.opt_C.zero_grad()
            loss2.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.C1.parameters()) + list(self.model.C2.parameters()),
                self.grad_clip,
            )
            self.opt_C.step()

            # ================================================================
            # Step 3: 冻结双头 (只改 requires_grad!), 最小化分歧 + 稳定项
            # ================================================================
            unfreeze(self.model.G)
            # 关键: 只改 requires_grad, 不能用 torch.no_grad()
            for p in self.model.C1.parameters():
                p.requires_grad = False
            for p in self.model.C2.parameters():
                p.requires_grad = False
            self.model.C1.eval()
            self.model.C2.eval()

            # 注意: Step 3 必须重新前向, 不能复用 Step 2 的 no_grad 特征!
            fs = self.model.G(xs)
            ft = self.model.G(xt)
            h1t = self.model.C1(ft)       # 双头 frozen 但仍参与前向图
            h2t = self.model.C2(ft)
            h1s = self.model.C1(fs)

            d_e_step3, _ = evidence_discrepancy(h1t["o"], h2t["o"], self.K)
            loss_unc = unc_anchor_loss(h1t["u"].mean(), self._ref_uncertainty, self.unc_margin)
            loss_bal = balance_loss(h1t["p"])

            loss3 = (
                d_e_step3
                + self.lam_unc * loss_unc
                + self.lam_bal * loss_bal
            )

            self.opt_G.zero_grad()
            loss3.backward()
            torch.nn.utils.clip_grad_norm_(self.model.G.parameters(), self.grad_clip)
            self.opt_G.step()

            # ── 恢复双头 requires_grad (为下一个 iteration 准备) ──
            for p in self.model.C1.parameters():
                p.requires_grad = True
            for p in self.model.C2.parameters():
                p.requires_grad = True
            self.model.C1.train()
            self.model.C2.train()

            # ── 累计指标 ──
            with torch.no_grad():
                metrics_sum["loss_total"] += (loss1.item() + loss2.item() + loss3.item())
                metrics_sum["loss1_edl"] += loss_edl.item()
                metrics_sum["loss1_mmd"] += loss_mmd.item()
                metrics_sum["loss1_cdan"] += loss_cdan.item()
                metrics_sum["loss1_coral"] += loss_coral.item()
                metrics_sum["loss1_agree"] += loss_agree.item()
                metrics_sum["loss2_edl"] += loss_edl_s2.item()
                metrics_sum["loss2_adv"] += d_e_batch2.item()
                metrics_sum["loss3_disc"] += d_e_step3.item()
                metrics_sum["loss3_unc"] += loss_unc.item()
                metrics_sum["loss3_bal"] += loss_bal.item()
                metrics_sum["d_e"] += d_e_batch2.item()

                metrics_sum["mean_S_src"] += h1s["S"].mean().item()
                metrics_sum["mean_S_tgt"] += h1t["S"].mean().item()
                metrics_sum["mean_u_src"] += h1s["u"].mean().item()
                metrics_sum["mean_u_tgt"] += h1t["u"].mean().item()

                metrics_sum["acc_src_head1"] += (h1s["p"].argmax(1) == ys).float().mean().item()
                metrics_sum["acc_src_head2"] += (h2s["p"].argmax(1) == ys).float().mean().item()

            n_batches += 1
            pbar.set_postfix({
                "loss": f"{(loss1.item()+loss2.item()+loss3.item()):.3f}",
                "d_e": f"{d_e_batch2.item():.4f}",
                "acc_s1": f"{metrics_sum['acc_src_head1']/n_batches:.3f}",
            })

        # ── 平均 ──
        for k in metrics_sum:
            metrics_sum[k] /= max(n_batches, 1)

        # ── LR 调度 ──
        self.sched_G.step()
        self.sched_C.step()
        self.sched_D.step()

        metrics_sum["lr_G"] = self.opt_G.param_groups[0]["lr"]
        metrics_sum["lr_C"] = self.opt_C.param_groups[0]["lr"]
        metrics_sum["lr_D"] = self.opt_D.param_groups[0]["lr"]

        return metrics_sum

    # ── 验证 ──
    @torch.no_grad()
    def validate(self, test_loader) -> dict:
        self.model.eval()
        all_prob, all_u, all_y = [], [], []

        for x, y in test_loader:
            x = x.to(self.device)
            h = self.model.C1(self.model.G(x))
            all_prob.append(h["p"].cpu().numpy())
            all_u.append(h["u"].squeeze(1).cpu().numpy())
            all_y.append(y.numpy())

        prob = np.concatenate(all_prob)
        u = np.concatenate(all_u)
        y = np.concatenate(all_y)

        return compute_all_metrics(prob, u, y)

    # ── 完整训练循环 ──
    def fit(self, src_loader, tgt_loader, test_loader, run_dir) -> dict:
        """完整训练: 返回最好的 epoch 及其指标"""
        import csv
        from pathlib import Path
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "epoch",
            "loss_total", "d_e", "mean_S_src", "mean_S_tgt",
            "mean_u_src", "mean_u_tgt",
            "acc_src_head1", "acc_src_head2",
            "lr_G",
            "val_Acc", "val_Macro-F1", "val_ECE", "val_Brier",
            "val_NLL", "val_ErrDet-AUROC", "val_AURC",
        ]
        csv_logger = CSVLogger(run_dir / "metrics.csv", fieldnames)

        best_val_acc = 0.0
        best_epoch = 0
        best_metrics = {}

        for epoch in range(self.epochs):
            train_metrics = self.train_epoch(src_loader, tgt_loader, epoch)
            val_metrics = self.validate(test_loader)

            # 记录
            row = {"epoch": epoch}
            row.update({k: train_metrics[k] for k in fieldnames if k.startswith("loss")
                        or k.startswith("d_e") or k.startswith("mean_")
                        or k.startswith("acc_src") or k.startswith("lr")})
            row.update({f"val_{k}": val_metrics[k] for k in ["Acc", "Macro-F1", "ECE",
                                                              "Brier", "NLL", "ErrDet-AUROC", "AURC"]})
            csv_logger.log(row)

            # 保存最好模型 (按验证集 Accuracy)
            if val_metrics["Acc"] > best_val_acc:
                best_val_acc = val_metrics["Acc"]
                best_epoch = epoch
                best_metrics = val_metrics
                save_checkpoint(run_dir, self.model,
                                {"G": self.opt_G, "C": self.opt_C, "D": self.opt_D},
                                epoch, val_metrics, "best.pt")

            print(f"  Epoch {epoch:3d}: train_loss={train_metrics['loss_total']:.3f}  "
                  f"d_e={train_metrics['d_e']:.4f}  S_tgt={train_metrics['mean_S_tgt']:.2f}  "
                  f"val_Acc={val_metrics['Acc']:.4f}  val_ECE={val_metrics['ECE']:.4f}")

        print(f"\n[Trainer] Best epoch: {best_epoch}, val_Acc: {best_val_acc:.4f}")
        return {"best_epoch": best_epoch, "best_metrics": best_metrics}
