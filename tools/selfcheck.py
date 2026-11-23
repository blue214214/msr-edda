"""
tools/selfcheck.py
MSR-EDDA 自检测脚本

检查清单 (与指南 3.4 对齐):
  1. 形状: MS-ResCNN 输出 B×256
  2. 参数量: 约 251K
  3. 梯度流向: 三步训练中各参数组是否收到梯度
  4. 损失量级: EDL ~ 0.5-2, d_e ≤ 0.4
  5. 单批过拟合: 10 步后 loss 应明显下降
  6. 证据头有效性: e > 0, α > 1, p 之和 = 1
  7. 冻结矩阵: Step 2 中 G 无梯度, Step 3 中 C 无梯度
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from src.models.backbone import MSResCNN
from src.models.heads import EvidenceHead, DomainDiscriminator
from src.models.msr_edda import MSREDDA
from src.losses import (
    edl_loss, evidence_discrepancy, mmd_loss, coral_loss,
    cdan_loss,
)
from src.models.heads import conditional_multilinear
from src.utils import count_parameters, freeze, unfreeze


def test_backbone_shapes():
    """1. 主干形状自检"""
    print("=" * 60)
    print("[1] MS-ResCNN 形状检测")
    model = MSResCNN(in_channels=1, num_channels=64, feature_dim=256)
    B = 4
    x = torch.randn(B, 1, 2048)
    out = model(x)
    print(f"  Input:  {tuple(x.shape)}")
    print(f"  Output: {tuple(out.shape)}")
    assert out.shape == (B, 256), f"FAIL: 期望 (4, 256), 得到 {tuple(out.shape)}"
    print(f"  PASS ✓")


def test_param_count():
    """2. 参数量自检 (应 ~251K)"""
    print("\n" + "=" * 60)
    print("[2] 参数量检测")
    model = MSResCNN(in_channels=1, num_channels=64, feature_dim=256)
    n = count_parameters(model)
    print(f"  MS-ResCNN: {n:,} 参数")
    # 251K 左右, 容差 ±5K
    if 245_000 <= n <= 260_000:
        print(f"  PASS ✓ (符合 ~251K 预期)")
    else:
        print(f"  WARN ⚠ (预期 251K, 实际 {n})")


def test_evidence_head():
    """3. 证据头形状 + 数学一致性"""
    print("\n" + "=" * 60)
    print("[3] EvidenceHead 检测")
    head = EvidenceHead(d_in=256, num_classes=4, hidden=128, p_drop=0.2)
    f = torch.randn(8, 256)
    h = head(f)

    print(f"  e  shape: {tuple(h['e'].shape)}")
    print(f"  alpha shape: {tuple(h['alpha'].shape)}")
    print(f"  S  shape: {tuple(h['S'].shape)}")
    print(f"  p  shape: {tuple(h['p'].shape)}")
    print(f"  b  shape: {tuple(h['b'].shape)}")
    print(f"  u  shape: {tuple(h['u'].shape)}")
    print(f"  o  shape: {tuple(h['o'].shape)}")

    # 数学检查
    assert (h["e"] > 0).all(), "e 必须严格正"
    assert (h["alpha"] > 1).all(), "alpha 必须 > 1"
    assert torch.allclose(h["p"].sum(1), torch.ones(8)), "p 之和应为 1"
    # b.sum() + u = 1
    b_sum = h["b"].sum(1) + h["u"].squeeze(1)
    assert torch.allclose(b_sum, torch.ones(8), atol=1e-5), f"b.sum() + u 应为 1, 实际 {b_sum}"
    # o = [b, u], 所以 o 的 L1 范数恒为 1
    o_norm = h["o"].abs().sum(1)
    assert torch.allclose(o_norm, torch.ones(8), atol=1e-5), f"o 的 L1 应为 1"

    print(f"  e > 0:   PASS ✓")
    print(f"  α > 1:   PASS ✓")
    print(f"  Σp = 1:  PASS ✓")
    print(f"  Σb+u = 1: PASS ✓")
    print(f"  ‖o‖₁ = 1: PASS ✓")


def test_discrepancy_range():
    """4. d_e 量级检测 (K=4 时 ≤ 0.4)"""
    print("\n" + "=" * 60)
    print("[4] 证据分歧 d_e 量级检测")
    K = 4
    o1 = torch.rand(10, K + 1)
    o2 = torch.rand(10, K + 1)
    # 归一化 (o 的 L1 应 = 1)
    o1 = o1 / o1.abs().sum(1, keepdim=True)
    o2 = o2 / o2.abs().sum(1, keepdim=True)
    d_e, d_per_sample = evidence_discrepancy(o1, o2, K)
    print(f"  K={K}: d_e mean={d_e.item():.4f}, max={d_per_sample.max().item():.4f}")
    assert d_per_sample.max() <= 2.0 / (K + 1) + 1e-6, f"d_e 上界应为 {2.0/(K+1)}"
    print(f"  上界检查: PASS ✓ (理论上界 {2.0/(K+1):.3f})")


def test_gradient_flow():
    """5. 三步训练梯度流向检测 (最重要)"""
    print("\n" + "=" * 60)
    print("[5] 三步训练梯度流向检测")

    model = MSREDDA(num_classes=4, feature_dim=256)
    B = 4
    xs = torch.randn(B, 1, 2048)
    ys = torch.randint(0, 4, (B,))
    xt = torch.randn(B, 1, 2048)

    # ── Step 1: 全部解冻, 全部应有梯度 ──
    print("\n  ── Step 1: 全部解冻 ──")
    unfreeze(model.G)
    unfreeze(model.C1)
    unfreeze(model.C2)
    unfreeze(model.D)

    fs = model.G(xs)
    ft = model.G(xt)
    h1s, h2s = model.C1(fs), model.C2(fs)
    h1t, h2t = model.C1(ft), model.C2(ft)

    loss = (edl_loss(h1s["alpha"], ys) + edl_loss(h2s["alpha"], ys)
            + mmd_loss(fs, ft)
            + cdan_loss(model.D, model.grl, fs, h1s["p"], ft, h1t["p"],
                        grl_lambda=1.0, multilinear_fn=conditional_multilinear))
    loss.backward()

    def _check_grads(module, name):
        has_grad = all(p.grad is not None for p in module.parameters())
        status = "✓ 有梯度" if has_grad else "✗ 无梯度"
        print(f"    {name:8s}: {status}")
        return has_grad

    ok1 = _check_grads(model.G, "G")
    ok2 = _check_grads(model.C1, "C1")
    ok3 = _check_grads(model.C2, "C2")
    ok4 = _check_grads(model.D, "D")
    print(f"    Step 1: {'PASS ✓' if all([ok1,ok2,ok3,ok4]) else 'FAIL ✗'}")

    # 清零
    model.zero_grad()

    # ── Step 2: 冻结 G, C 应有梯度, G 不应有 ──
    print("\n  ── Step 2: 冻结 G ──")
    freeze(model.G)
    unfreeze(model.C1)
    unfreeze(model.C2)

    with torch.no_grad():
        fs = model.G(xs)
        ft = model.G(xt)
    h1s, h2s = model.C1(fs), model.C2(fs)
    h1t, h2t = model.C1(ft), model.C2(ft)

    d_e_step2, _ = evidence_discrepancy(h1t["o"], h2t["o"], K=4)
    loss2 = (edl_loss(h1s["alpha"], ys) + edl_loss(h2s["alpha"], ys)
             - 3.0 * d_e_step2)
    loss2.backward()

    g_has_grad = any(p.grad is not None for p in model.G.parameters())
    ok5 = _check_grads(model.C1, "C1")
    ok6 = _check_grads(model.C2, "C2")
    print(f"    G 应无梯度: {'PASS ✓' if not g_has_grad else 'FAIL ✗ G 仍有梯度!'}")
    print(f"    Step 2: {'PASS ✓' if (not g_has_grad and ok5 and ok6) else 'FAIL ✗'}")

    model.zero_grad()

    # ── Step 3: 冻结 C (只改 requires_grad!), G 应有梯度 ──
    print("\n  ── Step 3: 冻结 C (只改 requires_grad) ──")
    unfreeze(model.G)
    for p in model.C1.parameters():
        p.requires_grad = False
    for p in model.C2.parameters():
        p.requires_grad = False
    model.C1.eval()
    model.C2.eval()

    fs = model.G(xs)          # 不能 no_grad!
    ft = model.G(xt)
    h1t, h2t = model.C1(ft), model.C2(ft)

    d_e_step3, _ = evidence_discrepancy(h1t["o"], h2t["o"], K=4)
    loss3 = d_e_step3
    loss3.backward()

    ok7 = _check_grads(model.G, "G")
    c_has_grad = any(p.grad is not None for p in model.C1.parameters()
                     ) or any(p.grad is not None for p in model.C2.parameters())
    print(f"    C 应无梯度: {'PASS ✓' if not c_has_grad else 'FAIL ✗ C 仍有梯度!'}")
    print(f"    Step 3: {'PASS ✓' if (ok7 and not c_has_grad) else 'FAIL ✗'}")

    # 恢复
    for p in model.C1.parameters():
        p.requires_grad = True
    for p in model.C2.parameters():
        p.requires_grad = True
    model.C1.train()
    model.C2.train()


def test_single_batch_overfit():
    """6. 单批过拟合检测 (10步后 loss 应下降)"""
    print("\n" + "=" * 60)
    print("[6] 单批过拟合检测 (10 步)")
    model = MSREDDA(num_classes=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    xs = torch.randn(8, 1, 2048)
    ys = torch.randint(0, 4, (8,))

    losses = []
    for step in range(10):
        f, h1, h2 = model(xs)
        loss = edl_loss(h1["alpha"], ys) + edl_loss(h2["alpha"], ys)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f"  Step 1 loss: {losses[0]:.4f}")
    print(f"  Step 10 loss: {losses[-1]:.4f}")
    if losses[-1] < losses[0] * 0.8:
        print(f"  PASS ✓ (loss 下降 {(1 - losses[-1]/losses[0])*100:.1f}%)")
    else:
        print(f"  WARN ⚠ (loss 未明显下降)")


def main():
    print("MSR-EDDA 自检测脚本")
    print("=" * 60)
    test_backbone_shapes()
    test_param_count()
    test_evidence_head()
    test_discrepancy_range()
    test_gradient_flow()
    test_single_batch_overfit()

    print("\n" + "=" * 60)
    print("自检测完成。如果都 PASS，可以开始跑真实数据了。")


if __name__ == "__main__":
    main()
