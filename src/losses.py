"""
src/losses.py
EDDA 的五个损失 + 两个稳定项

损失列表:
  ① edl_loss              —— 证据深度学习损失 (Sensoy et al., 2018)
  ② evidence_discrepancy  —— 证据分歧 d_e (意见向量 L1 距离 / (K+1))
  ③ mmd_loss              —— 多核最大均值差异 (MK-MMD)
  ④ coral_loss            —— CORAL 二阶统计对齐
  ⑤ cdan_loss             —— 条件域对抗损失
  ⑥ unc_anchor_loss       —— 不确定性锚定 (仅罚目标域更不确定)
  ⑦ balance_loss          —— 类别多样性 (防止单类坍塌)
"""
import torch
import torch.nn.functional as F
from typing import Tuple


# ============================================================
# ① EDL: 证据深度学习损失
# ============================================================
def edl_loss(alpha: torch.Tensor, y: torch.Tensor,
             kl_coeff: float = 0.1, kl_coeff_max: float = 0.1) -> torch.Tensor:
    """
    EDL 损失 = MSE + Variance + KL(alpha || Uniform)
    
    Args:
        alpha:    (B, K) Dirichlet 参数 (>=1)
        y:        (B,)  类别标签 (0..K-1)
        kl_coeff: 当前 KL 系数 (应使用升温后的值)
    Returns:
        标量损失
    """
    K = alpha.size(1)
    S = alpha.sum(dim=1, keepdim=True)
    p = alpha / S

    # MSE 项: (y_onehot - p)^2 期望
    y_onehot = F.one_hot(y, K).float()            # (B, K)
    mse = ((y_onehot - p) ** 2).sum(dim=1)        # (B,)

    # 方差项: 证据不足带来的预测方差
    var = (p * (1 - p) / (S + 1.0)).sum(dim=1)    # (B,)

    # KL 正则: 把非目标类的证据往 1 (均匀) 拉
    # a_t = y1 + (1-y1) * alpha → 目标类抹掉, 其他类保留
    a_t = y_onehot + (1.0 - y_onehot) * alpha     # (B, K)
    S_t = a_t.sum(dim=1, keepdim=True)             # (B, 1)

    # KL(Dirichlet(a_t) || Dirichlet(1))
    kl = (
        torch.lgamma(S_t).squeeze(1)
        - torch.lgamma(a_t).sum(dim=1)
        + ((a_t - 1.0) * (torch.digamma(a_t) - torch.digamma(S_t))).sum(dim=1)
    )  # (B,)

    return (mse + var + kl_coeff * kl).mean()


# ============================================================
# ② 证据分歧 d_e
# ============================================================
def evidence_discrepancy(o1: torch.Tensor, o2: torch.Tensor, K: int
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    逐样本计算意见向量的 L1 距离 / (K+1)
    
    Args:
        o1, o2: (B, K+1) 意见向量
        K:      类别数
    Returns:
        batch_mean: 标量, 批均值
        per_sample: (B,) 逐样本值 (可用于挑高分歧样本)
    """
    per_sample = (o1 - o2).abs().sum(dim=1) / (K + 1)
    return per_sample.mean(), per_sample


# ============================================================
# ③ MK-MMD: 多核最大均值差异
# ============================================================
def _pairwise_sqdist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """(B, d) × (B, d) → (B, B) 欧氏距离平方"""
    return (x.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(-1)


def mmd_loss(x: torch.Tensor, y: torch.Tensor,
             sigmas: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
             ) -> torch.Tensor:
    """
    多核 MMD
    
    Args:
        x, y: (B, d) 源域/目标域特征
        sigmas: 核带宽 (相对中位数启发式的倍率)
    Returns:
        标量 MMD²
    """
    dxx = _pairwise_sqdist(x, x)
    dyy = _pairwise_sqdist(y, y)
    dxy = _pairwise_sqdist(x, y)

    # 带宽由两域距离的中位数启发式确定
    med = dxy.detach().median().clamp_min(1e-6)

    loss = torch.tensor(0.0, device=x.device)
    for s in sigmas:
        gamma = 1.0 / (2.0 * (s * med) ** 2)
        kxx = torch.exp(-gamma * dxx).mean()
        kyy = torch.exp(-gamma * dyy).mean()
        kxy = torch.exp(-gamma * dxy).mean()
        loss = loss + (kxx + kyy - 2 * kxy)

    return loss / len(sigmas)


# ============================================================
# ④ CORAL: 二阶统计对齐
# ============================================================
def coral_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    CORAL: 最小化两域协方差矩阵的 Frobenius 距离
    
    默认作用在 logits 上 (与 MCDDFA 基线一致), d=num_classes
    """
    d = x.size(1)
    xm = x - x.mean(0, keepdim=True)
    ym = y - y.mean(0, keepdim=True)

    cx = xm.t() @ xm / max(x.size(0) - 1, 1)
    cy = ym.t() @ ym / max(y.size(0) - 1, 1)

    return (cx - cy).pow(2).sum() / (4.0 * d * d)


# ============================================================
# ⑤ CDAN: 条件域对抗损失
# ============================================================
def cdan_loss(D: torch.nn.Module, GRL_layer: torch.nn.Module,
              fs: torch.Tensor, ps: torch.Tensor,
              ft: torch.Tensor, pt: torch.Tensor,
              grl_lambda: float,
              multilinear_fn) -> torch.Tensor:
    """
    CDAN 损失 = L_DANN_source + L_DANN_target
    
    Args:
        D:          域判别器
        GRL_layer:  梯度反转层 (已设好 lambda)
        fs, ps:     源域特征 + 概率 (B, d), (B, K)
        ft, pt:     目标域特征 + 概率
        grl_lambda: GRL 系数
        multilinear_fn: 条件特征构造函数
    Returns:
        标量 CDAN 损失
    """
    GRL_layer.set_lambda(grl_lambda)

    # 条件多线性映射 → GRL → 判别器
    cond_s = multilinear_fn(fs, ps)              # (B, d*K)
    cond_t = multilinear_fn(ft, pt)

    adv_s = GRL_layer(cond_s)
    adv_t = GRL_layer(cond_t)

    logit_s = D(adv_s).squeeze(-1)               # (B,)
    logit_t = D(adv_t).squeeze(-1)

    # 源域标签 0, 目标域标签 1
    loss = (
        F.binary_cross_entropy_with_logits(logit_s, torch.zeros_like(logit_s))
        + F.binary_cross_entropy_with_logits(logit_t, torch.ones_like(logit_t))
    )
    return loss


# ============================================================
# ⑥ 不确定性锚定 (单边 hinge)
# ============================================================
def unc_anchor_loss(mean_ut: torch.Tensor, mean_us_ref: torch.Tensor,
                    margin: float = 0.05) -> torch.Tensor:
    """
    只惩罚 "目标域比源域更不确定" 的退化方向
    mean_us_ref 应是 detach 后的源域批均值 (Step 1 结束时记录)
    
    loss = hinge(mean_ut - mean_us_ref - margin)
    
    Args:
        mean_ut:    目标域当前批的平均不确定性
        mean_us_ref: 源域参考不确定性 (detached)
        margin:      hinge 边界
    """
    diff = mean_ut - mean_us_ref - margin
    return F.relu(diff)


# ============================================================
# ⑦ 类别多样性 (防止目标域单类坍塌)
# ============================================================
def balance_loss(p_target: torch.Tensor) -> torch.Tensor:
    """
    鼓励目标域各类别预测均匀分布 (信息量最大)
    
    使用均匀分布上的熵归一化: -H(p̄_t) / log(K)
    
    Args:
        p_target: (B, K) 目标域概率预测
    """
    K = p_target.size(1)
    p_mean = p_target.mean(dim=0)         # (K,) 平均类概率
    # 熵 H(p) = -sum p log p
    entropy = -(p_mean * (p_mean + 1e-8).log()).sum()
    max_entropy = torch.tensor(float(K)).log()  # log(K)
    return -entropy / max_entropy            # 范围 [-1, 0], 越小越均匀
