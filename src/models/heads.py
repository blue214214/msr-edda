"""
src/models/heads.py
证据分类头、GRL (梯度反转层)、域判别器
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# ============================================================
# GRL: 梯度反转层
# ============================================================
class GradientReverse(Function):
    @staticmethod
    def forward(ctx, x, lambda_adv):
        ctx.lambda_adv = lambda_adv
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_adv * grad_output, None


class GRL(nn.Module):
    """包装成 nn.Module 方便使用"""
    def __init__(self, lambda_adv: float = 1.0):
        super().__init__()
        self.lambda_adv = lambda_adv

    def forward(self, x):
        return GradientReverse.apply(x, self.lambda_adv)

    def set_lambda(self, lambda_adv: float):
        self.lambda_adv = lambda_adv


# ============================================================
# EvidenceHead: Dirichlet 证据分类头
# ============================================================
class EvidenceHead(nn.Module):
    """
    输入: (B, d_in) 特征
    输出: dict，包含 e, alpha, S, p, b, u, o, logit
    
    关键定义:
        e = softplus(net(f)) + 1e-6          证据 (严格正)
        alpha = e + 1                        Dirichlet 参数 (≥1)
        S = alpha.sum(dim=1, keepdim=True)   总证据
        p = alpha / S                        期望概率
        b = e / S                            信念质量 (sum(b) = 1-u)
        u = K / S                            总体不确定性
        o = cat([b, u], dim=1)               意见向量 (B, K+1)
    """
    def __init__(self, d_in: int = 256, num_classes: int = 4,
                 hidden: int = 128, p_drop: float = 0.2):
        super().__init__()
        self.K = num_classes

        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, f: torch.Tensor) -> dict:
        """
        Returns:
            dict with keys:
                logit:  raw network output
                e:      evidence (B, K), strictly positive
                alpha:  Dirichlet param (B, K), >= 1
                S:      total evidence (B, 1)
                p:      expected probability (B, K), sum=1
                b:      belief mass (B, K), sum = 1 - u
                u:      uncertainty (B, 1)
                o:      opinion vector (B, K+1) = [b; u]
        """
        logit = self.net(f)                          # (B, K)
        e = F.softplus(logit) + 1e-6                 # 证据严格正
        alpha = e + 1.0                              # Dirichlet 参数
        S = alpha.sum(dim=1, keepdim=True)           # 总证据
        p = alpha / S                                # 期望概率
        b = e / S                                    # 信念质量
        u = self.K / S                               # 总体不确定性
        o = torch.cat([b, u], dim=1)                 # 意见向量

        return {
            "logit": logit,
            "e": e,
            "alpha": alpha,
            "S": S,
            "p": p,
            "b": b,
            "u": u,
            "o": o,
        }


# ============================================================
# DomainDiscriminator: 域判别器 (用于 CDAN)
# ============================================================
class DomainDiscriminator(nn.Module):
    """
    输入: 条件特征 (B, d_feat)，通常是 feature ⊗ p (多线性映射)
    输出: (B, 1) 域预测 logit (0=源域, 1=目标域)
    """
    def __init__(self, d_in: int, hidden: int = 256, p_drop: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, 1),
        )

    def forward(self, cond_feat: torch.Tensor) -> torch.Tensor:
        return self.net(cond_feat)


# ============================================================
# ConditionedFeature: 多线性映射 (CDAN 用)
# ============================================================
def conditional_multilinear(feature: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """
    多线性映射: feature ⊗ p → (B, d_feat * K)
    
    Args:
        feature: (B, d_feat) 特征
        p:       (B, K) 类别概率
    Returns:
        (B, d_feat * K) 条件特征
    """
    # feature: (B, d, 1), p: (B, 1, K) → outer product
    outer = feature.unsqueeze(2) * p.unsqueeze(1)  # (B, d, K)
    return outer.reshape(feature.size(0), -1)       # (B, d*K)
