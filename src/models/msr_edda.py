"""
src/models/msr_edda.py
EDDA 完整模型: MS-ResCNN 主干 + 双 EvidenceHead + DomainDiscriminator

暴露 Step 1/2/3 各自需要的接口，避免反复构建前向
"""
import torch
import torch.nn as nn
from .backbone import MSResCNN
from .heads import EvidenceHead, DomainDiscriminator, GRL, conditional_multilinear


class MSREDDA(nn.Module):
    """
    MSR-EDDA 完整模型
    
    三步训练时的梯度流向:
      Step 1: G  + C1 + C2 + D 全部训练
      Step 2: 冻结 G (no_grad 包前向), 只训练 C1 + C2
      Step 3: 冻结 C1 + C2 (只改 requires_grad), 只训练 G
    """
    def __init__(self, num_classes: int = 4, feature_dim: int = 256,
                 head_hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.K = num_classes
        self.feature_dim = feature_dim

        # 共享主干 (num_channels=64 → ~251K 参数)
        self.G = MSResCNN(in_channels=1, num_channels=64,
                          feature_dim=feature_dim, window_length=2048)

        # 双证据头 (结构相同、初始化不同)
        self.C1 = EvidenceHead(d_in=feature_dim, num_classes=num_classes,
                               hidden=head_hidden, p_drop=dropout)
        self.C2 = EvidenceHead(d_in=feature_dim, num_classes=num_classes,
                               hidden=head_hidden, p_drop=dropout)
        # 确保两个头初始化不同
        self.C2.apply(self._init_different)

        # 域判别器 (CDAN 用)
        self.D = DomainDiscriminator(d_in=feature_dim * num_classes,
                                     hidden=256, p_drop=0.3)

        # 梯度反转层
        self.grl = GRL(lambda_adv=1.0)

    def _init_different(self, m):
        """给 C2 单独的初始化"""
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    # ── 前向接口 ──
    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """仅跑主干，返回特征"""
        return self.G(x)

    def forward_heads(self, f: torch.Tensor) -> tuple:
        """给定特征，跑两个证据头"""
        return self.C1(f), self.C2(f)

    def forward(self, x: torch.Tensor) -> tuple:
        """完整前向: x → 特征 → 双头"""
        f = self.G(x)
        h1 = self.C1(f)
        h2 = self.C2(f)
        return f, h1, h2

    # ── 推理接口 ──
    def predict(self, x: torch.Tensor, head_idx: int = 0) -> dict:
        """
        推理时用单个头 (默认 C1)
        
        Returns:
            {
                'pred': (B,)   预测类别
                'prob': (B, K) 期望概率
                'u':    (B, 1) 不确定性
                'e':    (B, K) 证据量
            }
        """
        self.eval()
        with torch.no_grad():
            f = self.G(x)
            h = self.C1(f) if head_idx == 0 else self.C2(f)
        return {
            "pred": h["p"].argmax(dim=1),
            "prob": h["p"],
            "u":    h["u"].squeeze(1),
            "e":    h["e"],
        }
