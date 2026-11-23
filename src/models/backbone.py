"""
src/models/backbone.py
MS-ResCNN: 多尺度残差卷积神经网络
输入: (B, 1, 2048)  振动信号切片
输出: (B, 256)       共享特征
参数量: ~251K
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv1d(nn.Module):
    """深度可分离卷积: DW + PW"""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 groups: int = None, padding: int = 0):
        super().__init__()
        if groups is None:
            groups = in_ch
        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size,
                            padding=padding, groups=groups, bias=False)
        self.pw = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.norm = nn.GroupNorm(8, out_ch)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.norm(x)
        return x


class MultiScaleResidualBlock(nn.Module):
    """
    三分支并行残差块
    ┌─ k=7  分支 ─┐
    ├─ k=15 分支 ─┼─ 1×1 融合 ─ (+ shortcut)
    └─ k=31 分支 ─┘
    每个分支都用深度可分离卷积 + GELU
    """
    def __init__(self, channels: int, num_groups: int = 8):
        super().__init__()
        self.branch_7  = DepthwiseSeparableConv1d(channels, channels,
                                                   kernel_size=7,  padding=3)
        self.branch_15 = DepthwiseSeparableConv1d(channels, channels,
                                                   kernel_size=15, padding=7)
        self.branch_31 = DepthwiseSeparableConv1d(channels, channels,
                                                   kernel_size=31, padding=15)
        # 融合层: 3×channels → channels
        self.fuse = nn.Sequential(
            nn.Conv1d(channels * 3, channels, 1, bias=False),
            nn.GroupNorm(num_groups, channels),
        )
        self.act = nn.GELU()
        self.shortcut_norm = nn.GroupNorm(num_groups, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b7  = self.branch_7(x)
        b15 = self.branch_15(x)
        b31 = self.branch_31(x)
        out = torch.cat([b7, b15, b31], dim=1)  # (B, 3C, L)
        out = self.fuse(out)                     # (B, C, L)
        out = self.act(out)
        # 残差连接
        shortcut = self.shortcut_norm(x)
        return out + shortcut


class MSResCNN(nn.Module):
    """
    MS-ResCNN 主干网络 (参数量 ~251K, num_channels=64)
    
    数据流:
        Input:  (B, 1, 2048)
        Stem:   Conv1d(1,64,k=31,s=2) + GN → (B, 64, 1024)
        Res1+Pool → (B, 64, 512)
        Res2+Pool → (B, 64, 256)
        Res3+Pool → (B, 64, 128)
        GAP + Proj(64→256) → (B, 256)
    """
    def __init__(self, in_channels: int = 1, num_channels: int = 64,
                 feature_dim: int = 256, window_length: int = 2048):
        super().__init__()
        self.feature_dim = feature_dim

        # ── Stem: 宽卷积建立大感受野 ──
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, num_channels, kernel_size=31, stride=2,
                      padding=15, bias=False),
            nn.GroupNorm(8, num_channels),
            nn.GELU(),
        )

        # ── 三个多尺度残差块 ──
        self.res_block_1 = MultiScaleResidualBlock(num_channels)
        self.res_block_2 = MultiScaleResidualBlock(num_channels)
        self.res_block_3 = MultiScaleResidualBlock(num_channels)

        # 块间下采样
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # ── 全局池化 + 投影 ──
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(num_channels, feature_dim),
            nn.GELU(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, 2048) 振动信号
        Returns:
            features: (B, 256) 跨域不变特征
        """
        # Stem
        x = self.stem(x)          # (B, 32, 1024)

        # 三个残差块 + 下采样
        x = self.res_block_1(x)   # (B, 32, 1024)
        x = self.pool(x)          # (B, 32, 512)

        x = self.res_block_2(x)   # (B, 32, 512)
        x = self.pool(x)          # (B, 32, 256)

        x = self.res_block_3(x)   # (B, 32, 256)
        x = self.pool(x)          # (B, 32, 128)

        # GAP + 投影
        x = self.gap(x)           # (B, 32, 1)
        x = x.squeeze(-1)         # (B, 32)
        x = self.proj(x)          # (B, 256)
        return x
