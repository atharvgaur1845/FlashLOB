"""In-repo reimplementations of the snapshot baselines (DeepLOB, TLOB).

These consume FI-2010-style snapshots [B, L, 40] and predict 3-class direction.
Reimplemented (not vendored) so every baseline trains under the *identical*
evaluation protocol — the whole point of Table 3. Architectures follow the papers
closely enough for comparable parameter counts and accuracy.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DeepLOB(nn.Module):
    """DeepLOB (Zhang et al., 2019): CNN + Inception + LSTM.

    Input [B, L, 40] -> reshaped to [B, 1, L, 40]. Convolutions first contract the
    40 LOB columns (price/size structure), an inception block mixes scales, then an
    LSTM over time, then a 3-way head. ~140k params.
    """

    def __init__(self, n_classes: int = 3, lstm_hidden: int = 64):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, (1, 2), stride=(1, 2)), nn.LeakyReLU(0.01), nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, (4, 1), padding=(3, 0)), nn.LeakyReLU(0.01), nn.BatchNorm2d(16),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, (1, 2), stride=(1, 2)), nn.LeakyReLU(0.01), nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, (4, 1), padding=(3, 0)), nn.LeakyReLU(0.01), nn.BatchNorm2d(16),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, (1, 10)), nn.LeakyReLU(0.01), nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, (4, 1), padding=(3, 0)), nn.LeakyReLU(0.01), nn.BatchNorm2d(16),
        )
        # inception: three parallel branches
        self.inc1 = nn.Sequential(nn.Conv2d(16, 32, (1, 1)), nn.LeakyReLU(0.01))
        self.inc2 = nn.Sequential(nn.Conv2d(16, 32, (3, 1), padding=(1, 0)), nn.LeakyReLU(0.01))
        self.inc3 = nn.Sequential(nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
                                  nn.Conv2d(16, 32, (1, 1)), nn.LeakyReLU(0.01))
        self.lstm = nn.LSTM(input_size=96, hidden_size=lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, n_classes)

    def forward(self, x):                         # x: [B, L, 40]
        x = x.unsqueeze(1)                        # [B,1,L,40]
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x)  # [B,16,L',1]
        i = torch.cat([self.inc1(x), self.inc2(x), self.inc3(x)], dim=1)  # [B,96,L',1]
        i = i.squeeze(-1).transpose(1, 2)         # [B, L', 96]
        out, _ = self.lstm(i)
        return self.fc(out[:, -1, :])             # [B, 3]


class TLOB(nn.Module):
    """TLOB (Berti & Kasneci, 2025): dual (temporal + feature) attention.

    Compact reimplementation: embed the 40 features, alternate temporal
    self-attention (over L) and feature self-attention (over the 40 channels),
    then pool and classify. ~1-2M params depending on depth/width.
    """

    def __init__(self, n_features: int = 40, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, n_classes: int = 3, seq_len: int = 100):
        super().__init__()
        self.feat_embed = nn.Linear(1, d_model)   # per-feature scalar -> d_model
        self.pos = nn.Parameter(torch.zeros(1, seq_len, n_features, d_model))
        self.blocks = nn.ModuleList([
            _DualAttentionBlock(d_model, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                nn.Linear(d_model, n_classes))
        self.seq_len = seq_len

    def forward(self, x):                          # x: [B, L, F]
        B, L, F = x.shape
        h = self.feat_embed(x.unsqueeze(-1))       # [B, L, F, d]
        if L <= self.pos.shape[1]:
            h = h + self.pos[:, :L]
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h).mean(dim=(1, 2))          # pool over time + features
        return self.fc(h)


class _DualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.temporal = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.feature = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.n1 = nn.LayerNorm(d_model); self.n2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(),
                                nn.Linear(2 * d_model, d_model))
        self.n3 = nn.LayerNorm(d_model)

    def forward(self, h):                           # [B, L, F, d]
        B, L, F, d = h.shape
        # temporal attention: sequences over L, batched across features
        t = h.permute(0, 2, 1, 3).reshape(B * F, L, d)
        t = t + self.temporal(self.n1(t), self.n1(t), self.n1(t))[0]
        h = t.reshape(B, F, L, d).permute(0, 2, 1, 3)
        # feature attention: sequences over F, batched across time
        f = h.reshape(B * L, F, d)
        f = f + self.feature(self.n2(f), self.n2(f), self.n2(f))[0]
        h = f.reshape(B, L, F, d)
        h = h + self.ff(self.n3(h))
        return h


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
