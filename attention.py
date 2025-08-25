import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Bias modules (all nn.Module) ----------

class RelPosBias1D(nn.Module):
    """
    1D relative position bias (additive to logits).
    Returns shape: (1, H, N, N)
    """
    def __init__(self, seq_len: int, num_heads: int):
        super().__init__()
        self.seq_len = int(seq_len)
        self.num_heads = int(num_heads)

        # (2*L-1, H)
        self.bias_table = nn.Parameter(torch.zeros(2 * self.seq_len - 1, self.num_heads))
        nn.init.trunc_normal_(self.bias_table, std=0.02)

        # (L, L) indices in [0, 2L-2]
        coords = torch.arange(self.seq_len)
        rel = coords[None, :] - coords[:, None]            # (L, L) in [-L+1..L-1]
        rel = (rel + (self.seq_len - 1)).long()            # shift to [0..2L-2]
        self.register_buffer("index_1d", rel, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = self.seq_len
        # (L, L, H) -> (1, H, L, L)
        bias = self.bias_table[self.index_1d.view(-1)].view(L, L, self.num_heads)
        return bias.permute(2, 0, 1).unsqueeze(0)


class RelPosBias2D(nn.Module):
    """
    2D relative position bias (from SwinT) over an h*w grid (N = h*w).
    Returns shape: (1, H, N, N)
    """
    def __init__(self, seq_len: int, num_heads: int, hw: tuple[int, int] | None = None):
        super().__init__()
        self.num_heads = int(num_heads)
        self.seq_len = int(seq_len)

        if hw is None:
            h, w = self._compute_hw(self.seq_len)
        else:
            h, w = hw
            assert h * w == seq_len, "hw must multiply to seq_len"
        self.h, self.w = int(h), int(w)

        # ((2h-1)*(2w-1), H)
        size = (2 * self.h - 1) * (2 * self.w - 1)
        self.bias_table = nn.Parameter(torch.zeros(size, self.num_heads))
        nn.init.trunc_normal_(self.bias_table, std=0.02)

        # build pairwise relative index (N, N)
        rel_index = self._build_index_2d(self.h, self.w)   # (N, N)
        self.register_buffer("index_2d", rel_index, persistent=False)

    @staticmethod
    def _compute_hw(n: int) -> tuple[int, int]:
        root = int(math.sqrt(n))
        for i in range(root, 0, -1):
            if n % i == 0:
                return (i, n // i)
        raise ValueError(f"Cannot factor seq_len={n} into an integer grid")

    @staticmethod
    def _build_index_2d(h: int, w: int) -> torch.Tensor:
        # coords: (2, h, w) with 'ij' indexing
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        coords = torch.stack([yy, xx], dim=0).reshape(2, -1)  # (2, N)
        rel = coords[:, :, None] - coords[:, None, :]         # (2, N, N)
        rel = rel.permute(1, 2, 0).contiguous()               # (N, N, 2)
        # shift to start from 0
        rel[:, :, 0] += h - 1
        rel[:, :, 1] += w - 1
        rel[:, :, 0] *= (2 * w - 1)
        index = rel.sum(-1)                                   # (N, N)
        return index.long()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = self.seq_len
        # (N, N, H) -> (1, H, N, N)
        bias = self.bias_table[self.index_2d.view(-1)].view(N, N, self.num_heads)
        return bias.permute(2, 0, 1).unsqueeze(0)


class ChannelCovBias(nn.Module):
    """
    Content-conditioned channel×channel bias for token-mixing attention.
    - Input x expected as (B, N, Cfeat), where N is the "sequence" (channels in transposed attention)
    - Pearson correlation across the last dim (Cfeat), tanh bound, zero diagonal
    - Per-head learned scale (init 0)
    - Learned positive temperature via softplus + floor
    Returns: (B, H, N, N)
    """
    def __init__(self, num_heads: int, eps: float = 1e-6, tau_init: float = 1.0, tau_floor: float = 1e-2):
        super().__init__()
        self.num_heads = int(num_heads)
        self.eps = float(eps)

        # Per-head scale, safe residual start
        self.bias_scale = nn.Parameter(torch.zeros(self.num_heads))

        # Positive temperature via softplus
        self.register_buffer("_tau_floor", torch.tensor(float(tau_floor)), persistent=False)
        tau_eff = max(float(tau_init) - float(tau_floor), 1e-6)
        log_tau_init = math.log(math.expm1(tau_eff))  # inverse softplus
        self.log_tau = nn.Parameter(torch.tensor(log_tau_init, dtype=torch.float32))

    @property
    def temperature(self) -> torch.Tensor:
        return F.softplus(self.log_tau) + self._tau_floor  # scalar tensor > 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, Cfeat = x.shape

        # 1) center across content/features
        xc = x - x.mean(dim=-1, keepdim=True)                      # (B, N, Cfeat)

        # 2) normalize to unit variance per "channel" -> Pearson correlation
        var = xc.square().mean(dim=-1, keepdim=True).clamp_min(self.eps)
        xc = xc / var.sqrt()

        # 3) correlation across channels
        corr = (xc @ xc.transpose(1, 2)) / float(Cfeat)            # (B, N, N)

        # 4) stabilize & remove self-bias
        corr = F.tanh(corr)
        corr = corr - torch.diag_embed(torch.diagonal(corr, dim1=-2, dim2=-1))

        # 5) per-head scale and temperature
        bias = corr.unsqueeze(1)                                   # (B, 1, N, N)
        bias = bias * self.bias_scale.view(1, self.num_heads, 1, 1).to(bias.dtype)
        bias = bias / self.temperature.to(bias.dtype)              # (B, H, N, N)
        return bias


class AttentionBlock(nn.Module):
    """
    Multi-head self-attention with pluggable bias module.

    Args:
        seq_len: sequence length N
        embed_dim: model width D
        num_heads: number of heads H
        bias_module: nn.Module that returns (B|1, H, N, N) when called.
                     - For content-free bias (1D/2D rel-pos), it can ignore x and return (1, H, N, N).
                     - For content-based bias (e.g., ChannelCovBias), it should consume x (B, N, D).
        expansion_ratio: expansion for qkv/proj inner dim.
    """
    def __init__(
        self,
        seq_len: int,
        embed_dim: int,
        num_heads: int,
        bias_module: nn.Module | None = None,
        expansion_ratio: float = 1.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.expanded_dim = int(embed_dim * expansion_ratio)
        self.head_dim = self.expanded_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.expanded_dim, "expanded_dim must be divisible by num_heads"

        self.qkv_proj = nn.Linear(self.dim, 3 * self.expanded_dim)
        self.proj = nn.Linear(self.expanded_dim, self.dim)
        self.attn_dropout = float(attn_dropout)

        # pluggable bias module (can be None)
        self.bias_module = bias_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D)
        """
        B, N, D = x.shape
        assert N == self.seq_len and D == self.dim, f"Expected (B,{self.seq_len},{self.dim}), got {x.shape}"

        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        bias = None
        if self.bias_module is not None:
            bias = self.bias_module(x)
            bias = bias.to(q.dtype)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=bias,               # additive logit bias
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        x = attn.transpose(1, 2).reshape(B, N, self.expanded_dim)
        x = self.proj(x)
        return x