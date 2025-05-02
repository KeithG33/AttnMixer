import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from timm.layers import DropPath, PatchEmbed


# Fixed sinusoidal positional encoding for 2D inputs
def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with "
            "odd dimension (got dim={:d})".format(d_model)
        )
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0.0, width).unsqueeze(1)
    pos_h = torch.arange(0.0, height).unsqueeze(1)
    pe[0:d_model:2, :, :] = (torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1))
    pe[1:d_model:2, :, :] = (torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1))
    pe[d_model::2, :, :] = (torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width))
    pe[d_model + 1 :: 2, :, :] = (torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width))
    return pe


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.0):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.norm(x)
        out = self.linear1(out)  # (B, N, C) -> (B, N, Hidden)
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.linear2(out)  # (B, N, Hidden) -> (B, N, C)
        out = out + x
        return out


class AttentionBlock(nn.Module):
    """Multi-head self-attention module with optional 1D or 2D relative position bias.

    Using timm Swin Transformer implementation as a reference for the 2d relative position bias. And
    uses a 1D relative position bias for the transposed attention in the mixer block.
    """

    def __init__(
        self,
        seq_len,
        embed_dim,
        num_heads,
        dropout=0.0,
        use_2d_relative_position=True,
        expansion_ratio=1.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim = embed_dim
        self.seq_len = seq_len
        self.use_2d_relative_position = use_2d_relative_position
        self.expanded_dim = int(embed_dim * expansion_ratio)
        self.head_dim = self.expanded_dim // num_heads
        assert (
            self.head_dim * num_heads == self.expanded_dim
        ), "expanded_dim must be divisible by num_heads"

        self.scale = nn.Parameter(torch.ones(1, 1, seq_len, 1))
        self.qkv_proj = nn.Linear(embed_dim, 3 * self.expanded_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(self.expanded_dim, embed_dim)

        if self.use_2d_relative_position:
            self.h, self.w = self.compute_grid_dimensions(seq_len)
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * self.h - 1) * (2 * self.w - 1), num_heads)
            )
            self.register_buffer(
                "relative_position_index",
                self.get_2d_relative_position_index(self.h, self.w),
            )
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        else:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(2 * seq_len - 1, num_heads)
            )
            self.register_buffer(
                "relative_position_index", self.get_1d_relative_position_index(seq_len)
            )
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def compute_grid_dimensions(self, n):
        """
        Compute grid dimensions (h, w) for 2D relative position bias. In our case, this will be the
        height and width of the chess board (8x8).
        """
        root = int(math.sqrt(n))
        for i in range(root, 0, -1):
            if n % i == 0:
                return (i, n // i)

    def get_2d_relative_position_index(self, h, w):
        """Create pairwise relative position index for 2D grid."""

        coords = torch.stack(
            torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        )  # 2, h, w
        coords_flatten = coords.reshape(2, -1)  # 2, h*w
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, h*w, h*w
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # h*w, h*w, 2
        relative_coords[:, :, 0] += h - 1  # Shift to start from 0
        relative_coords[:, :, 1] += w - 1
        relative_coords[:, :, 0] *= 2 * w - 1
        relative_position_index = relative_coords.sum(-1)  # Shape: (h*w, h*w)
        return relative_position_index  # h*w, h*w

    def get_1d_relative_position_index(self, seq_len):
        # Compute relative position indices for 1D sequences
        coords = torch.arange(seq_len)
        relative_coords = coords[None, :] - coords[:, None]  # seq_len, seq_len
        relative_coords += seq_len - 1  # Shift to start from 0
        return relative_coords  # seq_len, seq_len

    def _get_rel_pos_bias(self):
        """Retrieve relative position bias based on precomputed indices for the attention scores."""
        # Retrieve and reshape the relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1).long()
        ]
        relative_position_bias = relative_position_bias.view(
            self.seq_len, self.seq_len, -1
        )  # seq_len, seq_len, num_heads
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # num_heads, seq_len, seq_len
        return relative_position_bias.unsqueeze(0)  # 1, num_heads, seq_len, seq_len

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0) 

        relative_position_bias = self._get_rel_pos_bias() # 1, H, N, N
        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=relative_position_bias,
        )
        attn = self.scale * attn
        x = attn.transpose(1, 2).reshape(B, N, self.expanded_dim) 
        x = self.proj(x) 
        x = self.dropout(x)
        return x


class MixerBlock(nn.Module):
    def __init__(self, num_patches, embed_dim, num_heads=16, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.num_patches = num_patches
        self.seq_mixing_norm = nn.LayerNorm(embed_dim)
        self.seq_mixing_attn = AttentionBlock(
            num_patches,
            embed_dim,
            num_heads,
            dropout=0,
            use_2d_relative_position=True,
            expansion_ratio=1.25,
        )
        self.token_mixing_norm = nn.LayerNorm(embed_dim)
        self.token_mixing_attn = AttentionBlock(
            embed_dim,
            num_patches,
            7,
            dropout=0,
            use_2d_relative_position=False,
            expansion_ratio=1.0,
        )

        self.mlp = ResidualBlock(
            embed_dim, int(4 * embed_dim), dropout=dropout
        )

    def forward(self, x):
        # x shape: (B, 64, piece_embed)
        x = x + self.drop_path(self.token_mixing_attn(self.token_mixing_norm(x).transpose(1, 2)).transpose(1, 2))
        x = x + self.drop_path(self.seq_mixing_attn(self.seq_mixing_norm(x)))
        x = self.mlp(x)
        return x


class AttnMixer(nn.Module):
    """ Attn-Mixer model for image classification.

    This model is based off the mixer-MLP architecture, but uses attention blocks instead of MLPs.
    Additionally, both the sequence and token attention modules use relative position bias, where the
    token attention uses a 1D relative position bias and the sequence attention uses a 2D relative
    position bias corresponding to the patch grid.
    """

    def __init__(
        self,
        num_classes,
        img_size=(224, 224),
        patch_size=16,
        patch_embed_dim=96,
        num_blocks=(2,2,6,2),
        num_heads=(3,6,12,15),
        dropout=(0.025, 0.05, 0.075, 0.1),
        device='cuda',
    ):
        super().__init__()
        self.device = device

        # Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=patch_embed_dim,
            norm_layer=nn.LayerNorm,
        )
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        # Position encoding
        pos_encoding = positionalencoding2d(
            patch_embed_dim, self.patch_embed.grid_size[0], self.patch_embed.grid_size[1]
        ).view(1, patch_embed_dim, -1).permute(0, 2, 1)  # (1, C, N)
        self.register_buffer("pos_encoding", pos_encoding)

        # Mixer backbone
        self.stages = nn.ModuleList()
        for i in range(len(num_blocks)):
            blocks = nn.Sequential(
                *[
                    MixerBlock(
                        num_patches,
                        patch_embed_dim,
                        num_heads=num_heads[i],
                        dropout=dropout[i],
                        drop_path=0.0,
                    )
                    for _ in range(num_blocks[i])
                ]
            )
            self.stages.append(blocks)

        print(f"Num mixer params: {sum(p.numel() for p in self.stages.parameters())}")

        self.head_dim = patch_embed_dim * 8
        self.pred = nn.Sequential(
            nn.LayerNorm(patch_embed_dim),
            nn.Linear(patch_embed_dim, self.head_dim),
            nn.GELU(),
            nn.Linear(self.head_dim, num_classes),
        )

    def embed_image(self, x):
        x = self.patch_embed(x) # patch embed
        x = x + self.pos_encoding # sinusoidal encodings
        return x 

    def forward(self, x):
        features = self.embed_image(x)  # (B, N, C)
       
        for stage in self.stages:
            features = stage(features)

        features = features.mean(dim=1)  # (B, C)
        out = self.pred(features)
        return out




# model = AttnMixer(img_size=224, num_classes=4).cuda()

# # Generate synthetic data for demonstration
# image = torch.randn(1, 3, 224, 224).cuda()  # Shape: [batch_size, channels, height, width]

# output = model(image)  # Shape: [batch_size, output_dim]
# print(output)