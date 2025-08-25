import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from timm.layers import DropPath, PatchEmbed
from attention import AttentionBlock, RelPosBias1D, RelPosBias2D, ChannelCovBias


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


class MixerBlock(nn.Module):
    def __init__(self, num_patches, embed_dim, num_heads=16, dropout=0.0):
        super().__init__()
        self.num_patches = num_patches
        self.seq_mixing_norm = nn.LayerNorm(embed_dim)
        self.seq_mixing_attn = AttentionBlock(
            num_patches,
            embed_dim,
            num_heads,
            # use_2d_relative_position=True,
            bias_module=RelPosBias2D(num_patches, num_heads),
            expansion_ratio=1.25,
        )
        self.token_mixing_norm = nn.LayerNorm(embed_dim)
        self.token_mixing_attn = AttentionBlock(
            embed_dim,
            num_patches,
            7,
            # use_2d_relative_position=False,
            bias_module=ChannelCovBias(7),
            expansion_ratio=1.25,
        )

        self.mlp = ResidualBlock(
            embed_dim, int(4 * embed_dim), dropout=dropout
        )

    def forward(self, x):
        # x shape: (B, 64, piece_embed)
        x = x + self.token_mixing_attn(self.token_mixing_norm(x).transpose(1, 2)).transpose(1, 2)
        x = x + self.seq_mixing_attn(self.seq_mixing_norm(x))
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
        num_classes=1000,
        img_size=(224, 224),
        patch_size=16,
        patch_embed_dim=128,
        num_blocks=(2,2,6,2),
        num_heads=(4,4,8,8),
        device='cuda',
    ):
        super().__init__()
        self.device = device

        # Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=patch_embed_dim,
            norm_layer=nn.LayerNorm,
        )
        num_patches = self.patch_embed.num_patches

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
                        dropout=0,
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