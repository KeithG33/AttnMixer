""" Vision Transformer (ViT) Pico registration using timm's VisionTransformer.

This defines a pico-sized ViT following the same sizing used by
the VitMixer-Pico in this repo: patch_size=16, embed_dim=128, depth=12, num_heads=4.

Uses timm's VisionTransformer default config (borrowed from vit_tiny_patch16_224)
to ensure preprocessing and metadata are aligned for proper comparison.
"""

from timm.models._builder import build_model_with_cfg
from timm.models._registry import register_model
from timm.models.vision_transformer import VisionTransformer
from timm.models.vision_transformer import default_cfgs as vit_default_cfgs
from timm.models.vision_transformer import _create_vision_transformer


@register_model
def vit_pico_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ ViT-Pico (ViT-Pi/16) - Ultra small baseline ViT for comparison.

    Mirrors VitMixer-Pico sizing: patch_size=16, embed_dim=128, depth=12, num_heads=4.
    """
    model_args = dict(
        patch_size=16,
        embed_dim=128,
        depth=12,
        num_heads=4,
        mlp_ratio=4,
        qkv_bias=True,
    )
    # Use timm's VisionTransformer factory for parity with stock ViTs
    return _create_vision_transformer('vit_pico_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))


# Register default cfg for vit_pico by borrowing timm's vit_tiny_patch16_224 cfg
default_cfgs = {
    'vit_pico_patch16_224': vit_default_cfgs.get('vit_tiny_patch16_224'),
}
