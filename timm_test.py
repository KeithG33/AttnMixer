"""Lightweight tests for VitMixer and VitMixerRelPos models.

Run as a script to exercise model creation, forward pass, and intermediates.
"""

import argparse
import sys
from typing import List

import torch
import timm

# Ensure model entrypoints are registered
import vitmixer  # noqa: F401
import vitmixer_relpos  # noqa: F401
import vit_pico  # noqa: F401  # ensure ViT-Pico is registered

from timm.data import resolve_model_data_config


def _assert_shape(t: torch.Tensor, shape: tuple, msg: str = "") -> None:
    assert tuple(t.shape) == shape, f"{msg} expected {shape}, got {tuple(t.shape)}"


def test_model_forward(model_name: str, device: str = "cpu", batch_size: int = 2) -> None:
    print(f"\n[TEST] {model_name}")
    m = timm.create_model(model_name, pretrained=False).to(device)
    m.eval()

    # Resolve default input size per-model
    data_cfg = resolve_model_data_config(m)
    c, h, w = data_cfg.get("input_size", (3, 224, 224))

    # Forward pass
    x = torch.randn(batch_size, c, h, w, device=device)
    with torch.no_grad():
        y = m(x)
    _assert_shape(y, (batch_size, m.num_classes), "logits")
    print(f"- forward OK, logits: {tuple(y.shape)}")

    # Intermediates (if available)
    if hasattr(m, "forward_intermediates"):
        try:
            with torch.no_grad():
                inter = m.forward_intermediates(x, indices=[2, 5], output_fmt="NCHW", intermediates_only=True)
            assert isinstance(inter, list) and len(inter) == 2, "unexpected intermediates output"
            for idx, feat in enumerate(inter):
                assert isinstance(feat, torch.Tensor), "intermediate not a tensor"
                assert feat.dim() == 4, "intermediate should be BCHW when output_fmt='NCHW'"
            print("- intermediates OK: 2 layers, BCHW")
        except Exception as e:
            print(f"- intermediates failed (non-fatal): {e}")

    # Parameter count
    n_params = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"- params: {n_params:.2f}M")


def test_model_overrides(model_name: str, device: str = "cpu") -> None:
    print(f"\n[TEST] overrides on {model_name}")
    m = timm.create_model(
        model_name,
        pretrained=False,
        embed_dim=128,
        depth=8,
        num_heads=2,
    ).to(device)
    m.eval()
    data_cfg = resolve_model_data_config(m)
    c, h, w = data_cfg.get("input_size", (3, 224, 224))
    x = torch.randn(1, c, h, w, device=device)
    with torch.no_grad():
        y = m(x)
    _assert_shape(y, (1, m.num_classes), "override logits")
    print("- overrides forward OK")


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="*",
        default=[
            # VitMixer
            "vitmixer_pico_patch16_224",
            "vitmixer_tiny_patch16_224",
            # VitMixerRelPos
            "vitmixer_relpos_pico_patch16_224",
            "vitmixer_relpos_tiny_patch16_224",
        ],
        help="Model names to test",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args(argv)

    torch.manual_seed(0)

    for name in args.models:
        test_model_forward(name, device=args.device, batch_size=args.batch_size)

    # Try overrides on one representative of each family
    test_model_overrides("vitmixer_small_patch32_224", device=args.device)
    test_model_overrides("vitmixer_relpos_small_patch32_224", device=args.device)

    print("\nAll tests completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
