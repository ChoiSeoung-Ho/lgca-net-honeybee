#!/usr/bin/env python3
"""Sanity-check the proposed model: parameter count, forward shape, ablation variants.

Examples
--------
    python verify_model.py --check-params
    python verify_model.py --check-params --forward
    python verify_model.py --dry-run          # no torch needed: prints the expected budget
"""

from __future__ import annotations

import argparse
import sys

from common.models import EXPECTED_FULL_PARAMS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check-params", action="store_true",
                   help="Instantiate the full model and assert it has "
                        f"{EXPECTED_FULL_PARAMS:,} trainable parameters.")
    p.add_argument("--forward", action="store_true",
                   help="Run a forward pass on a random 2x3x256x256 batch.")
    p.add_argument("--ablations", action="store_true",
                   help="Also report the parameter counts of the ablation variants.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the expected budget without importing torch.")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        print(f"Expected trainable parameters (full LGCA-Net): {EXPECTED_FULL_PARAMS:,}")
        print("Run without --dry-run (and with PyTorch installed) to verify.")
        return 0

    try:
        import torch
    except Exception:
        print("PyTorch is not installed; use --dry-run for the expected budget.",
              file=sys.stderr)
        return 2

    from common.models import LGCANet, count_parameters
    from common.seed import set_seed

    set_seed(42)
    model = LGCANet()
    n = count_parameters(model)
    print(f"LGCA-Net trainable parameters: {n:,}")

    ok = True
    if args.check_params:
        if n != EXPECTED_FULL_PARAMS:
            print(f"MISMATCH: expected {EXPECTED_FULL_PARAMS:,}, got {n:,}", file=sys.stderr)
            ok = False
        else:
            print(f"OK: matches the reported {EXPECTED_FULL_PARAMS:,}")

    if args.ablations:
        for kw, label in (
            (dict(use_transformer=True, use_cross_attn=False), "w/o cross-attention"),
            (dict(use_transformer=False, use_cross_attn=False), "CNN only"),
        ):
            m = LGCANet(**kw)
            print(f"  {label:<22s}: {count_parameters(m):,}")

    if args.forward:
        model.eval()
        with torch.no_grad():
            y = model(torch.rand(2, 3, 256, 256))
        print(f"forward output shape: {tuple(y.shape)} (expected (2, 2))")
        ok = ok and tuple(y.shape) == (2, 2)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
