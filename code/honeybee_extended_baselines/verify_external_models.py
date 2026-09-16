#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check every registered model WITHOUT training, and record what was loaded.

For each model this builds it, counts parameters, runs one forward pass on a
``(2, 3, 256, 256)`` tensor and checks the output is ``(2, 2)`` logits.  It
writes ``results/model_provenance.json`` (consumed by the table generator, so
that the parameter column in the manuscript is measured rather than quoted) and
prints a readable status table.

Run this BEFORE launching a multi-day training job:

    python verify_external_models.py --models all
    python verify_external_models.py --models conformer mobileformer lsnet_t mambavision

Exit code is non-zero if any requested model failed, so it can gate a shell
pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_SET = ["lgca_net", "lgca_net_no_cross_attn", "lgca_net_cnn_only",
               "resnet50", "densenet121", "efficientnetv2", "coatnet",
               "crossvit", "nextvit_small",
               "conformer", "mobileformer", "lsnet_t", "mambavision"]


def _count(m):
    return (int(sum(p.numel() for p in m.parameters())),
            int(sum(p.numel() for p in m.parameters() if p.requires_grad)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-root", default=".",
                    help="directory containing common/models.py")
    ap.add_argument("--models", nargs="+", default=["all"])
    ap.add_argument("--num-classes", type=int, default=2)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--allow-random-init", action="store_true",
                    help="Permit a backbone with no public ImageNet checkpoint "
                         "(Mobile-Former) to be built randomly initialised. The "
                         "provenance records pretrained_loaded=false and the "
                         "manuscript table must then say 'none' in the "
                         "pretraining column for that row.")
    ap.add_argument("--out", default="./results/model_provenance.json")
    a = ap.parse_args()

    sys.path.insert(0, os.path.abspath(a.code_root))
    from common import models as M                      # noqa: E402
    try:
        from models.external import register as register_external   # noqa: E402
        register_external(M.MODEL_REGISTRY, M.DISPLAY_NAME)
        from models.external import adapters as EXT                  # noqa: E402
    except Exception as exc:
        print(f"[warn] external adapters not importable: {exc}")
        EXT = None

    if a.allow_random_init:
        os.environ["ALLOW_RANDOM_INIT"] = "1"

    names = DEFAULT_SET if a.models == ["all"] else a.models
    rows, failures = {}, []

    for name in names:
        entry = {"key": name, "display_name": M.display_name(name), "ok": False}
        try:
            pretrained = not a.no_pretrained and name not in M.FROM_SCRATCH
            model = M.build_model(name, pretrained=pretrained,
                                  num_classes=a.num_classes)
            total, trainable = _count(model)
            model.eval()
            try:
                with torch.no_grad():
                    out = model(torch.zeros(2, 3, 256, 256))
            except Exception as _cpu_exc:
                if not torch.cuda.is_available():
                    raise
                model = model.cuda()
                model.eval()   # LeViT식 self.ab 캐시 재생성
                with torch.no_grad():
                    out = model(torch.zeros(2, 3, 256, 256, device="cuda"))
                if torch.is_tensor(out):
                    out = out.detach().cpu()
                model = model.cpu()
                model.eval()
                entry["cpu_forward"] = (
                    f"failed ({type(_cpu_exc).__name__}); verified on CUDA")
            if not (torch.is_tensor(out) and tuple(out.shape) == (2, a.num_classes)):
                raise RuntimeError(f"forward returned {getattr(out,'shape',type(out))}, "
                                   f"expected (2, {a.num_classes})")
            entry.update(ok=True, n_params_total=total, n_params_trainable=trainable,
                         params_millions=round(total / 1e6, 3),
                         pretrained_requested=bool(pretrained),
                         pretrained_loaded=bool(pretrained),
                         source="timm/bundled")
            if EXT is not None:
                prov = EXT.last_provenance(name)
                if prov:                                  # external model: richer record
                    entry.update({k: v for k, v in prov.items() if k != "key"})
                    entry["ok"] = True
                    entry["params_millions"] = round(prov["n_params_total"] / 1e6, 3)
            del model
        except Exception as exc:
            entry.update(ok=False, error=f"{type(exc).__name__}: {exc}",
                         traceback=traceback.format_exc(limit=3))
            failures.append(name)
        rows[name] = entry

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    prior = {}
    if os.path.isfile(a.out):
        try:
            prior = json.load(open(a.out, encoding="utf-8"))
        except Exception:
            prior = {}
    prior.update(rows)
    json.dump(prior, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    w = max(len(n) for n in names) + 2
    print(f"\n{'model':<{w}}{'status':<10}{'params (M)':>12}  {'pretrained':<12}source")
    print("-" * (w + 52))
    for n in names:
        r = rows[n]
        if r["ok"]:
            pre = "yes" if r.get("pretrained_loaded") else "NO (random)"
            print(f"{n:<{w}}{'OK':<10}{r.get('params_millions', 0):>12.3f}  "
                  f"{pre:<12}{r.get('source', '')[:40]}")
        else:
            print(f"{n:<{w}}{'FAILED':<10}{'--':>12}  {'--':<12}{r['error'][:60]}")
    print(f"\nwrote {a.out}")

    if failures:
        print("\nFailed: " + ", ".join(failures))
        print("Install instructions: models/external/README.md")
        return 1
    randoms = [n for n, r in rows.items()
               if r["ok"] and r.get("pretrained_requested") and not r.get("pretrained_loaded")]
    if randoms:
        print("\n[!] Randomly initialised despite pretrained=True: " + ", ".join(randoms))
        print("    These are NOT ImageNet-pretrained baselines. Report them as such.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
