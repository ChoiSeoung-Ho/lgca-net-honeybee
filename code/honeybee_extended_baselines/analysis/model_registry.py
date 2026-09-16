# -*- coding: utf-8 -*-
"""Model discovery for the analysis layer.

The analysis scripts must not carry a hard-coded model list: adding a baseline
should change every table and figure without editing any of them.  Everything
here is derived from what is actually on disk.

    from model_registry import discover_models, display_name, order_models
    models = order_models(discover_models("./results", "you_chalk_brood", "original"))
"""
from __future__ import annotations

import glob
import json
import os
import re
from typing import Dict, List, Optional

# Manuscript-facing names.  A key that is absent falls back to a readable form
# derived from the key itself, so an unknown model still appears in the tables.
DISPLAY: Dict[str, str] = {
    "lgca_net": "LGCA-Net (proposed)",
    "lgca_net_no_cross_attn": "LGCA-Net w/o cross-attention",
    "lgca_net_cnn_only": "LGCA-Net CNN branch only",
    "resnet50": "ResNet-50",
    "densenet121": "DenseNet-121",
    "efficientnetv2": "EfficientNetV2-S",
    "coatnet": "CoAtNet-0",
    "crossvit": "CrossViT-S",
    "nextvit_small": "Next-ViT-S",
    "conformer": "Conformer-S",
    "mobileformer": "Mobile-Former-294M",
    "lsnet_t": "LSNet-T",
    "mambavision": "MambaVision-T",
}

SHORT: Dict[str, str] = {
    "lgca_net": "LGCA-Net\n(proposed)",
    "lgca_net_no_cross_attn": "LGCA-Net\nw/o cross-att.",
    "lgca_net_cnn_only": "LGCA-Net\nCNN only",
    "efficientnetv2": "EfficientNetV2-S",
    "mobileformer": "Mobile-Former",
}

#: Presentation order: proposed model, its ablations, then baselines.
ORDER_HINT: List[str] = [
    "lgca_net", "lgca_net_no_cross_attn", "lgca_net_cnn_only",
    "resnet50", "densenet121", "efficientnetv2",
    "coatnet", "crossvit", "nextvit_small",
    "conformer", "mobileformer", "lsnet_t", "mambavision",
]

FROM_SCRATCH = {"lgca_net", "lgca_net_no_cross_attn", "lgca_net_cnn_only"}

#: Architectures reported as hybrid / attention-based rather than plain CNNs.
HYBRID = {"coatnet", "crossvit", "nextvit_small", "conformer", "mobileformer",
          "lsnet_t", "mambavision", "lgca_net", "lgca_net_no_cross_attn"}

ABLATION_ORDER = ["lgca_net_cnn_only", "lgca_net_no_cross_attn", "lgca_net"]
REFERENCE_MODEL = "lgca_net"


# --------------------------------------------------------------------------- #
def discover_models(results_dir: str, task: str, run_tag: str = "original") -> List[str]:
    """Model keys that have a per-image out-of-fold prediction file for a task."""
    suffix = f"_{run_tag}" if run_tag else ""
    pat = os.path.join(results_dir, f"oof_{task}_*{suffix}.csv")
    out = []
    for p in glob.glob(pat):
        stem = os.path.basename(p)[:-4]
        name = stem[len(f"oof_{task}_"):]
        if suffix and name.endswith(suffix):
            name = name[: -len(suffix)]
        if name.startswith("trivial") or not name:
            continue
        out.append(name)
    return sorted(set(out))


def common_models(results_dir: str, tasks, run_tag: str = "original") -> List[str]:
    """Models present for *every* task, so tables are rectangular."""
    sets = [set(discover_models(results_dir, t, run_tag)) for t in tasks]
    return order_models(set.intersection(*sets)) if sets else []


def order_models(models) -> List[str]:
    models = list(models)
    rank = {m: i for i, m in enumerate(ORDER_HINT)}
    return sorted(models, key=lambda m: (rank.get(m, 10_000 + len(m)), m))


def display_name(key: str, latex: bool = False, bold_reference: bool = True) -> str:
    name = DISPLAY.get(key)
    if name is None:                       # readable fallback for an unknown key
        name = re.sub(r"[_\-]+", " ", key).strip()
        name = name[:1].upper() + name[1:]
    if latex and bold_reference and key == REFERENCE_MODEL:
        return r"\textbf{" + name + "}"
    return name


def short_name(key: str) -> str:
    return SHORT.get(key, display_name(key))


# --------------------------------------------------------------------------- #
def load_provenance(path: str = "./results/model_provenance.json") -> Dict[str, Dict]:
    if os.path.isfile(path):
        try:
            return json.load(open(path, encoding="utf-8"))
        except Exception:
            return {}
    return {}


def params_millions(key: str, provenance: Optional[Dict] = None) -> Optional[float]:
    """Measured parameter count in millions, or None if it was never measured.

    Never guessed: a missing entry prints as ``--`` so the table cannot quote a
    number that no run produced.  Populate it with verify_external_models.py.
    """
    if provenance and key in provenance and provenance[key].get("ok"):
        n = provenance[key].get("n_params_total")
        if n:
            return round(n / 1e6, 2)
    return None


def pretraining_label(key: str, provenance: Optional[Dict] = None) -> str:
    if key in FROM_SCRATCH:
        return "--"
    if provenance and key in provenance:
        p = provenance[key]
        if p.get("pretrained_requested") and not p.get("pretrained_loaded"):
            return "none (random)"
        if p.get("pretrained_loaded"):
            return "ImageNet-1K"
    return "ImageNet-1K"
