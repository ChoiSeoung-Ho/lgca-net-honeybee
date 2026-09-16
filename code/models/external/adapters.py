# -*- coding: utf-8 -*-
"""Adapters for the four baseline backbones that are not available from ``timm``.

Registry keys provided here: ``conformer``, ``mobileformer``, ``lsnet_t``,
``mambavision``.

Design rules, all of which exist to keep the comparison honest:

*   **Never silently substitute.** Every loader tries a list of documented
    sources in order.  If none succeeds it raises ``ExternalModelUnavailable``
    with the exact install commands for that architecture.  A run either uses
    the intended architecture or fails loudly.
*   **Never silently drop pretraining.** ``pretrained=True`` that cannot be
    honoured raises, unless the caller explicitly passes
    ``allow_random_init=True``.  The outcome is recorded in the returned
    provenance so that the manuscript table can state, per model, whether
    ImageNet weights were actually loaded.  This matters for Mobile-Former,
    which has no official public ImageNet checkpoint.
*   **Honour the pipeline contract.** Each factory returns an ``nn.Module`` that
    takes ``(B, 3, 256, 256)`` float32 in ``[0, 1]`` and returns ``(B, 2)``
    logits.  Backbones with a fixed input resolution are wrapped in
    ``ResizeWrapper``; backbones with multiple classifier heads (Conformer) are
    wrapped in ``MultiHeadSum``.
*   **Record what was loaded.** ``last_provenance()`` returns a dict describing
    source, checkpoint, parameter count and whether pretrained weights were
    applied, so that nothing in the manuscript rests on an assumption about
    which file was on disk.

The normalisation question is deliberate: this pipeline feeds ``[0, 1]`` tensors
with no ImageNet mean/std to *every* model, including the six timm baselines
already reported.  The new backbones are fed the same way.  Applying ImageNet
normalisation to only the new models would make them non-comparable with the
published rows; if you want normalisation, enable it for all models at once in
the dataloader and re-run everything.
"""

from __future__ import annotations

import importlib
import os
import sys
import hashlib
from dataclasses import dataclass, asdict, field
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# errors and provenance
# --------------------------------------------------------------------------- #
class ExternalModelUnavailable(ImportError):
    """Raised when none of the documented sources for a backbone is installed."""


class PretrainedWeightsUnavailable(RuntimeError):
    """Raised when ``pretrained=True`` cannot be honoured for a backbone."""


@dataclass
class Provenance:
    key: str
    display_name: str
    source: str = ""
    detail: str = ""
    checkpoint: str = ""
    checkpoint_sha256: str = ""
    pretrained_requested: bool = True
    pretrained_loaded: bool = False
    input_size: int = 256
    n_params_total: int = 0
    n_params_trainable: int = 0
    forward_ok: bool = False
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict:
        return asdict(self)


_LAST: Dict[str, Provenance] = {}


def last_provenance(key: Optional[str] = None):
    """Provenance of the most recent build (one key, or all)."""
    if key is None:
        return {k: v.as_dict() for k, v in _LAST.items()}
    return _LAST[key].as_dict() if key in _LAST else None


# --------------------------------------------------------------------------- #
# wrappers
# --------------------------------------------------------------------------- #
class ResizeWrapper(nn.Module):
    """Bilinearly resize the 256x256 pipeline input to a backbone's resolution."""

    def __init__(self, model: nn.Module, size: int):
        super().__init__()
        self.model = model
        self.size = int(size)

    def forward(self, x):
        if x.shape[-1] != self.size or x.shape[-2] != self.size:
            x = F.interpolate(x, size=(self.size, self.size),
                              mode="bilinear", align_corners=False)
        return self.model(x)


class MultiHeadSum(nn.Module):
    """Reduce a multi-head backbone to a single logit tensor.

    Conformer returns ``[conv_logits, trans_logits]``.  The official evaluation
    code sums them, so that is what is done here; using only one head would be a
    different model from the published one.
    """

    def __init__(self, model: nn.Module, mode: str = "sum"):
        super().__init__()
        self.model = model
        self.mode = mode

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (list, tuple)):
            outs = [o for o in out if torch.is_tensor(o) and o.dim() == 2]
            if not outs:
                raise RuntimeError("multi-head backbone returned no 2-D logits")
            if self.mode == "sum":
                return torch.stack(outs, 0).sum(0)
            if self.mode == "mean":
                return torch.stack(outs, 0).mean(0)
            return outs[0]
        return out


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _sha256(path: str, limit: int = 1 << 24) -> str:
    h = hashlib.sha256()
    read = 0
    with open(path, "rb") as fh:
        while read < limit:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            read += len(chunk)
    return h.hexdigest()[:16]


def _strip_classifier(state: Dict[str, torch.Tensor],
                      prefixes: Sequence[str]) -> Dict[str, torch.Tensor]:
    """Drop 1000-class classifier tensors before loading into a 2-class model."""
    return {k: v for k, v in state.items()
            if not any(k.startswith(p) for p in prefixes)}


def _unwrap_state(obj):
    for key in ("model", "state_dict", "model_state_dict", "module"):
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
            obj = obj[key]
    if isinstance(obj, dict):
        # strip a DataParallel prefix if present
        if all(k.startswith("module.") for k in obj):
            obj = {k[len("module."):]: v for k, v in obj.items()}
    return obj


def _find_checkpoint(candidates: Sequence[str]) -> Optional[str]:
    roots = [os.environ.get("EXTERNAL_CKPT_DIR", ""), ".", "./checkpoints",
             "./models/external/checkpoints", os.path.expanduser("~/.cache/honeybee")]
    for root in roots:
        if not root:
            continue
        for name in candidates:
            p = os.path.join(root, name)
            if os.path.isfile(p):
                return p
    for name in candidates:                       # absolute paths given directly
        if os.path.isfile(name):
            return name
    return None


def _load_into(model: nn.Module, state: Dict, drop_prefixes: Sequence[str],
               prov: Provenance) -> None:
    state = _unwrap_state(state)
    state = _strip_classifier(state, drop_prefixes)
    missing, unexpected = model.load_state_dict(state, strict=False)
    kept = len(state)
    prov.notes.append(
        f"loaded {kept} tensors; {len(missing)} missing, {len(unexpected)} unexpected"
    )
    # Everything missing must be a classifier parameter; anything else means the
    # checkpoint does not match the architecture and the run must not proceed.
    unexplained = [m for m in missing
                   if not any(m.startswith(p) for p in drop_prefixes)]
    if unexplained:
        raise PretrainedWeightsUnavailable(
            "checkpoint does not match the architecture; parameters missing "
            f"outside the classifier: {unexplained[:8]}"
            f"{' ...' if len(unexplained) > 8 else ''}"
        )
    prov.pretrained_loaded = True


def _finalise(model: nn.Module, prov: Provenance, input_size: int,
              num_classes: int, multihead: bool = False) -> nn.Module:
    """Wrap, count parameters, run one forward pass, store provenance."""
    m: nn.Module = model
    if multihead:
        m = MultiHeadSum(m, mode="sum")
    if input_size != 256:
        m = ResizeWrapper(m, input_size)
    prov.input_size = input_size
    prov.n_params_total = int(sum(p.numel() for p in m.parameters()))
    prov.n_params_trainable = int(sum(p.numel() for p in m.parameters()
                                      if p.requires_grad))
    m.eval()
    try:
        with torch.no_grad():
            out = m(torch.zeros(2, 3, 256, 256))
    except Exception as _cpu_exc:
        if not torch.cuda.is_available():
            raise
        m = m.cuda()
        m.eval()   # LeViT식 self.ab 캐시를 CUDA에서 재생성 (.cuda()는 일반 속성을 안 옮김)
        with torch.no_grad():
            out = m(torch.zeros(2, 3, 256, 256, device="cuda"))
        out = _logits_from(out)
        if torch.is_tensor(out):
            out = out.detach().cpu()
        m = m.cpu()
        m.eval()
        prov.notes.append(
            f"CPU forward failed ({type(_cpu_exc).__name__}); verified on CUDA")
    if not (torch.is_tensor(out) and out.shape == (2, num_classes)):
        raise RuntimeError(
            f"{prov.key}: forward pass returned {type(out)} / "
            f"{getattr(out, 'shape', None)}, expected a tensor of shape (2, {num_classes})"
        )
    prov.forward_ok = True
    _LAST[prov.key] = prov
    return m


def _guard_pretrained(prov: Provenance, allow_random_init: bool, hint: str) -> None:
    if prov.pretrained_requested and not prov.pretrained_loaded:
        if not allow_random_init:
            raise PretrainedWeightsUnavailable(
                f"{prov.display_name}: ImageNet weights were requested but could "
                f"not be obtained.\n{hint}\n"
                "Pass allow_random_init=True (or --allow-random-init) to train this "
                "backbone from scratch instead -- but then it is NOT an "
                "ImageNet-pretrained baseline and the manuscript table must say so."
            )
        prov.notes.append("RANDOM INITIALISATION -- not an ImageNet-pretrained baseline")


def _try_import(module: str):
    try:
        return importlib.import_module(module)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Conformer  (Peng et al., ICCV 2021)
# --------------------------------------------------------------------------- #
CONFORMER_HINT = """\
  git clone https://github.com/pengzhiliang/Conformer
  export PYTHONPATH=$PWD/Conformer:$PYTHONPATH
  # download Conformer_small_patch16.pth from that repository's README and put it
  # in ./checkpoints/ (or set EXTERNAL_CKPT_DIR)."""


def build_conformer(pretrained: bool = True, num_classes: int = 2,
                    allow_random_init: bool = False, variant: str = "small") -> nn.Module:
    prov = Provenance(key="conformer", display_name="Conformer-S",
                      pretrained_requested=bool(pretrained))
    mod = _try_import("conformer") or _try_import("Conformer.conformer") or _try_import("models.conformer")
    if mod is None or not hasattr(mod, "Conformer"):
        raise ExternalModelUnavailable(
            "Conformer is not importable.\n" + CONFORMER_HINT)
    prov.source = f"repo:{getattr(mod, '__name__', 'conformer')}"

    cfg = {"tiny":  dict(patch_size=16, channel_ratio=1, embed_dim=384, depth=12,
                         num_heads=6, mlp_ratio=4, qkv_bias=True),
           "small": dict(patch_size=16, channel_ratio=4, embed_dim=384, depth=12,
                         num_heads=6, mlp_ratio=4, qkv_bias=True),
           "base":  dict(patch_size=16, channel_ratio=6, embed_dim=576, depth=12,
                         num_heads=9, mlp_ratio=4, qkv_bias=True)}[variant]
    prov.detail = f"Conformer-{variant} " + ", ".join(f"{k}={v}" for k, v in cfg.items())
    model = mod.Conformer(num_classes=num_classes, **cfg)

    if pretrained:
        ckpt = _find_checkpoint([f"Conformer_{variant}_patch16.pth",
                                 f"conformer_{variant}_patch16.pth"])
        if ckpt:
            prov.checkpoint = os.path.abspath(ckpt)
            prov.checkpoint_sha256 = _sha256(ckpt)
            state = torch.load(ckpt, map_location="cpu")
            _load_into(model, state, ("conv_cls_head", "trans_cls_head"), prov)
    _guard_pretrained(prov, allow_random_init, CONFORMER_HINT)
    # Conformer returns [conv_logits, trans_logits]; the official evaluation sums them.
    prov.notes.append("dual classifier head reduced by summation (official protocol)")
    return _finalise(model, prov, input_size=224, num_classes=num_classes, multihead=True)


# --------------------------------------------------------------------------- #
# Mobile-Former  (Chen et al., CVPR 2022)
# --------------------------------------------------------------------------- #
MOBILEFORMER_HINT = """\
  Microsoft has not released official code or ImageNet weights for Mobile-Former.
  Install a reference implementation, e.g.
     git clone https://github.com/slwang9353/MobileFormer
     export PYTHONPATH=$PWD/MobileFormer:$PYTHONPATH
  If that implementation ships no ImageNet checkpoint, this backbone cannot be
  reported as an ImageNet-pretrained baseline."""


def build_mobileformer(pretrained: bool = True, num_classes: int = 2,
                       allow_random_init: bool = False,
                       variant: str = "294m") -> nn.Module:
    prov = Provenance(key="mobileformer", display_name="Mobile-Former-294M",
                      pretrained_requested=bool(pretrained))
    model = None
    for mod_name, fns in [
        ("mobile_former", (f"mobile_former_{variant}", "mobile_former_294m",
                           "MobileFormer_294M", "mobile_former")),
        ("MobileFormer",  (f"mobile_former_{variant}", "mobile_former_294m",
                           "MobileFormer_294M", "MobileFormer")),
        ("models.mobile_former", (f"mobile_former_{variant}", "mobile_former_294m")),
    ]:
        mod = _try_import(mod_name)
        if mod is None:
            continue
        for fn in fns:
            ctor = getattr(mod, fn, None)
            if ctor is None:
                continue
            try:
                model = ctor(num_classes=num_classes)
            except TypeError:
                try:
                    model = ctor()
                except Exception:
                    continue
            prov.source = f"repo:{mod_name}.{fn}"
            break
        if model is not None:
            break
    if model is None:
        raise ExternalModelUnavailable(
            "Mobile-Former is not importable.\n" + MOBILEFORMER_HINT)
    prov.detail = f"Mobile-Former {variant} (reference implementation)"

    if pretrained:
        ckpt = _find_checkpoint([f"mobile_former_{variant}.pth",
                                 f"mobileformer_{variant}.pth",
                                 "mobile_former_294m.pth.tar"])
        if ckpt:
            prov.checkpoint = os.path.abspath(ckpt)
            prov.checkpoint_sha256 = _sha256(ckpt)
            state = torch.load(ckpt, map_location="cpu")
            _load_into(model, state, ("classifier", "head", "fc"), prov)
        else:
            prov.notes.append("no ImageNet checkpoint found for Mobile-Former")
    _guard_pretrained(prov, allow_random_init, MOBILEFORMER_HINT)
    return _finalise(model, prov, input_size=224, num_classes=num_classes)


# --------------------------------------------------------------------------- #
# LSNet-T  (Wang et al., CVPR 2025)
# --------------------------------------------------------------------------- #
LSNET_HINT = """\
  git clone https://github.com/jameslahm/lsnet
  export PYTHONPATH=$PWD/lsnet:$PYTHONPATH
  # importing the repo registers lsnet_t/s/b with timm; weights come from the
  # HuggingFace hub (jameslahm/lsnet_t) or the repo's release assets."""


def build_lsnet(pretrained: bool = True, num_classes: int = 2,
                allow_random_init: bool = False, variant: str = "lsnet_t") -> nn.Module:
    prov = Provenance(key="lsnet_t", display_name="LSNet-T",
                      pretrained_requested=bool(pretrained))
    model = None
    # (a) the repo registers its models with timm on import
    for repo_mod in ("model.lsnet", "lsnet", "model", "models.lsnet", "lsnet.model"):
        if _try_import(repo_mod) is not None:
            try:
                import timm
                model = timm.create_model(variant, pretrained=bool(pretrained),
                                          num_classes=num_classes)
                prov.source = f"timm-registered by repo module '{repo_mod}'"
                prov.pretrained_loaded = bool(pretrained)
                break
            except Exception:
                model = None
    # (b) direct constructor from the repo
    if model is None:
        for repo_mod in ("lsnet", "model", "models.lsnet"):
            mod = _try_import(repo_mod)
            ctor = getattr(mod, variant, None) if mod else None
            if ctor is None:
                continue
            try:
                model = ctor(num_classes=num_classes, pretrained=bool(pretrained))
                prov.source = f"repo:{repo_mod}.{variant}"
                prov.pretrained_loaded = bool(pretrained)
                break
            except Exception:
                try:
                    model = ctor(num_classes=num_classes)
                    prov.source = f"repo:{repo_mod}.{variant} (weights loaded separately)"
                    break
                except Exception:
                    continue
    if model is None:
        raise ExternalModelUnavailable("LSNet is not importable.\n" + LSNET_HINT)
    prov.detail = variant

    if pretrained and not prov.pretrained_loaded:
        ckpt = _find_checkpoint([f"{variant}.pth", f"{variant}.pth.tar", f"{variant}.bin"])
        if ckpt:
            prov.checkpoint = os.path.abspath(ckpt)
            prov.checkpoint_sha256 = _sha256(ckpt)
            _load_into(model, torch.load(ckpt, map_location="cpu"),
                       ("head", "classifier", "dist_head"), prov)
    _guard_pretrained(prov, allow_random_init, LSNET_HINT)
    return _finalise(model, prov, input_size=224, num_classes=num_classes)


# --------------------------------------------------------------------------- #
# MambaVision-T  (Hatamizadeh & Kautz, CVPR 2025)
# --------------------------------------------------------------------------- #
MAMBAVISION_HINT = """\
  pip install mambavision            # pulls mamba-ssm + causal-conv1d
  # these compile CUDA kernels against the local toolkit: a matching nvcc must be
  # on PATH.  Alternative, no custom kernels to build:
  pip install "transformers>=4.40"
  #   -> nvidia/MambaVision-T-1K is then loaded with trust_remote_code=True."""


def _logits_from(out):
    """HF/커스텀 모델의 다양한 반환형에서 (B, C) 로짓만 뽑아낸다."""
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, dict):
        for _k in ("logits", "out", "output", "prediction", "pred"):
            if _k in out and torch.is_tensor(out[_k]) and out[_k].dim() == 2:
                return out[_k]
        _c = [v for v in out.values() if torch.is_tensor(v) and v.dim() == 2]
        if _c:
            return _c[-1]
        raise RuntimeError(f"dict 반환에서 2-D 로짓을 못 찾음: keys={list(out)}")
    if isinstance(out, (tuple, list)):
        return _logits_from(out[0])
    return out


class _HFClassifierAdapter(nn.Module):
    """Expose a HuggingFace image-classification model as plain logits."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        try:
            out = self.model(pixel_values=x)
        except TypeError:
            out = self.model(x)
        return _logits_from(out)


def build_mambavision(pretrained: bool = True, num_classes: int = 2,
                      allow_random_init: bool = False,
                      variant: str = "T") -> nn.Module:
    prov = Provenance(key="mambavision", display_name=f"MambaVision-{variant}",
                      pretrained_requested=bool(pretrained))
    model = None
    # (a) official NVIDIA package
    mod = _try_import("mambavision.models.mamba_vision")
    ctor = getattr(mod, f"mamba_vision_{variant}", None) if mod else None
    if ctor is not None:
        try:
            model = ctor(pretrained=bool(pretrained), num_classes=num_classes)
            prov.source = f"mambavision.models.mamba_vision.mamba_vision_{variant}"
            prov.pretrained_loaded = bool(pretrained)
        except Exception as exc:
            prov.notes.append(f"official package present but failed: {exc}")
            model = None
    # (b) HuggingFace hub
    if model is None:
        tf = _try_import("transformers")
        if tf is not None and pretrained:
            try:
                hf = tf.AutoModelForImageClassification.from_pretrained(
                    f"nvidia/MambaVision-{variant}-1K", trust_remote_code=True)
                _target = None
                for _nm, _mod in hf.named_modules():
                    if isinstance(_mod, nn.Linear) and (
                            _nm.split(".")[-1] in ("head", "classifier", "fc")
                            or _mod.out_features == 1000):
                        _target = (_nm, _mod)
                if _target is None:
                    _lin = [(a, b) for a, b in hf.named_modules()
                            if isinstance(b, nn.Linear)]
                    _target = _lin[-1] if _lin else None
                if _target is not None and _target[1].out_features != num_classes:
                    _nm, _mod = _target
                    _parent, _parts = hf, _nm.split(".")
                    for _pp in _parts[:-1]:
                        _parent = getattr(_parent, _pp)
                    setattr(_parent, _parts[-1],
                            nn.Linear(_mod.in_features, num_classes))
                    prov.notes.append(
                        f"head '{_nm}': {_mod.out_features} -> {num_classes}")
                model = _HFClassifierAdapter(hf)
                prov.source = f"huggingface:nvidia/MambaVision-{variant}-1K"
                prov.pretrained_loaded = True
                prov.notes.append("classifier head replaced with a 2-class linear layer")
            except Exception as exc:
                prov.notes.append(f"huggingface path failed: {exc}")
                model = None
    if model is None:
        raise ExternalModelUnavailable(
            "MambaVision is not importable.\n" + MAMBAVISION_HINT)
    prov.detail = f"MambaVision-{variant}"
    _guard_pretrained(prov, allow_random_init, MAMBAVISION_HINT)
    return _finalise(model, prov, input_size=224, num_classes=num_classes)


# --------------------------------------------------------------------------- #
# registry integration
# --------------------------------------------------------------------------- #
def _factory(fn: Callable, **fixed):
    def _build(pretrained: bool = True, num_classes: int = 2, **kw):
        allow = bool(kw.pop("allow_random_init",
                            os.environ.get("ALLOW_RANDOM_INIT", "0") == "1"))
        return fn(pretrained=pretrained, num_classes=num_classes,
                  allow_random_init=allow, **fixed, **kw)
    _build.__doc__ = fn.__doc__
    return _build


EXTERNAL_FACTORIES: Dict[str, Callable] = {
    "conformer":    _factory(build_conformer,   variant="small"),
    "mobileformer": _factory(build_mobileformer, variant="294m"),
    "lsnet_t":      _factory(build_lsnet,       variant="lsnet_t"),
    "mambavision":  _factory(build_mambavision, variant="T"),
}

EXTERNAL_DISPLAY_NAMES: Dict[str, str] = {
    "conformer": "Conformer-S",
    "mobileformer": "Mobile-Former-294M",
    "lsnet_t": "LSNet-T",
    "mambavision": "MambaVision-T",
}


def register(registry: Optional[Dict[str, Callable]] = None,
             display: Optional[Dict[str, str]] = None) -> Dict[str, Callable]:
    """Install the four external factories into ``common.models.MODEL_REGISTRY``.

    Import-safe: call it as often as you like.  Returns the registry it wrote to.
    """
    if registry is None:
        from common import models as _m
        registry = _m.MODEL_REGISTRY
        display = _m.DISPLAY_NAME
    registry.update(EXTERNAL_FACTORIES)
    if display is not None:
        display.update(EXTERNAL_DISPLAY_NAMES)
    return registry
