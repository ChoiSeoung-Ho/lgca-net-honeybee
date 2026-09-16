"""LGCA-Net (proposed hybrid CNN-Transformer with cross-attention) and the baseline registry.

LGCA-Net = **L**ocal-query **G**lobal-key **C**ross-**A**ttention Network: the
cross-attention fusion draws its *queries* from the local CNN token grid and its
*keys/values* from the global transformer-encoded tokens, which is what the name
records. (Earlier drafts of this work called the architecture HCF-Net; that name
survives only as a backwards-compatible alias.)

The proposed architecture is reproduced here exactly as trained for the
manuscript. Layer-by-layer parameter budget of the *full* model
(``use_transformer=True, use_cross_attn=True``):

===========================================  ==========
component                                    parameters
===========================================  ==========
Conv 7x7 3->64 s2 p3 (+bias)                      9,472
BatchNorm2d(64)                                     128
Conv 3x3 64->256 s2 p1 (+bias)                  147,712
BatchNorm2d(256)                                    512
Linear 256->256 (token projection)               65,792
positional embedding (1, 196, 256)               50,176
nn.TransformerEncoderLayer(256, 8, 1024)        789,760
CrossAttention(256, 8, qkv_bias=False)          262,400
LayerNorm(256) (fusion)                             512
Linear 256->512                                 131,584
Linear 512->2                                     1,026
-------------------------------------------  ----------
**total trainable**                          **1,459,074**
===========================================  ==========

i.e. the full model has exactly **1,459,074** trainable parameters. Run
``python verify_model.py --check-params`` to assert this.

Baselines are built through :data:`MODEL_REGISTRY`. Entries whose weights come
from an external repository (MambaVision, LSNet-T, Conformer, Mobile-Former)
raise a clear ``ImportError`` telling the user what to install; see
``models/external/README.md``.

Torch/timm imports are guarded so this module can be *imported* (for the
registry listing and docstrings) on a CPU-only machine without torch.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

try:  # pragma: no cover - only when torch is installed
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
    _Module = nn.Module
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _HAS_TORCH = False

    class _Module:  # type: ignore[no-redef]
        """Placeholder so class definitions below still parse without torch."""

        def __init__(self, *a, **k):
            raise ImportError(
                "PyTorch is required to instantiate models. "
                "Install torch>=2.9.0 (CUDA 12.8 build used in the paper)."
            )


#: Expected trainable parameter count of the full proposed model.
EXPECTED_FULL_PARAMS = 1_459_074

EMBED_DIM = 256
N_TOKENS = 196  # 14 x 14
N_HEADS = 8
FFN_DIM = 1024


# --------------------------------------------------------------------------- #
# Cross-attention
# --------------------------------------------------------------------------- #
class CrossAttention(_Module):
    """Multi-head cross-attention: queries from the CNN tokens, keys/values from the ViT tokens.

    ``scale = head_dim ** -0.5``; softmax over the key axis; output projection
    ``Linear(dim, dim)``. ``qkv_bias=False``, ``attn_drop=0``, ``proj_drop=0``
    as trained.
    """

    def __init__(self, dim: int = EMBED_DIM, heads: int = N_HEADS, qkv_bias: bool = False,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_q, x_kv, return_attn: bool = False):
        """``x_q``: (B, Nq, C) CNN tokens. ``x_kv``: (B, Nk, C) transformer tokens."""
        B, Nq, C = x_q.shape
        Nk = x_kv.shape[1]
        q = self.q(x_q).reshape(B, Nq, self.heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(x_kv).reshape(B, Nk, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x_kv).reshape(B, Nk, self.heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        out = self.proj_drop(self.proj(out))
        if return_attn:
            return out, attn
        return out


# --------------------------------------------------------------------------- #
# Proposed model
# --------------------------------------------------------------------------- #
class LGCANet(_Module):
    """Hybrid CNN-Transformer with cross-attention fusion (the proposed LGCA-Net).

    Pipeline (input 3 x 256 x 256, values in [0, 1]):

    1. **CNN stem** ``Conv7x7(3->64, s2, p3) + BN + GELU``,
       ``Conv3x3(64->256, s2, p1) + BN``, ``AdaptiveAvgPool2d((14, 14))``
       -> flatten to 196 tokens of width 256.
    2. **Transformer path** ``Linear(256->256)`` + learned positional embedding
       ``(1, 196, 256)`` -> one ``nn.TransformerEncoderLayer`` (d_model=256,
       nhead=8, dim_feedforward=1024, dropout=0.1, activation='relu',
       batch_first=True).
    3. **Fusion** cross-attention (Q = CNN tokens, K/V = transformer tokens),
       residual add with the CNN tokens, ``LayerNorm``.
    4. Mean-pool over the 196 tokens -> ``Linear(256->512) + GELU +
       Dropout(0.3) + Linear(512->2)``.

    Ablations
    ---------
    ``use_transformer=False``
        CNN-only: fused = ``LayerNorm(cnn_tokens)`` (transformer path and
        cross-attention are not built).
    ``use_cross_attn=False``
        Additive fusion: fused = ``LayerNorm(trans_tokens + cnn_tokens)``.
    """

    def __init__(
        self,
        num_classes: int = 2,
        embed_dim: int = EMBED_DIM,
        num_tokens: int = N_TOKENS,
        num_heads: int = N_HEADS,
        ffn_dim: int = FFN_DIM,
        dropout: float = 0.1,
        head_dropout: float = 0.3,
        use_transformer: bool = True,
        use_cross_attn: bool = True,
    ):
        super().__init__()
        self.use_transformer = bool(use_transformer)
        # cross-attention is meaningless without the transformer branch
        self.use_cross_attn = bool(use_cross_attn) and self.use_transformer
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        self.grid = int(round(num_tokens ** 0.5))

        # ---- CNN stem -----------------------------------------------------
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(64, embed_dim, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(embed_dim)  # Grad-CAM target layer
        self.pool = nn.AdaptiveAvgPool2d((self.grid, self.grid))

        # ---- Transformer path ---------------------------------------------
        if self.use_transformer:
            self.token_proj = nn.Linear(embed_dim, embed_dim)
            self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                activation="relu",
                batch_first=True,
            )
        # ---- Cross-attention fusion ---------------------------------------
        if self.use_cross_attn:
            self.cross_attn = CrossAttention(
                dim=embed_dim, heads=num_heads, qkv_bias=False,
                attn_drop=0.0, proj_drop=0.0,
            )
        self.fuse_norm = nn.LayerNorm(embed_dim)

        # ---- Classifier ----------------------------------------------------
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(512, num_classes),
        )

    # -- pieces ------------------------------------------------------------- #
    def forward_cnn_tokens(self, x):
        """Stem forward -> (B, 196, 256) tokens."""
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.pool(x)                       # (B, C, 14, 14)
        return x.flatten(2).transpose(1, 2)    # (B, 196, C)

    def forward_features(self, x):
        """Fused, LayerNorm-ed token sequence (B, 196, 256)."""
        cnn_tokens = self.forward_cnn_tokens(x)
        if not self.use_transformer:
            return self.fuse_norm(cnn_tokens)

        t = self.token_proj(cnn_tokens) + self.pos_embed
        t = self.encoder_layer(t)
        if self.use_cross_attn:
            attn_out = self.cross_attn(cnn_tokens, t)
            return self.fuse_norm(attn_out + cnn_tokens)
        return self.fuse_norm(t + cnn_tokens)

    def forward(self, x):
        feats = self.forward_features(x)
        pooled = feats.mean(dim=1)
        return self.head(pooled)


#: Backwards-compatible aliases. ``HCFNet`` is the name used in earlier drafts of
#: the manuscript; ``Hybrid_CNN_Transformer_AblationModel`` is the name used by the
#: original training scripts. Both refer to the same class as ``LGCANet``.
HCFNet = LGCANet
Hybrid_CNN_Transformer_AblationModel = LGCANet


def count_parameters(model, trainable_only: bool = True) -> int:
    """Number of (trainable) parameters in a module."""
    params = model.parameters()
    if trainable_only:
        return int(sum(p.numel() for p in params if p.requires_grad))
    return int(sum(p.numel() for p in params))


# --------------------------------------------------------------------------- #
# Baseline registry
# --------------------------------------------------------------------------- #
def _require_timm():
    try:
        import timm  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "timm is required for the ImageNet-pretrained baselines. "
            "pip install timm>=1.0.0"
        ) from exc
    import timm

    return timm


def _timm_factory(timm_name: str, input_size: Optional[int] = None):
    """Factory building a timm model, optionally wrapped to resize the input."""

    def _build(pretrained: bool = True, num_classes: int = 2):
        timm = _require_timm()
        model = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)
        if input_size is not None:
            return ResizeWrapper(model, input_size)
        return model

    _build.__doc__ = f"timm.create_model('{timm_name}', pretrained=..., num_classes=2)"
    return _build


class ResizeWrapper(_Module):
    """Bilinearly resize the 256x256 input to a backbone's required resolution.

    Used for CrossViT (``crossvit_small_240`` needs 240x240). Keeping the resize
    *inside* the model means the shared 256x256 dataloader is untouched.
    """

    def __init__(self, model, size: int):
        super().__init__()
        self.model = model
        self.size = int(size)

    def forward(self, x):
        if x.shape[-1] != self.size or x.shape[-2] != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        return self.model(x)


def _external_factory(display_name: str, hint: str):
    """Factory placeholder for a backbone that lives in an external repository."""

    def _build(pretrained: bool = True, num_classes: int = 2):
        raise ImportError(
            f"'{display_name}' is not bundled with this package.\n{hint}\n"
            "See models/external/README.md for the exact steps, then replace this "
            "registry entry with a call into the installed package."
        )

    _build.__doc__ = f"{display_name} (external): {hint}"
    return _build


def _mambavision_factory():
    """MambaVision-T.

    NOTE: requires NVIDIA's official ``mambavision`` package (``pip install
    mambavision``), which in turn needs ``mamba-ssm`` + ``causal-conv1d`` built
    against the local CUDA toolkit. It is *not* obtainable from timm's hub in
    the version used for the paper.
    """

    def _build(pretrained: bool = True, num_classes: int = 2):
        try:
            from mambavision.models.mamba_vision import mamba_vision_T  # type: ignore
        except Exception as exc:
            raise ImportError(
                "MambaVision-T requires the official 'mambavision' package "
                "(pip install mambavision; needs mamba-ssm and causal-conv1d built "
                "against your CUDA toolkit). See models/external/README.md."
            ) from exc
        return mamba_vision_T(pretrained=pretrained, num_classes=num_classes)

    _build.__doc__ = "MambaVision-T via the official NVIDIA mambavision package"
    return _build


def _proposed_factory(use_transformer: bool = True, use_cross_attn: bool = True):
    def _build(pretrained: bool = False, num_classes: int = 2):
        # `pretrained` is ignored: the proposed model is trained from scratch.
        return LGCANet(num_classes=num_classes, use_transformer=use_transformer,
                      use_cross_attn=use_cross_attn)

    return _build


#: Per-model default learning rate (the centre of the search grid actually used).
DEFAULT_LR = {
    "lgca_net": 1e-3,
    "lgca_net_no_cross_attn": 1e-3,
    "lgca_net_cnn_only": 1e-3,
    "proposed": 1e-3,
    "proposed_no_cross_attn": 1e-3,
    "proposed_cnn_only": 1e-3,
    "resnet50": 1e-4,
    "densenet121": 1e-4,
    "efficientnetv2": 1e-4,
    "nextvit_small": 1e-4,
    "mambavision": 1e-4,
    "lsnet_t": 1e-4,
    "coatnet": 1e-4,
    "crossvit": 1e-4,
    "conformer": 1e-4,
    "mobileformer": 1e-4,
}

#: Which models are trained from scratch (narrower LR grid, no ImageNet weights).
FROM_SCRATCH = {"lgca_net", "lgca_net_no_cross_attn", "lgca_net_cnn_only",
                "proposed", "proposed_no_cross_attn", "proposed_cnn_only"}

#: name -> factory(pretrained: bool, num_classes: int) -> nn.Module
MODEL_REGISTRY: Dict[str, Callable] = {
    # ---- proposed + ablations (from scratch) ------------------------------
    "lgca_net": _proposed_factory(True, True),
    "lgca_net_no_cross_attn": _proposed_factory(True, False),
    "lgca_net_cnn_only": _proposed_factory(False, False),
    # ---- timm, ImageNet-1K pretrained -------------------------------------
    "resnet50": _timm_factory("resnet50"),
    "densenet121": _timm_factory("densenet121"),
    "efficientnetv2": _timm_factory("tf_efficientnetv2_s"),
    "nextvit_small": _timm_factory("nextvit_small"),
    "coatnet": _timm_factory("coatnet_0_rw_224", input_size=224),
    # CrossViT needs 240x240 -> resize inside the wrapper
    "crossvit": _timm_factory("crossvit_small_240", input_size=240),
    # ---- external repositories --------------------------------------------
    "mambavision": _mambavision_factory(),
    "lsnet_t": _external_factory(
        "LSNet-T",
        "Install the official LSNet repository (jameslahm/lsnet) and import "
        "`lsnet_t` from it.",
    ),
    "conformer": _external_factory(
        "Conformer",
        "Clone https://github.com/pengzhiliang/Conformer and add it to PYTHONPATH; "
        "use `Conformer_small_patch16` with num_classes=2.",
    ),
    "mobileformer": _external_factory(
        "Mobile-Former",
        "Use an official/reference Mobile-Former implementation (e.g. "
        "https://github.com/AAboys/MobileFormer) and build the 294M variant "
        "with num_classes=2.",
    ),
}

#: Backwards-compatible registry aliases. The canonical keys are the ``lgca_net*``
#: ones; scripts written against the earlier naming keep working. Note the key is
#: what appears in output filenames (``oof_<task>_<model>.csv``), so pick one
#: spelling per experiment rather than mixing ``proposed`` and ``lgca_net`` in the
#: same results directory.
MODEL_ALIASES = {
    "proposed": "lgca_net",
    "proposed_no_cross_attn": "lgca_net_no_cross_attn",
    "proposed_cnn_only": "lgca_net_cnn_only",
    "hcfnet": "lgca_net",          # name used in earlier drafts of the manuscript
    "hcf_net": "lgca_net",
}
for _alias, _target in MODEL_ALIASES.items():
    MODEL_REGISTRY.setdefault(_alias, MODEL_REGISTRY[_target])


def canonical_name(name: str) -> str:
    """Resolve a registry alias to its canonical key."""
    return MODEL_ALIASES.get(name, name)

#: Pretty names as they appear in the manuscript tables.
DISPLAY_NAME = {
    "lgca_net": "Proposed (LGCA-Net)",
    "lgca_net_no_cross_attn": "LGCA-Net (w/o cross-attn.)",
    "lgca_net_cnn_only": "LGCA-Net (CNN only)",
    "proposed": "Proposed (LGCA-Net)",
    "proposed_no_cross_attn": "LGCA-Net (w/o cross-attn.)",
    "proposed_cnn_only": "LGCA-Net (CNN only)",
    "resnet50": "ResNet-50",
    "densenet121": "DenseNet-121",
    "efficientnetv2": "EfficientNetV2-S",
    "nextvit_small": "Next-ViT-S",
    "mambavision": "MambaVision",
    "lsnet_t": "LSNet-T",
    "coatnet": "CoAtNet",
    "crossvit": "CrossViT",
    "conformer": "Conformer",
    "mobileformer": "Mobile-Former",
}


def display_name(name: str) -> str:
    """Manuscript-facing name for a registry key, resolving aliases first."""
    return DISPLAY_NAME.get(name) or DISPLAY_NAME.get(canonical_name(name), name)


def build_model(name: str, pretrained: bool = True, num_classes: int = 2):
    """Instantiate a registered model by name."""
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Available: {', '.join(sorted(MODEL_REGISTRY))}"
        )
    return MODEL_REGISTRY[name](pretrained=pretrained, num_classes=num_classes)


def default_lr(name: str) -> float:
    """Default learning rate for a model name."""
    return DEFAULT_LR.get(name, 1e-4)


def lr_grid_for(name: str, pretrained_grid=(1e-3, 3e-4, 1e-4, 3e-5),
                scratch_grid=(1e-3, 3e-4)):
    """Learning-rate grid: narrower for from-scratch models."""
    return tuple(scratch_grid) if name in FROM_SCRATCH else tuple(pretrained_grid)


# >>> honeybee external baselines (auto-added) >>>
# Replaces the four placeholder entries with real factories from
# models/external/adapters.py.  Kept at the very end of the file so that the
# registry above is fully built before it is amended.  Remove with
# `python patch_models_registry.py --revert`.
try:  # pragma: no cover - depends on the external repositories being installed
    import os as _os
    import sys as _sys
    _here = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _here not in _sys.path:
        _sys.path.insert(0, _here)
    from models.external.adapters import (
        EXTERNAL_FACTORIES as _EXT_FACTORIES,
        EXTERNAL_DISPLAY_NAMES as _EXT_NAMES,
    )
    MODEL_REGISTRY.update(_EXT_FACTORIES)
    DISPLAY_NAME.update(_EXT_NAMES)
except Exception as _exc:  # the placeholders stay in place and still raise clearly
    import warnings as _warnings
    _warnings.warn(f"external baseline adapters not loaded: {_exc}")
# <<< honeybee external baselines (auto-added) <<<
