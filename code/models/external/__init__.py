# -*- coding: utf-8 -*-
"""External baseline backbones (Conformer, Mobile-Former, LSNet-T, MambaVision).

Importing this package does nothing on its own.  Call :func:`register` (or run
``patch_models_registry.py`` once) to install the four factories into
``common.models.MODEL_REGISTRY``.
"""
from .adapters import (  # noqa: F401
    ExternalModelUnavailable,
    PretrainedWeightsUnavailable,
    EXTERNAL_FACTORIES,
    EXTERNAL_DISPLAY_NAMES,
    Provenance,
    last_provenance,
    register,
    build_conformer,
    build_mobileformer,
    build_lsnet,
    build_mambavision,
)

__all__ = [
    "ExternalModelUnavailable", "PretrainedWeightsUnavailable",
    "EXTERNAL_FACTORIES", "EXTERNAL_DISPLAY_NAMES", "Provenance",
    "last_provenance", "register",
    "build_conformer", "build_mobileformer", "build_lsnet", "build_mambavision",
]
