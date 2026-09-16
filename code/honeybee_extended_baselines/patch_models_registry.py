#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wire ``models/external`` into ``common/models.py``.

Appends a short, clearly delimited block to the end of ``common/models.py`` that
imports the external factories and overwrites the four placeholder registry
entries.  Idempotent: running it twice changes nothing.  ``--revert`` removes the
block again.

    python patch_models_registry.py --code-root /path/to/code
    python patch_models_registry.py --code-root /path/to/code --revert
"""
import argparse
import os
import shutil
import sys

BEGIN = "# >>> honeybee external baselines (auto-added) >>>"
END = "# <<< honeybee external baselines (auto-added) <<<"

BLOCK = f'''

{BEGIN}
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
    _warnings.warn(f"external baseline adapters not loaded: {{_exc}}")
{END}
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-root", default=".",
                    help="directory containing common/models.py")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()

    target = os.path.join(a.code_root, "common", "models.py")
    if not os.path.isfile(target):
        print(f"[error] not found: {target}", file=sys.stderr)
        return 2
    src = open(target, encoding="utf-8").read()

    if a.revert:
        if BEGIN not in src:
            print("[ok] nothing to revert")
            return 0
        head, rest = src.split(BEGIN, 1)
        _, tail = rest.split(END, 1)
        open(target, "w", encoding="utf-8").write(head.rstrip() + "\n" + tail.lstrip("\n"))
        print(f"[ok] reverted {target}")
        return 0

    if BEGIN in src:
        print("[ok] already patched")
        return 0
    shutil.copy2(target, target + ".orig")
    open(target, "a", encoding="utf-8").write(BLOCK)
    print(f"[ok] patched {target} (backup at {target}.orig)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
