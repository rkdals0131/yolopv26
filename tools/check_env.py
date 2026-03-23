from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _module_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    return getattr(module, "__version__", None)


def _check_torchvision_nms() -> dict[str, Any]:
    result = {
        "importable": False,
        "callable": False,
        "error": None,
        "fallback_available": True,
    }
    try:
        import torch
        from torchvision.ops import nms

        result["importable"] = True
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 9.0, 9.0]], dtype=torch.float32)
        scores = torch.tensor([0.9, 0.8], dtype=torch.float32)
        keep = nms(boxes, scores, 0.5)
        result["callable"] = bool(keep.numel() >= 1)
    except Exception as exc:
        result["error"] = str(exc)
    return result


def _check_yolo26(check_runtime: bool) -> dict[str, Any]:
    result = {
        "importable": False,
        "supported": False,
        "version": None,
        "runtime_load_ok": None,
        "error": None,
    }
    try:
        from model.trunk.ultralytics_yolo26 import (
            ULTRALYTICS_VERSION,
            build_yolo26n_trunk,
            ensure_yolo26_support,
        )

        result["importable"] = True
        result["version"] = ULTRALYTICS_VERSION
        ensure_yolo26_support()
        result["supported"] = True
        if check_runtime:
            adapter = build_yolo26n_trunk()
            result["runtime_load_ok"] = bool(adapter.detect_head is not None)
    except Exception as exc:
        result["error"] = str(exc)
        if result["runtime_load_ok"] is None:
            result["runtime_load_ok"] = False
    return result


def check_env(*, check_yolo_runtime: bool = False) -> dict[str, Any]:
    return {
        "repo_root": str(REPO_ROOT),
        "python": sys.version.split()[0],
        "versions": {
            "torch": _module_version("torch"),
            "torchvision": _module_version("torchvision"),
            "ultralytics": _module_version("ultralytics"),
            "numpy": _module_version("numpy"),
            "scipy": _module_version("scipy"),
            "PIL": _module_version("PIL"),
        },
        "checks": {
            "torchvision_nms": _check_torchvision_nms(),
            "yolo26": _check_yolo26(check_yolo_runtime),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check PV26 runtime environment portability prerequisites.")
    parser.add_argument("--check-yolo-runtime", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    report = check_env(check_yolo_runtime=args.check_yolo_runtime)
    print(json.dumps(report, indent=2, ensure_ascii=True))

    if not args.strict:
        return

    failures: list[str] = []
    if report["versions"]["torch"] is None:
        failures.append("torch missing")
    if report["checks"]["yolo26"]["importable"] is not True:
        failures.append("ultralytics missing")
    if args.check_yolo_runtime and report["checks"]["yolo26"]["runtime_load_ok"] is not True:
        failures.append("yolo26 runtime load failed")
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
