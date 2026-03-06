#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import is_dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

# Allow "python tools/debug/compare_weight_io.py" from repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pv26.constants import DET_CLASSES_CANONICAL
from pv26.multitask_model import PV26MultiHeadYOLO26


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run best.pt and yolopv2.pt on one image and compare I/O + feature format."
    )
    parser.add_argument("--best", type=Path, default=Path("datasets/weights/best.pt"))
    parser.add_argument("--yolopv2", type=Path, default=Path("datasets/weights/yolopv2.pt"))
    parser.add_argument("--image", type=Path, default=Path("datasets/weights/example.jpg"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--out-json", type=Path, default=None)
    return parser.parse_args()


def _pick_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _load_image_tensor(path: Path, imgsz: int, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((imgsz, imgsz))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    return x.to(device)


def _tensor_summary(t: torch.Tensor) -> dict[str, Any]:
    out: dict[str, Any] = {
        "type": "tensor",
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
    }
    if t.numel() > 0 and torch.is_floating_point(t):
        out["min"] = float(t.min().item())
        out["max"] = float(t.max().item())
        out["mean"] = float(t.mean().item())
    return out


def _summarize_obj(obj: Any, *, _depth: int = 0, _max_depth: int = 8) -> Any:
    if _depth > _max_depth:
        return {"type": type(obj).__name__, "truncated": True}
    if torch.is_tensor(obj):
        return _tensor_summary(obj)
    if isinstance(obj, dict):
        return {str(k): _summarize_obj(v, _depth=_depth + 1, _max_depth=_max_depth) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return {"type": "tuple", "items": [_summarize_obj(v, _depth=_depth + 1, _max_depth=_max_depth) for v in obj]}
    if isinstance(obj, list):
        return {"type": "list", "items": [_summarize_obj(v, _depth=_depth + 1, _max_depth=_max_depth) for v in obj]}
    if is_dataclass(obj):
        return {
            "type": type(obj).__name__,
            "fields": {f.name: _summarize_obj(getattr(obj, f.name), _depth=_depth + 1, _max_depth=_max_depth) for f in fields(obj)},
        }
    return {"type": type(obj).__name__, "repr": repr(obj)}


def _extract_best_feat_shapes(det_out: Any, backend_out: Any | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if backend_out is not None:
        p3_backbone = getattr(backend_out, "p3_backbone", None)
        p3_head = getattr(backend_out, "p3_head", None)
        if torch.is_tensor(p3_backbone):
            out["p3_backbone"] = list(p3_backbone.shape)
        if torch.is_tensor(p3_head):
            out["p3_head"] = list(p3_head.shape)
    try:
        if isinstance(det_out, tuple) and len(det_out) >= 2 and isinstance(det_out[1], dict):
            one2many = det_out[1].get("one2many", {})
            feats = one2many.get("feats", [])
            if isinstance(feats, list):
                out["det_one2many_feats"] = [list(f.shape) for f in feats if torch.is_tensor(f)]
    except Exception:
        pass
    return out


def _extract_yolopv2_feat_shapes(out: Any) -> dict[str, Any]:
    feat_shapes: dict[str, Any] = {}
    try:
        if isinstance(out, tuple) and len(out) > 0 and isinstance(out[0], tuple) and len(out[0]) > 0:
            det_part = out[0][0]
            if isinstance(det_part, list):
                feat_shapes["det_feats"] = [list(v.shape) for v in det_part if torch.is_tensor(v)]
    except Exception:
        pass
    return feat_shapes


def _shape_from_summary(summary: dict[str, Any]) -> list[int] | None:
    if summary.get("type") == "tensor":
        shape = summary.get("shape")
        if isinstance(shape, list):
            return [int(v) for v in shape]
    return None


def _run_best(best_path: Path, x: torch.Tensor, device: torch.device) -> dict[str, Any]:
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model = PV26MultiHeadYOLO26(num_det_classes=len(DET_CLASSES_CANONICAL), yolo26_cfg="yolo26n.yaml").to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    backend_out = None
    with torch.no_grad():
        out = model(x)
        if hasattr(model, "det_backend"):
            backend_out = model.det_backend(x)
    out_summary = _summarize_obj(out)
    return {
        "checkpoint_keys": sorted(list(ckpt.keys())),
        "output_summary": out_summary,
        "feature_shapes": _extract_best_feat_shapes(out.det, backend_out),
    }


def _run_yolopv2(ts_path: Path, x: torch.Tensor, device: torch.device) -> dict[str, Any]:
    model = torch.jit.load(str(ts_path), map_location=device)
    model.eval()
    with torch.no_grad():
        out = model(x)
    return {
        "output_summary": _summarize_obj(out),
        "feature_shapes": _extract_yolopv2_feat_shapes(out),
    }


def _build_comparison(best: dict[str, Any], yolopv2: dict[str, Any]) -> dict[str, Any]:
    best_fields = best["output_summary"]["fields"]
    best_da_shape = _shape_from_summary(best_fields["da"])
    best_rm_shape = _shape_from_summary(best_fields["rm"])

    y_out_items = yolopv2["output_summary"]["items"]
    y_seg2_shape = _shape_from_summary(y_out_items[1]) if len(y_out_items) > 1 else None
    y_seg1_shape = _shape_from_summary(y_out_items[2]) if len(y_out_items) > 2 else None

    best_feats = best["feature_shapes"].get("det_one2many_feats", [])
    y_feats = yolopv2["feature_shapes"].get("det_feats", [])

    return {
        "segmentation_head_shape_match": {
            "best_da_vs_yolopv2_seg1": best_da_shape == y_seg1_shape,
            "best_rm_vs_yolopv2_seg2": best_rm_shape == y_seg2_shape,
            "best_da_shape": best_da_shape,
            "best_rm_shape": best_rm_shape,
            "yolopv2_seg2_shape": y_seg2_shape,
            "yolopv2_seg1_shape": y_seg1_shape,
        },
        "detection_feature_shape_match": {
            "same": best_feats == y_feats,
            "best_det_feats": best_feats,
            "yolopv2_det_feats": y_feats,
        },
        "top_level_output_type_match": (
            best["output_summary"]["type"] == yolopv2["output_summary"]["type"]
        ),
        "notes": [
            "best.pt is a training checkpoint dict (model_state in pv26 multi-head model).",
            "yolopv2.pt is a TorchScript module with tuple-style outputs.",
        ],
    }


def main() -> int:
    args = _parse_args()
    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp")

    device = _pick_device(args.device)
    x = _load_image_tensor(args.image, args.imgsz, device)

    best_result = _run_best(args.best, x, device)
    yolopv2_result = _run_yolopv2(args.yolopv2, x, device)
    comparison = _build_comparison(best_result, yolopv2_result)

    report = {
        "input": {
            "image": str(args.image),
            "input_tensor": _tensor_summary(x),
        },
        "best_pt": best_result,
        "yolopv2_pt": yolopv2_result,
        "comparison": comparison,
    }

    print("=== Input ===")
    print(json.dumps(report["input"], indent=2))
    print("\n=== Comparison Summary ===")
    print(json.dumps(comparison, indent=2))
    print("\n=== best.pt output summary ===")
    print(json.dumps(best_result["output_summary"], indent=2))
    print("\n=== yolopv2.pt output summary ===")
    print(json.dumps(yolopv2_result["output_summary"], indent=2))

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2))
        print(f"\nSaved full report to: {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
