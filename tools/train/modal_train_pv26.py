#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tarfile

try:
    import modal
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guidance path
    raise SystemExit(
        "This script requires Modal. Install with: pip install modal"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = REPO_ROOT / "tools" / "train" / "train_pv26.py"

# =========================
# User-editable defaults
# =========================
APP_NAME             = os.getenv("PV26_MODAL_APP_NAME", "pv26-train")     # Modal app 이름 (보통 수정 불필요)
DATASET_VOLUME_NAME  = os.getenv("PV26_MODAL_DATASET_VOLUME", "pv26-datasets")	# 데이터 볼륨 이름
ARTIFACT_VOLUME_NAME = os.getenv("PV26_MODAL_ARTIFACT_VOLUME", "pv26-artifacts")	# 체크포인트/로그 볼륨 이름
GPU_NAME             = os.getenv("PV26_MODAL_GPU", "A10G")	# 예: "A10G", "L4", "A100"
TIMEOUT_SEC          = int(os.getenv("PV26_MODAL_TIMEOUT_SEC", str(24 * 60 * 60)))	# 예: 3600(1h), 86400(24h)

DEFAULT_EPOCHS       = 10
DEFAULT_BATCH_SIZE   = 8
DEFAULT_WORKERS      = 4
DEFAULT_LR           = "5e-4"


DEFAULT_DATASET_DIR_IN_VOLUME = "pv26_v1_bdd_full"	# 예: "pv26_v1_bdd_full"
DEFAULT_DATASET_TAR_IN_VOLUME = "pv26_v1_bdd_full.tar"	# 예: ".tar", ".tar.gz"
DEFAULT_ARTIFACT_ROOT_IN_VOLUME = "runs/pv26_train"	# 예: "runs/pv26_train", "experiments/pv26"

DATASET_MOUNT = "/vol/datasets"
ARTIFACT_MOUNT = "/vol/artifacts"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        "numpy",
        "Pillow",
        "tqdm",
        "tensorboard",
        "ultralytics",
    )
    .add_local_dir(str(REPO_ROOT), remote_path="/root/repo")
)

dataset_volume = modal.Volume.from_name(DATASET_VOLUME_NAME, create_if_missing=True)
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)


def _safe_rel_path(v: str) -> Path:
    p = Path(v)
    if p.is_absolute():
        raise ValueError(f"expected relative path, got absolute: {v}")
    normalized = Path(*[part for part in p.parts if part not in ("", ".")])
    if any(part == ".." for part in normalized.parts):
        raise ValueError(f"relative path must not include '..': {v}")
    return normalized


def _resolve_under(root: Path, rel: str) -> Path:
    rel_path = _safe_rel_path(rel)
    target = (root / rel_path).resolve()
    root_resolved = root.resolve()
    if target != root_resolved and root_resolved not in target.parents:
        raise ValueError(f"path escapes root: {rel}")
    return target


def _safe_extract_tar(tar_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tf:
        members = tf.getmembers()
        for m in members:
            mpath = Path(m.name)
            if mpath.is_absolute() or ".." in mpath.parts:
                raise RuntimeError(f"unsafe tar member path: {m.name}")
            out_path = (dest_dir / mpath).resolve()
            if out_path != dest_dir.resolve() and dest_dir.resolve() not in out_path.parents:
                raise RuntimeError(f"unsafe extraction target: {m.name}")
        tf.extractall(path=dest_dir)


def _find_dataset_root(dataset_mount: Path, dataset_dir_rel: str, dataset_tar_rel: str) -> Path:
    expected_root = _resolve_under(dataset_mount, dataset_dir_rel)
    manifest = expected_root / "meta" / "split_manifest.csv"
    if manifest.exists():
        return expected_root

    tar_path = _resolve_under(dataset_mount, dataset_tar_rel)
    if not tar_path.exists():
        raise FileNotFoundError(
            f"dataset not found. Missing both directory and tar:\n"
            f"- expected dir: {expected_root}\n"
            f"- expected tar: {tar_path}"
        )

    print(f"[modal] extracting dataset tar: {tar_path} -> {dataset_mount}", flush=True)
    _safe_extract_tar(tar_path, dataset_mount)

    if manifest.exists():
        return expected_root

    # Fallback: tar root name may differ. Detect unique dataset root by manifest.
    manifests = list(dataset_mount.rglob("meta/split_manifest.csv"))
    if not manifests:
        raise FileNotFoundError(
            "dataset extraction finished, but no meta/split_manifest.csv was found "
            f"under {dataset_mount}"
        )
    if len(manifests) > 1:
        found = "\n".join(str(p) for p in manifests[:10])
        raise RuntimeError(
            "multiple dataset roots detected after extraction; set a unique DEFAULT_DATASET_DIR_IN_VOLUME.\n"
            f"found:\n{found}"
        )
    return manifests[0].parents[1]


def _stream_subprocess(cmd: list[str], env: dict[str, str]) -> int:
    print(f"[modal] exec: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line.rstrip("\n"), flush=True)
    return int(proc.wait())


@app.function(
    image=image,
    gpu=GPU_NAME,
    timeout=TIMEOUT_SEC,
    volumes={
        DATASET_MOUNT: dataset_volume,
        ARTIFACT_MOUNT: artifact_volume,
    },
)
def train_remote(
    run_name: str,
    dataset_dir_in_volume: str = DEFAULT_DATASET_DIR_IN_VOLUME,
    dataset_tar_in_volume: str = DEFAULT_DATASET_TAR_IN_VOLUME,
    artifact_root_in_volume: str = DEFAULT_ARTIFACT_ROOT_IN_VOLUME,
) -> dict[str, str]:
    run_name = run_name.strip()
    if not run_name:
        raise ValueError("--run-name is required and cannot be empty.")

    dataset_mount = Path(DATASET_MOUNT)
    artifact_mount = Path(ARTIFACT_MOUNT)
    dataset_root = _find_dataset_root(dataset_mount, dataset_dir_in_volume, dataset_tar_in_volume)
    dataset_volume.commit()

    out_root = _resolve_under(artifact_mount, artifact_root_in_volume)
    out_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--dataset-root",
        str(dataset_root),
        "--out-dir",
        str(out_root),
        "--epochs",
        str(DEFAULT_EPOCHS),
        "--batch-size",
        str(DEFAULT_BATCH_SIZE),
        "--workers",
        str(DEFAULT_WORKERS),
        "--lr",
        str(DEFAULT_LR),
    ]
    cmd.extend(["--run-name", run_name])

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
    rc = _stream_subprocess(cmd, env=env)
    artifact_volume.commit()

    result = {
        "dataset_root": str(dataset_root),
        "artifact_root": str(out_root),
        "run_name": run_name,
        "return_code": str(rc),
    }
    if rc != 0:
        raise RuntimeError(f"training failed with return code {rc}: {result}")
    print(f"[modal] training done: {result}", flush=True)
    return result


@app.local_entrypoint()
def modal_entrypoint(
    run_name: str = "",
):
    run_name = run_name.strip()
    if not run_name:
        raise SystemExit("--run-name is required. Example: --run-name exp_modal_a10g")

    result = train_remote.remote(
        dataset_dir_in_volume=DEFAULT_DATASET_DIR_IN_VOLUME,
        dataset_tar_in_volume=DEFAULT_DATASET_TAR_IN_VOLUME,
        artifact_root_in_volume=DEFAULT_ARTIFACT_ROOT_IN_VOLUME,
        run_name=run_name,
    )
    print(result)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Submit PV26 training to Modal.")
    sub = p.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Run remote training on Modal")
    train.add_argument("--run-name", required=True, help="Required. Modal run name / checkpoint namespace.")
    return p


def _main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command != "train":
        parser.error(f"unsupported command: {args.command}")

    result = train_remote.remote(
        dataset_dir_in_volume=DEFAULT_DATASET_DIR_IN_VOLUME,
        dataset_tar_in_volume=DEFAULT_DATASET_TAR_IN_VOLUME,
        artifact_root_in_volume=DEFAULT_ARTIFACT_ROOT_IN_VOLUME,
        run_name=args.run_name.strip(),
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
