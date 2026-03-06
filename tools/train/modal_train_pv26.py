#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import threading
import time

try:
    import modal
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guidance path
    raise SystemExit(
        "This script requires Modal. Install with: pip install modal"
    ) from exc


def _detect_repo_root() -> Path:
    # Local execution path: .../tools/train/modal_train_pv26.py
    p = Path(__file__).resolve()
    for cand in [p.parent.parent.parent, Path("/root/repo"), Path.cwd()]:
        train_py = cand / "tools" / "train" / "train_pv26.py"
        if train_py.exists():
            return cand
    raise RuntimeError(
        "failed to locate repo root. expected tools/train/train_pv26.py in one of: "
        f"{p.parent.parent.parent}, /root/repo, {Path.cwd()}"
    )


REPO_ROOT = _detect_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TRAIN_SCRIPT = REPO_ROOT / "tools" / "train" / "train_pv26.py"
from tools.train.modal_train_common import (
    ModalDatasetDefaults,
    ModalRuntimeDefaults,
    ModalSyncDefaults,
    ModalTrainDefaults,
    build_train_command,
    default_torch_specs_for_gpu,
    format_modal_profile,
)


# =========================
# User-editable defaults
# =========================
_DEFAULT_TORCH_SPEC, _DEFAULT_TORCHVISION_SPEC = default_torch_specs_for_gpu(
    os.getenv("PV26_MODAL_GPU", "A10G")
)

RUNTIME_DEFAULTS = ModalRuntimeDefaults(
    app_name=os.getenv("PV26_MODAL_APP_NAME", "pv26-train-a10g"),            # Modal app 이름
    dataset_volume_name=os.getenv("PV26_MODAL_DATASET_VOLUME", "pv26-datasets"),  # 데이터 볼륨 이름
    artifact_volume_name=os.getenv("PV26_MODAL_ARTIFACT_VOLUME", "pv26-artifacts"),  # 산출물 볼륨 이름
    gpu_name=os.getenv("PV26_MODAL_GPU", "A10G"),                            # 예: A10G | L4 | A100 | B200
    torch_spec=os.getenv("PV26_MODAL_TORCH_SPEC", _DEFAULT_TORCH_SPEC),      # torch wheel spec
    torchvision_spec=os.getenv("PV26_MODAL_TORCHVISION_SPEC", _DEFAULT_TORCHVISION_SPEC),  # torchvision wheel spec
    timeout_sec=int(os.getenv("PV26_MODAL_TIMEOUT_SEC", str(24 * 60 * 60))), # Modal timeout(초)
    cpu=16.0,                                                                 # 컨테이너 CPU 코어 할당량
    memory_mb=65536,                                                          # 컨테이너 RAM(MB)
)

TRAIN_DEFAULTS = ModalTrainDefaults(
    epochs=5,                     # 총 학습 epoch 수
    batch_size=32,                # 스텝당 배치 크기
    workers=8,                    # DataLoader worker 수
    prefetch_factor=4,            # worker당 prefetch 배치 수
    persistent_workers=True,      # epoch 사이 worker 유지
    augment=True,                 # train augmentation on/off
    lr="0",                       # 0이면 optimizer별 자동 LR
    optimizer="adamw",            # adamw | adam | sgd
    weight_decay="1e-4",          # optimizer weight decay
    momentum="0.937",             # SGD momentum
    scheduler="cosine",           # cosine | none
    min_lr_ratio="0.05",          # cosine eta_min 비율
    warmup_epochs=3,              # warmup epoch 수
    warmup_start_factor="0.1",    # warmup 시작 LR 비율
    compile=False,                # model torch.compile on/off
    compile_mode="default",       # compile mode
    compile_fullgraph=False,      # fullgraph compile on/off
    compile_seg_loss=True,        # seg loss block만 compile
    seg_output_stride=2,          # segmentation output stride
    det_pretrained="",            # detection pretrained 경로
    log_every=20,                 # --no-progress일 때 로그 주기
    progress=False,               # tqdm progress bar 사용 여부
    tensorboard=True,             # TensorBoard writer 사용 여부
    profile_every=20,             # train profile 출력 주기
    profile_sync_cuda=False,      # profile 시 CUDA sync 여부
    profile_system=False,         # 시스템 메모리/GPU 통계 포함 여부
    eval_map_every=5,             # validation mAP 계산 주기
    train_drop_last=False,        # 마지막 ragged batch drop 여부
)

DATASET_DEFAULTS = ModalDatasetDefaults(
    dataset_dir_in_volume="pv26_v1_bdd_full",    # 볼륨 내 dataset 디렉토리
    dataset_tar_in_volume="pv26_v1_bdd_full.tar",  # 볼륨 내 dataset tar 파일
    artifact_root_in_volume="runs/pv26_train",   # 볼륨 내 산출물 루트
)

SYNC_DEFAULTS = ModalSyncDefaults(
    auto_download_artifacts=True,    # 학습 완료 후 로컬 자동 다운로드
    local_artifact_root="runs/pv26_train",  # 로컬 저장 루트
    sync_every_n_epochs=1,           # latest.pt 갱신 N회마다 checkpoint sync
    sync_poll_sec=30,                # sync polling 주기
)

APP_NAME = RUNTIME_DEFAULTS.app_name
DATASET_VOLUME_NAME = RUNTIME_DEFAULTS.dataset_volume_name
ARTIFACT_VOLUME_NAME = RUNTIME_DEFAULTS.artifact_volume_name
GPU_NAME = RUNTIME_DEFAULTS.gpu_name
TORCH_SPEC = RUNTIME_DEFAULTS.torch_spec
TORCHVISION_SPEC = RUNTIME_DEFAULTS.torchvision_spec
TIMEOUT_SEC = RUNTIME_DEFAULTS.timeout_sec
DEFAULT_CPU = RUNTIME_DEFAULTS.cpu
DEFAULT_MEMORY_MB = RUNTIME_DEFAULTS.memory_mb
AUTO_DOWNLOAD_ARTIFACTS = SYNC_DEFAULTS.auto_download_artifacts
LOCAL_ARTIFACT_ROOT = SYNC_DEFAULTS.local_artifact_root
SYNC_EVERY_N_EPOCHS = SYNC_DEFAULTS.sync_every_n_epochs
SYNC_POLL_SEC = SYNC_DEFAULTS.sync_poll_sec

DATASET_MOUNT = "/vol/datasets"
ARTIFACT_MOUNT = "/vol/artifacts"
LOCAL_DATASET_SSD_ROOT = "/tmp/pv26_dataset_cache"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
    )
    .pip_install(
        TORCH_SPEC,
        TORCHVISION_SPEC,
        "numpy",
        "Pillow",
        "tqdm",
        "tensorboard",
        "ultralytics",
    )
    .add_local_dir(
        str(REPO_ROOT),
        remote_path="/root/repo",
        ignore=[
            ".git/**",
            ".venv/**",
            "runs/**",
            "datasets/**",
            "docs/**",
            ".omx/**",
            "__pycache__/**",
            ".pytest_cache/**",
        ],
    )
)

dataset_volume = modal.Volume.from_name(DATASET_VOLUME_NAME, create_if_missing=True)
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)


def _preflight_cuda_compat_or_raise(*, requested_gpu: str) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available inside this Modal container. "
            f"requested_gpu={requested_gpu}"
        )

    capability = torch.cuda.get_device_capability(0)
    required_sm = f"sm_{capability[0]}{capability[1]}"
    supported_arches = sorted(set(torch.cuda.get_arch_list()))
    supported = any(a == required_sm or a.startswith(required_sm) for a in supported_arches)
    if supported:
        device_name = torch.cuda.get_device_name(0)
        print(
            "[modal] cuda preflight: "
            f"device={device_name} capability={required_sm} torch={torch.__version__}",
            flush=True,
        )
        return

    try:
        import torchvision

        tv_version = torchvision.__version__
    except Exception:
        tv_version = "unknown"

    device_name = torch.cuda.get_device_name(0)
    supported_str = " ".join(supported_arches) if supported_arches else "(none)"
    raise RuntimeError(
        "PyTorch CUDA arch mismatch for selected GPU.\n"
        f"- requested_gpu={requested_gpu}\n"
        f"- detected_gpu={device_name}\n"
        f"- required_arch={required_sm}\n"
        f"- torch_supported_arches={supported_str}\n"
        f"- torch_version={torch.__version__} torchvision_version={tv_version}\n"
        f"- image_specs: {TORCH_SPEC}, {TORCHVISION_SPEC}\n"
        "Fix: install a newer torch build with "
        "PV26_MODAL_TORCH_SPEC/PV26_MODAL_TORCHVISION_SPEC "
        "(for B200, start with torch>=2.7.0 and torchvision>=0.22.0)."
    )


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


def _find_dataset_root(local_search_root: Path, expected_root: Path) -> Path:
    manifest = expected_root / "meta" / "split_manifest.csv"
    if manifest.exists():
        return expected_root
    manifests = list(local_search_root.rglob("meta/split_manifest.csv"))
    if not manifests:
        raise FileNotFoundError(
            "dataset preparation finished, but no meta/split_manifest.csv was found "
            f"under {local_search_root}"
        )
    if len(manifests) > 1:
        found = "\n".join(str(p) for p in manifests[:10])
        raise RuntimeError(
            "multiple dataset roots detected; set a unique DEFAULT_DATASET_DIR_IN_VOLUME.\n"
            f"found:\n{found}"
        )
    return manifests[0].parents[1]


def _prepare_dataset_on_local_ssd(
    *,
    dataset_mount: Path,
    dataset_dir_rel: str,
    dataset_tar_rel: str,
) -> Path:
    local_root = Path(LOCAL_DATASET_SSD_ROOT)
    local_root.mkdir(parents=True, exist_ok=True)
    expected_local_root = _resolve_under(local_root, dataset_dir_rel)

    volume_dir = _resolve_under(dataset_mount, dataset_dir_rel)
    volume_manifest = volume_dir / "meta" / "split_manifest.csv"
    volume_tar = _resolve_under(dataset_mount, dataset_tar_rel)

    # Prefer tar path: single big-file read from volume, then local SSD extraction.
    if volume_tar.exists():
        local_tar = local_root / volume_tar.name
        print(f"[modal] copying dataset tar to local SSD: {volume_tar} -> {local_tar}", flush=True)
        shutil.copy2(volume_tar, local_tar)
        if expected_local_root.exists():
            shutil.rmtree(expected_local_root)
        print(f"[modal] extracting tar on local SSD: {local_tar} -> {local_root}", flush=True)
        _safe_extract_tar(local_tar, local_root)
        return _find_dataset_root(local_root, expected_local_root)

    # Fallback: if tar is missing but directory already exists in volume, copy dir once to local SSD.
    if volume_manifest.exists():
        print(f"[modal] copying dataset directory to local SSD: {volume_dir} -> {expected_local_root}", flush=True)
        if expected_local_root.exists():
            shutil.rmtree(expected_local_root)
        expected_local_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(volume_dir, expected_local_root)
        return _find_dataset_root(local_root, expected_local_root)

    raise FileNotFoundError(
        f"dataset not found in volume. Missing both tar and directory:\n"
        f"- expected tar: {volume_tar}\n"
        f"- expected dir manifest: {volume_manifest}"
    )


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


def _download_artifacts_to_local(*, run_name: str, artifact_root_in_volume: str) -> Path:
    remote_root = "/" + artifact_root_in_volume.strip("/")
    remote_path = f"{remote_root}/{run_name}"
    local_root = _resolve_under(REPO_ROOT, LOCAL_ARTIFACT_ROOT)
    local_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        "modal",
        "volume",
        "get",
        "--force",
        ARTIFACT_VOLUME_NAME,
        remote_path,
        str(local_root),
    ]
    print(f"[modal] download artifacts: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"artifact download failed (code={proc.returncode}). "
            f"Try manually: modal volume get {ARTIFACT_VOLUME_NAME} {remote_path} {local_root}"
        )
    return local_root / run_name


def _download_tb_to_local(*, run_name: str, artifact_root_in_volume: str) -> Path:
    remote_root = "/" + artifact_root_in_volume.strip("/")
    remote_path = f"{remote_root}/{run_name}/tb"
    local_root = _resolve_under(REPO_ROOT, LOCAL_ARTIFACT_ROOT)
    local_run_dir = local_root / run_name
    local_run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "modal",
        "volume",
        "get",
        "--force",
        ARTIFACT_VOLUME_NAME,
        remote_path,
        str(local_run_dir),
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"tb sync failed (code={proc.returncode}). "
            f"Try manually: modal volume get --force {ARTIFACT_VOLUME_NAME} {remote_path} {local_run_dir}"
        )
    return local_run_dir / "tb"


def _download_checkpoints_to_local(*, run_name: str, artifact_root_in_volume: str) -> Path:
    remote_root = "/" + artifact_root_in_volume.strip("/")
    remote_path = f"{remote_root}/{run_name}/checkpoints"
    local_root = _resolve_under(REPO_ROOT, LOCAL_ARTIFACT_ROOT)
    local_run_dir = local_root / run_name
    local_run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "modal",
        "volume",
        "get",
        "--force",
        ARTIFACT_VOLUME_NAME,
        remote_path,
        str(local_run_dir),
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"checkpoint sync failed (code={proc.returncode}). "
            f"Try manually: modal volume get --force {ARTIFACT_VOLUME_NAME} {remote_path} {local_run_dir}"
        )
    return local_run_dir / "checkpoints"


def _latest_ckpt_fingerprint(*, run_name: str, artifact_root_in_volume: str) -> str | None:
    remote_root = "/" + artifact_root_in_volume.strip("/")
    remote_path = f"{remote_root}/{run_name}/checkpoints/latest.pt"
    cmd = [
        "modal",
        "volume",
        "ls",
        "--json",
        ARTIFACT_VOLUME_NAME,
        remote_path,
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    payload = proc.stdout.strip()
    if not payload:
        return None
    try:
        obj = json.loads(payload)
    except Exception:
        return payload
    if isinstance(obj, list) and not obj:
        return None
    return json.dumps(obj, sort_keys=True, ensure_ascii=True)


def _sync_worker(
    *,
    stop_event: threading.Event,
    run_name: str,
    artifact_root_in_volume: str,
    every_n_epochs: int,
    poll_sec: int,
) -> None:
    n = max(1, int(every_n_epochs))
    poll = max(5, int(poll_sec))
    last_fp: str | None = None
    latest_updates = 0
    while not stop_event.wait(poll):
        try:
            fp = _latest_ckpt_fingerprint(
                run_name=run_name,
                artifact_root_in_volume=artifact_root_in_volume,
            )
            if fp is None:
                continue
            if fp != last_fp:
                last_fp = fp
                latest_updates += 1
                # TensorBoard logs: every epoch sync (requested behavior).
                tb_path = _download_tb_to_local(
                    run_name=run_name,
                    artifact_root_in_volume=artifact_root_in_volume,
                )
                print(
                    f"[modal] periodic tb sync: latest update #{latest_updates} -> {tb_path}",
                    flush=True,
                )
                # Checkpoints: sync every N epochs to reduce overhead.
                if latest_updates % n == 0:
                    ckpt_path = _download_checkpoints_to_local(
                        run_name=run_name,
                        artifact_root_in_volume=artifact_root_in_volume,
                    )
                    print(
                        f"[modal] periodic checkpoint sync: latest update #{latest_updates} -> {ckpt_path}",
                        flush=True,
                    )
        except Exception as exc:
            print(f"[modal] warning(sync): {exc}", flush=True)


def _maybe_download_artifacts(result: dict[str, str], artifact_root_in_volume: str) -> None:
    if not AUTO_DOWNLOAD_ARTIFACTS:
        return
    run_name = result.get("run_name", "").strip()
    if not run_name:
        print("[modal] skip artifact download: run_name missing", flush=True)
        return
    try:
        local_path = _download_artifacts_to_local(
            run_name=run_name,
            artifact_root_in_volume=artifact_root_in_volume,
        )
        print(f"[modal] artifacts synced to local: {local_path}", flush=True)
    except Exception as exc:
        print(f"[modal] warning: {exc}", flush=True)


@app.function(
    image=image,
    gpu=GPU_NAME,
    cpu=DEFAULT_CPU,
    memory=DEFAULT_MEMORY_MB,
    timeout=TIMEOUT_SEC,
    volumes={
        DATASET_MOUNT: dataset_volume,
        ARTIFACT_MOUNT: artifact_volume,
    },
)
def train_remote(
    run_name: str,
    augment: bool = TRAIN_DEFAULTS.augment,
    dataset_dir_in_volume: str = DATASET_DEFAULTS.dataset_dir_in_volume,
    dataset_tar_in_volume: str = DATASET_DEFAULTS.dataset_tar_in_volume,
    artifact_root_in_volume: str = DATASET_DEFAULTS.artifact_root_in_volume,
) -> dict[str, str]:
    run_name = run_name.strip()
    if not run_name:
        raise ValueError("--run-name is required and cannot be empty.")

    _preflight_cuda_compat_or_raise(requested_gpu=GPU_NAME)

    dataset_mount = Path(DATASET_MOUNT)
    artifact_mount = Path(ARTIFACT_MOUNT)
    dataset_root = _prepare_dataset_on_local_ssd(
        dataset_mount=dataset_mount,
        dataset_dir_rel=dataset_dir_in_volume,
        dataset_tar_rel=dataset_tar_in_volume,
    )

    out_root = _resolve_under(artifact_mount, artifact_root_in_volume)
    out_root.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable]
    cmd.extend(
        build_train_command(
            train_script=TRAIN_SCRIPT,
            dataset_root=dataset_root,
            out_root=out_root,
            run_name=run_name,
            train_defaults=TRAIN_DEFAULTS,
            augment=bool(augment),
        )
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
    print(
        format_modal_profile(
            runtime_defaults=RUNTIME_DEFAULTS,
            train_defaults=TRAIN_DEFAULTS,
            augment=bool(augment),
        ),
        flush=True,
    )
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
    augment: bool = TRAIN_DEFAULTS.augment,
):
    run_name = run_name.strip()
    if not run_name:
        raise SystemExit("--run-name is required. Example: --run-name exp_modal_a10g")

    stop_event = threading.Event()
    sync_thread = threading.Thread(
        target=_sync_worker,
        kwargs={
            "stop_event": stop_event,
            "run_name": run_name,
            "artifact_root_in_volume": DATASET_DEFAULTS.artifact_root_in_volume,
            "every_n_epochs": SYNC_EVERY_N_EPOCHS,
            "poll_sec": SYNC_POLL_SEC,
        },
        daemon=True,
    )
    sync_thread.start()
    try:
        result = train_remote.remote(
            dataset_dir_in_volume=DATASET_DEFAULTS.dataset_dir_in_volume,
            dataset_tar_in_volume=DATASET_DEFAULTS.dataset_tar_in_volume,
            artifact_root_in_volume=DATASET_DEFAULTS.artifact_root_in_volume,
            run_name=run_name,
            augment=bool(augment),
        )
    finally:
        stop_event.set()
        sync_thread.join(timeout=2.0)
    _maybe_download_artifacts(result, DATASET_DEFAULTS.artifact_root_in_volume)
    print(result)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Submit PV26 training to Modal.")
    sub = p.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Run remote training on Modal")
    train.add_argument("--run-name", required=True, help="Required. Modal run name / checkpoint namespace.")
    train.add_argument(
        "--augment",
        dest="augment",
        action="store_true",
        default=TRAIN_DEFAULTS.augment,
        help="Enable train-time augmentation (default: on)",
    )
    train.add_argument(
        "--no-augment",
        dest="augment",
        action="store_false",
        help="Disable train-time augmentation",
    )
    return p


def _main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command != "train":
        parser.error(f"unsupported command: {args.command}")

    run_name = args.run_name.strip()
    stop_event = threading.Event()
    sync_thread = threading.Thread(
        target=_sync_worker,
        kwargs={
            "stop_event": stop_event,
            "run_name": run_name,
            "artifact_root_in_volume": DATASET_DEFAULTS.artifact_root_in_volume,
            "every_n_epochs": SYNC_EVERY_N_EPOCHS,
            "poll_sec": SYNC_POLL_SEC,
        },
        daemon=True,
    )
    sync_thread.start()
    try:
        result = train_remote.remote(
            dataset_dir_in_volume=DATASET_DEFAULTS.dataset_dir_in_volume,
            dataset_tar_in_volume=DATASET_DEFAULTS.dataset_tar_in_volume,
            artifact_root_in_volume=DATASET_DEFAULTS.artifact_root_in_volume,
            run_name=run_name,
            augment=bool(args.augment),
        )
    finally:
        stop_event.set()
        sync_thread.join(timeout=2.0)
    _maybe_download_artifacts(result, DATASET_DEFAULTS.artifact_root_in_volume)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
