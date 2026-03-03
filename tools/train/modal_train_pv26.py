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
TRAIN_SCRIPT = REPO_ROOT / "tools" / "train" / "train_pv26.py"

# =========================
# User-editable defaults
# =========================
APP_NAME             = os.getenv("PV26_MODAL_APP_NAME", "pv26-train")     # Modal app 이름 (보통 수정 불필요)
DATASET_VOLUME_NAME  = os.getenv("PV26_MODAL_DATASET_VOLUME", "pv26-datasets")	# 데이터 볼륨 이름
ARTIFACT_VOLUME_NAME = os.getenv("PV26_MODAL_ARTIFACT_VOLUME", "pv26-artifacts")	# 체크포인트/로그 볼륨 이름
GPU_NAME             = os.getenv("PV26_MODAL_GPU", "A10G")	# 예: "A10G", "L4", "A100"
TIMEOUT_SEC          = int(os.getenv("PV26_MODAL_TIMEOUT_SEC", str(24 * 60 * 60)))	# 예: 3600(1h), 86400(24h)

DEFAULT_EPOCHS       = 1        # 총 학습 epoch 수 (비용/시간에 거의 선형 비례)
DEFAULT_BATCH_SIZE   = 40        # 스텝당 배치 크기 (크면 GPU 사용률↑, 너무 크면 OOM)
DEFAULT_WORKERS      = 12        # DataLoader 워커 수 (AB 비교용 기본값)
DEFAULT_AUGMENT      = True      # train 증강 on/off (False면 --no-augment 전달)
DEFAULT_LR           = "5e-4"    # 학습률 (수렴 안정성/속도에 직접 영향)
DEFAULT_OPTIMIZER    = "adamw"   # Optimizer: adamw|adam|sgd
DEFAULT_WEIGHT_DECAY = "1e-4"    # Optimizer weight decay
DEFAULT_MOMENTUM     = "0.937"   # SGD momentum (adam/adamw에서는 무시됨)
DEFAULT_SCHEDULER    = "cosine"  # LR 스케줄러: cosine|none
DEFAULT_MIN_LR_RATIO = "0.05"    # cosine eta_min = lr * ratio
DEFAULT_COMPILE      = True      # torch.compile on/off (A/B 측정 후 필요시 off)
DEFAULT_COMPILE_MODE = "reduce-overhead"  # compile mode: default|reduce-overhead|max-autotune
DEFAULT_DET_PRETRAINED = ""      # 선택: detection trunk pretrained 체크포인트 경로(비우면 미사용)
DEFAULT_LOG_EVERY    = 100       # --no-progress일 때 몇 step마다 로그 출력할지
DEFAULT_PROGRESS     = False     # True면 tqdm 진행바 사용(원격 로그 줄 수가 급증할 수 있음)
DEFAULT_CPU          = 16.0      # Modal 컨테이너 CPU 코어 할당량 (AB 비교용 기본값)
DEFAULT_MEMORY_MB    = 65536     # Modal 컨테이너 RAM(MB) 할당량

DEFAULT_PREFETCH_FACTOR = 4      # 워커당 prefetch 배치 수 (AB 비교용 기본값)
DEFAULT_PERSISTENT_WORKERS = True  # epoch 사이에 워커 프로세스 유지(재시작 오버헤드 감소)
DEFAULT_PROFILE_EVERY = 10       # N step 평균 프로파일 로그 주기(0이면 비활성화)
DEFAULT_PROFILE_SYNC_CUDA = True  # True면 CUDA 동기화 기반 정밀 타이밍(오버헤드 증가)


DEFAULT_DATASET_DIR_IN_VOLUME   = "pv26_v1_bdd_full"	# 예: "pv26_v1_bdd_full"
DEFAULT_DATASET_TAR_IN_VOLUME   = "pv26_v1_bdd_full.tar"	# 예: ".tar", ".tar.gz"
DEFAULT_ARTIFACT_ROOT_IN_VOLUME = "runs/pv26_train"	# 예: "runs/pv26_train", "experiments/pv26"
AUTO_DOWNLOAD_ARTIFACTS         = True	# True면 학습 성공 후 로컬 runs/로 자동 다운로드
LOCAL_ARTIFACT_ROOT             = "runs/pv26_train"	# 로컬 저장 루트 (repo 기준 상대경로)
SYNC_EVERY_N_EPOCHS             = 1	 # latest.pt 갱신 N회(대략 N epoch)마다 로컬 동기화
SYNC_POLL_SEC                   = 30 # 동기화 폴링 주기(초)

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
    augment: bool = DEFAULT_AUGMENT,
    dataset_dir_in_volume: str = DEFAULT_DATASET_DIR_IN_VOLUME,
    dataset_tar_in_volume: str = DEFAULT_DATASET_TAR_IN_VOLUME,
    artifact_root_in_volume: str = DEFAULT_ARTIFACT_ROOT_IN_VOLUME,
) -> dict[str, str]:
    run_name = run_name.strip()
    if not run_name:
        raise ValueError("--run-name is required and cannot be empty.")

    dataset_mount = Path(DATASET_MOUNT)
    artifact_mount = Path(ARTIFACT_MOUNT)
    dataset_root = _prepare_dataset_on_local_ssd(
        dataset_mount=dataset_mount,
        dataset_dir_rel=dataset_dir_in_volume,
        dataset_tar_rel=dataset_tar_in_volume,
    )

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
        "--prefetch-factor",
        str(DEFAULT_PREFETCH_FACTOR),
        "--lr",
        str(DEFAULT_LR),
        "--optimizer",
        str(DEFAULT_OPTIMIZER),
        "--weight-decay",
        str(DEFAULT_WEIGHT_DECAY),
        "--momentum",
        str(DEFAULT_MOMENTUM),
        "--scheduler",
        str(DEFAULT_SCHEDULER),
        "--min-lr-ratio",
        str(DEFAULT_MIN_LR_RATIO),
        "--compile-mode",
        str(DEFAULT_COMPILE_MODE),
        "--log-every",
        str(DEFAULT_LOG_EVERY),
    ]
    if not bool(DEFAULT_COMPILE):
        cmd.append("--no-compile")
    if str(DEFAULT_DET_PRETRAINED).strip():
        cmd.extend(["--det-pretrained", str(DEFAULT_DET_PRETRAINED).strip()])
    if not DEFAULT_PERSISTENT_WORKERS:
        cmd.append("--no-persistent-workers")
    if int(DEFAULT_PROFILE_EVERY) > 0:
        cmd.extend(["--profile-every", str(int(DEFAULT_PROFILE_EVERY))])
    if DEFAULT_PROFILE_SYNC_CUDA:
        cmd.append("--profile-sync-cuda")
    if not DEFAULT_PROGRESS:
        cmd.append("--no-progress")
    if not bool(augment):
        cmd.append("--no-augment")
    cmd.extend(["--run-name", run_name])

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
    print(
        "[modal] profile: "
        f"gpu={GPU_NAME} cpu={DEFAULT_CPU} memory_mb={DEFAULT_MEMORY_MB} "
        f"batch={DEFAULT_BATCH_SIZE} augment={bool(augment)} "
        f"optimizer={DEFAULT_OPTIMIZER} scheduler={DEFAULT_SCHEDULER} "
        f"compile={DEFAULT_COMPILE} compile_mode={DEFAULT_COMPILE_MODE} "
        f"workers={DEFAULT_WORKERS} prefetch={DEFAULT_PREFETCH_FACTOR} "
        f"persistent_workers={DEFAULT_PERSISTENT_WORKERS} progress={DEFAULT_PROGRESS} "
        f"log_every={DEFAULT_LOG_EVERY} profile_every={DEFAULT_PROFILE_EVERY} "
        f"profile_sync_cuda={DEFAULT_PROFILE_SYNC_CUDA}",
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
    augment: bool = DEFAULT_AUGMENT,
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
            "artifact_root_in_volume": DEFAULT_ARTIFACT_ROOT_IN_VOLUME,
            "every_n_epochs": SYNC_EVERY_N_EPOCHS,
            "poll_sec": SYNC_POLL_SEC,
        },
        daemon=True,
    )
    sync_thread.start()
    try:
        result = train_remote.remote(
            dataset_dir_in_volume=DEFAULT_DATASET_DIR_IN_VOLUME,
            dataset_tar_in_volume=DEFAULT_DATASET_TAR_IN_VOLUME,
            artifact_root_in_volume=DEFAULT_ARTIFACT_ROOT_IN_VOLUME,
            run_name=run_name,
            augment=bool(augment),
        )
    finally:
        stop_event.set()
        sync_thread.join(timeout=2.0)
    _maybe_download_artifacts(result, DEFAULT_ARTIFACT_ROOT_IN_VOLUME)
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
        default=DEFAULT_AUGMENT,
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
            "artifact_root_in_volume": DEFAULT_ARTIFACT_ROOT_IN_VOLUME,
            "every_n_epochs": SYNC_EVERY_N_EPOCHS,
            "poll_sec": SYNC_POLL_SEC,
        },
        daemon=True,
    )
    sync_thread.start()
    try:
        result = train_remote.remote(
            dataset_dir_in_volume=DEFAULT_DATASET_DIR_IN_VOLUME,
            dataset_tar_in_volume=DEFAULT_DATASET_TAR_IN_VOLUME,
            artifact_root_in_volume=DEFAULT_ARTIFACT_ROOT_IN_VOLUME,
            run_name=run_name,
            augment=bool(args.augment),
        )
    finally:
        stop_event.set()
        sync_thread.join(timeout=2.0)
    _maybe_download_artifacts(result, DEFAULT_ARTIFACT_ROOT_IN_VOLUME)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
