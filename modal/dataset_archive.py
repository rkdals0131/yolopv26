from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

from constants import DATASET_ARCHIVE_NAME, DATASET_ROOT_DIRNAME, REQUIRED_DATASET_DIRS


def _log(message: str) -> None:
    print(f"[modal-dataset] {message}", flush=True)


def _archive_top_level_names(archive_path: Path, *, max_entries: int = 512) -> set[str]:
    name = archive_path.name.lower()
    if name.endswith((".tar.zst", ".tzst")):
        import subprocess

        process = subprocess.Popen(
            ["tar", "--use-compress-program=unzstd", "-tf", str(archive_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        top_levels: set[str] = set()
        try:
            assert process.stdout is not None
            for index, line in enumerate(process.stdout):
                if line.strip():
                    top_levels.add(line.split("/", 1)[0])
                if DATASET_ROOT_DIRNAME in top_levels or index + 1 >= max_entries:
                    break
        finally:
            process.kill()
            process.communicate()
        return top_levels
    if name.endswith((".tar.gz", ".tgz", ".tar")):
        top_levels: set[str] = set()
        with tarfile.open(archive_path) as tar:
            for index, member in enumerate(tar):
                if member.name:
                    top_levels.add(member.name.split("/", 1)[0])
                if DATASET_ROOT_DIRNAME in top_levels or index + 1 >= max_entries:
                    break
        return top_levels
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            return {name.split("/", 1)[0] for name in zf.namelist()[:max_entries] if name}
    raise ValueError(f"unsupported dataset archive extension: {archive_path}")


def verify_archive_contract(archive_path: Path) -> dict[str, object]:
    if archive_path.name != DATASET_ARCHIVE_NAME:
        raise ValueError(
            f"dataset archive filename mismatch: got {archive_path.name}, expected {DATASET_ARCHIVE_NAME}"
        )
    if not archive_path.is_file():
        raise FileNotFoundError(f"dataset archive not found: {archive_path}")
    top_levels = sorted(_archive_top_level_names(archive_path))
    if DATASET_ROOT_DIRNAME not in top_levels:
        raise FileNotFoundError(
            "dataset archive top-level directory mismatch: "
            f"expected {DATASET_ROOT_DIRNAME}/, found {top_levels[:10]}"
        )
    payload = {
        "archive_path": str(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "top_level_dirs": top_levels,
    }
    _log(
        "OK archive contract "
        f"name={archive_path.name} size_bytes={payload['archive_size_bytes']} top_level={DATASET_ROOT_DIRNAME}/"
    )
    return payload


def extract_archive(archive_path: Path, destination: Path) -> None:
    verify_archive_contract(archive_path)
    if destination.name != DATASET_ROOT_DIRNAME:
        raise ValueError(
            f"destination dirname mismatch: got {destination.name}, expected {DATASET_ROOT_DIRNAME}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        _log(f"OK dataset already extracted destination={destination}")
        return
    name = archive_path.name.lower()
    _log(f"extracting archive={archive_path} destination_parent={destination.parent}")
    if name.endswith((".tar.zst", ".tzst")):
        import subprocess

        subprocess.run(
            ["tar", "--use-compress-program=unzstd", "-xf", str(archive_path), "-C", str(destination.parent)],
            check=True,
        )
    elif name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive_path) as tar:
            tar.extractall(destination.parent)
    elif name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(destination.parent)
    else:
        raise ValueError(f"unsupported dataset archive extension: {archive_path}")

    if not destination.exists():
        candidates = [path for path in destination.parent.iterdir() if path.is_dir()]
        if len(candidates) == 1:
            candidates[0].rename(destination)
    if not destination.exists():
        raise FileNotFoundError(f"archive extraction did not create expected dataset root: {destination}")
    _log(f"OK extracted dataset root={destination}")


def verify_layout(dataset_root: Path) -> dict[str, str]:
    if dataset_root.name != DATASET_ROOT_DIRNAME:
        raise ValueError(
            f"dataset root dirname mismatch: got {dataset_root.name}, expected {DATASET_ROOT_DIRNAME}"
        )
    missing = [rel for rel in REQUIRED_DATASET_DIRS if not (dataset_root / rel).exists()]
    if missing:
        raise FileNotFoundError(f"dataset layout missing under {dataset_root}: {missing}")
    payload = {rel: str(dataset_root / rel) for rel in REQUIRED_DATASET_DIRS}
    _log(f"OK dataset layout root={dataset_root} required_dirs={len(REQUIRED_DATASET_DIRS)}")
    stats_path = dataset_root / "meta" / "final_dataset_stats.json"
    if stats_path.is_file():
        _log(f"OK dataset stats path={stats_path}")
    else:
        _log(f"WARN dataset stats missing path={stats_path}")
    return payload
