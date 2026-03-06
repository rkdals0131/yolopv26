from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModalProfile:
    app_name: str
    gpu_name: str
    cpu: float
    memory_mb: int
    epochs: int
    batch_size: int
    workers: int
    prefetch_factor: int


PROFILES: dict[str, ModalProfile] = {
    "a10g": ModalProfile(
        app_name="pv26-train-a10g",
        gpu_name="A10G",
        cpu=16.0,
        memory_mb=65536,
        epochs=5,
        batch_size=32,
        workers=8,
        prefetch_factor=4,
    ),
    "a100": ModalProfile(
        app_name="pv26-train-a100",
        gpu_name="A100-80GB",
        cpu=16.0,
        memory_mb=65536,
        epochs=2,
        batch_size=128,
        workers=10,
        prefetch_factor=2,
    ),
    "h100": ModalProfile(
        app_name="pv26-train-h100-w12-pf3",
        gpu_name="H100",
        cpu=32.0,
        memory_mb=131072,
        epochs=2,
        batch_size=128,
        workers=14,
        prefetch_factor=3,
    ),
}


def apply_profile_env_defaults(profile_name: str) -> None:
    key = str(profile_name).strip().lower()
    profile = PROFILES.get(key)
    if profile is None:
        raise ValueError(f"unknown modal profile: {profile_name}")

    os.environ.setdefault("PV26_MODAL_APP_NAME", profile.app_name)
    os.environ.setdefault("PV26_MODAL_GPU", profile.gpu_name)
    os.environ.setdefault("PV26_MODAL_CPU", str(profile.cpu))
    os.environ.setdefault("PV26_MODAL_MEMORY_MB", str(profile.memory_mb))
    os.environ.setdefault("PV26_MODAL_EPOCHS", str(profile.epochs))
    os.environ.setdefault("PV26_MODAL_BATCH_SIZE", str(profile.batch_size))
    os.environ.setdefault("PV26_MODAL_WORKERS", str(profile.workers))
    os.environ.setdefault("PV26_MODAL_PREFETCH_FACTOR", str(profile.prefetch_factor))

