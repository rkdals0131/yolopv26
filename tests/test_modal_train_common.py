import unittest
from pathlib import Path

from tools.train.modal_train_common import (
    ModalRuntimeDefaults,
    ModalTrainDefaults,
    build_train_command,
    format_modal_profile,
)


class TestModalTrainCommon(unittest.TestCase):
    def test_build_train_command_explicitly_pins_critical_flags(self):
        defaults = ModalTrainDefaults(
            epochs=2,
            batch_size=32,
            workers=8,
            prefetch_factor=4,
            persistent_workers=True,
            augment=True,
            lr="0",
            optimizer="adamw",
            weight_decay="1e-4",
            momentum="0.937",
            scheduler="cosine",
            min_lr_ratio="0.05",
            warmup_epochs=3,
            warmup_start_factor="0.1",
            compile=False,
            compile_mode="default",
            compile_fullgraph=False,
            compile_seg_loss=True,
            seg_output_stride=2,
            det_pretrained="",
            log_every=20,
            progress=False,
            tensorboard=True,
            profile_every=20,
            profile_sync_cuda=False,
            profile_system=False,
            eval_map_every=5,
            train_drop_last=True,
        )

        cmd = build_train_command(
            train_script=Path("tools/train/train_pv26.py"),
            dataset_root=Path("/tmp/dataset"),
            out_root=Path("/tmp/out"),
            run_name="exp",
            train_defaults=defaults,
            augment=False,
        )

        self.assertIn("--no-compile", cmd)
        self.assertIn("--compile-seg-loss", cmd)
        self.assertIn("--no-compile-fullgraph", cmd)
        self.assertIn("--no-progress", cmd)
        self.assertIn("--tensorboard", cmd)
        self.assertIn("--no-augment", cmd)
        self.assertIn("--train-drop-last", cmd)
        self.assertEqual(cmd[cmd.index("--seg-output-stride") + 1], "2")
        self.assertEqual(cmd[cmd.index("--profile-every") + 1], "20")

    def test_format_modal_profile_reports_seg_compile_and_stride(self):
        runtime = ModalRuntimeDefaults(
            app_name="pv26-train",
            dataset_volume_name="pv26-datasets",
            artifact_volume_name="pv26-artifacts",
            gpu_name="A10G",
            torch_spec="torch==2.6.0",
            torchvision_spec="torchvision==0.21.0",
            timeout_sec=3600,
            cpu=16.0,
            memory_mb=65536,
        )
        train = ModalTrainDefaults(
            epochs=5,
            batch_size=32,
            workers=8,
            prefetch_factor=4,
            persistent_workers=True,
            augment=True,
            lr="0",
            optimizer="adamw",
            weight_decay="1e-4",
            momentum="0.937",
            scheduler="cosine",
            min_lr_ratio="0.05",
            warmup_epochs=3,
            warmup_start_factor="0.1",
            compile=False,
            compile_mode="default",
            compile_fullgraph=False,
            compile_seg_loss=True,
            seg_output_stride=2,
            det_pretrained="",
            log_every=20,
            progress=False,
            tensorboard=True,
            profile_every=20,
            profile_sync_cuda=False,
            profile_system=False,
            eval_map_every=5,
            train_drop_last=False,
        )

        profile = format_modal_profile(runtime_defaults=runtime, train_defaults=train, augment=True)

        self.assertIn("compile=False", profile)
        self.assertIn("compile_seg_loss=True", profile)
        self.assertIn("seg_output_stride=2", profile)
        self.assertIn("tensorboard=True", profile)


if __name__ == "__main__":
    unittest.main()
