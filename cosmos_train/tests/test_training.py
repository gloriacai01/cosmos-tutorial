from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import pytest

from cosmos_train.framework_setup import (
    driver_library_path,
    patch_framework_for_edge,
    train_command,
    training_environment,
)
from model.training import ingest_episodes, parse_args


def _episode_zip(
    path: Path,
    episode_id: str,
    *,
    manifest_id: str | None = None,
    member_prefix: str | None = None,
) -> None:
    prefix = member_prefix or episode_id
    manifest = json.dumps(
        {"schema_version": 2, "episode_id": manifest_id or episode_id}
    ).encode()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, payload in (
            (f"{prefix}/manifest.json", manifest),
            (f"{prefix}/samples.parquet", b"pq"),
        ):
            zf.writestr(name, payload)


def _jsonl(path: Path, image_paths: list[str]) -> None:
    path.write_text(
        "\n".join(
            json.dumps({"image_path": p, "binary_data_id": f"org/loc/{i}"})
            for i, p in enumerate(image_paths)
        )
    )


def test_ingest_unpacks_by_manifest_episode_id_and_hashes_content(
    tmp_path: Path,
) -> None:
    export = tmp_path / "ds-export"
    data = export / "data"
    data.mkdir(parents=True)
    # Cloud JSONL paths carry no extension (file_extension is stored separately on upload).
    _episode_zip(data / "episode-000002", "000002", member_prefix="episode-2")
    _episode_zip(data / "a.zip", "000001")
    (data / "notes.txt").write_text("ignored")
    dataset_file = export / "dataset.jsonl"
    # Absolute (cloud), root-relative (docs), and export-dir-prefixed (`viam dataset export`) paths.
    _jsonl(
        dataset_file,
        [str(data / "episode-000002"), "ds-export/data/a.zip", "data/notes.txt"],
    )

    result = ingest_episodes(dataset_file, tmp_path / "episodes")

    assert sorted(p.name for p in (tmp_path / "episodes").iterdir()) == [
        "000001",
        "000002",
    ]
    assert (tmp_path / "episodes/000002/manifest.json").is_file()
    assert [r["episode_id"] for r in result["episodes"]] == ["000001", "000002"]
    assert result["episodes"][0]["binary_data_id"] == "org/loc/1"
    assert len(result["content_sha256"]) == 64

    again = ingest_episodes(dataset_file, tmp_path / "episodes2")
    assert again["content_sha256"] == result["content_sha256"]


def test_ingest_rejects_duplicates_and_unsafe_members(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    _episode_zip(data / "a.zip", "000001")
    _episode_zip(data / "dup.zip", "000001", member_prefix="other")
    dataset_file = tmp_path / "dataset.jsonl"
    _jsonl(dataset_file, [str(data / "a.zip"), str(data / "dup.zip")])
    with pytest.raises(ValueError, match="duplicate episode_id"):
        ingest_episodes(dataset_file, tmp_path / "episodes")

    evil = data / "evil.zip"
    with zipfile.ZipFile(evil, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("../escape/manifest.json", b"{}")
    _jsonl(dataset_file, [str(evil)])
    with pytest.raises(ValueError, match="unsafe member"):
        ingest_episodes(dataset_file, tmp_path / "episodes-evil")


def test_parse_args_defaults_warmup_and_outcomes() -> None:
    args = parse_args(
        [
            "--dataset_file=d.jsonl",
            "--model_output_directory=out",
            "--max_iter=100",
            "--outcomes=success failure",
        ]
    )
    assert args.warmup_steps == 5
    assert args.outcomes == frozenset({"success", "failure"})
    assert args.stage == "train"
    assert args.save_iter is None
    assert args.num_workers == 4
    assert args.grad_accum == 2
    assert args.train_lr == pytest.approx(2.5e-5)
    assert args.train_lr_end_factor == pytest.approx(0.2)
    assert args.compile_vae is True


def test_parse_args_accepts_training_tunables() -> None:
    args = parse_args(
        [
            "--dataset_file=d.jsonl",
            "--model_output_directory=out",
            "--max_iter=100",
            "--save_iter=25",
            "--num_workers=8",
            "--grad_accum=2",
            "--train_lr=1e-4",
            "--train_lr_end_factor=0.1",
            "--compile_vae=true",
        ]
    )
    assert args.save_iter == 25
    assert args.num_workers == 8
    assert args.grad_accum == 2
    assert args.train_lr == pytest.approx(1e-4)
    assert args.train_lr_end_factor == pytest.approx(0.1)
    assert args.compile_vae is True


@pytest.mark.parametrize("value", ["false", "0", "no", "off", "False"])
def test_parse_args_compile_vae_can_be_disabled(value: str) -> None:
    args = parse_args(
        [
            "--dataset_file=d.jsonl",
            "--model_output_directory=out",
            "--max_iter=100",
            f"--compile_vae={value}",
        ]
    )
    assert args.compile_vae is False


def test_parse_args_compile_vae_rejects_garbage() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--dataset_file=d.jsonl",
                "--model_output_directory=out",
                "--max_iter=100",
                "--compile_vae=sure",
            ]
        )


@pytest.mark.parametrize(
    "override,message",
    [
        (["--save_iter=0"], "save_iter"),
        (["--num_workers=0"], "num_workers"),
        (["--grad_accum=0"], "grad_accum"),
        (["--train_lr=0"], "train_lr"),
        (["--train_lr_end_factor=1.5"], "train_lr_end_factor"),
    ],
)
def test_parse_args_rejects_out_of_range_tunables(
    override: list[str], message: str
) -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--dataset_file=d.jsonl",
                "--model_output_directory=out",
                "--max_iter=10",
                *override,
            ]
        )


def test_train_command_defaults(tmp_path: Path) -> None:
    cmd = train_command(tmp_path, gpus=2, max_iter=100, warmup_steps=5, fps=15.0)
    assert "trainer.max_iter=100" in cmd
    assert "checkpoint.save_iter=100" in cmd
    assert "dataloader_train.dataloader.num_workers=4" in cmd
    assert "dataloader_train.dataloader.batch_size=1" in cmd
    assert "trainer.grad_accum_iter=2" in cmd
    assert "optimizer.lr=2.5e-05" in cmd
    assert "scheduler.f_min=[0.2]" in cmd
    assert "trainer.callbacks.compile_tokenizer.enabled=True" in cmd
    assert "dataloader_train.dataloader.datasets.droid.dataset.fps=15.0" in cmd


def test_train_command_rejects_non_positive_fps(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fps"):
        train_command(tmp_path, gpus=1, max_iter=10, warmup_steps=1, fps=0)


def test_train_command_opt_out_overrides(tmp_path: Path) -> None:
    cmd = train_command(
        tmp_path,
        gpus=2,
        max_iter=100,
        warmup_steps=5,
        fps=20.0,
        grad_accum=None,
        train_lr=None,
        train_lr_end_factor=None,
        compile_vae=False,
    )
    assert not any(flag.startswith("trainer.grad_accum_iter=") for flag in cmd)
    assert not any(flag.startswith("optimizer.lr=") for flag in cmd)
    assert not any(flag.startswith("scheduler.f_min=") for flag in cmd)
    assert not any("compile_tokenizer" in flag for flag in cmd)


def test_train_command_appends_overrides_when_supplied(tmp_path: Path) -> None:
    cmd = train_command(
        tmp_path,
        gpus=4,
        max_iter=200,
        warmup_steps=10,
        fps=20.0,
        save_iter=50,
        num_workers=8,
        grad_accum=2,
        train_lr=1e-4,
        train_lr_end_factor=0.1,
        compile_vae=True,
    )
    assert "checkpoint.save_iter=50" in cmd
    assert "dataloader_train.dataloader.num_workers=8" in cmd
    assert "trainer.grad_accum_iter=2" in cmd
    assert "optimizer.lr=0.0001" in cmd
    assert "scheduler.f_min=[0.1]" in cmd
    assert "trainer.callbacks.compile_tokenizer.enabled=True" in cmd
    assert "trainer.callbacks.compile_tokenizer.warmup_resolutions=['480']" in cmd


def test_training_environment_sets_tokenizer_thread_limits(tmp_path: Path) -> None:
    env = training_environment(
        base_checkpoint=tmp_path / "ckpt",
        vae=tmp_path / "vae.pth",
        droid_root=tmp_path / "droid",
        output_root=tmp_path / "out",
    )
    assert env["TOKENIZERS_PARALLELISM"] == "false"
    assert env["RAYON_NUM_THREADS"] == "1"


def test_driver_library_path_is_driver_mount_plus_venv_nvidia_wheels(
    tmp_path: Path,
) -> None:
    npp = tmp_path / ".venv/lib/python3.13/site-packages/nvidia/npp/lib"
    cublas = tmp_path / ".venv/lib/python3.13/site-packages/nvidia/cublas/lib"
    for directory in (npp, cublas):
        directory.mkdir(parents=True)
    assert driver_library_path(tmp_path) == os.pathsep.join(
        ["/usr/local/nvidia/lib64", "/usr/local/nvidia/lib", str(cublas), str(npp)]
    )
    assert (
        driver_library_path(tmp_path / "missing")
        == "/usr/local/nvidia/lib64:/usr/local/nvidia/lib"
    )


def test_safe_extract_rejects_links_and_absolute_members(tmp_path: Path) -> None:
    def zip_with(name: str, info: zipfile.ZipInfo, payload: bytes = b"") -> Path:
        path = tmp_path / name
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(info, payload)
        return path

    link = zipfile.ZipInfo("ep/manifest.json")
    link.create_system = 3  # Unix, so external_attr's high bits are mode
    link.external_attr = (0o120777) << 16  # S_IFLNK
    absolute = zipfile.ZipInfo("/etc/x")
    dataset_file = tmp_path / "dataset.jsonl"
    for archive, message in (
        (zip_with("link.zip", link, b"/etc/passwd"), "links are not allowed"),
        (zip_with("abs.zip", absolute), "unsafe member"),
    ):
        _jsonl(dataset_file, [str(archive)])
        with pytest.raises(ValueError, match=message):
            ingest_episodes(dataset_file, tmp_path / f"out-{archive.stem}")


def test_ingest_rejects_episode_id_with_path_separators(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    _episode_zip(data / "a.zip", "000001", manifest_id="../escape")
    dataset_file = tmp_path / "dataset.jsonl"
    _jsonl(dataset_file, [str(data / "a.zip")])
    with pytest.raises(ValueError, match="not a plain name"):
        ingest_episodes(dataset_file, tmp_path / "episodes")


def _fake_framework(root: Path) -> Path:
    """The four files patch_framework_for_edge edits, holding only the pinned text it looks for."""
    package = root / "cosmos_framework"
    posttrain = package / "configs/base/experiment/action/posttrain_config"
    posttrain.mkdir(parents=True)
    (posttrain / "action_policy_droid_nano.py").write_text(
        "NAME = 'action_policy_droid_nano'\n"
    )
    (package / "configs/base/config.py").write_text(
        "    import cosmos_framework.configs.base.experiment.action.posttrain_config."
        "action_policy_droid_nano  # noqa: F401\n"
    )
    (package / "scripts").mkdir()
    (package / "scripts/export_model.py").write_text(
        """    edge_policy_metadata = (
        _build_edge_policy_metadata(checkpoint_args.load_config())
        if is_edge and model_dict["config"].get("action_gen")
        else None
    )
"""
    )
    dataset = package / "data/generator/action/datasets"
    dataset.mkdir(parents=True)
    (dataset / "action_sft_dataset.py").write_text(
        "    dataset: Dataset = DROIDLeRobotDataset(root=root, **shard_kwargs)\n"
    )
    return package / "scripts/export_model.py"


def test_patch_framework_records_the_dataset_fps_in_the_export(tmp_path: Path) -> None:
    exporter = _fake_framework(tmp_path)

    patch_framework_for_edge(tmp_path, fps=15.0)
    patch_framework_for_edge(tmp_path, fps=15.0)  # idempotent

    text = exporter.read_text()
    assert text.count('"conditioning_fps": 15.0') == 1
    assert "_build_edge_policy_metadata" not in text


def test_patch_framework_replaces_an_earlier_runs_fps(tmp_path: Path) -> None:
    exporter = _fake_framework(tmp_path)
    patch_framework_for_edge(tmp_path, fps=20.0)

    patch_framework_for_edge(tmp_path, fps=15.0)

    text = exporter.read_text()
    assert '"conditioning_fps": 15.0' in text
    assert '"conditioning_fps": 20.0' not in text
