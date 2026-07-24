from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_stage2_e5_reserves_gpu7_from_grpo() -> None:
    script = (ROOT / "searchr1_stage2" / "run_single_stack_grpo.sh").read_text(
        encoding="utf-8"
    )

    assert "DEFAULT_TRAIN_GPUS=0,1,2,3,4,5,6" in script
    assert "DEFAULT_N_GPUS=7" in script
    assert 'E5_GPU=${E5_GPU:-7}' in script
    assert 'export CUDA_VISIBLE_DEVICES=$TRAIN_GPUS' in script
    assert 'trainer.n_gpus_per_node="$N_GPUS"' in script
    assert 'E5_GPU="$E5_GPU" RETRIEVER_BACKENDS="$BACKEND"' in script
    assert "trainer.n_gpus_per_node=8" not in script
    assert "GPUs=$GPU_IDS" not in script


def test_stage2_run_all_selects_backend_specific_gpu_layouts() -> None:
    script = (ROOT / "searchr1_stage2" / "run_all.sh").read_text(
        encoding="utf-8"
    )

    assert "train_gpus=0,1,2,3,4,5,6" in script
    assert "train_gpus=0,1,2,3,4,5,6,7" in script
    assert 'TRAIN_GPUS="$train_gpus" N_GPUS="$n_gpus" E5_GPU=7' in script


def test_stage2_signature_tracks_gpu_geometry() -> None:
    script = (ROOT / "searchr1_stage2" / "run_single_stack_grpo.sh").read_text(
        encoding="utf-8"
    )

    assert '"schema": 7' in script
    assert '"train_gpus": train_gpus' in script
    assert '"n_gpus": int(n_gpus)' in script
    assert '"e5_gpu": int(e5_gpu) if backend == "e5" else None' in script
    assert '"log_prob_micro_batch": int(log_prob_micro_batch)' in script


def test_stage2_backends_keep_the_same_global_batch_protocol() -> None:
    script = (ROOT / "searchr1_stage2" / "run_single_stack_grpo.sh").read_text(
        encoding="utf-8"
    )

    assert "TRAIN_BATCH=${TRAIN_BATCH:-56}" in script
    assert "VAL_BATCH=${VAL_BATCH:-56}" in script
    assert "TRAIN_BATCH=${TRAIN_BATCH:-112}" in script
    assert "VAL_BATCH=${VAL_BATCH:-112}" in script
    assert "MINI_BATCH=${MINI_BATCH:-56}" in script
    assert "VAL_DATA_NUM=${VAL_DATA_NUM:-$VAL_BATCH}" in script
    assert "$((16 * N_GPUS))" not in script


def test_stage2_smoke_allows_a_partial_validation_batch() -> None:
    script = (ROOT / "searchr1_stage2" / "run_single_stack_grpo.sh").read_text(
        encoding="utf-8"
    )

    assert "if val_used < 1:" in script
    assert "if val_used < val_batch:" not in script


def test_stage2_clears_numbered_experiment_environment() -> None:
    single = (ROOT / "searchr1_stage2" / "run_single_stack_grpo.sh").read_text(
        encoding="utf-8"
    )
    full = (ROOT / "searchr1_stage2" / "run_all.sh").read_text(encoding="utf-8")

    for name in (
        "SEARCH_R1_MIXED_MODE",
        "SEARCH_R1_N_AGENT",
        "ANSWER_REWARD_WEIGHT",
        "EVIDENCE_REWARD_WEIGHT",
        "SEARCH_COST_WEIGHT",
    ):
        assert name in single
        assert name in full
    assert "export SEARCH_R1_REWARD_MODE=answer" in single
