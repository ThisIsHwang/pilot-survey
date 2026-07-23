from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import textwrap
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from stackpilot.hard_assets import (
    E5_INDEX_SIZE,
    E5_PARTS,
    EXPECTED_DOCUMENTS,
    check as check_hard_assets,
    decompress_gzip_counted,
    download as download_hard_assets,
    download_bm25,
    ensure_bm25_link,
    repair_e5_assembly_prefix,
    required_free_gib,
    source_identity,
)

from stackpilot.hard_query_analysis import main as query_analysis_main
from stackpilot.hard_query_report import main as query_report_main
from stackpilot.hard_policy_eval import (
    atomic_write_jsonl,
    balanced_limit,
    check_retriever,
    evaluation_context,
    parallel_job_results,
    prepare_result_cache,
    recall_at,
    run_signature,
)
from stackpilot.hard_rq0_report import (
    crossed_cluster_bootstrap,
    gain_over_base,
    home_excess,
    matched_hard_question_ids,
)
from stackpilot.hard_rq0_contract import (
    RESULT_SCHEMA,
    validate_policy_seed,
    validate_policy_selection,
)
from stackpilot.normalize_hard_results import normalize
from stackpilot.prepare_hard_rq0 import (
    DATA_PREP_SCHEMA,
    artifact_records,
    atomic_write_json,
    expected_artifacts,
    extract_support_titles,
    prepare_request,
    prepared_cache_valid,
    to_searchr1_row,
)
from stackpilot.retrieval_clients import normalize_document
from stackpilot.retrieval_concurrency import batch_search
from stackpilot.validate_hard_results import validate_frame


class HardRQ0Tests(unittest.TestCase):
    def test_hard_evaluation_runs_a_bounded_concurrent_window(self) -> None:
        barrier = threading.Barrier(4)
        state_lock = threading.Lock()
        active = 0
        peak = 0

        def worker(job: int) -> int:
            nonlocal active, peak
            with state_lock:
                active += 1
                peak = max(peak, active)
            if job < 4:
                barrier.wait(timeout=2)
            with state_lock:
                active -= 1
            return job * 2

        completed = dict(
            parallel_job_results(
                range(12), worker, max_workers=4, max_in_flight=4
            )
        )

        self.assertEqual(completed, {job: job * 2 for job in range(12)})
        self.assertEqual(peak, 4)
        self.assertFalse(
            [
                thread.name
                for thread in threading.enumerate()
                if thread.name.startswith("hard-rq0-eval-")
            ],
            "normal completion must join every evaluation worker",
        )
        with self.assertRaisesRegex(ValueError, "at least max_workers"):
            list(
                parallel_job_results(
                    [1], lambda value: value, max_workers=1, max_in_flight=0
                )
            )

    def test_parallel_evaluation_yields_successes_before_peer_error(self) -> None:
        barrier = threading.Barrier(2)
        success_finished = threading.Event()

        def worker(job: int) -> int:
            barrier.wait(timeout=2)
            if job == 1:
                if not success_finished.wait(timeout=2):
                    raise AssertionError("successful peer did not finish")
                raise RuntimeError("failed job")
            success_finished.set()
            return job

        completed = []
        with self.assertRaisesRegex(RuntimeError, "failed job"):
            for item in parallel_job_results(
                [0, 1], worker, max_workers=2, max_in_flight=2
            ):
                completed.append(item)
        self.assertEqual(completed, [(0, 0)])

    def test_parallel_evaluation_does_not_start_queued_jobs_after_error(
        self,
    ) -> None:
        started = []

        def worker(job: int) -> int:
            started.append(job)
            if job == 0:
                raise RuntimeError("failed first job")
            return job

        with self.assertRaisesRegex(RuntimeError, "failed first job"):
            list(
                parallel_job_results(
                    range(8), worker, max_workers=1, max_in_flight=4
                )
            )
        self.assertEqual(started, [0])

    def test_parallel_evaluation_aborts_blocked_daemon_workers_in_subprocess(
        self,
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        script = textwrap.dedent(
            """
            import _thread
            import sys
            import threading
            import time

            from stackpilot.hard_policy_eval import parallel_job_results

            mode = sys.argv[1]
            started = threading.Event()
            never = threading.Event()

            def worker(job):
                if job == "block":
                    started.set()
                    never.wait()
                    return job
                if not started.wait(timeout=2):
                    raise RuntimeError("blocking peer did not start")
                if job == "error":
                    raise RuntimeError("intentional worker failure")
                return job

            if mode == "error":
                try:
                    list(
                        parallel_job_results(
                            ["block", "error"],
                            worker,
                            max_workers=2,
                            max_in_flight=2,
                        )
                    )
                except RuntimeError:
                    pass
                else:
                    raise SystemExit("worker failure was not propagated")
            elif mode == "close":
                results = parallel_job_results(
                    ["block", "success"],
                    worker,
                    max_workers=2,
                    max_in_flight=2,
                )
                if next(results) != ("success", "success"):
                    raise SystemExit("successful peer result was not yielded")
                results.close()
            elif mode == "keyboard":
                def interrupt():
                    if not started.wait(timeout=2):
                        return
                    time.sleep(0.05)
                    _thread.interrupt_main()

                threading.Thread(target=interrupt, daemon=True).start()
                try:
                    list(
                        parallel_job_results(
                            ["block"],
                            worker,
                            max_workers=1,
                            max_in_flight=1,
                        )
                    )
                except KeyboardInterrupt:
                    pass
                else:
                    raise SystemExit("KeyboardInterrupt was not propagated")
            else:
                raise SystemExit(f"unknown mode: {mode}")

            print(f"aborted:{mode}", flush=True)
            """
        )
        for mode in ("error", "close", "keyboard"):
            with self.subTest(mode=mode):
                completed = subprocess.run(
                    [sys.executable, "-c", script, mode],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                self.assertEqual(
                    completed.returncode,
                    0,
                    completed.stdout + completed.stderr,
                )
                self.assertIn(f"aborted:{mode}", completed.stdout)

    def test_gpu_faiss_server_serializes_searches(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        second_started = threading.Event()

        class FakeRetriever:
            def __init__(self) -> None:
                self.calls = 0

            def batch_search(self, queries, topk, return_scores):
                self.calls += 1
                if self.calls == 1:
                    entered.set()
                    self.assertions(queries, topk, return_scores)
                    self.wait_for_release()
                return [[{"id": self.calls}]], [[1.0]]

            @staticmethod
            def assertions(queries, topk, return_scores) -> None:
                if queries != ["q"] or topk != 3 or return_scores is not True:
                    raise AssertionError((queries, topk, return_scores))

            @staticmethod
            def wait_for_release() -> None:
                if not release.wait(timeout=2):
                    raise AssertionError("test did not release the first search")

        retriever = FakeRetriever()
        search_lock = threading.Lock()

        def second_search():
            second_started.set()
            return batch_search(retriever, ["q"], 3, search_lock)

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(
                batch_search, retriever, ["q"], 3, search_lock
            )
            self.assertTrue(entered.wait(timeout=2))
            second = executor.submit(second_search)
            self.assertTrue(second_started.wait(timeout=2))
            self.assertEqual(retriever.calls, 1)
            release.set()
            first.result(timeout=2)
            second.result(timeout=2)
        self.assertEqual(retriever.calls, 2)

    def test_h100_evaluation_and_prefetch_defaults_are_wired(self) -> None:
        root = Path(__file__).resolve().parents[1]
        evaluator = (root / "hard_rq0" / "eval_policy.sh").read_text("utf-8")
        launcher = (root / "scripts" / "lib" / "vllm_launch.sh").read_text(
            "utf-8"
        )
        pipeline = (root / "scripts" / "run_full_pipeline.sh").read_text(
            "utf-8"
        )

        self.assertIn("LLM_GPUS=${LLM_GPUS:-0,1,2,3,4,5,6}", evaluator)
        self.assertIn("TP=${TP:-1}", evaluator)
        self.assertIn("DP=${DP:-7}", evaluator)
        self.assertIn("HARD_EVAL_WORKERS=${HARD_EVAL_WORKERS:-112}", evaluator)
        self.assertIn("VLLM_BATCH_INVARIANT=${VLLM_BATCH_INVARIANT:-1}", evaluator)
        self.assertIn('--data-parallel-size "$DP"', launcher)
        self.assertIn('--api-server-count "$VLLM_API_SERVER_COUNT"', launcher)
        self.assertLess(
            pipeline.index(
                'bash "$ROOT/scripts/prefetch_future_models.sh" --stage2'
            ),
            pipeline.index('if [[ "$RUN_STAGE0" == 1 ]]'),
        )
        stage2_boundary = pipeline.index(
            'SKIP_BOOTSTRAP=1 bash "$ROOT/searchr1_stage2/run_all.sh"'
        )
        self.assertLess(
            pipeline.index("wait_background_job STAGE2_MODEL_PREFETCH_PID"),
            stage2_boundary,
        )
        self.assertGreater(
            pipeline.index("wait_background_job HARD_MODEL_PREFETCH_PID"),
            stage2_boundary,
        )
        self.assertIn("SEARCHR1_DEFER_GPU_PROBE=1", pipeline)
        self.assertIn("CUDA_VISIBLE_DEVICES=", pipeline)
        self.assertIn('flock -n "$PIPELINE_LOCK_FD"', pipeline)
        self.assertIn('"$ROOT/scripts/session_runner.py"', pipeline)
        self.assertLess(
            pipeline.index('bash "$ROOT/hard_rq0/preflight_storage.sh"'),
            pipeline.index("start_background_job HARD_ASSET_PREFETCH_PID"),
        )

    def test_specialist_rollout_uses_pinned_searchr1_retrieval_budget(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = (root / "hard_rq0" / "train_specialist.sh").read_text(
            encoding="utf-8"
        )
        expected_constants = (
            "readonly MAX_PROMPT_LENGTH=4096",
            "readonly MAX_RESPONSE_LENGTH=500",
            "readonly MAX_START_LENGTH=2048",
            "readonly MAX_OBS_LENGTH=500",
            "readonly MAX_TURNS=4",
            "TOPK=${TOPK:-3}",
        )
        for assignment in expected_constants:
            self.assertIn(assignment, script)

        expected_overrides = (
            'data.max_prompt_length="$MAX_PROMPT_LENGTH"',
            'data.max_response_length="$MAX_RESPONSE_LENGTH"',
            'data.max_start_length="$MAX_START_LENGTH"',
            'data.max_obs_length="$MAX_OBS_LENGTH"',
            'max_turns="$MAX_TURNS"',
            'retriever.topk="$TOPK"',
        )
        for override in expected_overrides:
            self.assertIn(override, script)

        self.assertNotIn("data.max_obs_length=700", script)
        self.assertIn('"max_obs_length": int(max_obs_length)', script)
        self.assertIn('"schema": 2,', script)

        merger = (root / "hard_rq0" / "merge_specialist.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('if payload.get("schema") != 2:', merger)
        self.assertIn('"max_obs_length": 500', merger)
        self.assertIn(
            'if payload.get("rollout_protocol") != expected_rollout_protocol:',
            merger,
        )

    def test_hard_asset_contract_is_pinned_and_never_adopts_legacy_files(self) -> None:
        self.assertEqual(E5_INDEX_SIZE, sum(part[1] for part in E5_PARTS))
        self.assertEqual(source_identity()["corpus"]["documents"], EXPECTED_DOCUMENTS)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "manifest is missing"):
                check_hard_assets(Path(temporary), adopt_legacy=True)

    def test_interrupted_e5_part_keeps_the_verified_prefix(self) -> None:
        parts = (
            ("part_a", 4, "a" * 64),
            ("part_b", 6, "b" * 64),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / ".assembling"
            path.write_bytes(b"aaaabbb")

            completed, prefix = repair_e5_assembly_prefix(
                path, ["part_a"], parts
            )

            self.assertEqual(completed, ["part_a"])
            self.assertEqual(prefix, 4)
            self.assertEqual(path.read_bytes(), b"aaaa")

    def test_counted_corpus_decompression_requires_complete_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "sample.jsonl.gz"
            target = root / "sample.jsonl"
            payload = b'{"id": 1}\n{"id": 2}\n'
            with gzip.open(source, "wb") as handle:
                handle.write(payload)
            copied, rows = decompress_gzip_counted(source, target)
            self.assertEqual((copied, rows), (len(payload), 2))
            self.assertEqual(target.read_bytes(), payload)

    def test_hard_asset_download_reuses_completed_components(self) -> None:
        corpus = {"path": "wiki-18.jsonl", "size": 1}
        bm25 = {"path": "bm25-pinned-revision/bm25", "size": 2}
        e5 = {"path": "e5_Flat.index", "size": 3}
        checked = {
            "schema": 2,
            "sources": source_identity(),
            "artifacts": {"corpus": corpus, "bm25": bm25, "e5": e5},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                patch(
                    "stackpilot.hard_assets.reusable_artifacts",
                    return_value=(
                        {"corpus": corpus, "bm25": bm25},
                        {"e5": "missing"},
                    ),
                ),
                patch("stackpilot.hard_assets.ensure_free_space") as free_space,
                patch("stackpilot.hard_assets.ensure_bm25_link") as repair_link,
                patch("stackpilot.hard_assets.decompress_corpus") as corpus_build,
                patch("stackpilot.hard_assets.download_bm25") as bm25_build,
                patch(
                    "stackpilot.hard_assets.assemble_e5", return_value=e5
                ) as e5_build,
                patch("stackpilot.hard_assets.check", return_value=checked),
            ):
                result = download_hard_assets(
                    root, min_free_gib=150, keep_sources=False
                )

            self.assertEqual(result, checked)
            corpus_build.assert_not_called()
            bm25_build.assert_not_called()
            e5_build.assert_called_once_with(root, keep_sources=False)
            free_space.assert_called_once_with(root, 150)
            repair_link.assert_called_once_with(root, root / bm25["path"])
            manifest = json.loads(
                (root / ".hard-rq0-assets-manifest.json").read_text("utf-8")
            )
            self.assertEqual(manifest["artifacts"], checked["artifacts"])
            self.assertFalse((root / ".hard-rq0-assets-incomplete").exists())

    def test_hard_asset_manifest_keeps_progress_after_later_failure(self) -> None:
        corpus = {"path": "wiki-18.jsonl", "size": 1}
        bm25 = {"path": "bm25-pinned-revision/bm25", "size": 2}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                patch(
                    "stackpilot.hard_assets.reusable_artifacts",
                    return_value=(
                        {"corpus": corpus},
                        {"bm25": "missing", "e5": "missing"},
                    ),
                ),
                patch("stackpilot.hard_assets.ensure_free_space"),
                patch("stackpilot.hard_assets.ensure_bm25_link"),
                patch("stackpilot.hard_assets.decompress_corpus") as corpus_build,
                patch("stackpilot.hard_assets.download_bm25", return_value=bm25),
                patch(
                    "stackpilot.hard_assets.assemble_e5",
                    side_effect=RuntimeError("interrupted"),
                ),
                self.assertRaisesRegex(RuntimeError, "interrupted"),
            ):
                download_hard_assets(root, min_free_gib=150, keep_sources=False)

            corpus_build.assert_not_called()
            manifest = json.loads(
                (root / ".hard-rq0-assets-manifest.json").read_text("utf-8")
            )
            self.assertEqual(manifest["artifacts"], {"corpus": corpus, "bm25": bm25})
            self.assertTrue((root / ".hard-rq0-assets-incomplete").is_file())

    def test_required_free_space_tracks_only_missing_components(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = (
                ({"corpus": {}, "bm25": {}, "e5": {}}, 0),
                ({"corpus": {}, "bm25": {}}, 150),
                ({"bm25": {}, "e5": {}}, 40),
                ({"corpus": {}, "e5": {}}, 10),
            )
            for reusable, expected in cases:
                with (
                    self.subTest(reusable=set(reusable)),
                    patch(
                        "stackpilot.hard_assets.reusable_artifacts",
                        return_value=(reusable, {}),
                    ),
                ):
                    self.assertEqual(
                        required_free_gib(root, full_min_gib=150), expected
                    )

    def test_bm25_download_keeps_stable_staging_after_interruption(self) -> None:
        fake_hub = types.ModuleType("huggingface_hub")
        fake_hub.snapshot_download = Mock(side_effect=RuntimeError("network stopped"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                patch.dict(sys.modules, {"huggingface_hub": fake_hub}),
                self.assertRaisesRegex(RuntimeError, "network stopped"),
            ):
                download_bm25(root)
            staging = root / f".bm25-download-{source_identity()['bm25']['revision']}"
            self.assertTrue(staging.is_dir())

    @unittest.skipIf(os.name == "nt", "directory symlink creation targets Linux")
    def test_bm25_compatibility_link_is_repaired(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            installed = root / "bm25-pinned-test" / "bm25"
            installed.mkdir(parents=True)
            stale = root / "bm25"
            stale.mkdir()
            (stale / "legacy").write_text("old", encoding="utf-8")

            ensure_bm25_link(root, installed)

            self.assertTrue(stale.is_symlink())
            self.assertEqual(stale.resolve(), installed.resolve())

    def test_crossed_bootstrap_requires_a_complete_seed_question_grid(self) -> None:
        frame = pd.DataFrame(
            [
                {"seed": seed, "question_id": question, "value": seed + question}
                for seed in (1, 2)
                for question in (10, 20)
            ]
        )
        observed, low, high = crossed_cluster_bootstrap(
            frame, "value", 200, np.random.default_rng(7)
        )
        self.assertEqual(observed, 16.5)
        self.assertLessEqual(low, observed)
        self.assertGreaterEqual(high, observed)

        incomplete = frame.iloc[:-1]
        with self.assertRaisesRegex(RuntimeError, "complete crossed"):
            crossed_cluster_bootstrap(incomplete, "value", 10, np.random.default_rng(7))

    def test_retriever_health_requires_the_full_pinned_corpus(self) -> None:
        from stackpilot.hard_assets import EXPECTED_DOCUMENTS

        source_root = Path(__file__).resolve().parents[1] / "stackpilot"
        server_files = {
            name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
            for name in ("retrieval_concurrency.py", "searchr1_server.py")
        }
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "status": "ok",
            "backend": "bm25",
            "index_path": "/idx/bm25",
            "corpus_path": "/data/wiki-18.jsonl",
            "index_documents": EXPECTED_DOCUMENTS,
            "corpus_documents": EXPECTED_DOCUMENTS,
            "server_files": server_files,
        }
        with patch("stackpilot.hard_policy_eval.requests.get", return_value=response):
            identity = check_retriever(
                "bm25", 8101, Path("/idx/bm25"), Path("/data/wiki-18.jsonl")
            )
        self.assertEqual(identity["index_documents"], EXPECTED_DOCUMENTS)

        response.json.return_value["index_documents"] -= 1
        with (
            patch("stackpilot.hard_policy_eval.requests.get", return_value=response),
            self.assertRaisesRegex(RuntimeError, "expected 21,015,324"),
        ):
            check_retriever(
                "bm25", 8101, Path("/idx/bm25"), Path("/data/wiki-18.jsonl")
            )

        response.json.return_value.update(
            {
                "backend": "e5",
                "index_path": "/idx/e5",
                "index_documents": EXPECTED_DOCUMENTS,
                "faiss_gpu": True,
                "faiss_gpu_count": 1,
                "gpu_search_serialized": True,
                "cuda_empty_cache_disabled": True,
                "retriever_model": "/models/e5/snapshots/revision",
                "retriever_model_revision": "revision",
            }
        )
        with patch("stackpilot.hard_policy_eval.requests.get", return_value=response):
            identity = check_retriever(
                "e5", 8102, Path("/idx/e5"), Path("/data/wiki-18.jsonl")
            )
        self.assertTrue(identity["gpu_search_serialized"])
        self.assertTrue(identity["cuda_empty_cache_disabled"])

    def test_evaluation_limit_is_balanced_across_datasets(self) -> None:
        rows = [{"id": f"a:{index}", "dataset": "a"} for index in range(5)] + [
            {"id": f"b:{index}", "dataset": "b"} for index in range(5)
        ]

        selected = balanced_limit(rows, 5)

        self.assertEqual(
            [row["id"] for row in selected], ["a:0", "b:0", "a:1", "b:1", "a:2"]
        )

    def test_searchr1_group_ids_are_globally_unique_question_ids(self) -> None:
        rows = [
            {
                "id": "2wikimultihopqa:q1",
                "dataset": "2wikimultihopqa",
                "split": "train",
                "question": "Question one?",
                "answers": ["one"],
                "support_titles": ["One"],
            },
            {
                "id": "musique:q1",
                "dataset": "musique",
                "split": "train",
                "question": "Question two?",
                "answers": ["two"],
                "support_titles": ["Two"],
            },
        ]

        converted = [to_searchr1_row(row) for row in rows]

        group_ids = [row["extra_info"]["index"] for row in converted]
        self.assertEqual(group_ids, ["2wikimultihopqa:q1", "musique:q1"])
        self.assertEqual(len(group_ids), len(set(group_ids)))

    @staticmethod
    def valid_result_frame() -> pd.DataFrame:
        rows = []
        policies = [("base-qwen", 0)] + [
            (tag, seed)
            for tag in ("bm25-specialist", "e5-specialist")
            for seed in (13, 42, 87)
        ]
        for tag, seed in policies:
            for backend in ("bm25", "e5"):
                rows.append(
                    {
                        "schema": RESULT_SCHEMA,
                        "policy_tag": tag,
                        "seed": seed,
                        "run_signature": f"run-{tag}-{seed}",
                        "evaluation_signature": "shared-evaluation",
                        "question_id": "toy:q1",
                        "dataset": "toy",
                        "backend": backend,
                        "topk": 3,
                        "em": 0.0,
                        "f1": 0.25,
                        "support_recall": 0.5,
                        "turn1_support_recall": 0.0,
                        "turn2_support_recall": 0.5,
                        "turn3_support_recall": 0.5,
                        "turn2_evidence_gain": 0.5,
                        "turn3_evidence_gain": 0.0,
                        "recovery_at_2": 1.0,
                        "recovery_at_3": 1.0,
                        "full_recovery_at_2": 0.0,
                        "full_recovery_at_3": 0.0,
                        "search_count": 2.0,
                        "question": "Toy question?",
                        "answers": ["toy"],
                        "queries": ["first query", "second query"],
                        "turns": [
                            {
                                "turn": 1,
                                "query": "first query",
                                "retrieved_titles": [],
                                "new_support_titles": [],
                                "support_recall": 0.0,
                                "evidence_gain": 0.0,
                                "query_token_count": 2.0,
                                "query_question_overlap": 0.0,
                                "query_has_quotes": 0.0,
                                "query_capitalized_ratio": 0.0,
                                "query_numeric_ratio": 0.0,
                                "query_lexical_change": 1.0,
                            },
                            {
                                "turn": 2,
                                "query": "second query",
                                "retrieved_titles": ["Toy"],
                                "new_support_titles": ["toy"],
                                "support_recall": 0.5,
                                "evidence_gain": 0.5,
                                "query_token_count": 2.0,
                                "query_question_overlap": 0.0,
                                "query_has_quotes": 0.0,
                                "query_capitalized_ratio": 0.0,
                                "query_numeric_ratio": 0.0,
                                "query_lexical_change": 1.0,
                            },
                        ],
                    }
                )
        return pd.DataFrame(rows)

    def test_gain_over_base_and_home_excess(self) -> None:
        rows = []
        scores = {
            ("base-qwen", 0, "bm25"): 0.40,
            ("base-qwen", 0, "e5"): 0.60,
            ("bm25-specialist", 13, "bm25"): 0.55,
            ("bm25-specialist", 13, "e5"): 0.65,
            ("e5-specialist", 13, "bm25"): 0.45,
            ("e5-specialist", 13, "e5"): 0.75,
        }
        for (tag, seed, backend), score in scores.items():
            rows.append(
                {
                    "subset": "all",
                    "policy_tag": tag,
                    "seed": seed,
                    "question_id": "q1",
                    "dataset": "toy",
                    "backend": backend,
                    "topk": 3,
                    "support_recall": score,
                }
            )
        frame = pd.DataFrame(rows)
        gains = gain_over_base(frame, "support_recall")
        interactions = home_excess(gains).set_index("policy_tag")
        self.assertAlmostEqual(
            float(interactions.loc["bm25-specialist", "home_excess_gain"]),
            0.10,
        )
        self.assertAlmostEqual(
            float(interactions.loc["e5-specialist", "home_excess_gain"]),
            0.10,
        )

    def test_matched_hard_requires_base_difficulty_and_recovery(self) -> None:
        rows = []
        for backend, first in (("bm25", 0.0), ("e5", 0.5)):
            rows.append(
                {
                    "policy_tag": "base-qwen",
                    "seed": 0,
                    "question_id": "q1",
                    "dataset": "toy",
                    "backend": backend,
                    "topk": 3,
                    "turn1_support_recall": first,
                    "turn3_support_recall": 1.0 if backend == "bm25" else first,
                }
            )
        rows.append(
            {
                "policy_tag": "bm25-specialist",
                "seed": 13,
                "question_id": "q1",
                "dataset": "toy",
                "backend": "bm25",
                "topk": 3,
                "turn1_support_recall": 0.0,
                "turn3_support_recall": 1.0,
            }
        )
        matched = matched_hard_question_ids(pd.DataFrame(rows))
        self.assertTrue(bool(matched.loc[0, "base_hard"]))
        self.assertTrue(bool(matched.loc[0, "recoverable"]))
        self.assertTrue(bool(matched.loc[0, "matched_hard"]))

        specialist_only = pd.DataFrame(rows).copy()
        specialist_only.loc[
            specialist_only["policy_tag"] == "base-qwen", "turn3_support_recall"
        ] = specialist_only.loc[
            specialist_only["policy_tag"] == "base-qwen", "turn1_support_recall"
        ]
        unmatched = matched_hard_question_ids(specialist_only)
        self.assertFalse(bool(unmatched.loc[0, "recoverable"]))

    def test_missing_turn_recall_is_carried_forward(self) -> None:
        row = normalize(
            {"turns": [{"turn": 1, "support_recall": 0.5, "evidence_gain": 0.5}]}
        )
        self.assertEqual(row["turn2_support_recall"], 0.5)
        self.assertEqual(row["turn3_support_recall"], 0.5)
        self.assertEqual(row["turn2_evidence_gain"], 0.0)
        self.assertEqual(row["turn3_evidence_gain"], 0.0)

    def test_wiki18_title_is_recovered_from_contents(self) -> None:
        title, text = normalize_document(
            {"id": "1", "contents": '"Story of Your Life"\nA novella by Ted Chiang.'}
        )
        self.assertEqual(title, "Story of Your Life")
        self.assertEqual(text, "A novella by Ted Chiang.")

    def test_musique_support_titles_are_extracted_from_decomposition(self) -> None:
        metadata = {
            "question_decomposition": [
                {
                    "support_paragraph": {
                        "title": "Green (Steve Hillage album)",
                        "is_supporting": True,
                    }
                },
                {
                    "support_paragraph": {
                        "title": "Miquette Giraudy",
                        "is_supporting": True,
                    }
                },
                {
                    "support_paragraph": {
                        "title": "Miquette Giraudy",
                        "is_supporting": True,
                    }
                },
                {
                    "support_paragraph": {
                        "title": "Distractor",
                        "is_supporting": False,
                    }
                },
            ]
        }
        self.assertEqual(
            extract_support_titles(metadata),
            ["Green (Steve Hillage album)", "Miquette Giraudy"],
        )

    def test_prepared_manifest_cache_detects_changes(self) -> None:
        config = {
            "seed": 42,
            "data": {
                "repo_id": "example/data",
                "revision": "a" * 40,
                "datasets": ["toy"],
                "train_examples_per_dataset": 2,
                "eval_examples_per_dataset": 1,
                "split_train": "train",
                "split_eval": "dev",
            },
        }
        request = prepare_request(config)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = expected_artifacts(request)
            for index, relative_path in enumerate(paths):
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"artifact-{index}\n", encoding="utf-8")
            manifest_path = root / "data" / ".hard-rq0-data-manifest.json"
            atomic_write_json(
                manifest_path,
                {
                    "schema": DATA_PREP_SCHEMA,
                    "request": request,
                    "artifacts": artifact_records(root, paths),
                },
            )
            self.assertTrue(prepared_cache_valid(manifest_path, root, request))
            (root / paths[0]).write_text("changed\n", encoding="utf-8")
            self.assertFalse(prepared_cache_valid(manifest_path, root, request))

    def test_prepare_request_requires_immutable_revision(self) -> None:
        config = {
            "seed": 42,
            "data": {
                "repo_id": "example/data",
                "revision": "main",
                "datasets": ["toy"],
                "train_examples_per_dataset": 2,
                "eval_examples_per_dataset": 1,
                "split_train": "train",
                "split_eval": "dev",
            },
        }
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "immutable 40-character"):
                prepare_request(config)

    def test_policy_selection_rejects_unsafe_or_ambiguous_inputs(self) -> None:
        with self.assertRaises(ValueError):
            validate_policy_selection("../escape", None, ("bm25", "e5"), (3,))
        with self.assertRaises(ValueError):
            validate_policy_selection("base-qwen", 0, ("bm25", "e5"), (3,))
        with self.assertRaises(ValueError):
            validate_policy_selection("base-qwen", None, ("bm25", "bm25"), (3,))
        with self.assertRaises(ValueError):
            validate_policy_selection("base-qwen", None, ("bm25", "e5"), (0,))
        with self.assertRaises(ValueError):
            validate_policy_seed("base-qwen", 13, (13, 42, 87))
        with self.assertRaises(ValueError):
            validate_policy_seed("bm25-specialist", 99, (13, 42, 87))

    def test_result_validator_enforces_exact_shared_grid(self) -> None:
        frame = self.valid_result_frame()
        validated, cells, evaluation_signature = validate_frame(
            frame,
            expected_topks=(3,),
            expected_datasets=("toy",),
        )
        self.assertEqual(len(validated), 14)
        self.assertEqual(cells, 2)
        self.assertEqual(evaluation_signature, "shared-evaluation")

        extra = frame.iloc[[0]].copy()
        extra["policy_tag"] = "stale-extra"
        with self.assertRaisesRegex(RuntimeError, "Expected policy tags"):
            validate_frame(
                pd.concat([frame, extra], ignore_index=True),
                expected_topks=(3,),
                expected_datasets=("toy",),
            )

        mixed = frame.copy()
        mixed.loc[mixed.index[-1], "evaluation_signature"] = "other-evaluation"
        with self.assertRaisesRegex(RuntimeError, "share one evaluation signature"):
            validate_frame(
                mixed,
                expected_topks=(3,),
                expected_datasets=("toy",),
            )

        nonfinite = frame.copy()
        nonfinite.loc[nonfinite.index[-1], "f1"] = np.nan
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            validate_frame(
                nonfinite,
                expected_topks=(3,),
                expected_datasets=("toy",),
            )

        inconsistent_episode = frame.copy(deep=True)
        bad_turns = [dict(turn) for turn in inconsistent_episode.iloc[-1]["turns"]]
        bad_turns[-1]["support_recall"] = 0.25
        inconsistent_episode.at[inconsistent_episode.index[-1], "turns"] = bad_turns
        with self.assertRaisesRegex(RuntimeError, "inconsistent"):
            validate_frame(
                inconsistent_episode,
                expected_topks=(3,),
                expected_datasets=("toy",),
            )

        missing_cell = frame.drop(frame.index[-1])
        with self.assertRaisesRegex(RuntimeError, "different evaluation grid"):
            validate_frame(
                missing_cell,
                expected_topks=(3,),
                expected_datasets=("toy",),
            )

    def test_query_reports_are_zero_search_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "policies"
            output = root / "report"
            results.mkdir()
            with patch.object(
                sys,
                "argv",
                [
                    "hard_query_analysis",
                    "--results-dir",
                    str(results),
                    "--output-dir",
                    str(output),
                ],
            ):
                query_analysis_main()
            summary = output / "query_turn_summary.csv"
            self.assertTrue(summary.is_file())
            self.assertTrue(pd.read_csv(summary).empty)
            report = output / "QUERY_BEHAVIOR.md"
            with patch.object(
                sys,
                "argv",
                [
                    "hard_query_report",
                    "--summary",
                    str(summary),
                    "--output",
                    str(report),
                ],
            ):
                query_report_main()
            self.assertIn("_No matched-hard query rows._", report.read_text("utf-8"))

    def test_atomic_manifest_is_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            atomic_write_json(path, {"schema": 1, "value": "ok"})
            self.assertEqual(json.loads(path.read_text("utf-8"))["value"], "ok")

    def test_result_cache_archives_stale_rows_and_compacts_current(self) -> None:
        current = self.valid_result_frame().iloc[0].to_dict()
        current["question"] = "Question?"
        current["prediction"] = "answer"
        stale = dict(current)
        stale["run_signature"] = "old-run"
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "base-qwen-seed0.jsonl"
            atomic_write_jsonl(output, [stale, current, current])
            cached = prepare_result_cache(
                output,
                [stale, current, current],
                {("toy:q1", "bm25", 3)},
                {"toy:q1": "toy"},
                "run-base-qwen-0",
                "shared-evaluation",
                "base-qwen",
                0,
            )
            self.assertEqual(set(cached), {("toy:q1", "bm25", 3)})
            compacted = [
                json.loads(line)
                for line in output.read_text("utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(compacted), 1)
            archives = list((output.parent / "archive").glob("*.jsonl"))
            self.assertEqual(len(archives), 1)
            self.assertEqual(len(archives[0].read_text("utf-8").splitlines()), 2)

    def test_run_signature_tracks_local_model_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary)
            (model / "config.json").write_text("{}\n", encoding="utf-8")
            weights = model / "model.safetensors"
            weights.write_bytes(b"first")
            config = {"llm": {"model": "configured-model"}}
            with patch.dict(
                os.environ,
                {"MODEL_PATH": str(model), "SERVED_MODEL_NAME": "served-model"},
                clear=False,
            ):
                first = run_signature(config, "evaluation", "base-qwen", 0)
                weights.write_bytes(b"different-size")
                second = run_signature(config, "evaluation", "base-qwen", 0)
            self.assertNotEqual(first, second)

    def test_evaluation_context_commits_data_assets_and_cell_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            asset_dir = root / "assets"
            data_dir.mkdir()
            asset_dir.mkdir()
            data_file = data_dir / "eval_all.jsonl"
            data_file.write_text('{"id":"q1"}\n', encoding="utf-8")

            data_hash = hashlib.sha256(data_file.read_bytes()).hexdigest()
            atomic_write_json(
                data_dir / ".hard-rq0-data-manifest.json",
                {
                    "schema": DATA_PREP_SCHEMA,
                    "artifacts": {
                        "data/eval_all.jsonl": {
                            "size": data_file.stat().st_size,
                            "sha256": data_hash,
                        }
                    },
                },
            )
            atomic_write_json(
                asset_dir / ".hard-rq0-assets-manifest.json",
                {"schema": 1, "artifacts": {}},
            )
            config = {
                "assets": {"root": str(asset_dir)},
                "retrieval": {
                    "e5_model": "e5",
                    "e5_model_revision": "immutable-e5-revision",
                },
                "agent": {"max_search_turns": 4, "result_snippet_chars": 700},
                "llm": {"temperature": 0.0, "max_tokens": 512},
            }
            retriever_identities = {
                "bm25": {"backend": "bm25", "index_path": "/idx/bm25"},
                "e5": {
                    "backend": "e5",
                    "index_path": "/idx/e5",
                    "retriever_model": "/models/e5/snapshots/revision",
                    "retriever_model_revision": "revision",
                    "faiss_gpu": True,
                    "faiss_gpu_count": 1,
                },
            }
            with patch.dict(
                os.environ,
                {
                    "TP": "1",
                    "DP": "7",
                    "VLLM_API_SERVER_COUNT": "7",
                    "GPU_MEMORY_UTILIZATION": "0.88",
                    "MAX_MODEL_LEN": "16384",
                    "VLLM_BATCH_INVARIANT": "1",
                },
                clear=False,
            ):
                context = evaluation_context(
                    config,
                    data_file.resolve(),
                    [{"id": "q1"}],
                    ["e5", "bm25"],
                    [10, 3],
                    retriever_identities,
                    112,
                )
            self.assertEqual(
                context["protocol"]["serving"],
                {
                    "tensor_parallel_size": 1,
                    "data_parallel_size": 7,
                    "api_server_count": 7,
                    "gpu_memory_utilization": 0.88,
                    "max_model_len": 16384,
                    "batch_invariant": True,
                },
            )
            with patch.dict(
                os.environ,
                {
                    "TP": "1",
                    "DP": "7",
                    "VLLM_API_SERVER_COUNT": "7",
                    "GPU_MEMORY_UTILIZATION": "0.88",
                    "MAX_MODEL_LEN": "16384",
                    "VLLM_BATCH_INVARIANT": "0",
                },
                clear=False,
            ):
                non_invariant = [
                    evaluation_context(
                        config,
                        data_file.resolve(),
                        [{"id": "q1"}],
                        ["e5", "bm25"],
                        [10, 3],
                        retriever_identities,
                        workers,
                    )
                    for workers in (56, 112)
                ]
            self.assertEqual(
                [
                    item["protocol"]["serving"]["evaluation_workers"]
                    for item in non_invariant
                ],
                [56, 112],
            )
            self.assertEqual(context["question_ids"], ["q1"])
            self.assertEqual(context["backends"], ["bm25", "e5"])
            self.assertEqual(context["topks"], [3, 10])
            self.assertEqual(context["data"]["sha256"], data_hash)
            self.assertEqual(
                context["retrievers"]["e5"]["retriever_model"],
                "/models/e5/snapshots/revision",
            )
            self.assertEqual(
                context["protocol"]["retrieval_model_revision"],
                "immutable-e5-revision",
            )

    def test_specialist_seeds_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            validate_frame(self.valid_result_frame(), expected_seeds=(0, 42, 87))
        with self.assertRaisesRegex(ValueError, "positive"):
            validate_policy_seed("bm25-specialist", 13, (0, 13, 42))

    def test_recall_is_carried_forward_when_agent_answers_early(self) -> None:
        self.assertEqual(recall_at([], 2), 0.0)
        self.assertEqual(recall_at([0.5], 2), 0.5)


if __name__ == "__main__":
    unittest.main()
