from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from stackpilot.hard_rq0_contract import (
    NUMBERED_EVALUATION_MANIFEST_SCHEMA,
    RESULT_SCHEMA,
)
from stackpilot.numbered_policy_eval import (
    atomic_json,
    check_hybrid_retriever,
    load_cached_rows,
    numbered_run_signature,
    prepare_result_cache,
    require_valid_numbered_episode,
    stable_signature,
)
from stackpilot.react_agent_eval import file_digest


def valid_episode() -> dict:
    return {
        "schema": RESULT_SCHEMA,
        "experiment_id": "EXP-003",
        "run_id": "exp003-run",
        "run_signature": "run-signature",
        "evaluation_signature": "evaluation-signature",
        "profile": "pilot",
        "variant": "blind",
        "policy_tag": "mixed-blind",
        "seed": 13,
        "backend_id_injected": False,
        "served_model": "numbered-policy",
        "question_id": "toy:q1",
        "question": "Question?",
        "dataset": "toy",
        "backend": "bm25",
        "topk": 3,
        "prediction": "answer",
        "raw_text_prediction": "answer",
        "answers": ["answer"],
        "support_titles": ["Evidence"],
        "em": 1.0,
        "f1": 1.0,
        "raw_text_em": 1.0,
        "raw_text_f1": 1.0,
        "protocol_failure": 0,
        "invalid_action_count": 0,
        "support_recall": 0.0,
        "turn1_support_recall": 0.0,
        "turn2_support_recall": 0.0,
        "turn3_support_recall": 0.0,
        "turn2_evidence_gain": 0.0,
        "turn3_evidence_gain": 0.0,
        "recovery_at_2": 0.0,
        "recovery_at_3": 0.0,
        "full_recovery_at_2": 0.0,
        "full_recovery_at_3": 0.0,
        "search_count": 0,
        "queries": [],
        "turns": [],
    }


class NumberedPolicyEvaluationTests(unittest.TestCase):
    def test_new_numbered_episode_must_pass_full_contract_before_persisting(
        self,
    ) -> None:
        row = valid_episode()
        kwargs = {
            "label": "newly generated numbered episode",
            "key": ("toy:q1", "bm25", 3),
            "expected_keys": {("toy:q1", "bm25", 3)},
            "item_by_id": {
                "toy:q1": {
                    "id": "toy:q1",
                    "question": "Question?",
                    "dataset": "toy",
                    "answers": ["answer"],
                    "support_titles": ["Evidence"],
                }
            },
            "experiment_id": "EXP-003",
            "external_run_id": "exp003-run",
            "run_signature": "run-signature",
            "evaluation_signature": "evaluation-signature",
            "tag": "mixed-blind",
            "seed": 13,
            "profile": "pilot",
            "variant": "blind",
            "inject_backend_id": False,
            "served_model": "numbered-policy",
            "max_search_turns": 4,
        }
        require_valid_numbered_episode(row, **kwargs)

        corrupted = dict(row)
        corrupted["raw_text_f1"] = 0.0
        with self.assertRaisesRegex(RuntimeError, "newly generated"):
            require_valid_numbered_episode(corrupted, **kwargs)

    def test_cache_salvages_valid_rows_and_archives_corrupt_and_stale_data(
        self,
    ) -> None:
        current = valid_episode()
        stale = dict(current)
        stale["profile"] = "smoke"
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "episodes.jsonl"
            output.write_bytes(
                (
                    json.dumps(current)
                    + "\n"
                    + "{truncated"
                    + "\n"
                    + json.dumps(stale)
                    + "\n"
                ).encode("utf-8")
            )
            existing, corrupt = load_cached_rows(output)
            self.assertTrue(corrupt)
            self.assertEqual(len(existing), 2)
            cached = prepare_result_cache(
                output,
                existing,
                expected_keys={("toy:q1", "bm25", 3)},
                item_by_id={
                    "toy:q1": {
                        "id": "toy:q1",
                        "question": "Question?",
                        "dataset": "toy",
                        "answers": ["answer"],
                        "support_titles": ["Evidence"],
                    }
                },
                experiment_id="EXP-003",
                external_run_id="exp003-run",
                run_signature="run-signature",
                evaluation_signature="evaluation-signature",
                tag="mixed-blind",
                seed=13,
                profile="pilot",
                variant="blind",
                inject_backend_id=False,
                served_model="numbered-policy",
                max_search_turns=4,
            )
            self.assertEqual(set(cached), {("toy:q1", "bm25", 3)})
            compacted = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(compacted, [current])
            archives = list((output.parent / "archive").glob("*"))
            self.assertEqual(len(archives), 2)
            self.assertTrue(any(".corrupt-" in path.name for path in archives))
            self.assertTrue(any(".stale-" in path.name for path in archives))

    def test_cache_rejects_metric_corruption(self) -> None:
        row = valid_episode()
        row["f1"] = 0.5
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "episodes.jsonl"
            cached = prepare_result_cache(
                output,
                [row],
                expected_keys={("toy:q1", "bm25", 3)},
                item_by_id={
                    "toy:q1": {
                        "id": "toy:q1",
                        "question": "Question?",
                        "dataset": "toy",
                        "answers": ["answer"],
                        "support_titles": ["Evidence"],
                    }
                },
                experiment_id="EXP-003",
                external_run_id="exp003-run",
                run_signature="run-signature",
                evaluation_signature="evaluation-signature",
                tag="mixed-blind",
                seed=13,
                profile="pilot",
                variant="blind",
                inject_backend_id=False,
                served_model="numbered-policy",
                max_search_turns=4,
            )
            self.assertEqual(cached, {})
            archives = list((output.parent / "archive").glob("*.jsonl"))
            self.assertEqual(len(archives), 1)

    def test_hybrid_health_commits_config_and_rejects_wrong_e5_upstream(
        self,
    ) -> None:
        base = {
            "bm25": {
                "backend": "bm25",
                "index_path": "/indexes/bm25",
                "server_files": {"searchr1_server.py": "a"},
            },
            "e5": {
                "backend": "e5",
                "index_path": "/indexes/e5",
                "faiss_gpu": True,
                "faiss_gpu_count": 1,
                "server_files": {"searchr1_server.py": "a"},
            },
        }
        upstreams = {
            name: {"status": "ok", **identity} for name, identity in base.items()
        }
        payload = {
            "status": "ok",
            "backend": "hybrid-rrf",
            "upstream_topk": 100,
            "rrf_constant": 60.0,
            "default_topk": 3,
            "request_timeout_seconds": 180.0,
            "server_file_sha256": file_digest(
                Path(__file__).resolve().parents[1]
                / "stackpilot"
                / "hybrid_rrf_server.py"
            ),
            "upstream_urls": {
                "bm25": "http://127.0.0.1:8101/retrieve",
                "e5": "http://127.0.0.1:8102/retrieve",
            },
            "upstreams": upstreams,
        }
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = payload
        with patch(
            "stackpilot.numbered_policy_eval.requests.get",
            return_value=response,
        ):
            identity = check_hybrid_retriever(
                port=8300,
                bm25_port=8101,
                e5_port=8102,
                upstream_topk=100,
                rrf_constant=60.0,
                base_identities=base,
            )
        self.assertEqual(identity["rrf_constant"], 60.0)
        self.assertIn("server_file_sha256", identity)

        expected_server_digest = payload["server_file_sha256"]
        payload["server_file_sha256"] = "stale"
        with (
            patch(
                "stackpilot.numbered_policy_eval.requests.get",
                return_value=response,
            ),
            self.assertRaisesRegex(RuntimeError, "stale server code"),
        ):
            check_hybrid_retriever(
                port=8300,
                bm25_port=8101,
                e5_port=8102,
                upstream_topk=100,
                rrf_constant=60.0,
                base_identities=base,
            )
        payload["server_file_sha256"] = expected_server_digest
        payload["upstreams"]["e5"]["faiss_gpu"] = False
        with (
            patch(
                "stackpilot.numbered_policy_eval.requests.get",
                return_value=response,
            ),
            self.assertRaisesRegex(RuntimeError, "faiss_gpu"),
        ):
            check_hybrid_retriever(
                port=8300,
                bm25_port=8101,
                e5_port=8102,
                upstream_topk=100,
                rrf_constant=60.0,
                base_identities=base,
            )

    def test_run_identity_tracks_profile_and_service_provenance(self) -> None:
        config = {"llm": {"model": "numbered-policy"}}
        first_context = {
            "services": {
                "model": {"served_model": "numbered-policy"},
                "retrievers": {"hybrid": {"rrf_constant": 60.0}},
            }
        }
        second_context = {
            "services": {
                "model": {"served_model": "numbered-policy"},
                "retrievers": {"hybrid": {"rrf_constant": 40.0}},
            }
        }
        first_eval = stable_signature(first_context)
        second_eval = stable_signature(second_context)
        first = numbered_run_signature(
            cfg=config,
            evaluation_signature=first_eval,
            experiment_id="EXP-006",
            external_run_id="exp006-run",
            tag="base-qwen",
            seed=0,
            profile="pilot",
            variant="base-qwen",
            inject_backend_id=False,
        )
        changed_service = numbered_run_signature(
            cfg=config,
            evaluation_signature=second_eval,
            experiment_id="EXP-006",
            external_run_id="exp006-run",
            tag="base-qwen",
            seed=0,
            profile="pilot",
            variant="base-qwen",
            inject_backend_id=False,
        )
        changed_profile = numbered_run_signature(
            cfg=config,
            evaluation_signature=first_eval,
            experiment_id="EXP-006",
            external_run_id="exp006-run",
            tag="base-qwen",
            seed=0,
            profile="smoke",
            variant="base-qwen",
            inject_backend_id=False,
        )
        self.assertNotEqual(first, changed_service)
        self.assertNotEqual(first, changed_profile)

    def test_manifest_write_is_atomic_and_shell_uses_h100_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "evaluation_manifest.json"
            atomic_json(
                path,
                {
                    "schema": NUMBERED_EVALUATION_MANIFEST_SCHEMA,
                    "result_schema": RESULT_SCHEMA,
                    "status": "complete",
                    "profile": "pilot",
                    "episodes_sha256": "abc",
                },
            )
            manifest = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["episodes_sha256"], "abc")
            self.assertEqual(manifest["result_schema"], RESULT_SCHEMA)

        root = Path(__file__).resolve().parents[1]
        evaluator = (root / "experiments" / "eval_numbered_policy.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("LLM_GPUS=${LLM_GPUS:-0,1,2,3,4,5,6}", evaluator)
        self.assertIn("TP=${TP:-1}", evaluator)
        self.assertIn("DP=${DP:-7}", evaluator)
        self.assertIn("NUMBERED_EVAL_WORKERS=${NUMBERED_EVAL_WORKERS:-112}", evaluator)
        self.assertIn("VLLM_BATCH_INVARIANT=${VLLM_BATCH_INVARIANT:-1}", evaluator)
        self.assertIn('--profile "$PROFILE" --variant "$VARIANT"', evaluator)


if __name__ == "__main__":
    unittest.main()
