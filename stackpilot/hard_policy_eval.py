from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import re
import threading
import time
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, TypeVar

import pandas as pd
import requests
from openai import OpenAI
from tqdm import tqdm

from stackpilot.common import (
    answer_em,
    answer_f1,
    append_jsonl,
    ensure_dir,
    load_config,
    read_jsonl,
    read_jsonl_tolerant,
)
from stackpilot.hard_assets import EXPECTED_DOCUMENTS
from stackpilot.hard_rq0_contract import (
    METRICS,
    RESULT_SCHEMA,
    episode_validation_error,
    finite_number,
    validate_policy_seed,
    validate_policy_selection,
)
from stackpilot.prepare_hard_rq0 import DATA_MANIFEST_NAME, DATA_PREP_SCHEMA
from stackpilot.react_agent_eval import (
    SYSTEM_PROMPT,
    file_digest,
    format_results,
    model_identity,
    parse_action,
)
from stackpilot.retrieval_clients import RetrievalClient
from stackpilot.service_checks import check_vllm, effective_model_name

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
ASSET_MANIFEST_NAME = ".hard-rq0-assets-manifest.json"
JobT = TypeVar("JobT")
ResultT = TypeVar("ResultT")


def normalize_title(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value)).replace("_", " ").strip()
    text = text.strip("\"'")
    return " ".join(text.lower().split())


def token_set(value: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(value)}


def query_features(
    question: str, query: str, previous_query: str | None
) -> dict[str, float]:
    tokens = TOKEN_RE.findall(query)
    question_tokens = token_set(question)
    query_tokens = {token.lower() for token in tokens}
    previous_tokens = token_set(previous_query or "")
    overlap = len(question_tokens & query_tokens) / max(1, len(question_tokens))
    change = 1.0
    if previous_query is not None:
        change = 1.0 - len(query_tokens & previous_tokens) / max(
            1, len(query_tokens | previous_tokens)
        )
    return {
        "query_token_count": float(len(tokens)),
        "query_question_overlap": float(overlap),
        "query_has_quotes": float('"' in query or "'" in query),
        "query_capitalized_ratio": float(
            sum(token[:1].isupper() for token in tokens) / max(1, len(tokens))
        ),
        "query_numeric_ratio": float(
            sum(any(character.isdigit() for character in token) for token in tokens)
            / max(1, len(tokens))
        ),
        "query_lexical_change": float(change),
    }


def signature(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_json_manifest(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Missing {label} manifest: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid {label} manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} manifest must contain a JSON object: {path}")
    return payload


def evaluation_context(
    cfg: dict[str, Any],
    data_file: Path,
    rows: list[dict[str, Any]],
    backends: list[str] | tuple[str, ...],
    topks: list[int] | tuple[int, ...],
    retriever_identities: dict[str, dict[str, Any]] | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    data_manifest_path = data_file.parent / DATA_MANIFEST_NAME
    data_manifest = require_json_manifest(data_manifest_path, "hard-RQ0 data")
    if data_manifest.get("schema") != DATA_PREP_SCHEMA:
        raise RuntimeError(
            f"Expected hard-RQ0 data manifest schema {DATA_PREP_SCHEMA}: "
            f"{data_manifest_path}"
        )
    expected_data_file = data_file.parent / "eval_all.jsonl"
    if data_file != expected_data_file.resolve():
        raise RuntimeError(
            f"Hard-RQ0 evaluation must use the manifested eval_all.jsonl: {expected_data_file}"
        )
    data_record = (data_manifest.get("artifacts") or {}).get("data/eval_all.jsonl")
    if not isinstance(data_record, dict):
        raise RuntimeError(
            f"Data manifest does not describe data/eval_all.jsonl: {data_manifest_path}"
        )
    data_sha256 = file_digest(data_file)
    if (
        data_record.get("size") != data_file.stat().st_size
        or data_record.get("sha256") != data_sha256
    ):
        raise RuntimeError(
            f"Evaluation data does not match its manifest; rerun hard_rq0/prepare_data.sh: "
            f"{data_file}"
        )

    asset_root = (
        Path(os.environ.get("HARD_ASSET_ROOT") or cfg["assets"]["root"])
        .expanduser()
        .resolve()
    )
    asset_manifest_path = asset_root / ASSET_MANIFEST_NAME
    require_json_manifest(asset_manifest_path, "hard-RQ0 asset")
    evaluator_names = (
        "common.py",
        "faiss_gpu.py",
        "hard_policy_eval.py",
        "hard_rq0_contract.py",
        "react_agent_eval.py",
        "retrieval_concurrency.py",
        "retrieval_clients.py",
        "searchr1_server.py",
        "service_checks.py",
    )
    batch_invariant = os.environ.get("VLLM_BATCH_INVARIANT", "0") == "1"
    serving = {
        "tensor_parallel_size": int(os.environ.get("TP", "1")),
        "data_parallel_size": int(os.environ.get("DP", "1")),
        "api_server_count": int(
            os.environ.get(
                "VLLM_API_SERVER_COUNT",
                os.environ.get("DP", "1"),
            )
        ),
        "gpu_memory_utilization": float(
            os.environ.get("GPU_MEMORY_UTILIZATION", "0.88")
        ),
        "max_model_len": int(os.environ.get("MAX_MODEL_LEN", "16384")),
        "batch_invariant": batch_invariant,
        "attention_backend": (
            os.environ.get("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
            if batch_invariant
            else os.environ.get("VLLM_ATTENTION_BACKEND") or "auto"
        ),
    }
    if not batch_invariant:
        serving["evaluation_workers"] = workers or 1
    return {
        "schema": RESULT_SCHEMA,
        "data": {
            "path": str(data_file),
            "sha256": data_sha256,
            "manifest_path": str(data_manifest_path.resolve()),
            "manifest_sha256": file_digest(data_manifest_path),
        },
        "assets": {
            "root": str(asset_root),
            "manifest_path": str(asset_manifest_path),
            "manifest_sha256": file_digest(asset_manifest_path),
        },
        "evaluator_files": {
            name: file_digest(Path(__file__).with_name(name))
            for name in evaluator_names
        },
        "question_ids": [str(row["id"]) for row in rows],
        "backends": sorted(backends),
        "topks": sorted(topks),
        "retrievers": {
            backend: retriever_identities[backend]
            for backend in sorted(backends)
            if retriever_identities is not None
        },
        "protocol": {
            "request_seed_strategy": "sha256(policy-seed,question-id,backend,topk,attempt)",
            "retrieval_model": cfg["retrieval"].get("e5_model"),
            "retrieval_model_revision": cfg["retrieval"].get("e5_model_revision"),
            "agent": cfg["agent"],
            "llm_generation": {
                "temperature": cfg["llm"]["temperature"],
                "max_tokens": cfg["llm"]["max_tokens"],
            },
            "serving": serving,
        },
    }


def run_signature(
    cfg: dict[str, Any],
    evaluation_signature: str,
    tag: str,
    seed: int,
) -> str:
    payload = {
        "schema": RESULT_SCHEMA,
        "config": cfg,
        "evaluation_signature": evaluation_signature,
        "tag": tag,
        "seed": seed,
        "model": model_identity(cfg),
    }
    return signature(payload)


def check_retriever(
    name: str, port: int, expected_index: Path, expected_corpus: Path
) -> dict[str, Any]:
    response = requests.get(f"http://127.0.0.1:{port}/health", timeout=10)
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "ok" or payload.get("backend") != name:
        raise RuntimeError(f"Unexpected {name} health response: {payload}")
    actual_index = Path(str(payload.get("index_path", ""))).resolve()
    if actual_index != expected_index.resolve():
        raise RuntimeError(
            f"{name} retriever uses {actual_index}, expected {expected_index.resolve()}"
        )
    actual_corpus = Path(str(payload.get("corpus_path", ""))).resolve()
    if actual_corpus != expected_corpus.resolve():
        raise RuntimeError(
            f"{name} retriever uses {actual_corpus}, expected {expected_corpus.resolve()}"
        )
    index_documents = int(payload.get("index_documents", -1))
    corpus_documents = int(payload.get("corpus_documents", -1))
    if index_documents != EXPECTED_DOCUMENTS:
        raise RuntimeError(
            f"{name} index has {index_documents:,} documents, "
            f"expected {EXPECTED_DOCUMENTS:,}"
        )
    if corpus_documents != EXPECTED_DOCUMENTS:
        raise RuntimeError(
            f"{name} corpus has {corpus_documents:,} documents, "
            f"expected {EXPECTED_DOCUMENTS:,}"
        )
    expected_server_files = {
        filename: file_digest(Path(__file__).with_name(filename))
        for filename in (
            "faiss_gpu.py",
            "retrieval_concurrency.py",
            "searchr1_server.py",
        )
    }
    if payload.get("server_files") != expected_server_files:
        raise RuntimeError(
            f"{name} retriever is running stale server code; restart "
            f"hard_rq0/launch_retrievers.sh: {payload.get('server_files')}"
        )
    identity: dict[str, Any] = {
        "backend": name,
        "index_path": str(actual_index),
        "corpus_path": str(actual_corpus),
        "index_documents": index_documents,
        "corpus_documents": corpus_documents,
    }
    if name == "e5":
        if (
            payload.get("faiss_gpu") is not True
            or int(payload.get("faiss_gpu_count", 0)) != 1
        ):
            raise RuntimeError(f"E5 retriever is not using one FAISS GPU: {payload}")
        if payload.get("faiss_gpu_load_mode") != "paged-fp16-flat":
            raise RuntimeError(
                f"E5 retriever is not using the memory-safe paged FAISS loader: "
                f"{payload}"
            )
        if payload.get("faiss_storage_dtype") != "float16":
            raise RuntimeError(
                f"E5 retriever is not using FP16 FAISS storage: {payload}"
            )
        if payload.get("gpu_search_serialized") is not True:
            raise RuntimeError(
                f"E5 retriever does not serialize its shared GPU index: {payload}"
            )
        if payload.get("cuda_empty_cache_disabled") is not True:
            raise RuntimeError(
                f"E5 retriever still synchronizes CUDA allocator cleanup; restart "
                f"hard_rq0/launch_retrievers.sh: {payload}"
            )
        retriever_model = str(payload.get("retriever_model", "")).strip()
        retriever_model_revision = str(
            payload.get("retriever_model_revision", "")
        ).strip()
        if not retriever_model or not retriever_model_revision:
            raise RuntimeError(
                "E5 health response does not identify the loaded retriever model "
                "and immutable revision: "
                f"{payload}"
            )
        identity.update(
            {
                "retriever_model": retriever_model,
                "retriever_model_revision": retriever_model_revision,
                "faiss_gpu": payload.get("faiss_gpu") is True,
                "faiss_gpu_count": int(payload.get("faiss_gpu_count", 0)),
                "faiss_gpu_load_mode": payload.get("faiss_gpu_load_mode"),
                "faiss_storage_dtype": payload.get("faiss_storage_dtype"),
                "faiss_temp_memory_mib": int(
                    payload.get("faiss_temp_memory_mib", 0)
                ),
                "faiss_index_bytes": int(payload.get("faiss_index_bytes", 0)),
                "gpu_search_serialized": True,
                "cuda_empty_cache_disabled": payload.get(
                    "cuda_empty_cache_disabled"
                )
                is True,
                "server_files": expected_server_files,
            }
        )
    else:
        identity["server_files"] = expected_server_files
    return identity


def complete(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    cfg: dict[str, Any],
    request_seed: int,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=float(cfg["llm"]["temperature"]),
        max_tokens=int(cfg["llm"]["max_tokens"]),
        seed=request_seed,
    )
    return response.choices[0].message.content or ""


def best_answer_scores(prediction: str, answers: list[str]) -> tuple[float, float]:
    return (
        max(answer_em(prediction, answer) for answer in answers),
        max(answer_f1(prediction, answer) for answer in answers),
    )


def recall_at(values: list[float], index: int) -> float:
    if not values:
        return 0.0
    return values[index] if len(values) > index else values[-1]


def gain_at(values: list[float], index: int) -> float:
    return values[index] if len(values) > index else 0.0


def run_episode(
    client: OpenAI,
    model: str,
    retriever: RetrievalClient,
    item: dict[str, Any],
    cfg: dict[str, Any],
    topk: int,
    eval_seed: int,
) -> dict[str, Any]:
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": f"Question: {item['question']}"})

    gold_titles = {normalize_title(title) for title in item["support_titles"]}
    cumulative_titles: set[str] = set()
    previously_seen_titles: set[str] = set()
    turn_records: list[dict[str, Any]] = []
    searches: list[str] = []
    prediction = ""
    search_budget = int(cfg["agent"]["max_search_turns"])
    max_attempts = search_budget * 2 + 2
    attempts = 0
    previous_recall = 0.0

    while len(searches) < search_budget and attempts < max_attempts:
        attempts += 1
        seed_text = f"{eval_seed}:{item['id']}:{retriever.name}:{topk}:{attempts}"
        request_seed = int.from_bytes(
            hashlib.sha256(seed_text.encode("utf-8")).digest()[:4], "big"
        ) % (2**31)
        content = complete(client, model, messages, cfg, request_seed)
        messages.append({"role": "assistant", "content": content})
        action, value = parse_action(content)
        if action == "answer":
            prediction = value
            break
        if action != "search" or not value:
            messages.append(
                {
                    "role": "user",
                    "content": "Invalid format. Output <search>...</search> or <answer>...</answer>.",
                }
            )
            continue

        results = retriever.search(value, topk)
        previous_query = searches[-1] if searches else None
        searches.append(value)
        retrieved_titles = [str(result["title"]) for result in results]
        normalized_retrieved = {normalize_title(title) for title in retrieved_titles}
        cumulative_titles.update(normalized_retrieved)
        matched = gold_titles & cumulative_titles
        recall = len(matched) / max(1, len(gold_titles))
        gain = recall - previous_recall
        new_support = sorted(
            (gold_titles & normalized_retrieved) - previously_seen_titles
        )
        turn_records.append(
            {
                "turn": len(searches),
                "query": value,
                "retrieved_titles": retrieved_titles,
                "support_recall": recall,
                "evidence_gain": gain,
                "new_support_titles": new_support,
                **query_features(str(item["question"]), value, previous_query),
            }
        )
        previously_seen_titles.update(normalized_retrieved)
        previous_recall = recall
        observation = format_results(results, int(cfg["agent"]["result_snippet_chars"]))
        messages.append(
            {"role": "user", "content": f"<information>\n{observation}\n</information>"}
        )

    if not prediction:
        messages.append(
            {
                "role": "user",
                "content": "The search budget is exhausted. Give your best final answer now as <answer>short answer</answer>.",
            }
        )
        seed_text = f"{eval_seed}:{item['id']}:{retriever.name}:{topk}:final"
        request_seed = int.from_bytes(
            hashlib.sha256(seed_text.encode("utf-8")).digest()[:4], "big"
        ) % (2**31)
        content = complete(client, model, messages, cfg, request_seed)
        action, value = parse_action(content)
        prediction = value if action == "answer" else content.strip()

    answers = [str(answer) for answer in item.get("answers") or [item["answer"]]]
    em, f1 = best_answer_scores(prediction, answers)
    recalls = [float(record["support_recall"]) for record in turn_records]
    gains = [float(record["evidence_gain"]) for record in turn_records]
    turn1_recall = recall_at(recalls, 0)
    turn2_recall = recall_at(recalls, 1)
    turn3_recall = recall_at(recalls, 2)
    turn2_gain = gain_at(gains, 1)
    turn3_gain = gain_at(gains, 2)
    final_recall = recalls[-1] if recalls else 0.0
    first_miss = turn1_recall < 1.0
    return {
        "question_id": str(item["id"]),
        "question": str(item["question"]),
        "dataset": str(item["dataset"]),
        "backend": retriever.name,
        "topk": topk,
        "prediction": prediction,
        "answers": answers,
        "em": em,
        "f1": f1,
        "support_recall": final_recall,
        "turn1_support_recall": turn1_recall,
        "turn2_support_recall": turn2_recall,
        "turn3_support_recall": turn3_recall,
        "turn2_evidence_gain": turn2_gain,
        "turn3_evidence_gain": turn3_gain,
        "search_count": len(searches),
        "recovery_at_2": float(first_miss and turn2_recall > turn1_recall),
        "recovery_at_3": float(first_miss and turn3_recall > turn1_recall),
        "full_recovery_at_2": float(first_miss and turn2_recall >= 1.0),
        "full_recovery_at_3": float(first_miss and turn3_recall >= 1.0),
        "queries": searches,
        "turns": turn_records,
    }


def validate_data_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("Hard-RQ0 evaluation data is empty")
    required = {"id", "question", "dataset", "answers", "support_titles"}
    identifiers: list[str] = []
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise RuntimeError(
                f"Hard-RQ0 evaluation row {index} is missing fields: {sorted(missing)}"
            )
        identifier = str(row["id"]).strip()
        if not identifier:
            raise RuntimeError(f"Hard-RQ0 evaluation row {index} has an empty ID")
        identifiers.append(identifier)
        if not str(row["question"]).strip() or not str(row["dataset"]).strip():
            raise RuntimeError(
                f"Hard-RQ0 evaluation row {identifier!r} has an empty question or dataset"
            )
        answers = row["answers"]
        support_titles = row["support_titles"]
        if not isinstance(answers, list) or not any(
            str(value).strip() for value in answers
        ):
            raise RuntimeError(
                f"Hard-RQ0 evaluation row {identifier!r} has no usable answers"
            )
        if not isinstance(support_titles, list) or not any(
            str(value).strip() for value in support_titles
        ):
            raise RuntimeError(
                f"Hard-RQ0 evaluation row {identifier!r} has no supporting titles"
            )
    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError("Hard-RQ0 evaluation question IDs must be unique")


def balanced_limit(
    rows: list[dict[str, Any]], limit: int | None
) -> list[dict[str, Any]]:
    if limit is None or limit >= len(rows):
        return list(rows)
    dataset_order: list[str] = []
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        dataset = str(row["dataset"])
        if dataset not in buckets:
            dataset_order.append(dataset)
            buckets[dataset] = []
        buckets[dataset].append(row)
    selected: list[dict[str, Any]] = []
    offset = 0
    while len(selected) < limit:
        made_progress = False
        for dataset in dataset_order:
            bucket = buckets[dataset]
            if offset < len(bucket):
                selected.append(bucket[offset])
                made_progress = True
                if len(selected) == limit:
                    break
        if not made_progress:
            break
        offset += 1
    return selected


def result_key(row: dict[str, Any]) -> tuple[str, str, int] | None:
    try:
        return (
            str(row["question_id"]),
            str(row["backend"]),
            int(row["topk"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parallel_job_results(
    jobs: Iterable[JobT],
    worker: Callable[[JobT], ResultT],
    max_workers: int,
    max_in_flight: int | None = None,
) -> Iterator[tuple[JobT, ResultT]]:
    """Run bounded independent jobs, abandoning daemon workers on interruption."""
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    capacity = max_workers * 2 if max_in_flight is None else max_in_flight
    if capacity < max_workers:
        raise ValueError("max_in_flight must be at least max_workers")

    pending_jobs = iter(jobs)
    work_queue: queue.Queue[Any] = queue.Queue(maxsize=capacity)
    result_queue: queue.Queue[
        tuple[JobT, bool, ResultT | BaseException]
    ] = queue.Queue()
    sentinel = object()
    stop_event = threading.Event()
    start_lock = threading.Lock()
    exhausted = False
    outstanding = 0

    def run_worker() -> None:
        while True:
            entry = work_queue.get()
            try:
                if entry is sentinel:
                    return
                job = entry
                # Checking and declaring a job started share a lock with
                # request_stop(). Once stop is set, queued jobs can be removed
                # or observed by workers, but the user worker is never called.
                with start_lock:
                    if stop_event.is_set():
                        return
                try:
                    result = worker(job)
                except BaseException as exc:
                    # Stop queued work at the point the first failure is
                    # observed, before the coordinator receives its result.
                    with start_lock:
                        stop_event.set()
                    result_queue.put((job, False, exc))
                else:
                    result_queue.put((job, True, result))
            finally:
                work_queue.task_done()
            if stop_event.is_set():
                return

    threads = [
        threading.Thread(
            target=run_worker,
            name=f"hard-rq0-eval-{index}",
            daemon=True,
        )
        for index in range(max_workers)
    ]
    for thread in threads:
        thread.start()

    def fill() -> None:
        nonlocal exhausted, outstanding
        while (
            not exhausted
            and not stop_event.is_set()
            and outstanding < capacity
        ):
            try:
                job = next(pending_jobs)
            except StopIteration:
                exhausted = True
                return
            work_queue.put(job)
            outstanding += 1

    def request_stop() -> None:
        with start_lock:
            stop_event.set()
        # Discard work that no worker has taken. Workers that raced with this
        # drain re-check stop_event under start_lock before invoking worker().
        while True:
            try:
                work_queue.get_nowait()
            except queue.Empty:
                break
            else:
                work_queue.task_done()
        # Wake workers already blocked in get(). The bounded queue is at least
        # as large as the worker count, so all sentinels fit after the drain.
        for _ in threads:
            work_queue.put_nowait(sentinel)

    def stop_normally() -> None:
        for _ in threads:
            work_queue.put(sentinel)
        for thread in threads:
            thread.join()

    completed_normally = False
    try:
        fill()
        while outstanding:
            # A finite poll keeps KeyboardInterrupt responsive on platforms
            # where an unbounded Queue.get() cannot be interrupted.
            while True:
                try:
                    first_outcome = result_queue.get(timeout=0.1)
                    break
                except queue.Empty:
                    continue
            outcomes = [first_outcome]
            while True:
                try:
                    outcomes.append(result_queue.get_nowait())
                except queue.Empty:
                    break
            outstanding -= len(outcomes)

            successful: list[tuple[JobT, ResultT]] = []
            first_error: BaseException | None = None
            for job, succeeded, payload in outcomes:
                if succeeded:
                    successful.append((job, payload))  # type: ignore[arg-type]
                elif first_error is None:
                    first_error = payload  # type: ignore[assignment]

            if first_error is not None:
                request_stop()
                # Preserve durability semantics: results completing alongside
                # the error get yielded before it, without waiting on a stuck
                # peer. Queued jobs cannot start because stop_event is set.
                deadline = time.monotonic() + 0.25
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        job, succeeded, payload = result_queue.get(
                            timeout=remaining
                        )
                    except queue.Empty:
                        break
                    if succeeded:
                        successful.append(
                            (job, payload)  # type: ignore[arg-type]
                        )

            for item in successful:
                yield item
            if first_error is not None:
                raise first_error
            fill()
        completed_normally = True
    finally:
        if completed_normally:
            try:
                stop_normally()
            except BaseException:
                request_stop()
                raise
        else:
            request_stop()


def prepare_result_cache(
    output_path: Path,
    existing: list[dict[str, Any]],
    expected_keys: set[tuple[str, str, int]],
    expected_datasets: dict[str, str],
    run_id: str,
    evaluation_id: str,
    tag: str,
    seed: int,
    max_search_turns: int = 4,
) -> dict[tuple[str, str, int], dict[str, Any]]:
    current: dict[tuple[str, str, int], dict[str, Any]] = {}
    archived: list[dict[str, Any]] = []
    for row in existing:
        key = result_key(row)
        bounded_metrics = all(
            finite_number(row.get(metric)) and 0.0 <= float(row[metric]) <= 1.0
            for metric in METRICS
        )
        search_count = row.get("search_count")
        valid_search_count = (
            finite_number(search_count)
            and float(search_count).is_integer()
            and 0 <= float(search_count) <= max_search_turns
        )
        valid_episode = episode_validation_error(row, max_search_turns) is None
        matches = (
            key in expected_keys
            and row.get("schema") == RESULT_SCHEMA
            and row.get("run_signature") == run_id
            and row.get("evaluation_signature") == evaluation_id
            and row.get("policy_tag") == tag
            and row.get("seed") == seed
            and str(row.get("dataset", ""))
            == expected_datasets.get(str(row.get("question_id", "")))
            and bounded_metrics
            and valid_search_count
            and valid_episode
        )
        if not matches or key is None:
            archived.append(row)
            continue
        if key in current:
            archived.append(current[key])
        current[key] = row

    if archived:
        archive_dir = output_path.parent / "archive"
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        archive_path = archive_dir / (
            f"{output_path.stem}.stale-{timestamp}-{run_id[:12]}.jsonl"
        )
        atomic_write_jsonl(archive_path, archived)
        print(f"Archived {len(archived)} stale result rows: {archive_path}")
    if existing:
        atomic_write_jsonl(output_path, list(current.values()))
    return current


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hard_rq0.yaml")
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--backends", nargs="+", choices=("bm25", "e5"), default=("bm25", "e5")
    )
    parser.add_argument("--topks", nargs="+", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("HARD_EVAL_WORKERS", "112")),
        help="Concurrent evaluation episodes (default: 112, or HARD_EVAL_WORKERS)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    retrieval_cfg = cfg["retrieval"]
    topks = list(args.topks or [int(value) for value in retrieval_cfg["topks"]])
    backends = list(args.backends)
    validate_policy_selection(args.tag, args.limit, backends, topks)
    validate_policy_seed(args.tag, args.seed, cfg["training"]["seeds"])
    if args.workers < 1:
        raise RuntimeError("--workers must be a positive integer")

    data_file = Path(args.data_file).resolve()
    if not data_file.is_file():
        raise RuntimeError(f"Hard-RQ0 evaluation data is missing: {data_file}")
    all_rows = read_jsonl(data_file)
    validate_data_rows(all_rows)
    rows = balanced_limit(all_rows, args.limit)
    if not rows:
        raise RuntimeError("The selected hard-RQ0 evaluation set is empty")

    ports = {
        "bm25": int(retrieval_cfg["bm25_port"]),
        "e5": int(retrieval_cfg["e5_port"]),
    }
    asset_root = (
        Path(os.environ.get("HARD_ASSET_ROOT") or cfg["assets"]["root"])
        .expanduser()
        .resolve()
    )
    expected_indexes = {
        "bm25": asset_root / "bm25",
        "e5": asset_root / "e5_Flat.index",
    }
    expected_corpus = asset_root / "wiki-18.jsonl"
    check_vllm(cfg)
    retriever_identities: dict[str, dict[str, Any]] = {}
    for backend in backends:
        retriever_identities[backend] = check_retriever(
            backend, ports[backend], expected_indexes[backend], expected_corpus
        )

    context = evaluation_context(
        cfg, data_file, rows, backends, topks, retriever_identities, args.workers
    )
    evaluation_id = signature(context)
    run_id = run_signature(cfg, evaluation_id, args.tag, args.seed)
    work_dir = ensure_dir(Path(cfg["work_dir"]).resolve() / "results" / "policies")
    output_path = work_dir / f"{args.tag}-seed{args.seed}.jsonl"
    summary_path = work_dir / f"{args.tag}-seed{args.seed}_summary.csv"

    expected_keys = {
        (str(item["id"]), backend, int(topk))
        for item in rows
        for backend in backends
        for topk in topks
    }
    expected_datasets = {str(item["id"]): str(item["dataset"]) for item in rows}
    existing = read_jsonl_tolerant(output_path)
    row_by_key = prepare_result_cache(
        output_path,
        existing,
        expected_keys,
        expected_datasets,
        run_id,
        evaluation_id,
        args.tag,
        args.seed,
        int(cfg["agent"]["max_search_turns"]),
    )

    retrievers = {
        backend: RetrievalClient(backend, f"http://127.0.0.1:{ports[backend]}/retrieve")
        for backend in backends
    }
    model = effective_model_name(cfg)
    thread_state = threading.local()
    client_registry: list[OpenAI] = []
    client_registry_lock = threading.Lock()

    def llm_client() -> OpenAI:
        client = getattr(thread_state, "llm_client", None)
        if client is None:
            client = OpenAI(
                base_url=cfg["llm"]["api_base"],
                api_key=cfg["llm"]["api_key"],
                timeout=180.0,
                max_retries=5,
            )
            thread_state.llm_client = client
            with client_registry_lock:
                client_registry.append(client)
        return client

    EvaluationJob = tuple[dict[str, Any], str, int, tuple[str, str, int]]

    def evaluate(job: EvaluationJob) -> dict[str, Any]:
        item, backend, topk, key = job
        try:
            return run_episode(
                llm_client(),
                model,
                retrievers[backend],
                item,
                cfg,
                topk,
                args.seed,
            )
        except Exception as exc:
            raise RuntimeError(f"Hard-RQ0 episode failed: {key}") from exc

    total = len(expected_keys)
    progress = tqdm(total=total, desc=f"hard RQ0 eval: {args.tag}/seed{args.seed}")
    progress.update(len(row_by_key))
    jobs: list[EvaluationJob] = []
    for item in rows:
        for backend in backends:
            for topk in topks:
                key = (str(item["id"]), backend, int(topk))
                if key not in row_by_key:
                    jobs.append((item, backend, int(topk), key))

    checkpoint_rows: list[dict[str, Any]] = []
    checkpoint_size = min(32, args.workers)
    try:
        for job, result in parallel_job_results(jobs, evaluate, args.workers):
            key = job[3]
            result.update(
                {
                    "schema": RESULT_SCHEMA,
                    "run_signature": run_id,
                    "evaluation_signature": evaluation_id,
                    "policy_tag": args.tag,
                    "seed": args.seed,
                    "served_model": model,
                }
            )
            checkpoint_rows.append(result)
            row_by_key[key] = result
            if len(checkpoint_rows) >= checkpoint_size:
                append_jsonl(output_path, checkpoint_rows)
                checkpoint_rows.clear()
            progress.update(1)
        if checkpoint_rows:
            append_jsonl(output_path, checkpoint_rows)
            checkpoint_rows.clear()
    except BaseException:
        if checkpoint_rows:
            try:
                append_jsonl(output_path, checkpoint_rows)
                checkpoint_rows.clear()
            except Exception as checkpoint_error:
                print(
                    f"Unable to flush the final evaluation checkpoint: "
                    f"{checkpoint_error}",
                    flush=True,
                )
        with client_registry_lock:
            for active_client in client_registry:
                try:
                    active_client.close()
                except Exception:
                    pass
        raise
    finally:
        progress.close()

    actual_keys = set(row_by_key)
    if actual_keys != expected_keys or len(row_by_key) != total:
        missing = sorted(expected_keys - actual_keys)[:10]
        extra = sorted(actual_keys - expected_keys)[:10]
        raise RuntimeError(
            f"Expected exactly {total} hard-RQ0 rows, found {len(row_by_key)}; "
            f"missing={missing}, extra={extra}"
        )
    ordered_rows = [
        row_by_key[(str(item["id"]), backend, int(topk))]
        for item in rows
        for backend in backends
        for topk in topks
    ]
    atomic_write_jsonl(output_path, ordered_rows)

    frame = pd.DataFrame(ordered_rows)
    summary_groups = ["dataset", "backend", "topk"]
    summary = (
        frame.groupby(summary_groups)[[*METRICS, "search_count"]].mean().reset_index()
    )
    question_counts = (
        frame.groupby(summary_groups)["question_id"]
        .nunique()
        .rename("n_questions")
        .reset_index()
    )
    summary = summary.merge(
        question_counts, on=summary_groups, how="inner", validate="one_to_one"
    )
    summary.insert(0, "policy_tag", args.tag)
    summary.insert(1, "seed", args.seed)
    summary.insert(2, "run_signature", run_id)
    summary.insert(3, "evaluation_signature", evaluation_id)
    summary.to_csv(summary_path, index=False)
    print(summary.round(4).to_string(index=False))
    print(f"Raw results: {output_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
