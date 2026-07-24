from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests
from openai import OpenAI
from tqdm import tqdm

from stackpilot.common import append_jsonl, ensure_dir, load_config, read_jsonl
from stackpilot.experiment_registry import experiment_by_id, load_registry
from stackpilot.hard_policy_eval import (
    atomic_write_jsonl,
    balanced_limit,
    best_answer_scores,
    check_retriever,
    episode_matches_source,
    evaluation_context,
    expected_answer_strings,
    parallel_job_results,
    run_episode,
    validate_data_rows,
)
from stackpilot.hard_rq0_contract import (
    METRICS,
    NUMBERED_EVALUATION_MANIFEST_SCHEMA,
    RESULT_SCHEMA,
    episode_validation_error,
    finite_number,
)
from stackpilot.react_agent_eval import file_digest, model_identity
from stackpilot.retrieval_clients import RetrievalClient
from stackpilot.service_checks import configure_local_no_proxy


def stable_signature(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def archive_path(output_path: Path, kind: str, suffix: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return output_path.parent / "archive" / (
        f"{output_path.stem}.{kind}-{timestamp}.{suffix}"
    )


def load_cached_rows(output_path: Path) -> tuple[list[dict[str, Any]], bool]:
    """Salvage JSON objects while preserving any corrupt checkpoint verbatim."""
    if not output_path.is_file():
        return [], False
    raw = output_path.read_bytes()
    rows: list[dict[str, Any]] = []
    corrupt = False
    for raw_line in raw.splitlines(keepends=True):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            corrupt = True
            continue
        if not isinstance(value, dict):
            corrupt = True
            continue
        rows.append(value)
    if corrupt:
        archived = archive_path(output_path, "corrupt", "jsonl")
        atomic_bytes(archived, raw)
        print(f"Archived corrupt numbered-evaluation checkpoint: {archived}")
    return rows, corrupt


def result_key(row: dict[str, Any]) -> tuple[str, str, int] | None:
    try:
        return (
            str(row["question_id"]),
            str(row["backend"]),
            int(row["topk"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def valid_cached_episode(
    row: dict[str, Any],
    *,
    key: tuple[str, str, int] | None,
    expected_keys: set[tuple[str, str, int]],
    item_by_id: dict[str, dict[str, Any]],
    experiment_id: str,
    external_run_id: str,
    run_signature: str,
    evaluation_signature: str,
    tag: str,
    seed: int,
    profile: str,
    variant: str,
    inject_backend_id: bool,
    served_model: str,
    max_search_turns: int,
) -> bool:
    if key is None or key not in expected_keys:
        return False
    item = item_by_id.get(key[0])
    if item is None:
        return False
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
    if (
        row.get("schema") != RESULT_SCHEMA
        or row.get("experiment_id") != experiment_id
        or row.get("run_id") != external_run_id
        or row.get("run_signature") != run_signature
        or row.get("evaluation_signature") != evaluation_signature
        or row.get("policy_tag") != tag
        or row.get("seed") != seed
        or row.get("profile") != profile
        or row.get("variant") != variant
        or row.get("backend_id_injected") is not inject_backend_id
        or row.get("served_model") != served_model
        or not episode_matches_source(row, item)
        or not isinstance(row.get("prediction"), str)
        or not bounded_metrics
        or not valid_search_count
        or episode_validation_error(row, max_search_turns) is not None
    ):
        return False
    expected_em, expected_f1 = best_answer_scores(
        str(row["prediction"]), expected_answer_strings(item)
    )
    return (
        abs(float(row["em"]) - expected_em) <= 1e-9
        and abs(float(row["f1"]) - expected_f1) <= 1e-9
    )


def require_valid_numbered_episode(
    row: dict[str, Any],
    *,
    label: str,
    key: tuple[str, str, int] | None,
    expected_keys: set[tuple[str, str, int]],
    item_by_id: dict[str, dict[str, Any]],
    experiment_id: str,
    external_run_id: str,
    run_signature: str,
    evaluation_signature: str,
    tag: str,
    seed: int,
    profile: str,
    variant: str,
    inject_backend_id: bool,
    served_model: str,
    max_search_turns: int,
) -> None:
    """Fail closed before a numbered episode is persisted or completed."""
    if valid_cached_episode(
        row,
        key=key,
        expected_keys=expected_keys,
        item_by_id=item_by_id,
        experiment_id=experiment_id,
        external_run_id=external_run_id,
        run_signature=run_signature,
        evaluation_signature=evaluation_signature,
        tag=tag,
        seed=seed,
        profile=profile,
        variant=variant,
        inject_backend_id=inject_backend_id,
        served_model=served_model,
        max_search_turns=max_search_turns,
    ):
        return
    protocol_problem = episode_validation_error(row, max_search_turns)
    detail = protocol_problem or "provenance, source row, or metric contract mismatch"
    raise RuntimeError(f"{label} is invalid for key {key}: {detail}")


def prepare_result_cache(
    output_path: Path,
    existing_rows: list[dict[str, Any]],
    *,
    expected_keys: set[tuple[str, str, int]],
    item_by_id: dict[str, dict[str, Any]],
    experiment_id: str,
    external_run_id: str,
    run_signature: str,
    evaluation_signature: str,
    tag: str,
    seed: int,
    profile: str,
    variant: str,
    inject_backend_id: bool,
    served_model: str,
    max_search_turns: int,
) -> dict[tuple[str, str, int], dict[str, Any]]:
    current: dict[tuple[str, str, int], dict[str, Any]] = {}
    stale: list[dict[str, Any]] = []
    for row in existing_rows:
        key = result_key(row)
        if not valid_cached_episode(
            row,
            key=key,
            expected_keys=expected_keys,
            item_by_id=item_by_id,
            experiment_id=experiment_id,
            external_run_id=external_run_id,
            run_signature=run_signature,
            evaluation_signature=evaluation_signature,
            tag=tag,
            seed=seed,
            profile=profile,
            variant=variant,
            inject_backend_id=inject_backend_id,
            served_model=served_model,
            max_search_turns=max_search_turns,
        ):
            stale.append(row)
            continue
        assert key is not None
        if key in current:
            stale.append(current[key])
        current[key] = row

    if stale:
        archived = archive_path(output_path, "stale", "jsonl")
        atomic_write_jsonl(archived, stale)
        print(f"Archived {len(stale)} stale numbered-evaluation rows: {archived}")
    if output_path.exists():
        atomic_write_jsonl(output_path, [current[key] for key in sorted(current)])
    return current


def check_model_service(
    cfg: dict[str, Any], api_base: str, served_model: str
) -> dict[str, Any]:
    configure_local_no_proxy()
    parsed = urlparse(api_base)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Invalid vLLM API base: {api_base!r}")
    models_url = f"{api_base.rstrip('/')}/models"
    try:
        response = requests.get(models_url, timeout=10)
        response.raise_for_status()
        model_ids = sorted(
            {
                str(item.get("id"))
                for item in response.json().get("data", [])
                if item.get("id") is not None
            }
        )
    except Exception as exc:
        raise RuntimeError(f"vLLM is not ready at {models_url}") from exc
    if served_model not in model_ids:
        raise RuntimeError(
            f"vLLM serves {model_ids}, but numbered evaluation requests "
            f"{served_model!r}"
        )
    return {
        "api_base": api_base.rstrip("/"),
        "served_model": served_model,
        "advertised_models": model_ids,
        "model_revision": os.environ.get("MODEL_REVISION"),
        "model": model_identity(cfg),
    }


def _assert_upstream_identity(
    name: str, payload: Any, identity: dict[str, Any]
) -> None:
    if not isinstance(payload, dict):
        raise RuntimeError(f"Hybrid {name} health payload is not an object: {payload}")
    if payload.get("status") != "ok" or payload.get("backend") != name:
        raise RuntimeError(f"Hybrid uses an unhealthy {name} upstream: {payload}")
    for field, expected in identity.items():
        if payload.get(field) != expected:
            raise RuntimeError(
                f"Hybrid {name} upstream has unexpected {field}: "
                f"{payload.get(field)!r} != {expected!r}"
            )


def check_hybrid_retriever(
    *,
    port: int,
    bm25_port: int,
    e5_port: int,
    upstream_topk: int,
    rrf_constant: float,
    base_identities: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    configure_local_no_proxy()
    url = f"http://127.0.0.1:{port}/health"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise RuntimeError(f"Hybrid RRF retriever is not ready at {url}") from exc
    if payload.get("status") != "ok" or payload.get("backend") != "hybrid-rrf":
        raise RuntimeError(f"Unexpected hybrid health response: {payload}")
    if int(payload.get("upstream_topk", -1)) != upstream_topk:
        raise RuntimeError(
            f"Hybrid upstream_topk={payload.get('upstream_topk')}, "
            f"expected {upstream_topk}"
        )
    if abs(float(payload.get("rrf_constant", float("nan"))) - rrf_constant) > 1e-12:
        raise RuntimeError(
            f"Hybrid rrf_constant={payload.get('rrf_constant')}, "
            f"expected {rrf_constant}"
        )
    expected_urls = {
        "bm25": f"http://127.0.0.1:{bm25_port}/retrieve",
        "e5": f"http://127.0.0.1:{e5_port}/retrieve",
    }
    if payload.get("upstream_urls") != expected_urls:
        raise RuntimeError(
            f"Hybrid uses unexpected upstream URLs: {payload.get('upstream_urls')}"
        )
    expected_server_digest = file_digest(
        Path(__file__).with_name("hybrid_rrf_server.py")
    )
    if payload.get("server_file_sha256") != expected_server_digest:
        raise RuntimeError(
            "Hybrid retriever is running stale server code; restart "
            f"experiments/launch_hybrid_rrf.sh: {payload.get('server_file_sha256')}"
        )
    upstreams = payload.get("upstreams")
    if not isinstance(upstreams, dict):
        raise RuntimeError(f"Hybrid health omits upstream identities: {payload}")
    for name in ("bm25", "e5"):
        _assert_upstream_identity(name, upstreams.get(name), base_identities[name])
    return {
        "backend": "hybrid",
        "port": port,
        "rrf_constant": float(payload["rrf_constant"]),
        "upstream_topk": int(payload["upstream_topk"]),
        "default_topk": int(payload.get("default_topk", 3)),
        "request_timeout_seconds": float(
            payload.get("request_timeout_seconds", 180.0)
        ),
        "upstream_urls": expected_urls,
        "upstreams": {
            name: base_identities[name] for name in ("bm25", "e5")
        },
        "server_file_sha256": expected_server_digest,
    }


def check_retriever_services(
    cfg: dict[str, Any],
    backends: list[str],
    *,
    bm25_port: int,
    e5_port: int,
    hybrid_port: int,
    hybrid_upstream_topk: int,
    hybrid_rrf_constant: float,
) -> dict[str, dict[str, Any]]:
    asset_root = (
        Path(os.environ.get("HARD_ASSET_ROOT") or cfg["assets"]["root"])
        .expanduser()
        .resolve()
    )
    expected_corpus = asset_root / "wiki-18.jsonl"
    base_names = {
        name
        for backend in backends
        for name in (("bm25", "e5") if backend == "hybrid" else (backend,))
    }
    ports = {"bm25": bm25_port, "e5": e5_port}
    indexes = {"bm25": asset_root / "bm25", "e5": asset_root / "e5_Flat.index"}
    base_identities = {
        name: check_retriever(name, ports[name], indexes[name], expected_corpus)
        for name in sorted(base_names)
    }
    if "e5" in base_identities:
        expected_revision = str(
            cfg["retrieval"].get("e5_model_revision", "")
        ).strip()
        actual_revision = str(
            base_identities["e5"].get("retriever_model_revision", "")
        ).strip()
        if not expected_revision or actual_revision != expected_revision:
            raise RuntimeError(
                f"E5 retriever revision {actual_revision!r} does not match "
                f"the configured immutable revision {expected_revision!r}"
            )

    identities: dict[str, dict[str, Any]] = {}
    for backend in backends:
        if backend == "hybrid":
            identities[backend] = check_hybrid_retriever(
                port=hybrid_port,
                bm25_port=bm25_port,
                e5_port=e5_port,
                upstream_topk=hybrid_upstream_topk,
                rrf_constant=hybrid_rrf_constant,
                base_identities=base_identities,
            )
        else:
            identities[backend] = {
                **base_identities[backend],
                "port": ports[backend],
                "retrieve_url": (
                    f"http://127.0.0.1:{ports[backend]}/retrieve"
                ),
            }
    return identities


def numbered_evaluation_context(
    *,
    cfg: dict[str, Any],
    data_file: Path,
    rows: list[dict[str, Any]],
    backends: list[str],
    topks: list[int],
    retriever_identities: dict[str, dict[str, Any]],
    model_service: dict[str, Any],
    workers: int,
    inject_backend_id: bool,
) -> dict[str, Any]:
    context = evaluation_context(
        cfg,
        data_file,
        rows,
        backends,
        topks,
        retriever_identities,
        workers,
    )
    context["evaluator_files"].update(
        {
            name: file_digest(Path(__file__).with_name(name))
            for name in (
                "experiment_registry.py",
                "numbered_policy_eval.py",
            )
        }
    )
    context["protocol"]["inject_backend_id"] = inject_backend_id
    context["services"] = {
        "model": model_service,
        "retrievers": {
            name: retriever_identities[name] for name in sorted(retriever_identities)
        },
    }
    return context


def numbered_run_signature(
    *,
    cfg: dict[str, Any],
    evaluation_signature: str,
    experiment_id: str,
    external_run_id: str,
    tag: str,
    seed: int,
    profile: str,
    variant: str,
    inject_backend_id: bool,
) -> str:
    return stable_signature(
        {
            "schema": RESULT_SCHEMA,
            "experiment_id": experiment_id,
            "run_id": external_run_id,
            "tag": tag,
            "seed": seed,
            "profile": profile,
            "variant": variant,
            "evaluation_signature": evaluation_signature,
            "inject_backend_id": inject_backend_id,
            "model": model_identity(cfg),
        }
    )


def summarize(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    groups = [
        "experiment_id",
        "run_id",
        "run_signature",
        "evaluation_signature",
        "profile",
        "variant",
        "policy_tag",
        "seed",
        "dataset",
        "backend",
        "topk",
    ]
    summary = frame.groupby(groups, as_index=False)[[*METRICS, "search_count"]].mean()
    counts = (
        frame.groupby(groups, as_index=False)["question_id"]
        .nunique()
        .rename(columns={"question_id": "n_questions"})
    )
    return summary.merge(counts, on=groups, validate="one_to_one").sort_values(
        ["dataset", "backend", "topk"]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hard_rq0.yaml")
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("bm25", "e5", "hybrid"),
        default=("bm25", "e5"),
    )
    parser.add_argument("--topks", nargs="+", type=int, default=(3, 5, 10))
    parser.add_argument("--bm25-port", type=int, default=8101)
    parser.add_argument("--e5-port", type=int, default=8102)
    parser.add_argument("--hybrid-port", type=int, default=8300)
    parser.add_argument("--hybrid-upstream-topk", type=int, default=100)
    parser.add_argument("--hybrid-rrf-constant", type=float, default=60.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("NUMBERED_EVAL_WORKERS", "112")),
        help="Concurrent evaluation episodes (default: 112)",
    )
    parser.add_argument(
        "--inject-backend-id",
        action="store_true",
        help="prepend the current backend as retrieval_environment metadata",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_by_id(load_registry(), args.experiment_id)
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.hybrid_upstream_topk < 1 or args.hybrid_rrf_constant <= 0:
        raise ValueError("Hybrid upstream top-k and RRF constant must be positive")
    cfg = load_config(args.config)
    cfg["llm"]["api_base"] = args.api_base
    cfg["llm"]["model"] = args.model
    data_file = Path(args.data_file).resolve()
    if not data_file.is_file():
        raise RuntimeError(f"Numbered evaluation data is missing: {data_file}")
    rows = balanced_limit(read_jsonl(data_file), args.limit)
    validate_data_rows(rows)
    backends = list(args.backends)
    topks = list(args.topks)
    if len(set(backends)) != len(backends):
        raise ValueError("backends must be unique")
    if len(set(topks)) != len(topks) or any(value < 1 for value in topks):
        raise ValueError("topks must be unique positive integers")

    model_service = check_model_service(cfg, args.api_base, args.model)
    retriever_identities = check_retriever_services(
        cfg,
        backends,
        bm25_port=args.bm25_port,
        e5_port=args.e5_port,
        hybrid_port=args.hybrid_port,
        hybrid_upstream_topk=args.hybrid_upstream_topk,
        hybrid_rrf_constant=args.hybrid_rrf_constant,
    )
    context = numbered_evaluation_context(
        cfg=cfg,
        data_file=data_file,
        rows=rows,
        backends=backends,
        topks=topks,
        retriever_identities=retriever_identities,
        model_service=model_service,
        workers=args.workers,
        inject_backend_id=args.inject_backend_id,
    )
    evaluation_id = stable_signature(context)
    run_signature = numbered_run_signature(
        cfg=cfg,
        evaluation_signature=evaluation_id,
        experiment_id=args.experiment_id,
        external_run_id=args.run_id,
        tag=args.tag,
        seed=args.seed,
        profile=args.profile,
        variant=args.variant,
        inject_backend_id=args.inject_backend_id,
    )

    output_dir = ensure_dir(args.output_dir)
    output_path = Path(output_dir) / "episodes.jsonl"
    summary_path = Path(output_dir) / "summary.csv"
    manifest_path = Path(output_dir) / "evaluation_manifest.json"
    # A manifest is the sole completion marker. Remove it before any checkpoint
    # mutation; a crash leaves resumable rows but never advertises completion.
    manifest_path.unlink(missing_ok=True)

    expected_order = [
        (str(item["id"]), backend, int(topk))
        for item in rows
        for backend in backends
        for topk in topks
    ]
    expected_keys = set(expected_order)
    item_by_id = {str(item["id"]): item for item in rows}
    max_search_turns = int(cfg["agent"]["max_search_turns"])
    existing_rows, _ = load_cached_rows(output_path)
    current = prepare_result_cache(
        output_path,
        existing_rows,
        expected_keys=expected_keys,
        item_by_id=item_by_id,
        experiment_id=args.experiment_id,
        external_run_id=args.run_id,
        run_signature=run_signature,
        evaluation_signature=evaluation_id,
        tag=args.tag,
        seed=args.seed,
        profile=args.profile,
        variant=args.variant,
        inject_backend_id=args.inject_backend_id,
        served_model=args.model,
        max_search_turns=max_search_turns,
    )

    urls = {
        "bm25": f"http://127.0.0.1:{args.bm25_port}/retrieve",
        "e5": f"http://127.0.0.1:{args.e5_port}/retrieve",
        "hybrid": f"http://127.0.0.1:{args.hybrid_port}/retrieve",
    }
    retrievers = {
        name: RetrievalClient(name, urls[name]) for name in backends
    }
    thread_state = threading.local()
    client_registry: list[OpenAI] = []
    client_registry_lock = threading.Lock()

    def llm_client() -> OpenAI:
        client = getattr(thread_state, "llm_client", None)
        if client is None:
            client = OpenAI(
                base_url=args.api_base,
                api_key="EMPTY",
                timeout=180.0,
                max_retries=5,
            )
            thread_state.llm_client = client
            with client_registry_lock:
                client_registry.append(client)
        return client

    EvaluationJob = tuple[
        dict[str, Any], str, int, tuple[str, str, int]
    ]

    def evaluate(job: EvaluationJob) -> dict[str, Any]:
        item, backend, topk, key = job
        eval_item = dict(item)
        if args.inject_backend_id:
            eval_item["question"] = (
                f"<retrieval_environment>{backend}</retrieval_environment>\n"
                f"{item['question']}"
            )
        try:
            episode = run_episode(
                client=llm_client(),
                model=args.model,
                retriever=retrievers[backend],
                item=eval_item,
                cfg=cfg,
                topk=topk,
                eval_seed=args.seed,
            )
        except Exception as exc:
            raise RuntimeError(f"Numbered evaluation episode failed: {key}") from exc
        episode["question"] = str(item["question"])
        return episode

    jobs: list[EvaluationJob] = []
    for item in rows:
        for backend in backends:
            for topk in topks:
                key = (str(item["id"]), backend, int(topk))
                if key not in current:
                    jobs.append((item, backend, int(topk), key))

    progress = tqdm(
        total=len(expected_order),
        initial=len(current),
        desc=f"{args.experiment_id}:{args.tag}",
    )
    checkpoint_rows: list[dict[str, Any]] = []
    checkpoint_size = min(32, args.workers)
    try:
        for job, episode in parallel_job_results(jobs, evaluate, args.workers):
            key = job[3]
            episode.update(
                {
                    "schema": RESULT_SCHEMA,
                    "experiment_id": args.experiment_id,
                    "run_id": args.run_id,
                    "run_signature": run_signature,
                    "evaluation_signature": evaluation_id,
                    "profile": args.profile,
                    "variant": args.variant,
                    "policy_tag": args.tag,
                    "seed": args.seed,
                    "backend_id_injected": args.inject_backend_id,
                    "served_model": args.model,
                }
            )
            require_valid_numbered_episode(
                episode,
                label="newly generated numbered episode",
                key=key,
                expected_keys=expected_keys,
                item_by_id=item_by_id,
                experiment_id=args.experiment_id,
                external_run_id=args.run_id,
                run_signature=run_signature,
                evaluation_signature=evaluation_id,
                tag=args.tag,
                seed=args.seed,
                profile=args.profile,
                variant=args.variant,
                inject_backend_id=args.inject_backend_id,
                served_model=args.model,
                max_search_turns=max_search_turns,
            )
            current[key] = episode
            checkpoint_rows.append(episode)
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
                    f"Unable to flush the final numbered-evaluation checkpoint: "
                    f"{checkpoint_error}",
                    flush=True,
                )
        raise
    finally:
        progress.close()
        with client_registry_lock:
            for client in client_registry:
                try:
                    client.close()
                except Exception:
                    pass

    if set(current) != expected_keys or len(current) != len(expected_order):
        missing = sorted(expected_keys - set(current))[:10]
        extra = sorted(set(current) - expected_keys)[:10]
        raise RuntimeError(
            f"Numbered evaluation is incomplete; missing={missing}, extra={extra}"
        )
    final_rows = [current[key] for key in expected_order]
    for key, episode in zip(expected_order, final_rows, strict=True):
        require_valid_numbered_episode(
            episode,
            label="numbered completion episode",
            key=key,
            expected_keys=expected_keys,
            item_by_id=item_by_id,
            experiment_id=args.experiment_id,
            external_run_id=args.run_id,
            run_signature=run_signature,
            evaluation_signature=evaluation_id,
            tag=args.tag,
            seed=args.seed,
            profile=args.profile,
            variant=args.variant,
            inject_backend_id=args.inject_backend_id,
            served_model=args.model,
            max_search_turns=max_search_turns,
        )
    atomic_write_jsonl(output_path, final_rows)
    summary = summarize(final_rows)
    summary.to_csv(summary_path, index=False)
    manifest = {
        "schema": NUMBERED_EVALUATION_MANIFEST_SCHEMA,
        "result_schema": RESULT_SCHEMA,
        "status": "complete",
        "experiment_id": args.experiment_id,
        "run_id": args.run_id,
        "run_signature": run_signature,
        "evaluation_signature": evaluation_id,
        "profile": args.profile,
        "variant": args.variant,
        "policy_tag": args.tag,
        "seed": args.seed,
        "questions": len(rows),
        "episodes": len(final_rows),
        "backends": backends,
        "topks": topks,
        "backend_id_injected": args.inject_backend_id,
        "workers": args.workers,
        "episodes_sha256": file_digest(output_path),
        "summary_sha256": file_digest(summary_path),
        "evaluation_context": context,
    }
    atomic_json(manifest_path, manifest)
    print(summary.round(4).to_string(index=False))
    print(f"Numbered results: {output_dir}")


if __name__ == "__main__":
    main()
