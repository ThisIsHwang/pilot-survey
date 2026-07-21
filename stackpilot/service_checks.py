from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

import requests


def effective_model_name(cfg: dict) -> str:
    return os.environ.get("SERVED_MODEL_NAME") or str(cfg["llm"]["model"])


def configure_local_no_proxy() -> None:
    values = [
        value.strip()
        for value in os.environ.get("NO_PROXY", "").split(",")
        if value.strip()
    ]
    for host in ("127.0.0.1", "localhost"):
        if host not in values:
            values.append(host)
    value = ",".join(values)
    os.environ["NO_PROXY"] = value
    os.environ["no_proxy"] = value


def check_retrievers(cfg: dict) -> None:
    configure_local_no_proxy()
    retrieval = cfg["retrieval"]
    work_dir = Path(cfg["work_dir"]).resolve()
    expected_indexes = {
        "bm25": work_dir / "indexes" / "bm25" / "bm25",
        "e5": work_dir / "indexes" / "e5" / "e5_Flat.index",
        "colbert": work_dir
        / "indexes"
        / "colbert"
        / "colbert"
        / "indexes"
        / "hotpot_pilot_colbert",
    }
    for name in ("bm25", "e5", "colbert"):
        port = int(retrieval[f"{name}_port"])
        health_url = f"http://127.0.0.1:{port}/health"
        try:
            health = requests.get(health_url, timeout=5)
            health.raise_for_status()
            payload = health.json()
            if payload.get("status") != "ok" or payload.get("backend") != name:
                raise RuntimeError(f"unexpected health response: {payload}")
            actual_index = Path(str(payload.get("index_path", ""))).resolve()
            if actual_index != expected_indexes[name]:
                raise RuntimeError(
                    f"server uses {actual_index}, expected {expected_indexes[name]}"
                )
            probe = requests.post(
                f"http://127.0.0.1:{port}/retrieve",
                json={
                    "queries": ["Who wrote Hamlet?"],
                    "topk": 1,
                    "return_scores": True,
                },
                timeout=120,
            )
            probe.raise_for_status()
            results = probe.json().get("result")
            if not results or not results[0]:
                raise RuntimeError("retrieval probe returned no documents")
        except Exception as exc:
            raise RuntimeError(
                f"{name} retriever on port {port} is not ready; "
                "run bash scripts/launch_retrievers.sh"
            ) from exc
    print("Retriever readiness checks passed.")


def check_vllm(cfg: dict) -> None:
    configure_local_no_proxy()
    api_base = str(cfg["llm"]["api_base"]).rstrip("/")
    parsed = urlparse(api_base)
    root = f"{parsed.scheme}://{parsed.netloc}"
    models_url = f"{root}/v1/models"
    expected = effective_model_name(cfg)
    try:
        response = requests.get(models_url, timeout=10)
        response.raise_for_status()
        model_ids = {str(item.get("id")) for item in response.json().get("data", [])}
    except Exception as exc:
        raise RuntimeError(
            "vLLM is not ready; run MODEL_PATH=/absolute/model/path "
            "bash scripts/launch_vllm_bg.sh"
        ) from exc
    if expected not in model_ids:
        raise RuntimeError(
            f"vLLM serves {sorted(model_ids)}, but the config requests {expected!r}. "
            "Do not set SERVED_MODEL_NAME to the local filesystem path."
        )
    print(f"vLLM readiness check passed: {expected}")
