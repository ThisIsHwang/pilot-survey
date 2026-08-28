from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
import time
from typing import Any

from stackpilot.causal_query_replay import _client


class ThinkingLeakError(RuntimeError):
    """Raised when a supposedly non-thinking response exposes reasoning."""


def _sampling_config(cfg: Mapping[str, Any], temperature: float) -> dict[str, Any]:
    model_cfg = cfg.get("model", {})
    key = "sampling" if float(temperature) > 1e-8 else "deterministic_sampling"
    values = model_cfg.get(key, model_cfg.get("sampling", {}))
    return dict(values) if isinstance(values, Mapping) else {}


def build_request_options(
    cfg: Mapping[str, Any], *, temperature: float
) -> dict[str, Any]:
    """Build OpenAI-compatible request fields for fail-closed non-thinking."""
    model_cfg = cfg.get("model", {})
    chat_kwargs = dict(model_cfg.get("chat_template_kwargs", {}))
    chat_kwargs["enable_thinking"] = False
    sampling = _sampling_config(cfg, temperature)
    extra_body: dict[str, Any] = {"chat_template_kwargs": chat_kwargs}
    if sampling.get("top_k") is not None:
        extra_body["top_k"] = int(sampling["top_k"])
    if sampling.get("min_p") is not None:
        extra_body["min_p"] = float(sampling["min_p"])
    if sampling.get("repetition_penalty") is not None:
        extra_body["repetition_penalty"] = float(
            sampling["repetition_penalty"]
        )
    output: dict[str, Any] = {"extra_body": extra_body}
    if sampling.get("top_p") is not None:
        output["top_p"] = float(sampling["top_p"])
    if sampling.get("presence_penalty") is not None:
        output["presence_penalty"] = float(sampling["presence_penalty"])
    return output


def _reasoning_value(message: Any) -> str:
    values = [
        getattr(message, "reasoning_content", None),
        getattr(message, "reasoning", None),
    ]
    extra = getattr(message, "model_extra", None)
    if isinstance(extra, Mapping):
        values.extend([extra.get("reasoning_content"), extra.get("reasoning")])
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _append_leak_record(record: Mapping[str, Any]) -> None:
    path = os.environ.get("STACKPILOT_THINKING_LEAK_LOG", "").strip()
    if not path:
        return
    destination = os.path.abspath(os.path.expanduser(path))
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    payload = json.dumps(dict(record), sort_keys=True, ensure_ascii=False) + "\n"
    descriptor = os.open(destination, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, payload.encode("utf-8"))
    finally:
        os.close(descriptor)


def assert_non_thinking_message(
    message: Any,
    content: str,
    *,
    model: str = "Qwen/Qwen3.5-9B",
    seed: int | None = None,
) -> None:
    reasoning = _reasoning_value(message)
    lowered = str(content).lower()
    reason = ""
    if reasoning:
        reason = "non-empty-reasoning-field"
    elif "<think" in lowered or "</think" in lowered:
        reason = "think-tag"
    if reason:
        _append_leak_record(
            {
                "schema": 1,
                "created_at_unix": time.time(),
                "model": model,
                "seed": seed,
                "reason": reason,
                "reasoning_preview": reasoning[:500],
                "content_preview": str(content)[:500],
            }
        )
        raise ThinkingLeakError(
            f"Qwen3.5 thinking leakage detected ({reason}) while thinking is disabled"
        )


def complete_no_think(
    cfg: dict[str, Any],
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    seed: int,
) -> str:
    model_cfg = cfg.get("model", {})
    if model_cfg.get("enable_thinking", False) is not False:
        raise RuntimeError("Weekend Qwen3.5 runtime requires enable_thinking=false")
    request = {
        "model": str(model_cfg["served_model_name"]),
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "seed": int(seed),
        **build_request_options(cfg, temperature=float(temperature)),
    }
    response = _client(cfg).chat.completions.create(**request)
    message = response.choices[0].message
    content = message.content or ""
    if bool(model_cfg.get("require_non_thinking", True)):
        assert_non_thinking_message(
            message,
            content,
            model=str(model_cfg["served_model_name"]),
            seed=int(seed),
        )
    return content


def install_no_think_completion() -> None:
    """Patch only modules used by the isolated weekend collection entrypoint."""
    import stackpilot.causal_query_replay as causal_replay
    import stackpilot.query_credit_labels as credit_labels
    import stackpilot.query_credit_weekend_collect_support as collect_support

    causal_replay._complete = complete_no_think
    credit_labels._complete = complete_no_think
    collect_support._complete = complete_no_think


__all__ = [
    "ThinkingLeakError",
    "assert_non_thinking_message",
    "build_request_options",
    "complete_no_think",
    "install_no_think_completion",
]
