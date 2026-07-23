from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname


SCHEMA = 1
PIN_RE = re.compile(
    r"^\s*([A-Za-z0-9_.-]+)(?:\[[^]]+\])?\s*==\s*([^\s;#]+)\s*(?:#.*)?$"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def interpreter_identity(python: Path) -> dict[str, Any]:
    command = r"""
import json
import os
import platform
import sys
import sysconfig

print(json.dumps({
    "executable": os.path.realpath(sys.executable),
    "base_executable": os.path.realpath(getattr(sys, "_base_executable", sys.executable)),
    "version": list(sys.version_info[:3]),
    "implementation": sys.implementation.name,
    "soabi": sysconfig.get_config_var("SOABI"),
    "system": platform.system(),
    "machine": platform.machine(),
}))
"""
    try:
        result = subprocess.run(
            [str(python), "-c", command],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        payload = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to inspect Python interpreter: {python}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid Python identity from: {python}")
    return payload


def _parse_values(values: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in values:
        key, separator, value = item.partition("=")
        if not separator or not key or key in result:
            raise ValueError(f"Values must be unique KEY=VALUE pairs; got {item!r}")
        result[key] = value
    return dict(sorted(result.items()))


def signature_request(
    *,
    root: Path,
    environment: str,
    python: Path,
    inputs: Iterable[Path],
    values: Iterable[str],
) -> dict[str, Any]:
    root = root.resolve()
    files: dict[str, dict[str, Any]] = {}
    for raw_path in inputs:
        path = raw_path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Bootstrap signature input is missing: {path}")
        try:
            name = path.relative_to(root).as_posix()
        except ValueError:
            name = str(path)
        if name in files:
            raise ValueError(f"Duplicate bootstrap signature input: {name}")
        files[name] = {"size": path.stat().st_size, "sha256": file_sha256(path)}
    return {
        "schema": SCHEMA,
        "environment": environment,
        "python": interpreter_identity(python),
        "inputs": dict(sorted(files.items())),
        "values": _parse_values(values),
    }


def request_signature(request: dict[str, Any]) -> str:
    canonical = json.dumps(request, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def marker_matches(path: Path, environment: str, signature: str) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return bool(
        isinstance(payload, dict)
        and payload.get("schema") == SCHEMA
        and payload.get("environment") == environment
        and payload.get("signature") == signature
    )


def atomic_write_marker(path: Path, environment: str, signature: str) -> None:
    payload = {
        "schema": SCHEMA,
        "environment": environment,
        "signature": signature,
        "completed_at": datetime.now(UTC).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_pins(paths: Iterable[Path], extra: Iterable[str]) -> dict[str, str]:
    pins: dict[str, str] = {}
    lines: list[tuple[str, str]] = []
    for path in paths:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            lines.append((str(path), raw_line))
    lines.extend(("--require", value) for value in extra)
    for source, raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = PIN_RE.fullmatch(line)
        if match is None:
            raise ValueError(
                f"Bootstrap requirements must use exact NAME==VERSION pins; "
                f"got {raw_line!r} in {source}"
            )
        name, version = match.groups()
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        previous = pins.get(normalized)
        if previous is not None and previous != version:
            raise ValueError(
                f"Conflicting bootstrap pins for {normalized}: {previous} != {version}"
            )
        pins[normalized] = version
    return dict(sorted(pins.items()))


def verify_requirements(paths: Iterable[Path], extra: Iterable[str]) -> dict[str, str]:
    pins = parse_pins(paths, extra)
    errors = []
    installed: dict[str, str] = {}
    for name, expected in pins.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"{name} is missing (expected {expected})")
            continue
        installed[name] = actual
        if actual.split("+", 1)[0] != expected:
            errors.append(f"{name}=={actual} (expected {expected})")
    if errors:
        raise RuntimeError("Bootstrap package verification failed: " + "; ".join(errors))
    return installed


def editable_origin(distribution: str) -> Path:
    try:
        metadata = importlib.metadata.distribution(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"Editable distribution is missing: {distribution}") from exc
    try:
        payload = json.loads(metadata.read_text("direct_url.json") or "")
        url = str(payload["url"])
        editable = payload["dir_info"]["editable"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"{distribution} does not have valid editable-install metadata"
        ) from exc
    if editable is not True:
        raise RuntimeError(f"{distribution} is not installed as editable")
    parsed = urlparse(url)
    if parsed.scheme != "file":
        raise RuntimeError(f"{distribution} editable origin is not local: {url}")
    return Path(url2pathname(unquote(parsed.path))).resolve()


def verify_editables(values: Iterable[str]) -> None:
    errors = []
    for item in values:
        distribution, separator, expected_text = item.partition("=")
        if not separator or not distribution or not expected_text:
            raise ValueError(f"Editables must be DIST=PATH pairs; got {item!r}")
        expected = Path(expected_text).resolve()
        try:
            actual = editable_origin(distribution)
        except RuntimeError as exc:
            errors.append(str(exc))
            continue
        if actual != expected:
            errors.append(f"{distribution} editable origin is {actual}; expected {expected}")
    if errors:
        raise RuntimeError("Bootstrap editable verification failed: " + "; ".join(errors))


def interpreter_compatible(environment_python: Path, base_python: Path) -> bool:
    try:
        environment = interpreter_identity(environment_python)
        base = interpreter_identity(base_python)
    except RuntimeError:
        return False
    return bool(
        environment.get("version", [])[:2] == base.get("version", [])[:2]
        and environment.get("implementation") == base.get("implementation")
        and environment.get("soabi") == base.get("soabi")
        and environment.get("system") == base.get("system")
        and environment.get("machine") == base.get("machine")
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage verified bootstrap caches")
    subparsers = parser.add_subparsers(dest="command", required=True)

    signature = subparsers.add_parser("signature")
    signature.add_argument("--root", required=True)
    signature.add_argument("--environment", required=True)
    signature.add_argument("--python", required=True)
    signature.add_argument("--input", action="append", default=[])
    signature.add_argument("--value", action="append", default=[])

    matches = subparsers.add_parser("marker-matches")
    matches.add_argument("--marker", required=True)
    matches.add_argument("--environment", required=True)
    matches.add_argument("--signature", required=True)

    write = subparsers.add_parser("write-marker")
    write.add_argument("--marker", required=True)
    write.add_argument("--environment", required=True)
    write.add_argument("--signature", required=True)

    verify = subparsers.add_parser("verify-requirements")
    verify.add_argument("--requirements", action="append", default=[])
    verify.add_argument("--require", action="append", default=[])
    verify.add_argument("--editable", action="append", default=[])

    compatible = subparsers.add_parser("interpreter-compatible")
    compatible.add_argument("--environment-python", required=True)
    compatible.add_argument("--base-python", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "signature":
        request = signature_request(
            root=Path(args.root),
            environment=args.environment,
            python=Path(args.python),
            inputs=[Path(value) for value in args.input],
            values=args.value,
        )
        print(request_signature(request))
    elif args.command == "marker-matches":
        if not marker_matches(
            Path(args.marker), args.environment, args.signature
        ):
            raise SystemExit(1)
    elif args.command == "write-marker":
        atomic_write_marker(
            Path(args.marker), args.environment, args.signature
        )
    elif args.command == "verify-requirements":
        installed = verify_requirements(
            [Path(value) for value in args.requirements], args.require
        )
        verify_editables(args.editable)
        print(json.dumps(installed, sort_keys=True))
    else:
        if not interpreter_compatible(
            Path(args.environment_python), Path(args.base_python)
        ):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
