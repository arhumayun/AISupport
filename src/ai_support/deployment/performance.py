"""Small reproducible HTTP latency probe for deployment smoke measurements."""

import json
import time
import urllib.request
from pathlib import Path
from statistics import median


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("at least one measurement is required")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(quantile * len(ordered) + 0.999999) - 1))
    return ordered[index]


def measure(url: str, requests: int = 100, timeout: float = 5.0) -> dict[str, object]:
    if requests < 1:
        raise ValueError("requests must be positive")
    latencies: list[float] = []
    errors = 0
    for _ in range(requests):
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
                if response.status >= 400:
                    errors += 1
        except OSError:
            errors += 1
        latencies.append((time.perf_counter() - started) * 1000)
    return {
        "url": url,
        "requests": requests,
        "errors": errors,
        "error_rate": errors / requests,
        "latency_ms": {
            "p50": round(median(latencies), 3),
            "p95": round(percentile(latencies, 0.95), 3),
            "max": round(max(latencies), 3),
        },
    }


def write_measurement(result: dict[str, object], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path
