"""Structured logging tests."""

import json
import logging

from ai_support.logging import JsonFormatter


def test_json_formatter_emits_safe_core_fields() -> None:
    record = logging.LogRecord(
        name="ai_support.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=12,
        msg="service_ready",
        args=(),
        exc_info=None,
    )

    event = json.loads(JsonFormatter().format(record))

    assert event["level"] == "INFO"
    assert event["logger"] == "ai_support.test"
    assert event["message"] == "service_ready"
    assert "timestamp" in event
    assert set(event) == {"timestamp", "level", "logger", "message"}
