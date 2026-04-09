"""Tests for XVerifyHTTPClient retry and logging behavior."""

from __future__ import annotations

import http.client
import json
import socket
from unittest.mock import MagicMock, patch

import pytest

from phys_reasoner.verifier.xverify_client import XVerifyHTTPClient


def _make_response(status: int = 200, body: dict | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status = status
    resp.reason = "OK" if status == 200 else "Server Error"
    resp.read.return_value = json.dumps(body or {"correct": True}).encode("utf-8")
    return resp


def _make_conn(responses_or_errors):
    """Build a mock HTTPConnection whose successive request/getresponse cycles
    return (or raise) the given sequence. Each element is either an Exception
    to raise from getresponse()/request(), or a mock response to return."""
    conns = []
    for item in responses_or_errors:
        conn = MagicMock()
        if isinstance(item, Exception):
            conn.request.side_effect = item
        else:
            conn.getresponse.return_value = item
        conns.append(conn)

    it = iter(conns)

    def factory(*args, **kwargs):
        return next(it)

    return factory, conns


@pytest.fixture
def client():
    # Small backoffs so tests run fast.
    return XVerifyHTTPClient(
        "http://localhost:9999/judge",
        max_retries=3,
        backoff_min=0.0,
        backoff_max=0.0,
    )


def test_success_first_try(client, caplog):
    factory, conns = _make_conn([_make_response(200, {"correct": True})])
    with patch("http.client.HTTPConnection", side_effect=factory):
        with caplog.at_level("WARNING"):
            result = client("pred", "gold", "problem")
    assert result is True
    assert len(conns) == 1
    # No warnings on clean success.
    assert not caplog.records


def test_success_returns_false_when_payload_says_so(client):
    factory, _ = _make_conn([_make_response(200, {"correct": False})])
    with patch("http.client.HTTPConnection", side_effect=factory):
        assert client("p", "g") is False


def test_transient_error_then_success(client, caplog):
    factory, conns = _make_conn(
        [
            ConnectionRefusedError("connection refused"),
            _make_response(200, {"correct": True}),
        ]
    )
    with patch("http.client.HTTPConnection", side_effect=factory):
        with caplog.at_level("WARNING"):
            result = client("pred", "gold")
    assert result is True
    assert len(conns) == 2
    # One warning for the retry.
    assert any("transient error" in r.message for r in caplog.records)
    assert any("attempt=1/3" in r.message for r in caplog.records)


def test_transient_error_exhausts_retries(client, caplog):
    factory, conns = _make_conn(
        [
            socket.timeout("timed out"),
            ConnectionResetError("reset"),
            http.client.RemoteDisconnected("remote closed"),
        ]
    )
    with patch("http.client.HTTPConnection", side_effect=factory):
        with caplog.at_level("WARNING"):
            result = client("pred", "gold")
    assert result is False
    assert len(conns) == 3
    # Two retry warnings + one final "failed after N attempts" warning.
    messages = [r.message for r in caplog.records]
    assert sum("transient error" in m for m in messages) == 2
    assert sum("failed after 3 attempts" in m for m in messages) == 1


def test_5xx_fails_fast_no_retry(client, caplog):
    factory, conns = _make_conn([_make_response(500)])
    with patch("http.client.HTTPConnection", side_effect=factory):
        with caplog.at_level("WARNING"):
            result = client("pred", "gold")
    assert result is False
    # Exactly one connection attempt: 5xx is not retried.
    assert len(conns) == 1
    assert any("returned 500" in r.message for r in caplog.records)


def test_4xx_fails_fast_no_retry(client, caplog):
    factory, conns = _make_conn([_make_response(400)])
    with patch("http.client.HTTPConnection", side_effect=factory):
        with caplog.at_level("WARNING"):
            result = client("pred", "gold")
    assert result is False
    assert len(conns) == 1
    assert any("returned 400" in r.message for r in caplog.records)


def test_non_transient_exception_fails_fast(client, caplog):
    # A bad JSON payload triggers json.JSONDecodeError inside the try block.
    bad_resp = MagicMock()
    bad_resp.status = 200
    bad_resp.reason = "OK"
    bad_resp.read.return_value = b"this is not json"

    conn = MagicMock()
    conn.getresponse.return_value = bad_resp

    with patch("http.client.HTTPConnection", return_value=conn):
        with caplog.at_level("WARNING"):
            result = client("pred", "gold")
    assert result is False
    # Called once — JSON error is not transient.
    assert conn.request.call_count == 1
    assert any("JSONDecodeError" in r.message for r in caplog.records)


def test_transient_on_request_call_is_retried(client, caplog):
    # Error raised on conn.request() (not getresponse) should also retry.
    factory, conns = _make_conn(
        [
            BrokenPipeError("pipe broken"),
            _make_response(200, {"correct": True}),
        ]
    )
    with patch("http.client.HTTPConnection", side_effect=factory):
        with caplog.at_level("WARNING"):
            result = client("pred", "gold")
    assert result is True
    assert len(conns) == 2


def test_https_scheme_uses_https_connection():
    client = XVerifyHTTPClient("https://example.com/judge", max_retries=1)
    factory, _ = _make_conn([_make_response(200, {"correct": True})])
    with patch("http.client.HTTPSConnection", side_effect=factory) as https_mock:
        with patch("http.client.HTTPConnection") as http_mock:
            result = client("p", "g")
    assert result is True
    https_mock.assert_called_once()
    http_mock.assert_not_called()


def test_invalid_scheme_raises():
    with pytest.raises(ValueError, match="http"):
        XVerifyHTTPClient("ftp://example.com/judge")
