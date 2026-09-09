"""Testes para os modulos da Parte B - Parte 2: AgentProtocol + Finetune."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from brain.agent_protocol import (
    AgentHandler,
    AgentResponse,
    AgentServer,
    decode_frame,
    encode_frame,
    AgentFrameError,
)
from tools.prepare_finetune_dataset import FinetuneDatasetCollector


# ======================================================================
# AgentProtocol
# ======================================================================


class TestFrameCodec:
    def test_roundtrip(self):
        frame = {"type": "ping", "payload": {"data": 123}, "request_id": "abc"}
        raw = encode_frame(frame)
        decoded = decode_frame(raw.rstrip(b"\n"))
        assert decoded == frame

    def test_decode_invalid_json(self):
        with pytest.raises(AgentFrameError):
            decode_frame(b"not json{{{")

    def test_decode_too_large(self):
        huge = b"x" * (1_000_001)
        with pytest.raises(AgentFrameError):
            decode_frame(huge)

    def test_decode_missing_type(self):
        with pytest.raises(AgentFrameError):
            decode_frame(b'{"payload": {}}')


class TestAgentHandler:
    def test_unknown_type_returns_error(self):
        h = AgentHandler()
        result = h.handle({"type": "unknown_thing"})
        assert result["type"] == "error"

    def test_registered_handler_called(self):
        h = AgentHandler()
        h.register("ping", lambda p: AgentResponse(ok=True, payload={"result": "pong"}))
        result = h.handle({"type": "ping", "payload": {}, "request_id": "r1"})
        assert result["type"] == "ok"
        assert result["request_id"] == "r1"
        assert result["payload"]["result"] == "pong"

    def test_handler_exception_becomes_error(self):
        h = AgentHandler()
        h.register("bad", lambda p: 1 / 0)
        result = h.handle({"type": "bad", "payload": {}})
        assert result["type"] == "error"


class TestAgentServer:
    def test_disabled_does_not_start(self):
        server = AgentServer(enabled=False)
        assert server.start() is False
        assert server._running is False

    def test_start_and_stop(self):
        server = AgentServer(enabled=True, port=18765)
        assert server.start() is True
        assert server._running is True
        server.stop()
        assert server._running is False

    def test_request_response_cycle(self):
        handler = AgentHandler()
        handler.register(
            "echo",
            lambda p: AgentResponse(ok=True, payload={"echoed": p.get("text", "")}),
        )
        server = AgentServer(enabled=True, port=18766, handler=handler)
        assert server.start() is True
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            sock.connect(("127.0.0.1", 18766))
            frame = {"type": "echo", "payload": {"text": "hello"}, "request_id": "r42"}
            sock.sendall(encode_frame(frame))
            data = sock.recv(65536)
            response = decode_frame(data.rstrip(b"\n"))
            assert response["type"] == "ok"
            assert response["request_id"] == "r42"
            assert response["payload"]["echoed"] == "hello"
            sock.close()
        finally:
            server.stop()


# ======================================================================
# FinetuneDatasetCollector
# ======================================================================


class TestFinetuneDatasetCollector:
    def setup_method(self):
        self.tmp_dir = Path("tmp/test_finetune")
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.tmp_dir / "dataset.jsonl"
        if self.path.exists():
            self.path.unlink()

    def teardown_method(self):
        if self.path.exists():
            self.path.unlink()
        if self.tmp_dir.exists():
            self.tmp_dir.rmdir()

    def test_record_creates_file(self):
        col = FinetuneDatasetCollector(dataset_path=str(self.path))
        col.record("Oi", "Ola!", feedback="good")
        assert self.path.exists()

    def test_record_writes_valid_jsonl(self):
        col = FinetuneDatasetCollector(dataset_path=str(self.path))
        col.record("Oi", "Ola!", feedback="good")
        col.record("Tchau", "Ate!", feedback="bad")
        lines = self.path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        data = json.loads(lines[0])
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][1]["role"] == "assistant"
        assert data["feedback"] == "good"

    def test_count_works(self):
        col = FinetuneDatasetCollector(dataset_path=str(self.path))
        assert col.count() == 0
        col.record("a", "b")
        assert col.count() == 1
        col.record("c", "d")
        assert col.count() == 2

    def test_stats_by_feedback(self):
        col = FinetuneDatasetCollector(dataset_path=str(self.path))
        col.record("a", "b", feedback="good")
        col.record("c", "d", feedback="good")
        col.record("e", "f", feedback="bad")
        stats = col.stats()
        assert stats["good"] == 2
        assert stats["bad"] == 1

    def test_invalid_feedback_raises(self):
        col = FinetuneDatasetCollector(dataset_path=str(self.path))
        with pytest.raises(ValueError):
            col.record("a", "b", feedback="invalid")

    def test_export_returns_path(self):
        col = FinetuneDatasetCollector(dataset_path=str(self.path))
        col.record("a", "b")
        result = col.export()
        assert result == str(self.path)
