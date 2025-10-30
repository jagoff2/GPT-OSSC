import json
from typing import Any, Dict

import torch
from fastapi.testclient import TestClient

from gpt_oss_ws.api_server import create_app


class _Tokenizer:
  def decode(self, tokens, skip_special_tokens=True):
    return "stub-response"


class FakeModel:
  def __init__(self, cfg):
    self.cfg = cfg
    self.tokenizer = _Tokenizer()

  def tokenizer_encode(self, text: str):
    return torch.tensor([[1, 2, 3]])

  def tokenizer_decode(self, tokens: torch.Tensor) -> str:
    return "stub-response"

  def generate(self, request_ctx, **kwargs):
    if "stream_callback" in kwargs and kwargs["stream_callback"] is not None:
      kwargs["stream_callback"](torch.tensor([[4]]), torch.zeros(1, 10))
    return torch.tensor([[1, 2, 3, 4]])

  def close(self):
    pass


# Backwards compatibility for modules importing StubModel.
StubModel = FakeModel


def _setup_app(monkeypatch, tmp_path):
  cfg_path = tmp_path / "cfg.yaml"
  cfg_path.write_text("model_name: demo\n")

  def fake_load_config(path: str, overrides=None):
    from gpt_oss_ws.config import WorkspaceConfig

    cfg = WorkspaceConfig()
    cfg.model_name = "demo"
    return cfg

  monkeypatch.setattr("gpt_oss_ws.api_server.load_config", fake_load_config)
  monkeypatch.setattr("gpt_oss_ws.api_server.GPTOSSHookedModel", FakeModel)
  return create_app(str(cfg_path))


def test_chat_completion_returns_structured_response(monkeypatch, tmp_path):
  app = _setup_app(monkeypatch, tmp_path)
  client = TestClient(app)
  payload: Dict[str, Any] = {
    "model": "demo",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 5,
    "temperature": 0.1,
    "top_p": 0.9,
    "stream": False,
  }
  resp = client.post("/v1/chat/completions", json=payload)
  assert resp.status_code == 200
  data = resp.json()
  assert data["choices"][0]["message"]["content"] == "stub-response"
  assert data["usage"]["total_tokens"] == data["usage"]["prompt_tokens"] + data["usage"]["completion_tokens"]


def test_chat_completion_validation(monkeypatch, tmp_path):
  app = _setup_app(monkeypatch, tmp_path)
  client = TestClient(app)
  payload: Dict[str, Any] = {
    "model": "other",
    "messages": [{"role": "user", "content": "hello"}],
    "stream": False,
  }
  resp = client.post("/v1/chat/completions", json=payload)
  assert resp.status_code == 400
  assert resp.json()["detail"] == "model mismatch"


def test_models_endpoint(monkeypatch, tmp_path):
  app = _setup_app(monkeypatch, tmp_path)
  client = TestClient(app)
  resp = client.get("/v1/models")
  assert resp.status_code == 200
  data = resp.json()
  assert data["object"] == "list"
  assert data["data"][0]["id"] == "demo"


def test_completions_endpoint(monkeypatch, tmp_path):
  app = _setup_app(monkeypatch, tmp_path)
  client = TestClient(app)
  payload: Dict[str, Any] = {
    "model": "demo",
    "prompt": "hello",
    "max_tokens": 4,
    "stream": False,
  }
  resp = client.post("/v1/completions", json=payload)
  assert resp.status_code == 200
  data = resp.json()
  assert data["object"] == "text_completion"
  assert data["choices"][0]["text"] == "stub-response"
  assert data["choices"][0]["finish_reason"] == "stop"
  assert data["usage"]["prompt_tokens"] >= 1


def test_completions_streaming(monkeypatch, tmp_path):
  app = _setup_app(monkeypatch, tmp_path)
  client = TestClient(app)
  payload: Dict[str, Any] = {
    "model": "demo",
    "prompt": "hello",
    "stream": True,
  }
  with client.stream("POST", "/v1/completions", json=payload, timeout=5) as response:
    assert response.status_code == 200
    lines = [line for line in response.iter_lines() if line]
  assert any("text_completion.chunk" in line for line in lines)
  assert lines[-1] == "data: [DONE]"


def test_chat_streaming(monkeypatch, tmp_path):
  app = _setup_app(monkeypatch, tmp_path)
  client = TestClient(app)
  payload: Dict[str, Any] = {
    "model": "demo",
    "messages": [{"role": "user", "content": "hello"}],
    "stream": True,
  }
  with client.stream("POST", "/v1/chat/completions", json=payload, timeout=5) as response:
    assert response.status_code == 200
    lines = [line for line in response.iter_lines() if line]
  assert any("chat.completion.chunk" in line for line in lines)
  assert lines[-1] == "data: [DONE]"


def test_health_endpoint(monkeypatch, tmp_path):
  app = _setup_app(monkeypatch, tmp_path)
  client = TestClient(app)
  resp = client.get("/health")
  assert resp.status_code == 200
  assert resp.json() == {"status": "ok"}
