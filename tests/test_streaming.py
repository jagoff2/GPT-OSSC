from fastapi.testclient import TestClient
from unittest.mock import patch

from gpt_oss_ws.api_server import create_app
from tests.test_api_compat import FakeModel


@patch("gpt_oss_ws.api_server.GPTOSSHookedModel", return_value=FakeModel(None))
def test_streaming_sends_chunks(mock_model):
  app = create_app("configs/server.yaml")
  client = TestClient(app)
  payload = {
    "model": "openai/gpt-oss-20b",
    "messages": [
      {"role": "user", "content": "Hello"}
    ],
    "stream": True,
  }
  with client.stream("POST", "/v1/chat/completions", json=payload, timeout=5) as response:
    assert response.status_code == 200
    lines = [line for line in response.iter_lines() if line]
  assert any(line.startswith("data: ") for line in lines)
  assert "[DONE]" in lines[-1]
