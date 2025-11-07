import json
from fastapi.testclient import TestClient
from gpt_oss_ws.api_server import create_app

overrides = {
    "retro": {
        "enabled": True,
        "max_retracts": 4,
        "diffusion_blend": 0.3,
        "diffusion_temperature": 0.1,
        "window": 64,
        "margin": 1.0,
    },
    "chunk_size": 32,
    "log_level": "DEBUG",
}

app = create_app(
    "configs/server.yaml",
    overrides=overrides,
    base_seed=99,
    default_temperature=0.0,
    default_top_p=1.0,
)

client = TestClient(app)

payload = {
    "model": "openai/gpt-oss-20b",
    "messages": [
        {"role": "user", "content": "Summarize the benefits of retro refinement in one paragraph."}
    ],
    "max_tokens": 128,
    "temperature": 0.0,
    "top_p": 1.0,
}

response = client.post("/v1/chat/completions", json=payload)
print("status:", response.status_code)
try:
    print(json.dumps(response.json(), indent=2)[:2000])
except Exception as exc:
    print("Failed to decode response:", exc)
