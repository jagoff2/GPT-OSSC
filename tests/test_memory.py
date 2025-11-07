import json
from unittest.mock import MagicMock, patch

import numpy as np

from gpt_oss_ws.config import WorkspaceConfig
from gpt_oss_ws.memory import MemoryEntry, WorkspaceMemory


def test_workspace_memory_add_and_search(tmp_path):
  cfg = WorkspaceConfig()
  cfg.sqlite_path = str(tmp_path / "memory.sqlite")
  cfg.faiss_index_path = str(tmp_path / "memory.faiss")
  cfg.memory_embedding_dim = 8

  dummy_encoder = MagicMock()
  dummy_encoder.encode.side_effect = lambda text, convert_to_numpy=True: np.ones(cfg.memory_embedding_dim, dtype=np.float32)

  with patch("gpt_oss_ws.memory.SentenceTransformer", return_value=dummy_encoder):
    memory = WorkspaceMemory(cfg)
    entry = MemoryEntry(
      time=0.0,
      goal="goal",
      decision="decision",
      outcome="outcome",
      ws_snapshot=[0.0, 1.0],
      tags=["tag"],
      text="sample text",
    )
    memory.add(entry)
    results = memory.search("sample", k=1)
    assert results
    score, retrieved = results[0]
    assert retrieved.text == "sample text"
    memory.close()
