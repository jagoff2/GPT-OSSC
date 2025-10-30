from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

from .config import WorkspaceConfig

try:  # pragma: no cover - exercised on environments with FAISS
  import faiss  # type: ignore
except ImportError:  # pragma: no cover
  faiss = None  # type: ignore

try:  # pragma: no cover - exercised when sentence-transformers is present
  from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:  # pragma: no cover
  SentenceTransformer = None  # type: ignore


class _FallbackIndex:
  """Lightweight inner-product index used when FAISS is unavailable."""

  def __init__(self, dim: int, path: Path) -> None:
    self.dim = dim
    self.path = path
    self.vectors = np.empty((0, dim), dtype=np.float32)

  @property
  def ntotal(self) -> int:
    return int(self.vectors.shape[0])

  @property
  def is_trained(self) -> bool:
    return True

  def train(self, data: np.ndarray) -> None:
    return None

  def add(self, data: np.ndarray) -> None:
    array = np.asarray(data, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != self.dim:
      raise ValueError(f"Expected data with shape [*, {self.dim}], got {array.shape}")
    self.vectors = np.concatenate([self.vectors, array], axis=0)

  def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    if self.ntotal == 0:
      scores = np.zeros((query.shape[0], k), dtype=np.float32)
      ids = -np.ones((query.shape[0], k), dtype=np.int64)
      return scores, ids
    query = np.asarray(query, dtype=np.float32)
    sims = query @ self.vectors.T
    k = min(k, sims.shape[1]) if sims.size else 0
    if k == 0:
      scores = np.zeros((query.shape[0], 0), dtype=np.float32)
      ids = -np.ones((query.shape[0], 0), dtype=np.int64)
      return scores, ids
    topk_idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    row_indices = np.arange(query.shape[0])[:, None]
    topk_scores = sims[row_indices, topk_idx]
    order = np.argsort(-topk_scores, axis=1)
    sorted_scores = np.take_along_axis(topk_scores, order, axis=1)
    sorted_idx = np.take_along_axis(topk_idx, order, axis=1)
    return sorted_scores.astype(np.float32), sorted_idx.astype(np.int64)

  def save(self) -> None:
    self.path.parent.mkdir(parents=True, exist_ok=True)
    np.save(self.path, self.vectors)

  def load(self) -> None:
    if self.path.exists():
      self.vectors = np.load(self.path)


@dataclass
class MemoryEntry:
  time: float
  goal: str
  decision: str
  outcome: str
  ws_snapshot: List[float]
  tags: List[str]
  text: str


class WorkspaceMemory:
  def __init__(self, config: WorkspaceConfig) -> None:
    self.config = config
    self.sqlite_path = Path(config.sqlite_path)
    self.faiss_path = Path(config.faiss_index_path)
    self.embedding_dim = config.memory_embedding_dim
    self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    self.conn = sqlite3.connect(self.sqlite_path)
    self._init_db()
    if faiss is not None:
      self.index = faiss.IndexFlatIP(self.embedding_dim)
      if self.faiss_path.exists():
        self.index = faiss.read_index(str(self.faiss_path))
    else:
      fallback_path = self.faiss_path.with_suffix(".npy")
      self.index = _FallbackIndex(self.embedding_dim, fallback_path)
      self.index.load()
    if SentenceTransformer is None:
      raise ImportError(
        "sentence-transformers is required for WorkspaceMemory; install it or patch SentenceTransformer before use."
      )
    self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

  def _init_db(self) -> None:
    cur = self.conn.cursor()
    cur.execute(
      """
      CREATE TABLE IF NOT EXISTS memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        time REAL,
        goal TEXT,
        decision TEXT,
        outcome TEXT,
        ws_snapshot TEXT,
        tags TEXT,
        text TEXT
      )
      """
    )
    self.conn.commit()

  def add(self, entry: MemoryEntry) -> None:
    cur = self.conn.cursor()
    cur.execute(
      "INSERT INTO memory (time, goal, decision, outcome, ws_snapshot, tags, text) VALUES (?, ?, ?, ?, ?, ?, ?)",
      (
        entry.time,
        entry.goal,
        entry.decision,
        entry.outcome,
        json.dumps(entry.ws_snapshot),
        json.dumps(entry.tags),
        entry.text,
      ),
    )
    self.conn.commit()
    embedding = self.encoder.encode(entry.text, convert_to_numpy=True)
    if not self.index.is_trained:
      self.index.train(np.expand_dims(embedding, axis=0))
    self.index.add(np.expand_dims(embedding, axis=0))
    self._persist_index()

  def search(self, query: str, k: int = 5) -> List[Tuple[float, MemoryEntry]]:
    if self.index.ntotal == 0:
      return []
    embedding = self.encoder.encode(query, convert_to_numpy=True)
    scores, ids = self.index.search(np.expand_dims(embedding, axis=0), k)
    cur = self.conn.cursor()
    results: List[Tuple[float, MemoryEntry]] = []
    for score, idx in zip(scores[0], ids[0]):
      if idx < 0:
        continue
      cur.execute("SELECT time, goal, decision, outcome, ws_snapshot, tags, text FROM memory WHERE id=?", (int(idx) + 1,))
      row = cur.fetchone()
      if row:
        results.append((float(score), MemoryEntry(
          time=row[0],
          goal=row[1],
          decision=row[2],
          outcome=row[3],
          ws_snapshot=json.loads(row[4]),
          tags=json.loads(row[5]),
          text=row[6]
        )))
    return results

  def close(self) -> None:
    self.conn.close()

  def _persist_index(self) -> None:
    if faiss is not None:
      faiss.write_index(self.index, str(self.faiss_path))
    else:
      self.index.save()
