from __future__ import annotations

from typing import Dict, List


TASKS: Dict[str, List[str]] = {
  "tool_plan": [
    "Identify tools needed for multi-hop QA and explain sequencing.",
    "Select memory entries that improve upcoming reasoning steps.",
    "Revise plan based on feedback from previous iteration."
  ],
  "long_context": [
    "Summarize the persistent workspace state over 8 turns.",
    "Describe retention policy effects on current virtual KV."
  ]
}
