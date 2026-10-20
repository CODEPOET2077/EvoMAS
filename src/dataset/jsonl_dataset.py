"""Minimal task dataset: one JSON object per line.

Required fields: ``question`` and ``true_answer``. Optional: ``task_id`` and ``file_name``
(path to an attached file, appended to the question as a hint).

To evaluate on a benchmark (GAIA, HLE, ...), export its tasks to this format or register a new
dataset class with ``@DATASET.register_module``.
"""

import json
from typing import Any, Dict, List

from src.registry import DATASET
from src.utils import assemble_project_path


@DATASET.register_module(name="jsonl_dataset", force=True)
class JsonlDataset:
    def __init__(self, path: str, **kwargs):
        self.path = assemble_project_path(path)
        self.data: List[Dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "question" not in record:
                    raise ValueError(f"{self.path}:{line_no + 1} has no 'question' field.")
                record.setdefault("task_id", f"task_{line_no}")
                record.setdefault("true_answer", None)
                record.setdefault("file_name", "")
                self.data.append(record)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.data[index]
