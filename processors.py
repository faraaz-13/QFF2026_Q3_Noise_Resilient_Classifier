"""Processor JSON loading helpers.
The canonical configs are processors/processor_A.json and processor_B.json.
"""
import json
from pathlib import Path

def load_processor(name: str):
    if name not in {"processor_A", "processor_B"}:
        raise ValueError("name must be processor_A or processor_B")
    path = Path(__file__).resolve().parents[1] / "processors" / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))
