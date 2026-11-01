"""Offline, small-matrix M0 tests; never read or overwrite formal results."""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
torch.set_num_threads(1)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import requests

    def blocked(*args, **kwargs):
        raise AssertionError("M0 tests must not access the network")

    monkeypatch.setattr(requests.sessions.Session, "request", blocked)
