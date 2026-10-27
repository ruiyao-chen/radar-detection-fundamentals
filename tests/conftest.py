"""Shared fixtures. Every test runs offline and checks a result against an
independent answer: a closed form, a direct numerical integral, or Monte Carlo."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


def binomial_tol(n: int, p: float, k: float = 4.0) -> float:
    """Allowed |measured rate - p| for n independent trials: k binomial sigmas."""
    return k * np.sqrt(p * (1.0 - p) / n)
