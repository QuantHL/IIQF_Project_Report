import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def params():
    """Indicative Vasicek parameters (report Table 2)."""
    return dict(a=0.25, b=0.04, sigma=0.015)
