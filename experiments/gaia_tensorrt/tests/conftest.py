"""CPU-only tests of the experiment's metadata: no GPU, no torch, no TensorRT, no shelling out."""

import sys
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / "exporter"))
