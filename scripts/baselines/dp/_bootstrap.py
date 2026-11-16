"""Put the repository root on sys.path so the scripts in this directory import forceharness without installing it."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
