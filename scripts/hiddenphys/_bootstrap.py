"""Make ``forceharness`` importable from the scripts in scripts/hiddenphys/ without installing the package, and load
``.env`` from the repository root (the same as scripts/_bootstrap.py, one directory deeper)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass
