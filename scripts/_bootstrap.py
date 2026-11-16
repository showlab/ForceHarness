"""Make ``forceharness`` importable from the scripts in scripts/ without installing the package, and load ``.env`` from the
repository root (API keys; see .env.example)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass
