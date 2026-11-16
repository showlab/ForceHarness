"""Layer 2: self-terminating force primitives (shared by every backend)."""
from .report import ContactEvent, ContactReport  # noqa: F401
from .events import EventDetector  # noqa: F401
from .library import FORCE_ONLY_EXTRA, INTERFACES, Primitives  # noqa: F401
