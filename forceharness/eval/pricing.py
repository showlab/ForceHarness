"""Model prices in $ per 1M tokens, read from ``prices.yaml`` next to this file and matched by the longest model-name
prefix.  Note the source and date of every price you add to that file."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

DEFAULT = Path(__file__).resolve().parent / "prices.yaml"


def price_for(model: str, path: Path = DEFAULT) -> Tuple[float, float]:
    """(input, output) price in $ per 1M tokens; (0, 0) for an unknown model."""
    if not Path(path).exists():
        return 0.0, 0.0
    table: Dict[str, Dict[str, float]] = yaml.safe_load(Path(path).read_text()) or {}
    best = None
    for k, v in table.items():
        if model == k or model.startswith(k):
            if best is None or len(k) > len(best):
                best = k
    if best is None:
        return 0.0, 0.0
    return float(table[best].get("in", 0.0)), float(table[best].get("out", 0.0))


def _best_key(model: str, table: Dict[str, Dict[str, float]]) -> Optional[str]:
    best = None
    for k in table:
        if model == k or model.startswith(k):
            if best is None or len(k) > len(best):
                best = k
    return best


def cache_prices_for(model: str, path: Path = DEFAULT) -> Tuple[Optional[float], Optional[float]]:
    """Cache read / write prices ($ per 1M tokens); None where the table has none (the caller then uses the input
    list price)."""
    if not Path(path).exists():
        return None, None
    table: Dict[str, Dict[str, float]] = yaml.safe_load(Path(path).read_text()) or {}
    best = _best_key(model, table)
    if best is None:
        return None, None
    row = table[best]
    cr, cw = row.get("cache_read"), row.get("cache_write")
    return (None if cr is None else float(cr)), (None if cw is None else float(cw))
