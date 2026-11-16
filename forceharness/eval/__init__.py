"""Batch evaluation: expand a matrix config, run episodes in parallel, aggregate with Wilson intervals."""
from .pricing import price_for  # noqa: F401
from .runner import expand_matrix, run_matrix, run_one  # noqa: F401
from .metrics import aggregate, wilson  # noqa: F401
