"""Notebook memory utilities.

Helpers for cleaning interactive notebook namespaces while keeping selected
objects required for downstream sections.
"""

from __future__ import annotations

import gc
import inspect
from typing import Any, Iterable, MutableMapping


def _is_drop_candidate(value: Any) -> bool:
    """Return True when an object is a common heavy notebook temporary."""
    cls = getattr(value, "__class__", None)
    module_name = getattr(cls, "__module__", "")
    return (
        isinstance(value, (dict, list, tuple, set))
        or module_name.startswith("pandas")
        or module_name.startswith("numpy")
    )


def cleanup_notebook_namespace(
    namespace: MutableMapping[str, Any] | None = None,
    *,
    keep: Iterable[str] = (),
) -> list[str]:
    """Clean notebook globals and return removed variable names.

    Args:
        namespace: Mapping to clean (typically ``globals()``). If omitted,
            uses the caller's global namespace.
        keep: Variable names to preserve.

    Returns:
        List[str]: Removed variable names.
    """
    if namespace is None:
        frame = inspect.currentframe()
        caller_frame = frame.f_back if frame is not None else None
        if caller_frame is None:
            raise RuntimeError("Could not resolve caller namespace.")
        namespace = caller_frame.f_globals

    protected = set(keep)
    protected.update({"cleanup_notebook_namespace"})

    to_remove: list[str] = []
    for name, value in list(namespace.items()):
        if name.startswith("_") or name in protected:
            continue
        if _is_drop_candidate(value):
            to_remove.append(name)

    for name in to_remove:
        namespace.pop(name, None)

    gc.collect()
    return to_remove

