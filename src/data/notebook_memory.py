"""Notebook namespace cleanup helpers."""

from __future__ import annotations

import gc
import inspect
from typing import Any, Iterable, Literal, MutableMapping


SECTION61_BASE_KEEP = {
    "src",
    "pd",
    "feature_data",
    "SAVE_RESULTS",
    "N_RANDOM_FEATURE_SETS",
    "FEATURE_SET_SEED",
    "RESULTS_DIR",
}

SECTION61_PHASE_KEEP = {
    "after_continuous": {"zoo_inputs"},
    "final": set(),
}


def _is_drop_candidate(value: Any) -> bool:
    """Return `True` for heavy temporary notebook objects."""
    cls = getattr(value, "__class__", None)
    module_name = getattr(cls, "__module__", "")
    if any(module_name.startswith(prefix) for prefix in ("pandas", "numpy", "matplotlib")):
        return True

    for attr in ("results", "feature_audit_summary", "feature_set_preview"):
        nested = getattr(value, attr, None)
        nested_cls = getattr(nested, "__class__", None)
        nested_module = getattr(nested_cls, "__module__", "")
        if nested_module.startswith("pandas"):
            return True

    return (
        isinstance(value, (dict, list, tuple, set))
    )


def cleanup_notebook_namespace(
    namespace: MutableMapping[str, Any] | None = None,
    *,
    keep: Iterable[str] = (),
) -> list[str]:
    """Clean notebook globals and return removed variable names."""
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


def cleanup_section62_memory(
    namespace: MutableMapping[str, Any] | None = None,
    *,
    phase: Literal["after_continuous", "final"] = "after_continuous",
    keep_extra: Iterable[str] = (),
) -> list[str]:
    """Apply the predefined Section 6.2 notebook cleanup policy."""
    if phase not in SECTION61_PHASE_KEEP:
        raise ValueError(f"Unknown section 6.2 cleanup phase: {phase}")

    if namespace is None:
        frame = inspect.currentframe()
        caller_frame = frame.f_back if frame is not None else None
        if caller_frame is None:
            raise RuntimeError("Could not resolve caller namespace.")
        namespace = caller_frame.f_globals

    keep = set(SECTION61_BASE_KEEP)
    keep.update(SECTION61_PHASE_KEEP[phase])
    keep.update(keep_extra)
    return cleanup_notebook_namespace(namespace=namespace, keep=keep)
