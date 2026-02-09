"""Schema validation helpers.

Default behavior is a lightweight integrity gate for notebook startup.
Strict validation can be enabled explicitly for deeper schema checks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


SCHEMA_DIR = Path(__file__).parent / "schemas"
LEGACY_SCHEMA_DIR = Path(__file__).parent


def load_schema(json_name: str) -> Dict[str, Any]:
    """Load schema file from the canonical schema directory with legacy fallback."""
    schema_file = Path(json_name).name
    candidates = [SCHEMA_DIR / schema_file, LEGACY_SCHEMA_DIR / schema_file]
    for schema_path in candidates:
        if schema_path.exists():
            with open(schema_path, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(
        f"Schema '{schema_file}' was not found in {SCHEMA_DIR} or {LEGACY_SCHEMA_DIR}."
    )


def _add_issue(
    issues: List[Dict[str, str]],
    severity: str,
    message: str,
    field: Optional[str] = None,
) -> None:
    issue = {"severity": severity, "message": message}
    if field is not None:
        issue["field"] = field
    issues.append(issue)


def _normalize_column_map(columns: Iterable[str]) -> Dict[str, str]:
    return {c.lower(): c for c in columns}


def _find_column(df: pd.DataFrame, name: str) -> Optional[str]:
    col_map = _normalize_column_map(df.columns)
    return col_map.get(name.lower())


def _get_datetime_series(df: pd.DataFrame) -> tuple[pd.Series, str]:
    date_col = _find_column(df, "date")
    if date_col is not None:
        parsed = pd.to_datetime(df[date_col], errors="coerce")
        return pd.Series(parsed, index=df.index), f"column '{date_col}'"

    parsed = pd.to_datetime(df.index, errors="coerce")
    return pd.Series(parsed), "index"


def _validate_temporal_axis(df: pd.DataFrame, issues: List[Dict[str, str]]) -> None:
    parsed_dates, source = _get_datetime_series(df)
    if parsed_dates.empty:
        _add_issue(issues, "warning", "Temporal axis is empty.", "date")
        return

    invalid_dates = int(parsed_dates.isna().sum())
    total_dates = len(parsed_dates)

    if invalid_dates == total_dates:
        _add_issue(
            issues,
            "error",
            f"Temporal validation failed: {source} could not be parsed as datetime.",
            "date",
        )
        return

    if invalid_dates > 0:
        _add_issue(
            issues,
            "warning",
            f"Temporal validation: {source} contains {invalid_dates} invalid datetime values.",
            "date",
        )

    valid_dates = parsed_dates.dropna()
    if valid_dates.duplicated().any():
        _add_issue(
            issues,
            "warning",
            f"Temporal validation: {source} contains duplicate datetime values.",
            "date",
        )

    if not valid_dates.is_monotonic_increasing:
        _add_issue(
            issues,
            "warning",
            f"Temporal validation: {source} is not sorted in ascending time.",
            "date",
        )


def _normalize_expected_type(col_def: Dict[str, Any]) -> Optional[str]:
    col_type = col_def.get("type")
    if isinstance(col_type, list):
        non_null_types = [t for t in col_type if t != "null"]
        return non_null_types[0] if non_null_types else None
    if isinstance(col_type, str):
        return col_type
    return None


def _is_dtype_compatible(series: pd.Series, expected_type: Optional[str], col_def: Dict[str, Any]) -> bool:
    if expected_type is None:
        return True

    if expected_type == "float":
        return pd.api.types.is_numeric_dtype(series)
    if expected_type == "int":
        return pd.api.types.is_integer_dtype(series)
    if expected_type == "boolean":
        return pd.api.types.is_bool_dtype(series)
    if expected_type == "string":
        if col_def.get("format") == "date-time":
            parsed = pd.to_datetime(series, errors="coerce")
            return not parsed.isna().all()
        return pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)
    return True


def _validate_required_columns(
    df: pd.DataFrame, schema: Dict[str, Any], issues: List[Dict[str, str]]
) -> None:
    required = schema.get("required", [])
    for col in required:
        if col.lower() == "date":
            continue
        if _find_column(df, col) is None:
            _add_issue(issues, "error", f"Missing required column: '{col}'.", col)


def _validate_strict_columns(
    df: pd.DataFrame,
    schema: Dict[str, Any],
    issues: List[Dict[str, str]],
) -> None:
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    for schema_col, col_def in properties.items():
        actual_col = _find_column(df, schema_col)
        if actual_col is None:
            if schema_col not in required:
                _add_issue(issues, "warning", f"Optional column '{schema_col}' is missing.", schema_col)
            continue

        series = df[actual_col]
        expected_type = _normalize_expected_type(col_def)
        if not _is_dtype_compatible(series, expected_type, col_def):
            _add_issue(
                issues,
                "error",
                f"Column '{actual_col}' has dtype '{series.dtype}', expected '{expected_type}'.",
                actual_col,
            )

        if schema_col in required and series.isna().any():
            _add_issue(
                issues,
                "warning",
                f"Required column '{actual_col}' contains {int(series.isna().sum())} missing values.",
                actual_col,
            )

        if "minimum" in col_def:
            numeric = pd.to_numeric(series, errors="coerce")
            below_min = int((numeric < col_def["minimum"]).fillna(False).sum())
            if below_min > 0:
                _add_issue(
                    issues,
                    "error",
                    f"Column '{actual_col}' has {below_min} values below minimum={col_def['minimum']}.",
                    actual_col,
                )


def _build_report(
    *,
    schema_name: str,
    strict: bool,
    issues: List[Dict[str, str]],
    rows: int,
    columns: int,
) -> Dict[str, Any]:
    errors = [i for i in issues if i["severity"] == "error"]
    warnings = [i for i in issues if i["severity"] == "warning"]
    return {
        "schema": schema_name,
        "mode": "strict" if strict else "lightweight",
        "rows": rows,
        "columns": columns,
        "passed": len(errors) == 0,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "issues": issues,
    }


def _print_report(report: Dict[str, Any], *, label: Optional[str] = None) -> None:
    target = label or "DataFrame"
    status = "PASSED" if report["passed"] else "FAILED"
    print(
        f"[{target}] Schema validation ({report['mode']}): "
        f"{status} | errors={report['error_count']} warnings={report['warning_count']}"
    )
    for issue in report["issues"]:
        prefix = "ERROR" if issue["severity"] == "error" else "WARN"
        field = f"[{issue['field']}]" if "field" in issue else ""
        print(f"  - {prefix}{field} {issue['message']}")


def validate_schema(
    df: pd.DataFrame,
    json_name: str,
    *,
    strict: bool = False,
    raise_on_error: bool = False,
    verbose: bool = True,
    label: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate a DataFrame against schema.

    Modes:
    - strict=False (default): lightweight startup gate.
    - strict=True: full schema checks (types, optional fields, minimum constraints).
    """
    schema = load_schema(json_name)
    issues: List[Dict[str, str]] = []

    _validate_temporal_axis(df, issues)
    _validate_required_columns(df, schema, issues)
    if strict:
        _validate_strict_columns(df, schema, issues)

    report = _build_report(
        schema_name=Path(json_name).name,
        strict=strict,
        issues=issues,
        rows=len(df),
        columns=len(df.columns),
    )

    if verbose:
        _print_report(report, label=label)

    if raise_on_error and not report["passed"]:
        errors = "; ".join(issue["message"] for issue in report["errors"])
        raise ValueError(f"Schema validation failed: {errors}")

    return report


def validate_schema_all_dfs(
    dfs: Dict[str, pd.DataFrame],
    *,
    schema_name: str = "schema_yf.json",
    strict: bool = False,
    raise_on_error: bool = False,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Validate multiple dataframes and return per-key reports."""
    reports: Dict[str, Dict[str, Any]] = {}
    for name, df in dfs.items():
        reports[name] = validate_schema(
            df,
            schema_name,
            strict=strict,
            raise_on_error=False,
            verbose=verbose,
            label=name,
        )

    if raise_on_error:
        failed = [name for name, report in reports.items() if not report["passed"]]
        if failed:
            raise ValueError(
                "Schema validation failed for: " + ", ".join(sorted(failed))
            )

    return reports
