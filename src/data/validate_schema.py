import json
from pathlib import Path
import pandas as pd

def load_schema():
    schema_path = Path(__file__).parent / "schema.json"
    with open(schema_path, 'r') as f:
        return json.load(f)

def validate_schema(df):
    schema = load_schema()
    errors = []
    properties = schema.get("properties", {})

    if "Date" not in df.columns:
        try:
            idx = pd.to_datetime(df.index, errors="raise")
        except Exception:
            errors.append(f"Index could not be parsed as datetime")
        else:
            # check for NaT
            if idx.isnull().any():
                errors.append(f"Index contains missing datetime values (NaT).")
            # check uniqueness
            if not idx.is_unique:
                errors.append(f"Index contains duplicate datetime values.")
            if not idx.is_monotonic_increasing:
                errors.append(f"Index is not sorted (not monotonic increasing).")

    for col, col_def in properties.items():
        if col == "Date":
            continue
        if col not in df.columns:
            errors.append(f"Missing column: {col}")
        else:
            col_type = col_def.get("type", "")
            # Handle when the type can be [float, null] for example
            if isinstance(col_type, list):
                col_type = col_type[0]
            
            # Check type (relaxed check)
            if col_type == "float" and "float" not in str(df[col].dtype):
                errors.append(f"{col} should be float, got {df[col].dtype}")
            elif col_type == "int" and "int" not in str(df[col].dtype):
                errors.append(f"{col} should be int, got {df[col].dtype}")
            elif col_type == "string" and col_def.get("format") == "date-time" and "datetime" not in str(df[col].dtype):
                errors.append(f"{col} should be datetime, got {df[col].dtype}")


    if errors:
        print("❌ Schema validation failed:")
        for e in errors:
            print(" •", e)
    else:
        print("✅ Schema validation passed!")