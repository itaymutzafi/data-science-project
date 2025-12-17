import json
from pathlib import Path
import pandas as pd

def load_schema(json_name):
    schema_path = Path(__file__).parent / json_name
    with open(schema_path, 'r') as f:
        return json.load(f)

def date_column_check(df, errors):
    if "date" not in {c.lower() for c in df.columns}:
        try:
            idx = pd.to_datetime(df.index, errors="raise")
        except Exception:
            errors.append(f"Index could not be parsed as datetime")
        else:
            # Check basic properties pandas require
            if idx.isnull().any():
                errors.append(f"Index contains missing datetime values (NaT)")
            if not idx.is_unique:
                errors.append(f"Index contains duplicate datetime values")
            if not idx.is_monotonic_increasing:
                errors.append(f"Index is not sorted (not monotonic increasing)")
    
    return df, errors

def validate_column_type(df, col, col_def, errors):
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
    
    return df, errors

def print_result(df, errors):
    message = "Schema validation"
    if errors:
        print(message + " failed:\n")
        for e in errors:
            print(" •", e)
    else:
        print(message + " passed!\n")

def validate_schema(df, json_name):
    schema = load_schema(json_name)
    errors = []
    properties = schema.get("properties", {})

    print(f"Dataframe info:")
    print(df.info())

    df, errors = date_column_check(df, errors)

    # For future graphs add columns
    if json_name == "schema_yf.json":        
        if 'Day' not in df.columns:
            df['Day'] = df.index.day_name()
        if 'Month' not in df.columns:
            df['Month'] = df.index.month

    for col, col_def in properties.items():
        if col.lower() == "date":
            continue
        if col not in df.columns:
            errors.append(f"Missing column: {col}")
        else:
            df, errors = validate_column_type(df, col, col_def, errors)

    print_result(df, errors)
