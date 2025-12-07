import json
from pathlib import Path

from pandas._libs import properties

def load_schema():
    schema_path = Path(__file__).parent / "schema.json"
    with open(schema_path, 'r') as f:
        return json.load(f)

def validate_schema(df):
    schema = load_schema()
    errors = []
    properties = schema.get("properties", {})

    for col, col_def in properties.items():
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