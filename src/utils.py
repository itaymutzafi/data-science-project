import pandas as pd


def configure_pandas_display() -> None:
    """
    Set convenient pandas display options for exploration.
    """
    pd.set_option("display.max_rows", 60)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", "{:.4f}".format)
