import os
import pandas as pd


def load_csv(filename: str, folder: str = "raw") -> pd.DataFrame:
    """
    Load a CSV file from a subfolder inside the data directory.

    Parameters
    ----------
    filename : str
        CSV file name.
    folder : str
        Subfolder under 'data' ('raw', 'processed', etc.)

    Returns
    -------
    pd.DataFrame
        Loaded dataset.

    Notes
    -----
    The expected structure will be finalized once the dataset is known.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(project_root, "data", folder, filename)
    return pd.read_csv(path)
