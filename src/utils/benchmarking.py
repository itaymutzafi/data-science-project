import time
import sys
import logging

logger = logging.getLogger(__name__)

def benchmark_import(name: str):
    """Benchmarks the import time of a module."""
    print(f"Importing {name}...", end=" ", flush=True)
    start = time.time()
    try:
        __import__(name)
        end = time.time()
        duration = end - start
        print(f"Done in {duration:.4f} seconds")
        return duration
    except ImportError as e:
        print(f"Failed: {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def run_import_benchmarks(modules=None):
    if modules is None:
        modules = ["numpy", "pandas", "scipy"]
    
    print(f"Python: {sys.version}")
    for mod in modules:
        benchmark_import(mod)
