from .statistic_tests import display_seasonality_results, test_seasonality

# Backward compatibility alias to keep prior API stable.
perform_statistic_tests = test_seasonality

__all__ = [
    "test_seasonality",
    "display_seasonality_results",
    "perform_statistic_tests",
]
