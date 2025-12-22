"""Technical indicators module.

Vectorized implementations of core indicators for the pipeline.
"""

import numpy as np
import pandas as pd


class TechnicalIndicators:
    """Collection of static indicator calculators."""

    @staticmethod
    def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = data.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=window, min_periods=window).mean()
        avg_loss = loss.rolling(window=window, min_periods=window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_moving_average(data: pd.Series, window: int = 20) -> pd.Series:
        """Simple moving average."""
        return data.rolling(window=window, min_periods=window).mean()

    @staticmethod
    def calculate_macd(
        data: pd.Series,
        short_window: int = 12,
        long_window: int = 26,
        signal_window: int = 9,
    ) -> pd.DataFrame:
        """Moving Average Convergence Divergence."""
        exp_short = data.ewm(span=short_window, adjust=False, min_periods=short_window).mean()
        exp_long = data.ewm(span=long_window, adjust=False, min_periods=long_window).mean()
        macd = exp_short - exp_long
        macd_signal = macd.ewm(span=signal_window, adjust=False, min_periods=signal_window).mean()
        macd_hist = macd - macd_signal
        return pd.DataFrame(
            {
                "MACD": macd,
                "MACD_Signal": macd_signal,
                "MACD_Hist": macd_hist,
            }
        )

    @staticmethod
    def calculate_bollinger_bands(
        data: pd.Series,
        window: int = 20,
        num_std: float = 2.0,
    ) -> pd.DataFrame:
        """Bollinger Bands (upper/lower/width)."""
        rolling_mean = data.rolling(window=window, min_periods=window).mean()
        rolling_std = data.rolling(window=window, min_periods=window).std()
        bb_upper = rolling_mean + num_std * rolling_std
        bb_lower = rolling_mean - num_std * rolling_std
        bb_width = bb_upper - bb_lower
        return pd.DataFrame(
            {
                "BB_Upper": bb_upper,
                "BB_Lower": bb_lower,
                "BB_Width": bb_width,
            }
        )

    @staticmethod
    def calculate_atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14,
    ) -> pd.Series:
        """Average True Range."""
        prev_close = close.shift(1)
        tr_components = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        )
        true_range = tr_components.max(axis=1)
        atr = true_range.rolling(window=window, min_periods=window).mean()
        return atr.rename("ATR")

    @staticmethod
    def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume."""
        direction = np.sign(close.diff()).fillna(0)
        obv = (direction * volume).fillna(0).cumsum()
        return obv.rename("OBV")
