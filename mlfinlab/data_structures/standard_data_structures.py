"""
Advances in Financial Machine Learning, Marcos Lopez de Prado
Chapter 2: Financial Data Structures

This module contains the functions to help users create structured financial data from raw unstructured data,
in the form of time, tick, volume, and dollar bars.

These bars are used throughout the text book (Advances in Financial Machine Learning, By Marcos Lopez de Prado, 2018,
pg 25) to build the more interesting features for predicting financial time series data.

These financial data structures have better statistical properties when compared to those based on fixed time interval
sampling. A great paper to read more about this is titled: The Volume Clock: Insights into the high frequency paradigm,
Lopez de Prado, et al.

Many of the projects going forward will require Dollar and Volume bars.
"""

# Imports
from typing import Union, Iterable, Optional

import numpy as np
import pandas as pd

from mlfinlab.data_structures.base_bars import BaseBars


class StandardBars(BaseBars):
    """
    Contains all of the logic to construct the standard bars from chapter 2. This class shouldn't be used directly.
    We have added functions to the package such as get_dollar_bars which will create an instance of this
    class and then construct the standard bars, to return to the user.

    This is because we wanted to simplify the logic as much as possible, for the end user.
    """

    def __init__(self, metric: str, threshold: int = 50000, batch_size: int = 20000000, enable_footprint: bool = False):
        """
        Constructor

        :param metric: (str) Type of run bar to create. Example: "dollar_run"
        :param threshold: (int) Threshold at which to sample
        :param batch_size: (int) Number of rows to read in from the csv, per batch
        :param enable_footprint: (bool) Enable footprint tracking with bid/ask volume per price level.
        """
        BaseBars.__init__(self, metric, batch_size, enable_footprint)

        # Threshold at which to sample
        self.threshold = threshold

    def _reset_cache(self):
        """
        Implementation of abstract method _reset_cache for standard bars
        """
        self.open_price = None
        self.high_price, self.low_price = -np.inf, np.inf
        self.cum_statistics = {'cum_ticks': 0, 'cum_dollar_value': 0, 'cum_volume': 0, 'cum_buy_volume': 0}

    def _extract_bars(self, data: Union[list, tuple, np.ndarray]) -> list:
        """
        For loop which compiles the various bars: dollar, volume, or tick.
        We did investigate the use of trying to solve this in a vectorised manner but found that a For loop worked well.

        :param data: (tuple) Contains 3, 4, or 5 columns - date_time, price, volume (or bid_qty, ask_qty).
        :return: (list) Extracted bars
        """

        # Iterate over rows
        list_bars = []

        for row in data:
            # Set variables and detect input format
            date_time = row[0]
            self.tick_num += 1
            price = np.float(row[1])

            # Detect format: 3-column, 4-column, or 5+ column
            if len(row) == 3:
                # Standard format: [date_time, price, volume]
                volume = row[2]
                bid_qty, ask_qty = None, None
            elif len(row) == 4:
                # Bid/ask format: [date_time, price, bid_qty, ask_qty]
                bid_qty = row[2]
                ask_qty = row[3]
                volume = bid_qty + ask_qty
            elif len(row) >= 5:
                # Full format: [date_time, price, volume, bid_qty, ask_qty, ...]
                volume = row[2]
                bid_qty = row[3]
                ask_qty = row[4]
            else:
                raise ValueError(f"Invalid row format: expected 3, 4, or 5+ columns, got {len(row)}")

            dollar_value = price * volume
            signed_tick = self._apply_tick_rule(price)

            if isinstance(self.threshold, (int, float)):
                # If the threshold is fixed, it's used for every sampling
                threshold = self.threshold
            else:
                # If the threshold is changing, then the threshold defined just before
                # sampling time is used
                threshold = self.threshold.iloc[self.threshold.index.get_loc(date_time, method='pad')]

            if self.open_price is None:
                self.open_price = price

            # Update high low prices
            self.high_price, self.low_price = self._update_high_low(price)

            # Calculations
            self.cum_statistics['cum_ticks'] += 1
            self.cum_statistics['cum_dollar_value'] += dollar_value
            self.cum_statistics['cum_volume'] += volume
            if signed_tick == 1:
                self.cum_statistics['cum_buy_volume'] += volume

            # Update footprint with current tick
            self._update_footprint(price, volume, signed_tick, date_time, bid_qty, ask_qty)

            # If threshold reached then take a sample
            if self.cum_statistics[self.metric] >= threshold:  # pylint: disable=eval-used
                # Finalize footprint for completed bar
                self._finalize_footprint(date_time)

                self._create_bars(date_time, price,
                                  self.high_price, self.low_price, list_bars)

                # Reset cache
                self._reset_cache()
        return list_bars


def get_dollar_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], threshold: Union[float, pd.Series] = 70000000,
                    batch_size: int = 20000000, verbose: bool = True, to_csv: bool = False, output_path: Optional[str] = None,
                    enable_footprint: bool = False):
    """
    Creates the dollar bars: date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value.

    Following the paper "The Volume Clock: Insights into the high frequency paradigm" by Lopez de Prado, et al,
    it is suggested that using 1/50 of the average daily dollar value, would result in more desirable statistical
    properties.

    :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing raw tick data
                            in the format[date_time, price, volume] or [date_time, price, bid_qty, ask_qty]
    :param threshold: (float, or pd.Series) A cumulative value above this threshold triggers a sample to be taken.
                      If a series is given, then at each sampling time the closest previous threshold is used.
                      (Values in the series can only be at times when the threshold is changed, not for every observation)
    :param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: (bool) Print out batch numbers (True or False)
    :param to_csv: (bool) Save bars to csv after every batch run (True or False)
    :param output_path: (str) Path to csv file, if to_csv is True
    :param enable_footprint: (bool) Enable footprint tracking with bid/ask volume per price level.
    :return: (pd.DataFrame or dict) Dataframe of dollar bars, or dict {'bars': df, 'footprint': df} if enable_footprint=True
    """

    bars = StandardBars(metric='cum_dollar_value', threshold=threshold, batch_size=batch_size, enable_footprint=enable_footprint)
    dollar_bars = bars.batch_run(file_path_or_df=file_path_or_df, verbose=verbose, to_csv=to_csv, output_path=output_path)

    if enable_footprint:
        footprint = bars.get_footprint()
        return {'bars': dollar_bars, 'footprint': footprint}
    return dollar_bars


def get_volume_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], threshold: Union[float, pd.Series] = 70000000,
                    batch_size: int = 20000000, verbose: bool = True, to_csv: bool = False, output_path: Optional[str] = None,
                    enable_footprint: bool = False):
    """
    Creates the volume bars: date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value.

    Following the paper "The Volume Clock: Insights into the high frequency paradigm" by Lopez de Prado, et al,
    it is suggested that using 1/50 of the average daily volume, would result in more desirable statistical properties.

    :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing raw tick data
                            in the format[date_time, price, volume] or [date_time, price, bid_qty, ask_qty]
    :param threshold: (float, or pd.Series) A cumulative value above this threshold triggers a sample to be taken.
                      If a series is given, then at each sampling time the closest previous threshold is used.
                      (Values in the series can only be at times when the threshold is changed, not for every observation)
    :param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: (bool) Print out batch numbers (True or False)
    :param to_csv: (bool) Save bars to csv after every batch run (True or False)
    :param output_path: (str) Path to csv file, if to_csv is True
    :param enable_footprint: (bool) Enable footprint tracking with bid/ask volume per price level.
    :return: (pd.DataFrame or dict) Dataframe of volume bars, or dict {'bars': df, 'footprint': df} if enable_footprint=True
    """
    bars = StandardBars(metric='cum_volume', threshold=threshold, batch_size=batch_size, enable_footprint=enable_footprint)
    volume_bars = bars.batch_run(file_path_or_df=file_path_or_df, verbose=verbose, to_csv=to_csv, output_path=output_path)

    if enable_footprint:
        footprint = bars.get_footprint()
        return {'bars': volume_bars, 'footprint': footprint}
    return volume_bars


def get_tick_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], threshold: Union[float, pd.Series] = 70000000,
                  batch_size: int = 20000000, verbose: bool = True, to_csv: bool = False, output_path: Optional[str] = None,
                  enable_footprint: bool = False):
    """
    Creates the tick bars: date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value.

    :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing raw tick data
                             in the format[date_time, price, volume] or [date_time, price, bid_qty, ask_qty]
    :param threshold: (float, or pd.Series) A cumulative value above this threshold triggers a sample to be taken.
                      If a series is given, then at each sampling time the closest previous threshold is used.
                      (Values in the series can only be at times when the threshold is changed, not for every observation)
    :param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: (bool) Print out batch numbers (True or False)
    :param to_csv: (bool) Save bars to csv after every batch run (True or False)
    :param output_path: (str) Path to csv file, if to_csv is True
    :param enable_footprint: (bool) Enable footprint tracking with bid/ask volume per price level.
    :return: (pd.DataFrame or dict) Dataframe of tick bars, or dict {'bars': df, 'footprint': df} if enable_footprint=True
    """
    bars = StandardBars(metric='cum_ticks', threshold=threshold, batch_size=batch_size, enable_footprint=enable_footprint)
    tick_bars = bars.batch_run(file_path_or_df=file_path_or_df, verbose=verbose, to_csv=to_csv, output_path=output_path)

    if enable_footprint:
        footprint = bars.get_footprint()
        return {'bars': tick_bars, 'footprint': footprint}
    return tick_bars
