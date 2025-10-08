.. _implementations-data_structures:

===============
Data Structures
===============

When analyzing financial data, unstructured data sets, in this case tick data, are commonly transformed into a structured
format referred to as bars, where a bar represents a row in a table. mlfinlab implements tick, volume, and dollar bars
using traditional standard bar methods as well as the less common information driven bars.

Standard Bars
#############

The four standard bar methods implemented share a similar underlying idea in that they take a sample of data after a
certain threshold is reached and they all result in a time series of Open, High, Low, and Close data.

1. Time bars, are sampled after a fixed interval of time has passed.
2. Tick bars, are sampled after a fixed number of ticks have taken place.
3. Volume bars, are sampled after a fixed number of contracts (volume) has been traded.
4. Dollar bars, are sampled after a fixed monetary amount has been traded.

These bars are used throughout the text book (Advances in Financial Machine Learning, By Marcos Lopez de Prado, 2018,
pg 25) to build the more interesting features for predicting financial time series data.

.. tip::
   A fundamental paper that you need to read to have a better grasp on these concepts is:
   `Easley, David, Marcos M. López de Prado, and Maureen O’Hara. "The volume clock: Insights into the high-frequency
   paradigm." The Journal of Portfolio Management 39.1 (2012): 19-29. <https://jpm.pm-research.com/content/39/1/19.abstract>`_

.. tip::
   A threshold can be either fixed (given as ``float``) or dynamic (given as ``pd.Series``). If a dynamic threshold is used
   then there is no need to declare threshold for every observation. Values are needed only for the first observation
   (or any time before it) and later at times when the threshold is changed to a new value.
   Whenever sampling is made, the most recent threshold level is used.

   **An example for volume bars**
   We have daily observations of prices and volumes:

   +------------+------------+-----------+
   | Time       | Price      | Volume    |
   +============+============+===========+
   | 20.04.2020 | 1000       | 10        |
   +------------+------------+-----------+
   | 21.04.2020 | 990        | 10        |
   +------------+------------+-----------+
   | 22.04.2020 | 1000       | 20        |
   +------------+------------+-----------+
   | 23.04.2020 | 1100       | 10        |
   +------------+------------+-----------+
   | 24.04.2020 | 1000       | 10        |
   +------------+------------+-----------+

   And we set a dynamic threshold:

   +------------+------------+
   | Time       | Threshold  |
   +============+============+
   | 20.04.2020 | 20         |
   +------------+------------+
   | 23.04.2020 | 10         |
   +------------+------------+

   The data will be sampled as follows:

   - 20.04.2020 and 21.04.2020 into one bar, as their volume is 20.
   - 22.04.2020 as a single bar, as its volume is 20.
   - 23.04.2020 as a single bar, as it now fills the lower volume threshold of 10.
   - 24.04.2020 as a single bar again.

Time Bars
*********

These are the traditional open, high, low, close bars that traders are used to seeing. The problem with using this sampling
technique is that information doesn't arrive to market in a chronological clock, i.e. news event don't occur on the hour - every hour.

It is for this reason that Time Bars have poor statistical properties in comparison to the other sampling techniques.

.. py:currentmodule:: mlfinlab.data_structures.time_data_structures
.. autofunction:: get_time_bars


Tick Bars
*********

.. py:currentmodule:: mlfinlab.data_structures.standard_data_structures
.. autofunction:: get_tick_bars

.. code-block::

	from mlfinlab.data_structures import standard_data_structures

	# Tick Bars
	tick = standard_data_structures.get_tick_bars('FILE_PATH', threshold=5500,
	                                               batch_size=1000000, verbose=False)


Volume Bars
***********

.. py:currentmodule:: mlfinlab.data_structures.standard_data_structures
.. autofunction:: get_volume_bars


.. code-block::

	from mlfinlab.data_structures import standard_data_structures

	# Volume Bars
	volume = standard_data_structures.get_volume_bars('FILE_PATH', threshold=28000,
                                                      batch_size=1000000, verbose=False)


Dollar Bars
***********

.. py:currentmodule:: mlfinlab.data_structures.standard_data_structures
.. autofunction::  get_dollar_bars

.. code-block::

	from mlfinlab.data_structures import standard_data_structures

	# Dollar Bars
	dollar = standard_data_structures.get_dollar_bars('FILE_PATH', threshold=70000000,
	                                                   batch_size=1000000, verbose=True)

Statistical Properties
**********************

The chart below that tick, volume, and dollar bars all exhibit a distribution significantly closer to normal - versus
standard time bars:

.. image:: normality_graph.png
   :scale: 70 %
   :align: center

|

------------------------------------

|

Information-Driven Bars
#######################

Information-driven bars are based on the notion of sampling a bar when new information arrives to the market. The two
types of information-driven bars implemented are imbalance bars and run bars. For each type, tick, volume, and dollar bars
are included.


Imbalance Bars
**************

2 types of imbalance bars are implemented in mlfinlab:

    1. Expected number of ticks, defined as EMA (book implementation)
    2. Constant number of expected number of ticks.

Imbalance Bars Generation Algorithm
===================================

Let's discuss the generation of imbalance bars on an example of volume imbalance bars. As it is described in
Advances in Financial Machine Learning book:

First let's define what is the tick rule:

.. math::
    b_t = \begin{cases} b_{t-1},\;\;\;\;\;\;\;\;\;\; \Delta p_t \mbox{=0} \\ |\Delta p_t| / \Delta p_{t},\;\;\;	\Delta p_t \neq\mbox{0} \end{cases}

For any given :math:`t`, where :math:`p_t` is the price associated with :math:`t` and :math:`v_t` is volume, the tick rule :math:`b_t` is defined as:

Tick rule is used as a proxy of trade direction, however, some data providers already provide customers with tick direction, in this case we don't need to calculate tick rule, just use the provided tick direction instead.

Cumulative volume imbalance from :math:`1` to :math:`T` is defined as:

.. math::
    \theta_t = \sum_{t=1}^T b_t*v_t`

Where :math:`T` is the time when the bar is sampled.

Next we need to define :math:`E_0[T]` as the expected number of ticks, the book suggests to use a exponentially weighted moving average (EWMA)
of the expected number of ticks from previously generated bars. Let's introduce the first hyperparameter for imbalance bars generation:
**num_prev_bars** which corresponds to the window used for EWMA calculation.

Here we face the problem of the first bar's generation, because we don't know the expected number of ticks upfront.
To solve this we introduce the second hyperparameter: expected_num_ticks_init which corresponds to initial guess for
**expected number of ticks** before the first imbalance bar is generated.

Bar is sampled when:

.. math::
    |\theta_t| \geq E_0[T]*[2v^+ - E_0[v_t]]

To estimate (expected imbalance) we simply calculate the EWMA of volume imbalance from previous bars, that is why we need
to store volume imbalances in an imbalance array, the window for estimation is either **expected_num_ticks_init** before
the first bar is sampled, or expected number of ticks(:math:`E_0[T]`) * **num_prev_bars** when the first bar is generated.

Note that when we have at least one imbalance bar generated we update :math:`2v^+ - E_0[v_t]` only when the next bar is
sampled and not on every trade observed

Algorithm Logic
===============

Now that we have understood the logic of the imbalance bar generation, let's understand the process in further detail.

.. code-block::

	num_prev_bars = 3
	expected_num_ticks_init = 100000
	expected_num_ticks = expected_num_ticks_init
	cum_theta = 0
	num_ticks = 0
	imbalance_array = []
	imbalance_bars = []
	bar_length_array = []
	for row in data.rows:
	    #track high,low,close, volume info
	    num_ticks += 1
	    tick_rule = get_tick_rule(price, prev_price)
	    volume_imbalance = tick_rule * row['volume']
	    imbalance_array.append(volume_imbalance)
	    cum_theta += volume_imbalance
	    if len(imbalance_bars) == 0 and len(imbalance_array) >= expected_num_ticks_init:
	        expected_imbalance = ewma(imbalance_array, window=expected_num_ticks_init)

	    if abs(cum_theta) >= expected_num_ticks * abs(expected_imbalance):
	        bar = form_bar(open, high, low, close, volume)
	        imbalance_bars.append(bar)
	        bar_length_array.append(num_ticks)
	        cum_theta, num_ticks = 0, 0
	        expected_num_ticks = ewma(bar_lenght_array, window=num_prev_bars)
	        expected_imbalance = ewma(imbalance_array, window = num_prev_bars*expected_num_ticks)


Note that in algorithm pseudo-code we reset :math:`\theta_t` when bar is formed, in our case the formula for :math:`\theta_t` is:

.. math::
    \theta_t = \sum_{t=t^*}^T b_t*v_t


Let's look at dynamics of :math:`|\theta_t|` and :math:`E_0[T] * |2v^+ - E_0[v_t]|` to understand why we decided to
reset :math:`\theta_t` when a bar is formed. The following figure highlights the dynamics when theta value is reset:

.. image:: imbalance_images/theta_reset.png
   :scale: 70 %
   :align: center


Note that on the first set of ticks, the threshold condition is not stable. Remember, before the first bar is generated,
the expected imbalance is calculated on every tick with window = expected_num_ticks_init, that is why it changes with every tick.
After the first bar was generated both expected number of ticks (:math:`E_0[T]`) and expected volume imbalance
(:math:`2v^+ - E_0[v_t]`) are updated only when the next bar is generated

When theta is not reset:

.. image:: imbalance_images/theta_not_reset.png
   :scale: 70 %
   :align: center


The reason for that is due to the fact that theta is accumulated when several bars are generated theta value is not
reset :math:`\Rightarrow` condition is met on small number of ticks :math:`\Rightarrow` length of the next bar converges
to 1 :math:`\Rightarrow` bar is sampled on the next consecutive tick.

The logic described above is implemented in the **mlfinlab** package under ImbalanceBars

Implementation
==============

.. py:currentmodule:: mlfinlab.data_structures.imbalance_data_structures
.. autofunction::  get_ema_dollar_imbalance_bars
.. autofunction:: get_ema_volume_imbalance_bars
.. autofunction:: get_ema_tick_imbalance_bars
.. autofunction:: get_const_dollar_imbalance_bars
.. autofunction:: get_const_volume_imbalance_bars
.. autofunction:: get_const_tick_imbalance_bars


Example
=======

.. code-block::

   from mlfinlab.data_structures import get_ema_dollar_imbalance_bars, get_const_dollar_imbalance_bars

   # EMA, Const Dollar Imbalance Bars
   dollar_imbalance_ema = get_ema_dollar_imbalance_bars('FILE_PATH', num_prev_bars=3, exp_num_ticks_init=100000,
                                                        exp_num_ticks_constraints=[100, 1000], expected_imbalance_window=10000)

   dollar_imbalance_const = get_const_dollar_imbalance_bars('FILE_PATH', exp_num_ticks_init=100000, expected_imbalance_window=10000)

|

-----------------------------

|

Run Bars
********

Run bars share the same mathematical structure as imbalance bars, however, instead of looking at each individual trade,
we are looking at sequences of trades in the same direction. The idea is that we are trying to detect order flow imbalance
caused by actions such as large traders sweeping the order book or iceberg orders.

2 types of run bars are implemented in mlfinlab:

    1. Expected number of ticks, defined as EWMA (book implementation)
    2. Constant number of expected number of ticks.

Implementation
==============

.. py:currentmodule:: mlfinlab.data_structures.run_data_structures
.. autofunction:: get_ema_dollar_run_bars
.. autofunction:: get_ema_volume_run_bars
.. autofunction:: get_ema_tick_run_bars
.. autofunction:: get_const_dollar_run_bars
.. autofunction:: get_const_volume_run_bars
.. autofunction:: get_const_tick_run_bars

Example
=======

.. code-block::

   from mlfinlab.data_structures import get_ema_dollar_run_bars, get_const_dollar_run_bars

   # EMA, Const Dollar Imbalance Bars
   dollar_imbalance_ema = get_ema_dollar_run_bars('FILE_PATH', num_prev_bars=3, exp_num_ticks_init=100000,
                                                   exp_num_ticks_constraints=[100, 1000], expected_imbalance_window=10000)

   dollar_imbalance_const = get_const_dollar_run_bars('FILE_PATH', num_prev_bars=3, exp_num_ticks_init=100000,
                                                      expected_imbalance_window=10000)

|

Footprint Bars
##############

Footprint bars extend MLFinLab's bar sampling capabilities by tracking bid/ask volume at each price level within a bar.
This provides a granular view of order flow and market microstructure that is not available in standard OHLCV bars.

Overview
********

Traditional bars aggregate tick data into Open, High, Low, Close, and Volume (OHLCV) format, but they lose information about:

* How volume was distributed across different price levels
* The aggressiveness of buyers vs sellers at each price
* The Point of Control (POC) - the price with the most volume
* Value Area - the price range where most volume occurred

Footprint bars solve this by maintaining a detailed record of bid/ask volume for every price level traded within each bar.

Key Features
************

* **Price-level granularity**: Track volume at each price within a bar
* **Bid/Ask separation**: Distinguish aggressive buyers from aggressive sellers
* **Universal support**: Works with all bar types (time, dollar, volume, tick, imbalance, run)
* **Multiple input formats**: Supports standard 3-column format with tick rule inference, or 4/5-column format with real bid/ask data
* **OHLC flags**: Mark which prices correspond to Open, High, Low, Close
* **Delta calculation**: Automatically compute bid_vol - ask_vol for order flow analysis

Input Data Formats
******************

Footprint bars support three input formats:

**Format 1: Standard (3 columns)**

.. code-block::

   date_time, price, volume

Volume direction is inferred using the tick rule:
- Price increase → classified as aggressive buy (bid_vol)
- Price decrease → classified as aggressive sell (ask_vol)
- No change → inherits previous tick direction or splits evenly

**Format 2: Bid/Ask (4 columns)** - Recommended

.. code-block::

   date_time, price, bid_qty, ask_qty

Provides real bid/ask separation from exchange data. Total volume = bid_qty + ask_qty.

**Format 3: Full (5 columns)**

.. code-block::

   date_time, price, volume, bid_qty, ask_qty

Includes both total volume and bid/ask breakdown for validation.

Usage Example
*************

**Enable footprint tracking for any bar type:**

.. code-block:: python

   from mlfinlab.data_structures import get_dollar_bars
   import pandas as pd

   # Load tick data with bid/ask information
   data = pd.DataFrame({
       'date_time': pd.date_range('2021-01-01 09:30', periods=1000, freq='1s'),
       'price': [...],      # tick prices
       'bid_qty': [...],    # aggressive buy volume
       'ask_qty': [...]     # aggressive sell volume
   })

   # Generate footprint bars
   result = get_dollar_bars(
       data,
       threshold=70000000,
       enable_footprint=True  # Enable footprint tracking
   )

   # Access results
   bars = result['bars']          # Standard OHLCV DataFrame
   footprint = result['footprint']  # MultiIndex footprint DataFrame

**Footprint DataFrame Structure:**

.. code-block:: python

   # MultiIndex: (bar_timestamp, price)
   # Columns:
   #   bid_vol      - Aggressive buy volume at this price level
   #   ask_vol      - Aggressive sell volume at this price level
   #   total_vol    - Total volume (bid_vol + ask_vol)
   #   delta        - Order flow delta (bid_vol - ask_vol)
   #   is_open      - True if this price is the bar's Open
   #   is_high      - True if this price is the bar's High
   #   is_low       - True if this price is the bar's Low
   #   is_close     - True if this price is the bar's Close

   # Example output:
   #                              bid_vol  ask_vol  total_vol  delta  is_open  is_high  is_low  is_close
   # bar_timestamp       price
   # 2021-01-01 09:30:00 99.95      150      50        200     100     True    False    True     False
   #                     100.00     300     200        500     100    False     True    False      True

**Analyzing footprint data:**

.. code-block:: python

   # Get footprint for a specific bar
   bar_time = footprint.index.get_level_values(0)[0]
   bar_footprint = footprint.loc[bar_time]

   # Find Point of Control (POC) - price with most volume
   poc_price = bar_footprint['total_vol'].idxmax()
   poc_volume = bar_footprint['total_vol'].max()

   # Calculate cumulative delta for the bar
   bar_delta = bar_footprint['delta'].sum()

   # Find value area (70% of volume)
   sorted_by_vol = bar_footprint.sort_values('total_vol', ascending=False)
   total_volume = bar_footprint['total_vol'].sum()
   cumsum = sorted_by_vol['total_vol'].cumsum()
   value_area = sorted_by_vol[cumsum <= total_volume * 0.7]

Supported Bar Types
*******************

Footprint tracking works with all bar sampling methods:

**Standard Bars:**

.. code-block:: python

   from mlfinlab.data_structures import (
       get_dollar_bars, get_volume_bars, get_tick_bars
   )

   # Each supports enable_footprint=True
   result = get_volume_bars(data, threshold=50000, enable_footprint=True)

**Time Bars:**

.. code-block:: python

   from mlfinlab.data_structures import get_time_bars

   result = get_time_bars(
       data,
       resolution='MIN',
       num_units=5,
       enable_footprint=True
   )

**Information-Driven Bars:**

.. code-block:: python

   from mlfinlab.data_structures.imbalance_data_structures import (
       get_ema_dollar_imbalance_bars
   )

   # Footprint + imbalance bars = powerful combination
   bars, thresholds = get_ema_dollar_imbalance_bars(
       data,
       num_prev_bars=3,
       expected_imbalance_window=10000,
       exp_num_ticks_init=20000,
       enable_footprint=True
   )
   # Returns: (dict with bars & footprint, thresholds DataFrame)

Market Microstructure Insights
*******************************

Footprint bars enable advanced order flow analysis:

**1. Aggressive vs Passive Volume**

- ``bid_vol`` represents aggressive buyers (market orders lifting the offer)
- ``ask_vol`` represents aggressive sellers (market orders hitting the bid)
- Delta shows the battle between buyers and sellers

**2. Price Acceptance/Rejection**

.. code-block:: python

   # Prices with high volume = acceptance
   # Prices with low volume = rejection

   for bar_time in footprint.index.get_level_values(0).unique():
       bar_fp = footprint.loc[bar_time]

       # High volume node (HVN) - price acceptance
       hvn = bar_fp[bar_fp['total_vol'] > bar_fp['total_vol'].quantile(0.8)]

       # Low volume node (LVN) - price rejection
       lvn = bar_fp[bar_fp['total_vol'] < bar_fp['total_vol'].quantile(0.2)]

**3. Imbalance at Price Levels**

.. code-block:: python

   # Strong buying at a price level
   strong_buying = bar_footprint[bar_footprint['delta'] > 0].sort_values('delta', ascending=False)

   # Strong selling at a price level
   strong_selling = bar_footprint[bar_footprint['delta'] < 0].sort_values('delta')

   # Balanced (auction in progress)
   balanced = bar_footprint[abs(bar_footprint['delta']) < threshold]

Implementation Notes
********************

**Precision Handling**

Prices are rounded to 8 decimal places to avoid floating-point comparison issues:

.. code-block:: python

   # Internally:
   price = round(price, 8)  # Ensures consistent dictionary keys

**Performance**

- Footprint tracking adds minimal overhead when ``enable_footprint=False`` (default)
- When enabled, memory usage increases proportional to number of unique price levels
- Typical overhead: 10-100x more rows than standard bars (one row per price level per bar)

**Backward Compatibility**

.. code-block:: python

   # Existing code works unchanged
   bars = get_dollar_bars(data, threshold=70000000)
   # Returns: DataFrame (backward compatible)

   # New functionality is opt-in
   result = get_dollar_bars(data, threshold=70000000, enable_footprint=True)
   # Returns: dict with 'bars' and 'footprint' keys

Requirements
************

- **scikit-learn >= 1.2.0** (for compatibility)
- **numpy >= 1.20.0** (for modern numpy API)

References
**********

For more information on order flow and market microstructure:

* Easley, D., López de Prado, M. M., & O'Hara, M. (2012). "Flow toxicity and liquidity in a high-frequency world."
  *The Review of Financial Studies*, 25(5), 1457-1493.

* Steidlmayer, J. P., & Koy, K. (1986). *Markets and Market Logic*. Porcupine Press.

|

-----------------------

|

Research Notebooks
##################

The following research notebooks can be used to better understand the previously discussed data structures

Standard Bars
*************

* `Getting Started`_
* `Sample Techniques`_

.. _Getting Started: https://github.com/hudson-and-thames/research/blob/master/Advances%20in%20Financial%20Machine%20Learning/Financial%20Data%20Structures/Getting%20Started.ipynb
.. _Sample Techniques: https://github.com/hudson-and-thames/research/blob/master/Advances%20in%20Financial%20Machine%20Learning/Financial%20Data%20Structures/Sample_Techniques.ipynb

Imbalance Bars
**************

* `Imbalance Bars`_

.. _Imbalance Bars: https://github.com/hudson-and-thames/research/blob/master/Advances%20in%20Financial%20Machine%20Learning/Financial%20Data%20Structures/Dollar-Imbalance-Bars.ipynb

|

---------------------

|

Data Preparation Tutorial
#########################

First import your tick data.

.. code-block::

   # Required Imports
   import numpy as np
   import pandas as pd

   data = pd.read_csv('data.csv')

In order to utilize the bar sampling methods presented below, our data must first be formatted properly.
Many data vendors will let you choose the format of your raw tick data files. We want to only focus on the following
3 columns: date_time, price, volume. The reason for this is to minimise the size of the csv files and the amount of time
when reading in the files.

Our data is sourced from TickData LLC which provides software called TickWrite, to aid in the formatting of saved files.
This allows us to save csv files in the format date_time, price, volume. (If you don't use TickWrite then make sure to pre-format your files)

For this tutorial we will assume that you need to first do some pre-processing and then save your data to a csv file.

.. code-block::

   # Don't convert to datetime here, it will take forever to convert
   # on account of the sheer size of tick data files.
   date_time = data['Date'] + ' ' + data['Time']
   new_data = pd.concat([date_time, data['Price'], data['Volume']], axis=1)
   new_data.columns = ['date', 'price', 'volume']


Initially, your instinct may be to pass an in-memory DataFrame object but the truth is when you're running the function
in production, your raw tick data csv files will be way too large to hold in memory. We used the subset 2011 to 2019 and
it was more than 25 gigs. It is for this reason that the mlfinlab package suggests using a file path to read the raw data
files from disk.

.. code-block::

	# Save to csv
	new_data.to_csv('FILE_PATH', index=False)
