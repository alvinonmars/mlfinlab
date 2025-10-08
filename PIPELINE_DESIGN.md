# Footprint-Based Event Detection Pipeline - Design Document

> **Document Status**: Design Phase
> **Created**: 2025-10-06
> **Last Updated**: 2025-10-06
> **Related Project**: Independent footprint event detection system

---

## Executive Summary

**Objective**: Design and implement a complete ML pipeline for strategy development based on footprint bar event detection.

**Core Insight**: Traditional OHLCV shows "results" (price movement), footprint shows "process" (order flow dynamics).

**Pipeline Must Answer**:
1. **What**: What events are worth trading?
2. **When**: When to enter/exit?
3. **How Much**: What position size?

---

## Architecture Decision: Three Modes

### Mode A: Research-Driven
```
Explore → Validate → Productionize
```
- **Suitable for**: Strategy still in exploration phase
- **Advantage**: High flexibility, fast iteration
- **Challenge**: Research code hard to productionize

### Mode B: Production-First
```
Define interfaces → Implement → Backtest validation
```
- **Suitable for**: Event detection logic already clear
- **Advantage**: High code quality, easy deployment
- **Challenge**: Slow early iteration

### Mode C: Hybrid (⭐ RECOMMENDED)
```
Research Environment (Jupyter) ←→ Production Code (Shared Core Library)
```
- **Research**: Fast experiments, visualization
- **Production**: Strict testing, performance optimization
- **Shared**: Event detectors, feature extractors, evaluation metrics

**Your Choice**: [ ] Mode A  [ ] Mode B  [ ] Mode C

---

## Pipeline: 6-Stage Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: Data Pipeline                                         │
│  Tick Data → Footprint Bars → Storage/Versioning                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2: Event Detection (🎯 Core)                             │
│  Footprint Bars → Event Detector → Event Timestamps + Metadata  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 3: Feature Engineering                                   │
│  Events + Footprint → Raw/Derived/Cross-Bar Features            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 4: Labeling Strategy                                     │
│  Events + Price Data → Labels (Direction/Magnitude/Outcome)     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 5: Sample Management                                     │
│  Events + Labels → Weighted Samples (Uniqueness/Importance)     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 6: Model Training & Validation                           │
│  Samples → Purged CV → Model → Backtest Statistics              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: Data Pipeline

### Core Decisions

#### 1.1 Data Storage Format

**Challenge**: Footprint data volume >> regular OHLCV (each bar has N price levels)

**Options**:

| Format      | Pros                          | Cons                        | Use Case                    |
|-------------|-------------------------------|-----------------------------|-----------------------------|
| **Parquet** | Columnar, fast filtering      | Not great for time-series   | Multi-symbol batch research |
| **HDF5**    | Fast time-series queries      | Complex nested structure    | Single-symbol high-freq     |
| **Database**| Query flexibility, concurrent | Setup overhead              | Production, multiple users  |
| **CSV**     | Simple, human-readable        | Slow, large files           | Small datasets only         |

**Your Choice**: ________________

**Rationale**: ________________

---

#### 1.2 Computation Strategy

**Options**:

**A. Pre-compute Everything**
- Generate all footprint bars upfront
- Store both bars and footprint DataFrames
- Pros: Fast research iteration
- Cons: Storage cost, re-compute on parameter change

**B. On-Demand Computation**
- Calculate footprint bars when needed
- Pros: No storage, always fresh
- Cons: Slow, not suitable for iteration

**C. Hybrid (⭐ RECOMMENDED)**
- Pre-compute and store bars (OHLCV + metadata)
- Compute footprint features on-demand during research
- Cache frequently used derived features
- Pros: Balance between speed and flexibility

**Your Choice**: [ ] A  [ ] B  [ ] C

---

#### 1.3 Data Versioning

**Challenge**: Parameter changes require re-computation
- Different thresholds (dollar bar: $1M vs $500K)
- Different input formats (with/without bid_qty, ask_qty)
- Algorithm updates (bug fixes, improvements)

**Strategy**:
```
data/
  ├── raw/
  │   └── tick_data/
  ├── processed/
  │   ├── v1_dollar_1M/
  │   │   ├── bars.parquet
  │   │   └── metadata.json  # stores parameters
  │   └── v2_dollar_500K/
  │       ├── bars.parquet
  │       └── metadata.json
  └── features/
      └── v1_dollar_1M/
          └── footprint_features.parquet
```

**Your Versioning Scheme**: ________________

---

### Key Questions for Stage 1

**Q1.1**: Tick data volume? (GB/day, time range)
**Answer**: ________________

**Q1.2**: Single symbol or multi-symbol?
**Answer**: ________________

**Q1.3**: Real-time requirements or batch only?
**Answer**: ________________

---

## Stage 2: Event Detection (🎯 CORE)

### Event Taxonomy

#### 2.1 Delta Events (Order Flow Imbalance)

| Event Type            | Definition                                      | Signal                          |
|-----------------------|-------------------------------------------------|---------------------------------|
| **Delta Divergence**  | Price ↑ but cumulative delta ↓                  | Weakness, potential reversal    |
| **Delta Spike**       | Sudden one-sided pressure                       | Strong directional move pending |
| **Delta Exhaustion**  | Large delta but price doesn't move              | Absorption, potential reversal  |
| **Delta Reversal**    | Delta changes from strongly +/- to opposite     | Regime change                   |

#### 2.2 Volume Profile Events

| Event Type            | Definition                                      | Signal                          |
|-----------------------|-------------------------------------------------|---------------------------------|
| **POC Migration**     | Point of Control moves rapidly                  | Acceptance of new price level   |
| **Value Area Breakout**| Price breaks out of VA High/Low                | Potential trend                 |
| **Volume Node Break** | Price breaks through high-volume level          | Significant move                |
| **Low Volume Node**   | Price stuck in low-volume area                  | Fast move likely                |

#### 2.3 Microstructure Events

| Event Type               | Definition                                   | Signal                          |
|--------------------------|----------------------------------------------|---------------------------------|
| **Bid/Ask Ratio Shift**  | Ratio suddenly changes regime                | Order flow direction change     |
| **Single Price Accumulation** | Large volume at one price level         | Institution building position   |
| **Volume at Touch**      | Price repeatedly tests a level with volume  | Support/Resistance              |
| **Iceberg Detection**    | Continuous absorption at one price           | Hidden large order              |

#### 2.4 Time-Based Events

| Event Type            | Definition                                      | Signal                          |
|-----------------------|-------------------------------------------------|---------------------------------|
| **Session Open**      | Specific footprint pattern at market open       | Day's direction                 |
| **Pre-News**          | Footprint changes before scheduled news         | Informed flow                   |
| **Post-News**         | Footprint reaction to news                      | Market interpretation           |

---

### Event Detector Design Patterns

#### Pattern 1: Rule-Based Detector

```python
# Pseudo-code
if delta > threshold and price_change < min_move:
    event = "absorption"
    strength = delta / threshold
```

**Pros**: Interpretable, debuggable, fast
**Cons**: Parameter sensitive, can't capture complex patterns
**Suitable for**: Well-understood events, quick prototyping

---

#### Pattern 2: Pattern-Based Detector

```python
# Pseudo-code
if match_sequence(footprint_history, pattern="accumulation → breakout → retest"):
    event = "pattern_detected"
```

**Pros**: Matches trader intuition, captures sequences
**Cons**: Requires labeled examples, pattern library maintenance
**Suitable for**: Events with clear visual signatures

---

#### Pattern 3: Statistical Detector

```python
# Pseudo-code
delta_zscore = (delta - rolling_mean) / rolling_std
if abs(delta_zscore) > 3:
    event = "anomaly"
```

**Pros**: Adaptive to market regime, no manual thresholds
**Cons**: Requires sufficient history, lag in regime change
**Suitable for**: Anomaly detection, regime-aware strategies

---

#### Pattern 4: ML-Based Detector (Future Extension)

```python
# Pseudo-code
features = extract_footprint_features(recent_bars)
probability = classifier.predict_proba(features)
if probability > threshold:
    event = "ml_detected"
```

**Pros**: Captures complex non-linear patterns
**Cons**: Black box, overfitting risk, requires labeled data
**Suitable for**: Mature strategies with large labeled dataset

---

### Your Event Detection Design

**Q2.1**: Which event types are you detecting? (Check all that apply)
- [ ] Delta Divergence
- [ ] Delta Spike
- [ ] Delta Exhaustion
- [ ] POC Migration
- [ ] Value Area Breakout
- [ ] Volume Node Break
- [ ] Bid/Ask Ratio Shift
- [ ] Single Price Accumulation
- [ ] Other: ________________

**Q2.2**: Which detector pattern(s)? (Check all that apply)
- [ ] Rule-Based
- [ ] Pattern-Based
- [ ] Statistical
- [ ] ML-Based
- [ ] Hybrid: ________________

**Q2.3**: Is event detection logic already implemented?
- [ ] Yes, well-tested
- [ ] Partially, needs refinement
- [ ] No, still exploring

**Q2.4**: Do events provide direction signals?
- [ ] Yes (long/short)
- [ ] No (only "anomaly detected")
- [ ] Sometimes (depends on event type)

---

## Stage 3: Feature Engineering

### Feature Hierarchy

#### Level 1: Raw Footprint Features (Direct Extraction)

```python
# From each bar's footprint DataFrame
features = {
    'delta_sum': footprint['delta'].sum(),
    'delta_mean': footprint['delta'].mean(),
    'delta_std': footprint['delta'].std(),
    'delta_max': footprint['delta'].max(),
    'delta_min': footprint['delta'].min(),

    'bid_vol_sum': footprint['bid_vol'].sum(),
    'ask_vol_sum': footprint['ask_vol'].sum(),
    'bid_ask_ratio': bid_vol_sum / ask_vol_sum,

    'poc_price': footprint.loc[footprint['total_vol'].idxmax(), 'price'],
    'poc_volume': footprint['total_vol'].max(),
    'poc_delta': footprint.loc[footprint['total_vol'].idxmax(), 'delta'],

    'value_area_high': calculate_value_area_high(footprint),  # 70% volume
    'value_area_low': calculate_value_area_low(footprint),
    'value_area_range': va_high - va_low,

    'price_levels': len(footprint),  # Number of distinct prices
    'price_range': footprint.index.get_level_values('price').max() - min(),
}
```

#### Level 2: Derived Footprint Features (Computed)

```python
# Computed from raw features
features = {
    # Delta-based
    'cumulative_delta_5': last_5_bars['delta_sum'].sum(),
    'delta_momentum': (delta_sum_now - delta_sum_prev) / delta_sum_prev,
    'volume_weighted_delta': (delta_sum / volume_sum),
    'delta_at_open': footprint[footprint['is_open']]['delta'].sum(),
    'delta_at_close': footprint[footprint['is_close']]['delta'].sum(),

    # Volume profile
    'poc_distance_from_close': abs(poc_price - close_price) / close_price,
    'va_percentile': where_in_value_area(close_price, va_high, va_low),

    # Microstructure
    'bid_concentration': gini_coefficient(footprint['bid_vol']),
    'ask_concentration': gini_coefficient(footprint['ask_vol']),
    'imbalance_at_extremes': (delta_at_high + delta_at_low) / 2,
}
```

#### Level 3: Cross-Bar Features (Time Series)

```python
# Comparing current bar with history
features = {
    # Divergence
    'price_delta_divergence': correlation(price_changes[-10:], cumulative_delta[-10:]),

    # POC dynamics
    'poc_migration_speed': (poc_price_now - poc_price_5bars_ago) / 5,
    'poc_stability': std(poc_prices[-10:]),

    # Volume profile similarity
    'vp_similarity_to_prev': cosine_similarity(current_vp, previous_vp),
    'vp_shape_change': detect_shape_change(current_vp, previous_vp),

    # Regime features
    'bid_ask_regime': detect_regime_shift(bid_ask_ratios[-20:]),
}
```

---

### MLFinLab Integration

#### 3.1 Fractional Differentiation

**Challenge**: Footprint features may be non-stationary

```python
from mlfinlab.features.fracdiff import frac_diff_ffd

# Make cumulative_delta stationary while preserving memory
stationary_delta = frac_diff_ffd(cumulative_delta, d=0.5, threshold=1e-5)
```

**When to use**: Time-series features like cumulative delta, POC prices

---

#### 3.2 Entropy Features

**Challenge**: Quantify information content of footprint

```python
from mlfinlab.microstructural_features import get_shannon_entropy

# Encode footprint delta into discrete bins
delta_encoded = encode_array(footprint['delta'], num_letters=10)
delta_entropy = get_shannon_entropy(delta_encoded)
```

**When to use**: Measure market randomness vs structure

---

#### 3.3 Microstructural Features

**From footprint, you can calculate**:
- **Kyle's Lambda**: Price impact per unit volume
- **VPIN**: Volume-synchronized probability of informed trading
- **Roll Measure**: Effective spread estimation

These are in `mlfinlab.microstructural_features` and can use footprint data as input.

---

### Feature Selection Strategy

**Challenge**: Footprint can generate 100+ features → overfitting risk

**MLFinLab Tools**:
1. **Clustered Feature Importance**: Group correlated features, analyze at cluster level
2. **MDA/MDI/SFI**: Identify most predictive features
3. **PCA validation**: Check if important features align with principal components

**Your Strategy**: ________________

---

### Your Feature Engineering Plan

**Q3.1**: Which feature levels will you use?
- [ ] Level 1: Raw only
- [ ] Level 1 + Level 2
- [ ] All three levels

**Q3.2**: Do you have domain knowledge about which features are most important?
**Answer**: ________________

**Q3.3**: Will you use fractional differentiation?
- [ ] Yes, for all time-series features
- [ ] Yes, selectively
- [ ] No, features are already stationary
- [ ] Unknown, need to test

**Q3.4**: Estimated total number of features?
**Answer**: ________________

---

## Stage 4: Labeling Strategy

### The Core Challenge: Time Misalignment

**Traditional Labeling Assumption**:
```
t0: Enter position → t1: Exit at barrier or time limit
Label = outcome at t1
```

**Event-Based Strategy Reality**:
```
t0: Event detected → t1: Enter position → t2: Event ends → t3: Exit position
Label = ??? (what time range?)
```

**This is a fundamental design choice!**

---

### Solution 1: Event-Driven Labeling

**Approach**: Label based on price movement during event's "active period"

```python
# Pseudo-code
event_start = event_timestamp
event_end = event_timestamp + event_duration  # How to determine?

price_change = (price_at_end - price_at_start) / price_at_start

if price_change > threshold:
    label = 1  # Successful event
elif price_change < -threshold:
    label = -1  # Failed event
else:
    label = 0  # No clear outcome
```

**MLFinLab Tool**: `trend_scanning_labels`
- Fits multiple regressions from t to t+L
- Selects regression with maximum t-value for slope
- Can use t-value magnitude as label

**Pros**: Aligned with strategy logic
**Cons**: Requires defining "event duration" (hard!)

**When to use**: Events have clear start/end

---

### Solution 2: Meta-Labeling Approach (⭐ RECOMMENDED)

**Approach**: Decouple event detection from trade filtering

```
Primary Model: Event Detector
  ↓ (outputs: event detected, suggested direction)

Secondary Model: Meta-Labeling
  ↓ (outputs: should we trade this event? probability)

Trade Decision: Combine both
```

**Implementation**:

```python
# Step 1: Event detector provides "side" (direction)
side_prediction = pd.Series({
    event_timestamp_1: 1,   # Event suggests long
    event_timestamp_2: -1,  # Event suggests short
    ...
})

# Step 2: Use Triple-Barrier with side prediction
from mlfinlab.labeling import get_events, get_bins

events = get_events(
    close=bars['close'],
    t_events=event_timestamps,  # From your detector
    pt_sl=[1, 1],  # Symmetric barriers
    target=daily_volatility,
    side_prediction=side_prediction  # From your detector
)

# Meta-labels: Did the event lead to profit?
meta_labels = get_bins(events, bars['close'])
# meta_labels['bin'] ∈ {0, 1}
#   1 = Event was correct, hit profit target
#   0 = Event was wrong, hit stop loss
```

**Step 3: Train secondary model**
```python
# Features: All your footprint features
# Target: meta_labels['bin']
# Goal: Predict which detected events will be profitable
```

**Pros**:
- Matches MLFinLab philosophy
- Separates event quality from trade filtering
- Can improve F1-score significantly

**Cons**:
- Requires two models
- More complex pipeline

**When to use**: Event detector has decent recall but poor precision

---

### Solution 3: Outcome-Based Labeling

**Approach**: Fixed time horizon after event, with volatility-adjusted barriers

```python
from mlfinlab.labeling import add_vertical_barrier, get_events

# Give each event a "validity period"
vertical_barriers = add_vertical_barrier(
    t_events=event_timestamps,
    close=bars['close'],
    num_days=5  # Or num_bars for non-daily data
)

# Triple-barrier with barriers scaled by recent volatility
events = get_events(
    close=bars['close'],
    t_events=event_timestamps,
    pt_sl=[2, 1],  # Asymmetric: 2x vol for profit, 1x for stop
    target=daily_volatility,  # Or event-specific volatility
    vertical_barrier_times=vertical_barriers
)

labels = get_bins(events, bars['close'])
# labels['bin'] ∈ {-1, 0, 1}
#   1 = Hit upper barrier (profit)
#  -1 = Hit lower barrier (stop loss)
#   0 = Hit time barrier (no clear outcome)
```

**Pros**:
- Simple, well-tested (Triple-Barrier is standard)
- Handles both direction and magnitude

**Cons**:
- Time barrier is arbitrary
- May not align with event lifecycle

**When to use**: Events don't have clear end points

---

### Your Labeling Choice

**Q4.1**: Which labeling approach?
- [ ] Solution 1: Event-Driven (Trend Scanning)
- [ ] Solution 2: Meta-Labeling (⭐ Recommended)
- [ ] Solution 3: Outcome-Based (Triple-Barrier)
- [ ] Custom: ________________

**Q4.2**: Does your event detector provide direction?
- [ ] Yes, always (long/short)
- [ ] Sometimes (depends on event type)
- [ ] No, only "anomaly detected"

**Q4.3**: How do you define "event ends"?
**Answer**: ________________

**Q4.4**: What is acceptable holding period range?
**Answer**: ________________ (e.g., "5-20 bars", "1-4 hours")

---

## Stage 5: Sample Management

### Footprint-Specific Challenges

1. **High-Frequency Events**: May detect events every few bars
2. **Event Duration**: An absorption event may span multiple bars
3. **Label Overlap**: Adjacent events' labels may overlap in time

**Impact**: Standard ML assumes i.i.d. samples, but financial events are NOT!

---

### MLFinLab Sample Management Tools

#### 5.1 Sample Uniqueness

**Concept**: Quantify how much unique information each sample contains

```python
from mlfinlab.sampling.concurrent import get_av_uniqueness_from_triple_barrier

# Input: events DataFrame from get_events()
# Output: Series with average uniqueness per event
uniqueness = get_av_uniqueness_from_triple_barrier(
    triple_barrier_events=events,
    close=bars['close'],
    num_threads=4
)

# uniqueness ranges from 0 (completely overlapped) to 1 (unique)
```

**Use case**:
- Filter out highly overlapped samples
- Weight samples by uniqueness in training

---

#### 5.2 Sequential Bootstrapping

**Concept**: When sampling for bagging, prioritize samples with high uniqueness

```python
from mlfinlab.sampling.bootstrapping import get_ind_matrix, seq_bootstrap

# Build indicator matrix (rows=time, cols=samples)
ind_mat = get_ind_matrix(events)

# Generate bootstrap samples that maximize uniqueness
bootstrap_indices = seq_bootstrap(
    ind_mat=ind_mat,
    sample_length=100,  # Number of samples to draw
    warmup_samples=None,
    verbose=True
)
```

**Use case**:
- Training ensemble models
- Creating diverse folds for cross-validation

---

#### 5.3 Sample Weights

**Three weighting schemes**:

**A. By Return and Uniqueness** (⭐ Recommended)
```python
from mlfinlab.sample_weights.attribution import get_weights_by_return

weights = get_weights_by_return(
    triple_barrier_events=events,
    close=bars['close'],
    num_threads=4
)
# Higher weight = larger return + higher uniqueness
```

**B. By Time Decay**
```python
from mlfinlab.sample_weights.attribution import get_weights_by_time_decay

weights = get_weights_by_time_decay(
    triple_barrier_events=events,
    close=bars['close'],
    decay=0.5,  # Exponential decay factor
    num_threads=4
)
# Recent samples get higher weight
```

**C. By Event Strength** (Custom)
```python
# Weight by footprint signal strength
event_strength = pd.Series({
    event_ts: abs(delta_at_event),  # Example: delta magnitude
    ...
})

weights = uniqueness * event_strength  # Combine uniqueness with domain knowledge
```

---

### Your Sample Management Strategy

**Q5.1**: Expected event frequency?
**Answer**: ________________ (e.g., "10-20 events per day")

**Q5.2**: Do events often overlap in time?
- [ ] Yes, frequently
- [ ] Sometimes
- [ ] Rarely

**Q5.3**: Which sample weighting scheme?
- [ ] By return and uniqueness
- [ ] By time decay
- [ ] By event strength
- [ ] Combination: ________________
- [ ] No weighting (equal weight)

**Q5.4**: Will you filter samples by minimum uniqueness?
- [ ] Yes, threshold = ________________
- [ ] No, keep all samples

---

## Stage 6: Model Training & Validation

### Cross-Validation Strategy

**Standard K-Fold WILL LEAK DATA!**

**Why?**
- Bar t's footprint contains info from bar t-1
- Events span multiple bars
- Train/test split in time creates overlap

---

### MLFinLab Solution: Purged K-Fold

```python
from mlfinlab.cross_validation import PurgedKFold

cv = PurgedKFold(
    n_splits=5,
    samples_info_sets=events['t1'],  # End time of each event
    pct_embargo=0.01  # 1% embargo after each test fold
)

# Use with sklearn
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=cv)
```

**What it does**:
1. **Purging**: Remove training samples that overlap with test samples
2. **Embargo**: Add gap after each test fold to prevent leakage

---

### Combinatorial Purged CV (Advanced)

**Concept**: Generate multiple backtest paths, get distribution of Sharpe ratios

```python
from mlfinlab.cross_validation import CombinatorialPurgedKFold

cpcv = CombinatorialPurgedKFold(
    n_splits=6,
    n_test_splits=2,
    samples_info_sets=events['t1']
)

# Generates C(6,2) = 15 different train/test combinations
# Each combination is a different backtest path
```

**Output**: Distribution of performance metrics instead of single number

**Use case**:
- Assess strategy robustness
- Detect overfitting (high variance in performance)

---

### Model Selection

#### Simple Models (⭐ Recommended Starting Point)

**A. Logistic Regression with L1**
```python
from sklearn.linear_model import LogisticRegressionCV

model = LogisticRegressionCV(
    penalty='l1',
    solver='saga',
    cv=cv,  # Use PurgedKFold
    class_weight='balanced'
)
```
**Pros**: Feature selection built-in, interpretable, fast
**Cons**: Assumes linear relationships

---

**B. Random Forest**
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,  # Prevent overfitting
    min_samples_leaf=50,  # Prevent overfitting
    class_weight='balanced'
)
```
**Pros**: Captures non-linearity, feature importance
**Cons**: Can overfit, harder to interpret

---

#### MLFinLab Ensemble

**Sequential Bootstrapped Bagging**
```python
from mlfinlab.ensemble import SequentiallyBootstrappedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier

base = DecisionTreeClassifier(max_depth=3)

model = SequentiallyBootstrappedBaggingClassifier(
    base_estimator=base,
    samples_info_sets=events['t1'],
    price_bars=bars,
    oob_score=True,
    n_estimators=100
)

model.fit(X, y, sample_weight=weights)
```

**Advantage**: Uses sequential bootstrapping → higher sample uniqueness

---

#### Complex Models (Use with Caution)

**C. Gradient Boosting**
- High risk of overfitting
- Use early stopping with validation set

**D. Neural Networks**
- Requires large dataset (1000+ samples)
- Very high overfitting risk
- Only if simple models fail

---

### Backtest Validation

**Beyond Accuracy**: Financial ML requires specialized metrics

#### MLFinLab Backtest Statistics

```python
from mlfinlab.backtest_statistics import (
    sharpe_ratio,
    probabilistic_sharpe_ratio,
    deflated_sharpe_ratio,
    bets_concentration
)

# Calculate returns from predictions
strategy_returns = calculate_returns(predictions, actual_prices)

# Sharpe Ratio
sr = sharpe_ratio(strategy_returns, entries_per_year=252)

# Probabilistic Sharpe Ratio (PSR)
# "What's the probability that true SR > benchmark SR?"
psr = probabilistic_sharpe_ratio(
    observed_sr=sr,
    benchmark_sr=1.0,  # Your target
    number_of_returns=len(strategy_returns)
)

# Deflated Sharpe Ratio (DSR)
# Corrects for multiple testing (you tried many strategies)
dsr = deflated_sharpe_ratio(
    observed_sr=sr,
    sr_estimates=[1.2, 0.9, 1.5, ...],  # All strategies you tested
    number_of_returns=len(strategy_returns)
)

# Bets Concentration
# "Are returns coming from few bets or many?"
concentration = bets_concentration(strategy_returns)
# Close to 0 = uniform, close to 1 = concentrated (overfitting sign!)
```

---

### Overfitting Detection Checklist

- [ ] **PSR > 0.95**: Strategy SR is statistically significant
- [ ] **DSR > 0.95**: Strategy survives multiple testing correction
- [ ] **Low concentration**: Returns distributed across many bets
- [ ] **Stable performance across folds**: Low variance in CPCV
- [ ] **Feature importance stable**: Top features consistent across folds
- [ ] **Out-of-sample period**: Test on completely held-out data

---

### Your Model Training Plan

**Q6.1**: Starting model choice?
- [ ] Logistic Regression
- [ ] Random Forest
- [ ] Sequential Bootstrapped Bagging
- [ ] Other: ________________

**Q6.2**: Expected sample size?
**Answer**: ________________ events

**Q6.3**: What is your target metric?
- [ ] Sharpe Ratio > ________________
- [ ] Accuracy > ________________%
- [ ] F1-Score > ________________
- [ ] Other: ________________

**Q6.4**: Overfitting prevention strategy?
- [ ] Purged K-Fold CV
- [ ] Combinatorial Purged CV
- [ ] Walk-forward analysis
- [ ] Multiple strategies: ________________

---

## Critical Design Decisions

### Decision Matrix

| Decision                  | Options                          | Your Choice | Rationale |
|---------------------------|----------------------------------|-------------|-----------|
| **Pipeline Mode**         | Research / Production / Hybrid   |             |           |
| **Data Storage**          | Parquet / HDF5 / Database / CSV  |             |           |
| **Computation Strategy**  | Pre-compute / On-demand / Hybrid |             |           |
| **Event Detector Type**   | Rule / Pattern / Statistical / ML|             |           |
| **Labeling Approach**     | Event-Driven / Meta / Outcome    |             |           |
| **Sample Weighting**      | Return-Uniqueness / Decay / Strength |         |           |
| **Model Type**            | Linear / Tree / Ensemble         |             |           |
| **CV Strategy**           | Purged K-Fold / CPCV / Both      |             |           |

---

## Implementation Phases

### Phase 1: Proof of Concept (2-4 weeks)

**Goal**: Validate that footprint events contain predictive signal

**Tasks**:
- [ ] Implement basic event detector (rule-based)
- [ ] Generate footprint bars for sample period (1-3 months)
- [ ] Label events with simple triple-barrier
- [ ] Extract 10-20 key footprint features
- [ ] Train simple model (Logistic Regression)
- [ ] Backtest with basic metrics (Sharpe, accuracy)

**Success Criteria**:
- SR > 0.5 (or your baseline)
- Accuracy > 55%
- Events occur frequently enough (10+ per week)

---

### Phase 2: Pipeline Hardening (4-6 weeks)

**Goal**: Build robust, maintainable pipeline

**Tasks**:
- [ ] Implement data versioning system
- [ ] Create modular components (detector, features, labeler)
- [ ] Add unit tests for each component
- [ ] Implement Purged K-Fold CV
- [ ] Add MLFinLab sample management (uniqueness, weights)
- [ ] Implement feature importance analysis
- [ ] Add comprehensive backtest statistics

**Success Criteria**:
- All components have >80% test coverage
- Pipeline runs end-to-end without manual intervention
- Can regenerate results with different parameters

---

### Phase 3: Optimization (4-8 weeks)

**Goal**: Maximize performance and robustness

**Tasks**:
- [ ] Expand event detector (add more event types)
- [ ] Feature engineering iteration (test 50+ features)
- [ ] Model selection (compare 3-5 models)
- [ ] Hyperparameter optimization
- [ ] Walk-forward analysis on longer period
- [ ] Implement CPCV for robustness check
- [ ] Deploy to paper trading (if applicable)

**Success Criteria**:
- PSR > 0.95
- DSR > 0.95 (if multiple strategies tested)
- Low bet concentration
- Stable performance across time periods

---

### Phase 4: Production (Ongoing)

**Goal**: Real-time deployment and monitoring

**Tasks**:
- [ ] Convert batch pipeline to streaming (if needed)
- [ ] Implement real-time event detection
- [ ] Set up monitoring dashboards
- [ ] Implement alerting for degradation
- [ ] Regular model retraining schedule
- [ ] Performance tracking and reporting

**Success Criteria**:
- Real-time latency < 1 second (or your requirement)
- Live performance matches backtest (accounting for costs)
- Zero downtime deployments

---

## Component Interfaces (For Modular Design)

### Interface 1: FootprintBarGenerator

```python
class FootprintBarGenerator:
    def __init__(self, threshold, metric='dollar'):
        pass

    def generate_bars(self, tick_data) -> Dict[str, pd.DataFrame]:
        """
        Returns: {
            'bars': DataFrame with OHLCV,
            'footprint': MultiIndex DataFrame (bar_timestamp, price)
        }
        """
        pass

    def save(self, path):
        pass

    @staticmethod
    def load(path):
        pass
```

---

### Interface 2: EventDetector

```python
class EventDetector:
    def __init__(self, detector_type, params):
        pass

    def detect(self, bars, footprint) -> pd.DataFrame:
        """
        Returns: DataFrame with columns:
            - timestamp: Event time
            - event_type: str (e.g., 'delta_divergence')
            - side: int (-1, 0, 1) if directional
            - strength: float (0-1 confidence/strength)
            - metadata: dict (event-specific data)
        """
        pass

    def visualize_event(self, event_id, bars, footprint):
        """For debugging"""
        pass
```

---

### Interface 3: FeatureExtractor

```python
class FeatureExtractor:
    def __init__(self, feature_config):
        pass

    def extract(self, events, bars, footprint) -> pd.DataFrame:
        """
        Returns: DataFrame with index=event_timestamps, columns=features
        """
        pass

    def get_feature_names(self) -> List[str]:
        pass

    def get_feature_importance(self, model) -> pd.Series:
        pass
```

---

### Interface 4: Labeler

```python
class Labeler:
    def __init__(self, labeling_strategy, params):
        pass

    def label(self, events, bars) -> pd.DataFrame:
        """
        Returns: DataFrame with:
            - index: event_timestamps
            - label: int or float
            - t1: label end time
            - confidence: optional
        """
        pass
```

---

### Interface 5: Pipeline Orchestrator

```python
class StrategyPipeline:
    def __init__(self, config):
        self.bar_generator = FootprintBarGenerator(**config['bars'])
        self.detector = EventDetector(**config['detector'])
        self.features = FeatureExtractor(**config['features'])
        self.labeler = Labeler(**config['labeler'])
        self.model = None

    def run_research(self, tick_data, train_period, test_period):
        """Full research workflow"""
        pass

    def train(self, tick_data, period):
        """Train model on period"""
        pass

    def backtest(self, tick_data, period):
        """Backtest on period"""
        pass

    def predict_live(self, recent_ticks):
        """Real-time prediction"""
        pass
```

---

## Next Steps

### Immediate Actions

1. **Answer the 8 key questions** throughout this document
2. **Fill in the Decision Matrix** with your choices
3. **Review Phase 1 tasks** and estimate timeline
4. **Set up development environment**:
   - Install MLFinLab
   - Prepare sample tick data
   - Set up Jupyter for research

### First Code to Write

**Priority 1**: Basic event detector
```python
# Start with simplest event: Delta Spike
def detect_delta_spike(footprint, threshold=2.0):
    delta_zscore = (footprint['delta'].sum() - rolling_mean) / rolling_std
    if abs(delta_zscore) > threshold:
        return True, np.sign(delta_zscore)
    return False, 0
```

**Priority 2**: Test footprint bar generation
```python
from mlfinlab.data_structures import get_dollar_bars

result = get_dollar_bars(
    tick_data,
    threshold=1000000,
    enable_footprint=True
)

bars = result['bars']
footprint = result['footprint']

# Verify correctness
assert len(bars) > 0
assert footprint.groupby('bar_timestamp').size().min() >= 1
```

**Priority 3**: Simple backtest
```python
# Detect events → Label → Calculate returns
events = detect_all_events(bars, footprint)
labels = simple_labeling(events, bars)
returns = calculate_returns(events, bars)
sr = sharpe_ratio(returns)
print(f"Sharpe: {sr:.2f}")
```

---

## References and Resources

### MLFinLab Documentation
- **Data Structures**: `docs/source/implementations/data_structures.rst`
- **Labeling**: `docs/source/labeling/tb_meta_labeling.rst`
- **Feature Importance**: `docs/source/implementations/feature_importance.rst`
- **Sampling**: `docs/source/implementations/sampling.rst`
- **Cross-Validation**: `docs/source/implementations/cross_validation.rst`
- **Backtest Statistics**: `docs/source/implementations/backtest_statistics.rst`

### Key Papers
- **Advances in Financial Machine Learning** - Marcos Lopez de Prado
  - Chapter 3: Meta-Labeling
  - Chapter 4: Sampling
  - Chapter 7: Cross-Validation
  - Chapter 8: Feature Importance
  - Chapter 18-19: Microstructural Features

### Example Workflow
See `CLAUDE.md` Section "Create Footprint Bars" for end-to-end example.

---

## Document Maintenance

**When to update this document**:
- [ ] After answering key questions
- [ ] After making major design decisions
- [ ] After completing each phase
- [ ] When discovering new challenges
- [ ] When changing architecture

**Versioning**:
- Current version: 1.0
- Last updated: 2025-10-06
- Next review: ________________

---

## Appendix: Quick Command Reference

```bash
# Activate environment
source /opt/homebrew/Caskroom/miniconda/base/bin/activate cs

# Test MLFinLab import
python -c "from mlfinlab.data_structures import get_dollar_bars; print('OK')"

# Run research notebook
cd research/
jupyter notebook footprint_pipeline.ipynb

# Run full pipeline
python pipeline/main.py --config configs/default.yaml

# Run tests
pytest tests/ -v

# Build documentation
cd docs/ && make html
```

---

**For AI Assistants**: When resuming this discussion, read this file completely before continuing. Pay special attention to answered questions and the Decision Matrix to understand the user's choices.
