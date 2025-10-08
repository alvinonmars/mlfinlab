# Strategy Specification: Footprint-Based Key Level Trading

> **Document Type**: Strategy Implementation Specification
> **Created**: 2025-10-06
> **Last Updated**: 2025-10-06
> **Status**: Design Phase - Technical Architecture Defined

---

## Strategy Overview

### Core Philosophy

**"Only trade at key levels"** - A discretionary trading strategy distilled from manual trading experience, focusing on high-probability setups at critical price zones.

### Key Characteristics

1. **Rule-Based Signal Generation**: Structure patterns are well-defined (2K continuation, engulfing reversal)
2. **External Dependency**: Key level detection is a separate, independent system
3. **Multi-Dimensional Filters**: Delta distribution, POC, volume, body size
4. **Risk-Reward Driven**: Stops and targets derived from key levels + ATR, minimum R:R = 1:1

**This is NOT a pattern discovery problem - it's a quantification problem.**

---

## Trading Instrument

- **Primary**: CME GC (Gold Futures) main contract
- **Timeframe**: 5-minute footprint bars
- **Signal Frequency**: 10-40 signals per day
- **Market Hours**: CME trading hours (need to specify session if relevant)

---

## Key Level System (External Component)

### Source
Independent detection system based on:
- Price extremes (swing highs/lows)
- Volume profile distribution
- Historical data analysis

### Key Level Definition

```
Key Level = {
    price: float,          // Core price level
    width: int,            // Width in ticks (typically ≤ 15 ticks)
    type: enum,            // 'support' | 'resistance'
    strength: float,       // 0-1 confidence score (optional)
    detected_at: datetime  // When this level was identified
}
```

### Level Types

**Resistance (R)**: Key levels above current price
- Acts as ceiling
- Price may reverse or consolidate when approaching

**Support (S)**: Key levels below current price
- Acts as floor
- Price may bounce or consolidate when approaching

**In-Zone**: Current bar overlaps with key level
- Critical decision zone
- May break through or reverse

### Narrow Width Characteristic

**Width typically ≤ 15 ticks** means:
- Precise levels, not broad zones
- Requires tick-accurate detection
- High signal-to-noise ratio

---

## Signal Generation Rules

### Pattern 1: 2K Continuation Structure

**Definition**: Two consecutive bars moving in the same direction

**Entry Conditions**:
1. Bar 1 and Bar 2 have same directional bias (both bullish or both bearish)
2. **Multi-dimensional filters** (all must pass):
   - Delta distribution: Favorable pattern (p-type for long, b-type for short)
   - POC delta: Aligns with direction
   - Volume: Above average
   - Body size: Above average

**Position Logic**:

**Long Setup**:
- 2K bullish continuation detected
- **AND** distance to nearest resistance > 2 ATR
- **OR** no resistance level above (clear path)
- **Interpretation**: Trend likely to continue

**Short Setup**:
- 2K bearish continuation detected
- **AND** distance to nearest support > 2 ATR
- **OR** no support level below

---

### Pattern 2: Engulfing Reversal Structure

**Definition**: Two consecutive bars, second bar engulfs first AND opposite direction

**Entry Conditions**:
1. Bar 1 and Bar 2 have opposite directional bias
2. Bar 2's body fully engulfs Bar 1's body (high/low range)
3. **Same multi-dimensional filters** as continuation

**Position Logic**:

**Long Setup**:
- Bearish→Bullish engulfing detected
- **AND** current bars are touching/penetrating support level below
- **Interpretation**: Support has held, reversal initiated

**Short Setup**:
- Bullish→Bearish engulfing detected
- **AND** current bars are touching/penetrating resistance level above

---

## Multi-Dimensional Filter Details

### Filter 1: Delta Distribution Pattern

**Objective**: Determine if aggressive orders align with direction

**P-Type (Bullish)**:
```
Upper half of bar: Positive delta (buyers aggressive at higher prices)
Lower half of bar: Neutral or negative delta
→ Buying pressure increasing with price = bullish
```

**B-Type (Bearish)**:
```
Upper half of bar: Neutral or positive delta
Lower half of bar: Negative delta (sellers aggressive at lower prices)
→ Selling pressure increasing with lower prices = bearish
```

**Implementation**:
```python
def analyze_delta_distribution(footprint, direction):
    mid_price = footprint['price'].median()
    upper_half_delta = footprint[footprint['price'] > mid_price]['delta'].sum()
    lower_half_delta = footprint[footprint['price'] <= mid_price]['delta'].sum()

    if direction == 'long':
        # Expect p-type: upper_delta > 0 and dominant
        return upper_half_delta > 0 and upper_half_delta > abs(lower_half_delta)
    else:
        # Expect b-type: lower_delta < 0 and dominant
        return lower_half_delta < 0 and abs(lower_half_delta) > upper_half_delta
```

---

### Filter 2: POC Delta Alignment

**Objective**: Ensure highest-volume price level shows directional conviction

```python
def check_poc_delta(footprint, direction):
    poc_price = footprint.loc[footprint['total_vol'].idxmax(), 'price']
    poc_delta = footprint.loc[footprint['price'] == poc_price, 'delta'].iloc[0]

    return np.sign(poc_delta) == direction  # 1 for long, -1 for short
```

**Rationale**: If most volume occurred at a price with opposite delta, the move may be weak.

---

### Filter 3: Volume Above Average

**Objective**: Ensure significant participation

```python
def check_volume(bar, lookback_bars=20):
    avg_volume = lookback_bars['volume'].mean()
    return bar.volume > avg_volume
```

**Adaptive**: Uses rolling average, adjusts to market activity

---

### Filter 4: Body Size Above Average

**Objective**: Ensure decisive price movement (not doji/indecision)

```python
def check_body_size(bar, lookback_bars=20):
    avg_body = abs(lookback_bars['close'] - lookback_bars['open']).mean()
    body_size = abs(bar.close - bar.open)
    return body_size > avg_body
```

---

## Position Relationship Analysis

### Critical Concept: Relative Position to Key Levels

The strategy's core insight is that **setup validity depends on WHERE the pattern occurs relative to key levels**.

### Distance Measurement

**Normalized by ATR**:
```python
distance_atr = abs(current_price - key_level_price) / atr_value
```

**Why ATR normalization?**
- Market volatility changes over time
- 10 ticks may be "far" in quiet markets, "near" in volatile markets
- ATR provides adaptive scale

---

### Long Signal Position Rules

| Pattern       | Key Level Requirement                              | Logic                                    |
|---------------|---------------------------------------------------|------------------------------------------|
| Continuation  | Distance to resistance > 2 ATR OR no resistance   | Trend has room to continue               |
| Reversal      | Touching support (within level width + tolerance) | Support has held, bounce likely          |

### Short Signal Position Rules (Symmetric)

| Pattern       | Key Level Requirement                              | Logic                                    |
|---------------|---------------------------------------------------|------------------------------------------|
| Continuation  | Distance to support > 2 ATR OR no support         | Downtrend has room to continue           |
| Reversal      | Touching resistance (within level width + tolerance) | Resistance has held, rejection likely |

---

### "Touching" Definition

```python
def is_touching_level(bar, key_level, tolerance_ticks=2):
    """
    Check if bar overlaps with key level zone

    Key level zone = [price - width/2, price + width/2]
    With tolerance = [price - width/2 - tolerance, price + width/2 + tolerance]
    """
    lower_bound = key_level.price - key_level.width / 2 - tolerance_ticks
    upper_bound = key_level.price + key_level.width / 2 + tolerance_ticks

    # Bar's high/low must penetrate the zone
    return bar.low <= upper_bound and bar.high >= lower_bound
```

---

## Risk Management

### Stop Loss Calculation

**Long Position**:
```python
def calculate_stop_loss_long(entry_price, support_level, atr):
    if support_level exists:
        # Below support zone
        stop = support_level.price - support_level.width / 2 - atr * 0.5
    else:
        # Fallback: 2 ATR below entry
        stop = entry_price - atr * 2

    return stop
```

**Short Position** (symmetric):
```python
def calculate_stop_loss_short(entry_price, resistance_level, atr):
    if resistance_level exists:
        # Above resistance zone
        stop = resistance_level.price + resistance_level.width / 2 + atr * 0.5
    else:
        # Fallback: 2 ATR above entry
        stop = entry_price + atr * 2

    return stop
```

---

### Take Profit Calculation

**Minimum Risk-Reward = 1:1**

```python
def calculate_take_profit_long(entry_price, stop_loss, resistance_level):
    risk = entry_price - stop_loss

    if resistance_level exists:
        potential_target = resistance_level.price - resistance_level.width / 2
        reward_to_resistance = potential_target - entry_price

        if reward_to_resistance >= risk:
            # Resistance is far enough, use it
            return potential_target

    # Fallback: 1.5:1 risk-reward
    return entry_price + risk * 1.5
```

**Logic**:
1. Calculate risk (distance to stop)
2. Check if next key level provides >= 1:1 R:R
3. If yes, target that level
4. If no, use 1.5:1 multiplier

---

## Technical Architecture

### Overall System Design

```
┌──────────────────────────────────────────────────────────────┐
│                     EXTERNAL SYSTEM                          │
│              Key Level Detection Engine                      │
│  (Independent, produces key levels from historical data)     │
└──────────────────────────────────────────────────────────────┘
                            ↓
                  (File / API / Database)
                            ↓
┌──────────────────────────────────────────────────────────────┐
│                   STRATEGY PIPELINE                          │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Layer 1: Data Ingestion                            │    │
│  │  • Footprint Bar Loader (MLFinLab)                 │    │
│  │  • Key Level Manager (reads external system)       │    │
│  │  • Market Context (ATR, volume averages)           │    │
│  └────────────────────────────────────────────────────┘    │
│                            ↓                                 │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Layer 2: Signal Generation (CORE RULES)            │    │
│  │  • Structure Detector                              │    │
│  │    - 2K Continuation                               │    │
│  │    - Engulfing Reversal                            │    │
│  │  • Filter Engine                                   │    │
│  │    - Delta Distribution (p/b type)                 │    │
│  │    - POC Delta Alignment                           │    │
│  │    - Volume vs Average                             │    │
│  │    - Body Size vs Average                          │    │
│  │  • Position Analyzer                               │    │
│  │    - Distance to nearest resistance/support        │    │
│  │    - Touching detection                            │    │
│  │  • Signal Validator                                │    │
│  │    - Continuation: check distance > 2 ATR          │    │
│  │    - Reversal: check touching level                │    │
│  └────────────────────────────────────────────────────┘    │
│                            ↓                                 │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Layer 3: OPTIONAL ML Meta-Labeling Filter          │    │
│  │  • Extract features from candidate signals         │    │
│  │  • Predict: "Should we trade this signal?"         │    │
│  │  • Filter: Only signals with prob > threshold      │    │
│  └────────────────────────────────────────────────────┘    │
│                            ↓                                 │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Layer 4: Risk Management                           │    │
│  │  • Stop Loss: Based on opposite key level + ATR    │    │
│  │  • Take Profit: Based on target level or R:R       │    │
│  │  • Position Size: (if applicable)                  │    │
│  └────────────────────────────────────────────────────┘    │
│                            ↓                                 │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Layer 5: Execution & Backtest                      │    │
│  │  • Backtest Engine (batch historical validation)   │    │
│  │  • Walk-Forward Validator                          │    │
│  │  • Live Trading Interface (optional)               │    │
│  └────────────────────────────────────────────────────┘    │
│                            ↓                                 │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Layer 6: Analysis & Reporting                      │    │
│  │  • MLFinLab Backtest Statistics                    │    │
│  │    - Sharpe Ratio, PSR, DSR                        │    │
│  │    - Bet Concentration                             │    │
│  │    - Drawdown Analysis                             │    │
│  │  • Signal Quality Metrics                          │    │
│  │    - Win rate, Avg R:R, Expectancy                 │    │
│  │  • Visualization                                   │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## ML Role: Three Possible Modes

### Mode 1: Pure Rule-Based (Phase 1 - RECOMMENDED START)

**No ML, just rules**

```
Key Levels + Footprint → Rule Engine → Signals
```

**Advantages**:
- ✅ Fully explainable (matches manual trading logic)
- ✅ Fast execution (milliseconds)
- ✅ Easy to debug
- ✅ No training data required initially

**Use this if**: Manual trading win rate is already good (>55%)

---

### Mode 2: Rule Engine + ML Meta-Labeling (Phase 2 - IF NEEDED)

**ML filters signals, doesn't generate them**

```
Rule Engine → Candidate Signals (30/day)
                    ↓
        ML Filter (Meta-Labeling)
                    ↓
         Final Signals (15/day, higher win rate)
```

**How it works**:

1. **Rule engine generates ALL candidate signals** (same rules as Phase 1)
2. **ML model answers**: "Should we trade this particular signal?"
3. **Only trade signals where ML confidence > threshold** (e.g., 70%)

**MLFinLab Implementation**:
```python
from mlfinlab.labeling import get_events, get_bins

# Label historical candidate signals
events = get_events(
    close=bars['close'],
    t_events=candidate_signals.index,
    pt_sl=[1, 1],  # Your stops
    target=atr,
    side_prediction=candidate_signals['direction']  # From rules
)

meta_labels = get_bins(events, bars['close'])
# Train model: X=signal_features, y=meta_labels['bin'] (0/1)
```

**Use this if**:
- Rule engine produces too many signals (quantity over quality)
- Win rate <55%
- Want to improve precision

---

### Mode 3: Adaptive Rule Engine (Phase 3 - ADVANCED)

**ML identifies market regime, rules adapt**

```
ML Regime Detector → Market State (trend/chop)
                            ↓
                    Adjust rule parameters
                            ↓
                    Rule Engine → Signals
```

**Example**:
- **Trend regime detected**: Favor continuation setups, relax "2 ATR distance" to 1.5 ATR
- **Choppy regime detected**: Only reversal setups, strict filters

**Use this if**: Strategy performance varies significantly across market conditions

---

## Key Level Integration Strategies

### Critical Issue: Avoiding Look-Ahead Bias

**Problem**: Key levels are detected using historical data

**Wrong** (❌ Look-ahead bias):
```python
# Using levels detected TODAY to backtest YESTERDAY
levels_today = load_key_levels('2025-01-06')
signal_yesterday = check_signal(bar_2025_01_05, levels_today)  # WRONG!
```

**Correct** (✅ Point-in-time):
```python
# Using levels that EXISTED at that time
levels_at_time = load_key_levels_as_of('2025-01-05')
signal_yesterday = check_signal(bar_2025_01_05, levels_at_time)  # Correct
```

---

### Integration Solution 1: File-Based (SIMPLE - RECOMMENDED START)

**External System Produces**:
```
data/key_levels/
    ├── GC_2025-01-01.json
    ├── GC_2025-01-02.json
    ├── GC_2025-01-03.json
    ...
```

**File Format**:
```json
{
  "symbol": "GC",
  "as_of_date": "2025-01-03",
  "levels": [
    {
      "id": "R1",
      "type": "resistance",
      "price": 2050.5,
      "width": 10,
      "strength": 0.85,
      "detected_at": "2025-01-02T23:00:00Z"
    },
    {
      "id": "S1",
      "type": "support",
      "price": 2020.3,
      "width": 8,
      "strength": 0.92,
      "detected_at": "2025-01-01T23:00:00Z"
    }
  ]
}
```

**Strategy Loads**:
```python
class KeyLevelManager:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_levels_as_of(self, date):
        """Load levels that existed at this date"""
        filepath = f"{self.data_dir}/GC_{date.strftime('%Y-%m-%d')}.json"
        with open(filepath) as f:
            return json.load(f)['levels']

    def get_nearest_resistance(self, current_price, levels):
        resistances = [l for l in levels if l['type'] == 'resistance' and l['price'] > current_price]
        return min(resistances, key=lambda x: x['price']) if resistances else None
```

**Advantages**:
- ✅ Simple: No database, no API
- ✅ Decoupled: Two systems completely independent
- ✅ Version control: Easy to track changes
- ✅ Debugging: Can manually inspect/edit files

**Disadvantages**:
- ❌ Not real-time (file updates required)
- ❌ Polling overhead (check for new files)

**Best for**: Daily/intraday trading with end-of-day level updates

---

### Integration Solution 2: Database (REAL-TIME)

**Schema**:
```sql
CREATE TABLE key_levels (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10),
    type VARCHAR(20),  -- 'resistance' or 'support'
    price DECIMAL(10, 2),
    width_ticks INT,
    strength DECIMAL(3, 2),
    detected_at TIMESTAMP,
    valid_from TIMESTAMP,  -- When this level became active
    valid_until TIMESTAMP,  -- NULL if still active
    is_active BOOLEAN
);

CREATE INDEX idx_symbol_active_price ON key_levels(symbol, is_active, price);
CREATE INDEX idx_valid_time ON key_levels(valid_from, valid_until);
```

**Point-in-Time Query** (CRITICAL for backtesting):
```sql
-- Get levels that were active on 2025-01-05 at 14:30
SELECT * FROM key_levels
WHERE symbol = 'GC'
  AND valid_from <= '2025-01-05 14:30:00'
  AND (valid_until IS NULL OR valid_until > '2025-01-05 14:30:00')
ORDER BY price;
```

**Advantages**:
- ✅ Real-time updates
- ✅ Historical point-in-time queries (no look-ahead bias)
- ✅ Flexible filtering

**Disadvantages**:
- ❌ Requires database setup
- ❌ More complex

**Best for**: Real-time trading, multiple strategies sharing levels

---

### Integration Solution 3: API Service (MICROSERVICES)

**External System Exposes**:
```
GET /api/v1/key-levels?symbol=GC&as_of=2025-01-05T14:30:00Z

Response:
{
  "symbol": "GC",
  "as_of": "2025-01-05T14:30:00Z",
  "levels": [...],
  "updated_at": "2025-01-05T14:25:00Z"
}
```

**Advantages**:
- ✅ Standard interface (REST/gRPC)
- ✅ Supports multiple clients
- ✅ Versioned API

**Disadvantages**:
- ❌ Network latency
- ❌ Requires API server

**Best for**: Production systems, multiple teams

---

### Recommended Approach: Start File-Based, Design for Database

```python
class KeyLevelManager:
    def __init__(self, backend='file', **kwargs):
        if backend == 'file':
            self.backend = FileLevelBackend(kwargs['data_dir'])
        elif backend == 'database':
            self.backend = DatabaseLevelBackend(kwargs['db_conn'])
        elif backend == 'api':
            self.backend = APILevelBackend(kwargs['api_url'])

    # Uniform interface regardless of backend
    def get_levels_as_of(self, symbol, datetime):
        return self.backend.get_levels_as_of(symbol, datetime)
```

**Start simple, scale when needed.**

---

## MLFinLab Integration Points

### 1. Data Structures (✅ COMPLETED)

```python
from mlfinlab.data_structures import get_time_bars

result = get_time_bars(
    tick_data,
    resolution='5Min',
    enable_footprint=True
)

bars = result['bars']          # OHLCV + metadata
footprint = result['footprint'] # MultiIndex (bar_timestamp, price)
```

**Usage**: Generate 5-min footprint bars for GC

---

### 2. Backtest Statistics (⭐ PRIMARY VALUE)

```python
from mlfinlab.backtest_statistics import (
    sharpe_ratio,
    probabilistic_sharpe_ratio,
    deflated_sharpe_ratio,
    bets_concentration
)

# After running backtest
strategy_returns = backtest_engine.run(bars, key_levels)

# MLFinLab metrics
sr = sharpe_ratio(strategy_returns, entries_per_year=252)

# PSR: "What's probability that true SR > 1.0?"
psr = probabilistic_sharpe_ratio(
    observed_sr=sr,
    benchmark_sr=1.0,
    number_of_returns=len(strategy_returns)
)

# DSR: Corrects for multiple testing (if you tried many variations)
dsr = deflated_sharpe_ratio(
    observed_sr=sr,
    sr_estimates=[1.2, 0.9, 1.1, ...],  # All variants tested
    number_of_returns=len(strategy_returns)
)

# Concentration: Are returns from few trades or many?
conc = bets_concentration(strategy_returns)
# If conc close to 1 → overfitting risk
```

**Overfitting Detection**:
- ✅ PSR > 0.95: Statistically significant
- ✅ DSR > 0.95: Survives multiple testing
- ✅ Low concentration: Returns distributed
- ❌ High concentration: Few trades dominate (red flag)

---

### 3. Meta-Labeling (OPTIONAL - If win rate needs improvement)

```python
from mlfinlab.labeling import get_events, get_bins
from mlfinlab.cross_validation import PurgedKFold
from sklearn.ensemble import RandomForestClassifier

# Step 1: Your rules generate candidate signals
candidates = rule_engine.generate_all_signals(bars, key_levels)
# E.g., 1000 signals over backtest period

# Step 2: Label outcomes
events = get_events(
    close=bars['close'],
    t_events=candidates.index,
    pt_sl=[1, 1],  # Based on your stop/target logic
    target=atr_series,
    side_prediction=candidates['direction']  # 1 or -1
)

meta_labels = get_bins(events, bars['close'])
# meta_labels['bin']:
#   1 = signal was correct (hit profit target)
#   0 = signal was wrong (hit stop loss)

# Step 3: Extract features from each signal
features = pd.DataFrame({
    'delta_strength': candidates['delta_sum'] / candidates['volume'],
    'distance_to_level_atr': candidates['distance_to_nearest_level'] / candidates['atr'],
    'volume_ratio': candidates['volume'] / candidates['avg_volume'],
    'body_ratio': candidates['body_size'] / candidates['avg_body'],
    'key_level_strength': candidates['nearest_level_strength'],
    # ... more features
})

# Step 4: Train Meta-Labeling model with Purged CV
X = features
y = meta_labels['bin']

cv = PurgedKFold(n_splits=5, samples_info_sets=events['t1'])
model = RandomForestClassifier(max_depth=3, min_samples_leaf=50, class_weight='balanced')

# Cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.3f}")

# Step 5: Use in production
model.fit(X, y)

new_candidate = rule_engine.generate_signal(latest_bars, latest_levels)
new_features = extract_features(new_candidate)
prob = model.predict_proba(new_features)[0, 1]

if prob > 0.7:  # Only trade high-confidence signals
    execute_trade(new_candidate)
```

**Expected Improvement**:
- Before: 1000 signals, 52% win rate
- After: 500 signals, 65% win rate
- **Fewer trades, higher quality**

---

### 4. Sequential Bootstrapping (OPTIONAL - If signals overlap)

If multiple signals' holding periods overlap (e.g., entered at 10:00, 10:05, 10:10 all still open):

```python
from mlfinlab.ensemble import SequentiallyBootstrappedBaggingClassifier

# When training meta-labeling model
clf = SequentiallyBootstrappedBaggingClassifier(
    base_estimator=RandomForestClassifier(max_depth=3),
    samples_info_sets=events['t1'],  # Signal exit times
    price_bars=bars,
    oob_score=True,
    n_estimators=100
)

clf.fit(X, y, sample_weight=sample_weights)
```

**Benefit**: Accounts for sample overlap, reduces overfitting

---

## Implementation Phases

### Phase 1: Rule Engine Validation (2-4 weeks)

**Goal**: Prove rules accurately replicate manual trading logic

**Tasks**:
1. ✅ Implement footprint bar generation (MLFinLab - already done)
2. ✅ Implement key level loader (file-based)
3. ✅ Implement structure detectors (2K continuation, engulfing)
4. ✅ Implement multi-dimensional filters (delta, POC, volume, body)
5. ✅ Implement position analyzer (distance, touching logic)
6. ✅ Implement risk calculator (stops/targets from levels)
7. ✅ Backtest on 1-3 months historical data
8. ✅ Compare with manual trading records (if available)

**Success Criteria**:
- [ ] Detects all manually traded signals (high recall)
- [ ] False positive rate acceptable (<30%)
- [ ] Win rate close to manual trading (±5%)
- [ ] Signals occur 10-40 per day (matches expectation)

**Deliverables**:
- Working rule engine
- Backtest report showing:
  - Total signals generated
  - Win rate, avg R:R
  - Sharpe ratio
  - Match rate vs manual trades (if records available)

---

### Phase 2: Optimization & ML Enhancement (4-6 weeks)

**Goal**: Improve signal quality using ML (if needed)

**Tasks**:
1. ✅ Implement meta-labeling
2. ✅ Feature engineering for signal filtering
3. ✅ Train ML filter with PurgedKFold CV
4. ✅ Implement MLFinLab backtest statistics (PSR, DSR, concentration)
5. ✅ Walk-forward validation
6. ✅ Compare ML-filtered vs pure rules

**Success Criteria**:
- [ ] ML filter improves win rate by ≥5%
- [ ] PSR > 0.95 (statistically significant)
- [ ] DSR > 0.95 (survives multiple testing)
- [ ] Low bet concentration (<0.3)
- [ ] Stable performance across walk-forward periods

**Deliverables**:
- ML-enhanced pipeline
- Comprehensive backtest report
- Feature importance analysis

---

### Phase 3: Production Deployment (4-8 weeks)

**Goal**: Real-time signal generation for live trading

**Tasks**:
1. ✅ Convert batch pipeline to streaming
2. ✅ Real-time key level integration (database or API)
3. ✅ Real-time footprint bar updates
4. ✅ Latency optimization (<1 second per signal)
5. ✅ Monitoring dashboards
6. ✅ Alerting system
7. ✅ Paper trading validation

**Success Criteria**:
- [ ] Signals generated within 10 seconds of bar close
- [ ] No missed signals (100% uptime)
- [ ] Paper trading matches backtest performance (±10%)
- [ ] Key level updates propagate in <30 seconds

**Deliverables**:
- Production-ready system
- Monitoring dashboard
- Alert system
- Paper trading results

---

## Core Questions to Answer

### 1. Key Level Interface

**Q1.1**: What format does your key level detector output?
- [ ] JSON files
- [ ] CSV files
- [ ] Database tables
- [ ] API endpoint
- [ ] Other: ________________

**Q1.2**: How frequently are key levels updated?
- [ ] Real-time (continuous)
- [ ] Every 5 minutes (with bars)
- [ ] Every hour
- [ ] Daily (end of session)
- [ ] Weekly
- [ ] On-demand

**Q1.3**: Are historical key levels available for backtesting?
- [ ] Yes, files for each historical date
- [ ] Yes, database with temporal queries
- [ ] No, need to regenerate
- [ ] Partial (last N days only)

**Q1.4**: Sample key level data structure
```
Please provide example:
{
  "price": ?,
  "width": ?,
  "type": ?,
  ...
}
```

**Your Answers**:
- Q1.1: ________________
- Q1.2: ________________
- Q1.3: ________________
- Q1.4: ________________

---

### 2. Historical Data & Manual Trading

**Q2.1**: Do you have manual trading records?
- [ ] Yes, detailed (entry/exit/reason)
- [ ] Yes, partial (entry/exit only)
- [ ] No

**Q2.2**: If yes, how many manual trades in records?
**Answer**: ________________

**Q2.3**: Approximate win rate from manual trading?
**Answer**: ________________%

**Q2.4**: Available historical tick data time range?
**Answer**: From ________________ to ________________

**Q2.5**: Data quality?
- [ ] Complete (no gaps)
- [ ] Minor gaps (<1%)
- [ ] Significant gaps (need cleaning)

---

### 3. ML Requirements

**Q3.1**: Is ML filtering needed immediately?
- [ ] No, start with pure rules
- [ ] Yes, manual win rate not high enough
- [ ] Unsure, let's test Phase 1 first

**Q3.2**: If using ML, what's the goal?
- [ ] Improve win rate
- [ ] Reduce signal quantity (keep best)
- [ ] Adapt to market regimes
- [ ] Other: ________________

**Q3.3**: Minimum acceptable win rate?
**Answer**: ________________%

**Q3.4**: Training data size expectation
- [ ] <1000 signals (need to be careful about overfitting)
- [ ] 1000-5000 signals (reasonable)
- [ ] >5000 signals (good)

---

### 4. Execution Requirements

**Q4.1**: Is this for live trading or research only?
- [ ] Research/backtesting only (no rush)
- [ ] Paper trading (simulated real-time)
- [ ] Live trading (production critical)

**Q4.2**: If live trading, latency requirement?
**Answer**: Signal must be generated within ________ seconds of bar close

**Q4.3**: Execution platform?
- [ ] Manual execution (visual alerts)
- [ ] Semi-automated (confirm before execute)
- [ ] Fully automated
- [ ] Not decided yet

---

### 5. Performance Targets

**Q5.1**: Target Sharpe Ratio?
**Answer**: ________________

**Q5.2**: Max acceptable drawdown?
**Answer**: ________________%

**Q5.3**: Minimum trades per day to be useful?
**Answer**: ________________

**Q5.4**: Maximum trades per day (too many)?
**Answer**: ________________

---

## Next Steps

1. **Answer the 5 sections of questions above**
2. **Provide sample key level data file** (if available)
3. **Provide sample manual trade log** (if available, for validation)
4. **Confirm integration strategy**: File-based to start?
5. **Set timeline**: When do you need Phase 1 complete?

Once you answer these, I can:
- Refine technical implementation details
- Create concrete code interfaces
- Estimate development timeline
- Identify potential blockers

---

## Appendix: Code Pseudocode

### Minimal Viable Signal Generator

```python
class StrategyEngine:
    def __init__(self, key_level_manager, atr_calculator):
        self.levels = key_level_manager
        self.atr = atr_calculator
        self.structure_detector = StructureDetector()
        self.filter = FilterEngine()
        self.position_analyzer = PositionAnalyzer(self.levels, self.atr)
        self.risk_mgr = RiskManager()

    def process_bar(self, current_bar, previous_bar):
        """Process new 5-min bar"""

        # Detect structures
        continuation_signal = self.structure_detector.detect_continuation(previous_bar, current_bar)
        reversal_signal = self.structure_detector.detect_reversal(previous_bar, current_bar)

        signals = []

        # Process continuation
        if continuation_signal:
            if self.filter.check_all(current_bar, continuation_signal.direction):
                position = self.position_analyzer.analyze(current_bar, continuation_signal)
                if position.is_valid:
                    stops = self.risk_mgr.calculate_stops(continuation_signal, position)
                    signals.append(TradingSignal(
                        type='continuation',
                        direction=continuation_signal.direction,
                        entry=current_bar.close,
                        stop=stops.stop_loss,
                        target=stops.take_profit,
                        timestamp=current_bar.timestamp
                    ))

        # Process reversal (similar logic)
        if reversal_signal:
            ...

        return signals
```

---

**For AI Assistants**: When continuing this discussion, read this file to understand the complete strategy specification, then refer to PIPELINE_DESIGN.md for general ML pipeline architecture.
