# MLFinLab - AI Assistant Context

> **For AI Assistants**: Read this file at the start of new conversations to understand the project architecture and recent changes.

---

## Project Overview

**MLFinLab** is a Python package for financial machine learning based on Dr. Marcos Lopez de Prado's research:
- *"Advances in Financial Machine Learning"*
- *"Machine Learning for Asset Managers"*

**Core Philosophy**: Financial markets require specialized ML tools distinct from general-purpose libraries, addressing non-IID data, market microstructure, and overfitting prevention.

---

## Architecture: 13 Core Modules

### 1. Data Structures (`mlfinlab/data_structures/`)
Information-driven bar sampling that goes beyond fixed time intervals.

**Classes**: `BaseBars`, `StandardBars`, `TimeBars`, `ImbalanceBars`, `RunBars`

**Key Features**:
- **Standard bars**: Tick, Volume, Dollar (fixed threshold sampling)
- **Imbalance bars**: Sample when order flow imbalance exceeds threshold
- **Run bars**: Sample on consecutive directional moves
- **Time bars**: Millisecond precision support
- **Footprint bars** ⭐ NEW: Price-level bid/ask volume tracking

**Footprint Implementation** (Added 2025-10):
- Enable with `enable_footprint=True` parameter
- Supports 3 input formats: (timestamp, price, volume), (timestamp, price, bid_qty, ask_qty), (timestamp, price, volume, bid_qty, ask_qty)
- Returns dict: `{'bars': DataFrame, 'footprint': DataFrame}` when enabled
- Footprint uses MultiIndex: (bar_timestamp, price)
- Columns: bid_vol, ask_vol, total_vol, delta, is_open, is_high, is_low, is_close

### 2. Labeling (`mlfinlab/labeling/`)
Generate training labels for supervised learning.

**Methods**:
- **Triple-barrier**: Dynamic profit/stop-loss based on volatility
- **Meta-labeling**: Secondary model to filter primary model's predictions (increases F1-score)
- **Trend scanning**: Regression-based trend detection
- **Fixed horizon, Excess returns, Tail sets, Raw returns**: Various labeling strategies

### 3. Feature Engineering (`mlfinlab/features/`)
Transform raw data into stationary, predictive features.

**Key Techniques**:
- **Fractional differentiation**: Achieve stationarity while preserving memory
- **Filters**: CUSUM, Z-score (event-driven sampling)
- **Structural breaks**: CUSUM tests, SADF explosiveness tests (bubble detection)
- **Microstructural features**: Entropy, Roll measure, Kyle/Amihud/Hasbrouck lambdas, VPIN

### 4. Feature Importance (`mlfinlab/feature_importance/`)
Identify which features drive model predictions.

**Methods**:
- **MDI** (Mean Decrease Impurity): Tree-based, in-sample
- **MDA** (Mean Decrease Accuracy): Out-of-sample performance
- **SFI** (Single Feature Importance): No substitution effects
- **Clustered importance**: Handle multicollinearity
- **Model fingerprints**: Decompose linear/non-linear/interaction effects
- **PCA analysis**: Validate feature importance patterns

### 5. Sampling (`mlfinlab/sampling/`)
Handle non-IID financial data.

**Techniques**:
- **Sample uniqueness**: Measure label concurrency
- **Sequential bootstrapping**: Maximize sample uniqueness during resampling
- **Sample weights**: By return, uniqueness, or time decay

### 6. Cross-Validation (`mlfinlab/cross_validation/`)
Prevent data leakage in backtesting.

**Methods**:
- **Purged K-Fold**: Remove overlapping samples from training set
- **Embargo**: Additional protection against leakage
- **Combinatorial Purged CV (CPCV)**: Multiple backtest paths for Sharpe ratio distribution

### 7. Ensemble (`mlfinlab/ensemble/`)
Ensemble methods tailored for financial data.

**Classes**:
- `SequentiallyBootstrappedBaggingClassifier`
- `SequentiallyBootstrappedBaggingRegressor`

Integrates with sequential bootstrapping and sample weights.

### 8. Bet Sizing (`mlfinlab/bet_sizing/`)
Position sizing from ML predictions.

**Approaches**:
- From probabilities: Size bets based on predicted probability
- Dynamic: Sigmoid/Power functions for adaptive sizing
- Budget/Reserve: Account for concurrent positions
- **EF3M**: Mixture of Gaussians for strategy drift detection

### 9. Backtest Statistics (`mlfinlab/backtest_statistics/`)
Robust performance metrics.

**Metrics**:
- **Sharpe ratios**: Annualized, Probabilistic (PSR), Deflated (DSR)
- **Information ratio**: Excess return vs benchmark
- **Minimum Track Record Length**: Statistical confidence threshold
- **Bets concentration**: HHI-inspired return uniformity
- **Drawdown & Time Under Water**
- **Average holding period**

### 10. Portfolio Optimization (`mlfinlab/portfolio_optimization/`)
Modern portfolio construction beyond mean-variance.

**Algorithms**:
- **HRP** (Hierarchical Risk Parity): ML + traditional optimization
- **HERC** (Hierarchical Equal Risk Contribution)
- **NCO** (Nested Clustered Optimization)
- **CLA** (Critical Line Algorithm)
- Returns/Risk estimators with shrinkage and denoising

### 11. Online Portfolio Selection (`mlfinlab/online_portfolio_selection/`)
Sequential allocation based on capital growth theory.

**Categories**:
- **Benchmarks**: Buy & Hold, Best Stock, CRIP
- **Momentum**: Follow the Winner, Follow the Loser
- **Mean Reversion**: PAMR, OLMAR, RMR
- **Pattern Matching**: CORN, CORNU, CORNK

### 12. Codependence (`mlfinlab/codependence/`)
Beyond Pearson correlation.

**Metrics**:
- Correlation-based: Distance correlation, angular distance
- Information theory: Mutual information, variation of information
- Optimal transport distance
- Codependence matrices: GPR, GNPR

### 13. Clustering (`mlfinlab/clustering/`)
Hierarchical and feature clustering.

**Methods**:
- **ONC** (Optimal Number of Clusters)
- **Feature clusters**: Reduce substitution effects in feature importance

---

## Recent Changes

### Footprint Bars Support (2025-10)
- **Files modified**:
  - `mlfinlab/data_structures/base_bars.py`: Core footprint logic (3 new methods)
  - `mlfinlab/data_structures/standard_data_structures.py`: StandardBars hooks
  - `mlfinlab/data_structures/time_data_structures.py`: TimeBars hooks
  - `mlfinlab/data_structures/imbalance_data_structures.py`: ImbalanceBars hooks
  - `mlfinlab/data_structures/run_data_structures.py`: RunBars hooks

- **Architecture decision**: Extend `BaseBars` with optional parameter, not create new base class
- **Backward compatible**: Default `enable_footprint=False`
- **Total code**: ~150 lines added across all files

### Sklearn Compatibility (2025-10)
- **Files modified**: `mlfinlab/ensemble/sb_bagging.py`, `requirements.txt`, `setup.cfg`
- **Decision**: No backward compatibility, require `scikit-learn>=1.2.0`, `numpy>=1.20.0`
- **Implementation**: Simple utility functions instead of complex try-except blocks

### Active Research Project (2025-10)
- **Project**: Footprint-based key level trading strategy
- **Strategy specification**: `STRATEGY_SPEC.md` (complete strategy rules, filters, risk management)
- **Pipeline design**: `PIPELINE_DESIGN.md` (general 6-stage ML architecture)
- **Status**: Technical architecture defined, ready for implementation
- **Strategy type**: Rule-based signal generation with optional ML meta-labeling
- **Core insight**: "Only trade at key levels" - quantifying manual trading experience
- **Next step**: Answer integration questions in STRATEGY_SPEC.md and begin Phase 1 implementation

---

## Key Design Patterns

### 1. Modular Architecture
Each algorithm is encapsulated in its own class with consistent interface:
```python
algorithm = SomeAlgorithm(params)
result = algorithm.allocate(data)  # or .fit(), .get_events(), etc.
```

### 2. Optional Parameters
Extend functionality without breaking existing code:
```python
bars = get_dollar_bars(data, threshold=1000)  # Basic usage
bars_dict = get_dollar_bars(data, threshold=1000, enable_footprint=True)  # Extended
```

### 3. Information-Driven Sampling
Bars are sampled based on **information arrival** (order flow) rather than fixed time:
- Imbalance bars: θ_t = Σ(b_t × v_t), sample when |θ_t| ≥ threshold
- Run bars: Sample on consecutive directional sequences
- Footprint: Track price-level microstructure within each bar

### 4. Non-IID Aware
Financial data is neither independent nor identically distributed:
- **Purged cross-validation**: Remove overlapping samples
- **Sequential bootstrapping**: Maximize sample uniqueness
- **Sample weights**: Account for label concurrency

---

## Common Tasks

### Read Documentation
```
MLFinLab docs:
  - Full docs: docs/source/index.rst
  - Data structures: docs/source/implementations/data_structures.rst
  - Labeling: docs/source/labeling/tb_meta_labeling.rst
  - Feature importance: docs/source/implementations/feature_importance.rst

Strategy project docs:
  - STRATEGY_SPEC.md: Complete strategy specification (rules, filters, risk mgmt)
  - PIPELINE_DESIGN.md: General ML pipeline architecture (6 stages)
  - CLAUDE.md: This file (quick context for AI assistants)
```

### For New Conversations
When starting a new conversation about the strategy project:
```
"Read CLAUDE.md and STRATEGY_SPEC.md to understand the footprint-based
key level trading strategy, then let's continue from [specific section]"
```

### Test Changes
```bash
conda activate cs  # or: source /opt/homebrew/Caskroom/miniconda/base/bin/activate cs
cd /Users/alvinma/Desktop/work/mlfinlab
python -m pytest tests/  # Run all tests
python -c "import mlfinlab; print(mlfinlab.__version__)"  # Test import
```

### Build Documentation
```bash
cd docs
make html  # Build static HTML
open build/html/index.html  # macOS
sphinx-autobuild source build/html --port 8000  # Live preview
```

### Create Footprint Bars
```python
from mlfinlab.data_structures import get_dollar_bars

# Basic dollar bars
bars = get_dollar_bars(data, threshold=1000000)

# Dollar bars with footprint
result = get_dollar_bars(data, threshold=1000000, enable_footprint=True)
bars = result['bars']
footprint = result['footprint']

# Footprint analysis
poc = footprint.groupby('bar_timestamp')['total_vol'].idxmax()  # Point of Control
delta = footprint['delta']  # Bid pressure - Ask pressure
```

---

## References

- **Main textbook**: "Advances in Financial Machine Learning" by Marcos Lopez de Prado
- **Secondary**: "Machine Learning for Asset Managers" by Marcos Lopez de Prado
- **Documentation**: https://mlfinlab.readthedocs.io/
- **Research notebooks**: https://github.com/hudson-and-thames/research

---

## For AI Assistants: Quick Start Checklist

When starting a new conversation about MLFinLab:

- [ ] Read this file (`CLAUDE.md`)
- [ ] Understand the 13 core modules
- [ ] Note recent changes (footprint bars, sklearn compatibility)
- [ ] Remember key design patterns (modular, optional parameters, information-driven)
- [ ] If working on specific module, read relevant docs in `docs/source/implementations/`

**Coding Style**:
- Follow existing architecture (extend `BaseBars`, not create new classes)
- Keep code simple and maintainable
- No backward compatibility unless explicitly requested
- Add comprehensive documentation to `docs/source/`
- Update this file when making significant architectural changes
