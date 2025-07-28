# Comprehensive Pattern Detection Test Suite

This test suite validates that all implemented candlestick patterns and gaps are detected correctly at the expected bar indices, based on the mathematical framework from the comprehensive trading strategy documentation.

## Quick Start

```bash
# Run all pattern tests
pytest -q tests/

# Run individual pattern test
pytest tests/test_patterns.py::test_pattern_detection

# Run integration test with report
pytest tests/test_patterns.py::test_all_patterns_integration

# Run mathematical validation tests
pytest tests/test_patterns.py::test_mathematical_validation
```

## Test Structure

- **`data_factory.py`**: Generates synthetic OHLCV datasets that trigger each pattern exactly once based on mathematical rules
- **`run_headless.py`**: Wraps GUI CLI invocation and extracts detection results
- **`test_patterns.py`**: Pytest file with parametrized tests for each pattern
- **`TEST_REPORT.md`**: Auto-generated test report with pass/fail status

## Supported Patterns

| Pattern | Type | Mathematical Rule | Expected Index |
|---------|------|-------------------|----------------|
| FVG | Gap | H_{t-1} < L_{t+1} (bullish) | 6 |
| Engulfing | Reversal | B₂ > B₁ and proper engulfing | 6 |
| Hammer | Reversal | Body ratio ≤ 0.4, lower wick ≥ 0.6 | 5 |
| Double Wick | Indecision | Body ratio ≤ 0.4, wicks ≥ 0.3 | 5 |
| II Bars | Continuation | Hᵢ ≤ H_{i-1} and Lᵢ ≥ L_{i-1} | 6 |
| Doji | Indecision | Body ratio ≤ 0.1, symmetrical wicks | 5 |
| Marubozu | Momentum | Body ratio ≥ 0.8, no wicks | 5 |
| Spinning Top | Indecision | Body ratio ≤ 0.4, long wicks | 5 |
| Weak Body | Indecision | Body ratio ≤ 0.3 | 5 |
| Strong Body | Momentum | Body ratio ≥ 0.7 | 5 |
| Breakout | Momentum | Price break with volume | 5 |
| Exhaustion | Reversal | Small body, long wicks | 5 |
| Accumulation | Consolidation | Small range, low volume | 5 |
| Distribution | Consolidation | Small range, low volume | 5 |

## Test Process

1. **Dataset Generation**: Creates synthetic CSV with pattern at bar index 5 (or 6 for two-bar patterns)
2. **Strategy Creation**: Builds single-pattern strategy configuration
3. **Headless Execution**: Runs GUI in headless mode with synthetic data
4. **Result Extraction**: Parses JSON output for detection index
5. **Validation**: Compares actual vs expected detection index
6. **Report Generation**: Creates comprehensive test report

## Mathematical Framework

All patterns are based on the mathematical framework from the comprehensive trading strategy documentation:

### Core Candlestick Quantification
- **Body Size**: Bₜ = |Cₜ - Oₜ|
- **Wick Components**: Wᵘₜ = Hₜ - max(Oₜ, Cₜ), Wˡₜ = min(Oₜ, Cₜ) - Lₜ
- **Normalized Wicks**: Ẇᵘₜ = Wᵘₜ / (Hₜ - Lₜ), Ẇˡₜ = Wˡₜ / (Hₜ - Lₜ)

### Pattern-Specific Rules
- **Doji-ness Score**: Dₜ = exp[−(Bₜ/(Hₜ − Lₜ))²/(2σ_b²)] × exp[−(Ẇᵘₜ − Ẇˡₜ)²/(2σ_w²)]
- **Two-Bar Reversal**: A₂bar = β_eng ⋅ (B₂ / B₁)
- **FVG Detection**: Gap size ≥ min_gap_size

## Expected Results

All patterns should be detected at the expected bar indices in the synthetic datasets:
- **Single-bar patterns**: Index 5 (6th bar, 0-based)
- **Two-bar patterns**: Index 6 (7th bar, 0-based)

## Debugging

If tests fail:
1. Check the generated CSV files in `tests/` directory
2. Review the JSON output from headless runs
3. Verify pattern detection logic in the pattern classes
4. Check the mathematical rules in the documentation files
5. Run with verbose output: `pytest -v tests/`

## Test Report

After running the integration test, a comprehensive report is generated in `tests/TEST_REPORT.md` with:
- Pattern-by-pattern results
- Expected vs actual detection indices
- Pass/fail status with notes
- Overall success rate

## Mathematical Validation

The test suite includes mathematical validation tests that verify:
- FVG gap rules: H_{t-1} < L_{t+1} for bullish FVG
- Doji rules: small body (≤10%), symmetrical wicks (≤30% difference)
- Engulfing rules: B₂ > B₁ and proper engulfing conditions

Run with: `pytest tests/test_patterns.py::test_mathematical_validation` 