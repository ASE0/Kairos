# Pattern Detection Test Report

| Pattern | Expected | Actual | Status | Notes |
|---------|----------|--------|--------|-------|
| fvg | 6 | 6 | PASS | Correct detection |
| engulfing | 6 | 5 | FAIL | Expected 6, got 5 |
| hammer | 5 | -1 | FAIL | No detection found |
| double_wick | 5 | 0 | FAIL | Expected 5, got 0 |
| ii_bars | 6 | 7 | FAIL | Expected 6, got 7 |
| doji | 5 | -1 | FAIL | No detection found |
| marubozu | 5 | -1 | FAIL | No detection found |
| spinning_top | 5 | 0 | FAIL | Expected 5, got 0 |