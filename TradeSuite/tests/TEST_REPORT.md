# Pattern Detection Test Report

| Pattern | Expected | Actual | Status | Notes |
|---------|----------|--------|--------|-------|
| fvg | 7 | 7 | PASS | Correct detection |
| engulfing | 6 | 6 | PASS | Correct detection |
| hammer | 5 | 5 | PASS | Correct detection |
| double_wick | 5 | 5 | PASS | Correct detection |
| ii_bars | 7 | 7 | PASS | Correct detection |
| doji | 5 | 5 | PASS | Correct detection |
| marubozu | 5 | 5 | PASS | Correct detection |
| spinning_top | 5 | 5 | PASS | Correct detection || order_block_zone | 3 | -1 | FAIL | Expected 3, got -1. Data: test_order_block_zone_data.csv |
| order_block_zone | 10 | 10 | PASS |  |
| vwap_mean_reversion_band | 3 | -1 | FAIL | Expected 3, got -1. Data: test_vwap_mean_reversion_band_data.csv |
| order_block_zone | 10 | 10 | PASS |  |
| vwap_mean_reversion_band | 3 | -1 | FAIL | Expected 3, got -1. Data: test_vwap_mean_reversion_band_data.csv |
| order_block_zone | 10 | 10 | PASS |  |
| vwap_mean_reversion_band | 20 | 20 | PASS |  |
| order_block_zone | 10 | 10 | PASS |  |
| vwap_mean_reversion_band | 20 | 20 | PASS |  |
| support_resistance_band | 60 | -1 | FAIL | Expected 60, got -1. Data: test_support_resistance_band_data.csv |
| order_block_zone | 10 | 10 | PASS |  |
| vwap_mean_reversion_band | 20 | 20 | PASS |  |
| support_resistance_band | 60 | 20 | FAIL | Expected 60, got 20. Data: test_support_resistance_band_data.csv |
| order_block_zone | 10 | 10 | PASS |  |
| vwap_mean_reversion_band | 20 | 20 | PASS |  |
| support_resistance_band | 60 | 20 | FAIL | Expected 60, got 20. Data: test_support_resistance_band_data.csv |
| order_block_zone | 10 | 10 | PASS |  |
| vwap_mean_reversion_band | 20 | 20 | PASS |  |
| support_resistance_band | 60 | 20 | FAIL | Expected 60, got 20. Data: test_support_resistance_band_data.csv |
| order_block_zone | 10 | 10 | PASS |  |
| vwap_mean_reversion_band | 20 | 20 | PASS |  |
| support_resistance_band | 60 | 20 | FAIL | Expected 60, got 20. Data: test_support_resistance_band_data.csv |
| order_block_zone | 10 | 10 | PASS |  |
| vwap_mean_reversion_band | 20 | 20 | PASS |  |
| support_resistance_band | 60 | 20 | FAIL | Expected 60, got 20. Data: test_support_resistance_band_data.csv |
| order_block_zone | NEGATIVE | -1 | PASS |  |
| vwap_mean_reversion_band | NEGATIVE | 20 | FAIL | Expected no detection, got 20. Data: test_vwap_mean_reversion_band_negative_data.csv |
| support_resistance_band | NEGATIVE | 20 | FAIL | Expected no detection, got 20. Data: test_support_resistance_band_negative_data.csv |
| order_block_zone | 10 | 10 | PASS |  |
| vwap_mean_reversion_band | 20 | 20 | PASS |  |
| support_resistance_band | 60 | 20 | FAIL | Expected 60, got 20. Data: test_support_resistance_band_data.csv |
| order_block_zone | NEGATIVE | -1 | PASS |  |
| vwap_mean_reversion_band | NEGATIVE | 20 | FAIL | Expected no detection, got 20. Data: test_vwap_mean_reversion_band_negative_data.csv |
| support_resistance_band | NEGATIVE | 20 | FAIL | Expected no detection, got 20. Data: test_support_resistance_band_negative_data.csv |
| order_block_zone | 10 | 10 | PASS |  |
| vwap_mean_reversion_band | 20 | 25 | FAIL | Expected 20, got 25. Data: test_vwap_mean_reversion_band_data.csv |
| support_resistance_band | 60 | 20 | FAIL | Expected 60, got 20. Data: test_support_resistance_band_data.csv |
| order_block_zone | NEGATIVE | -1 | PASS |  |
| vwap_mean_reversion_band | NEGATIVE | 20 | FAIL | Expected no detection, got 20. Data: test_vwap_mean_reversion_band_negative_data.csv |
| support_resistance_band | NEGATIVE | 20 | FAIL | Expected no detection, got 20. Data: test_support_resistance_band_negative_data.csv |
| order_block_zone | 10 | 10 | PASS |  |
| vwap_mean_reversion_band | 20 | 25 | FAIL | Expected 20, got 25. Data: test_vwap_mean_reversion_band_data.csv |
| support_resistance_band | 60 | 20 | FAIL | Expected 60, got 20. Data: test_support_resistance_band_data.csv |
| order_block_zone | NEGATIVE | -1 | PASS |  |
| vwap_mean_reversion_band | NEGATIVE | 20 | FAIL | Expected no detection, got 20. Data: test_vwap_mean_reversion_band_negative_data.csv |
| support_resistance_band | NEGATIVE | 20 | FAIL | Expected no detection, got 20. Data: test_support_resistance_band_negative_data.csv |
| order_block_zone | 10 | 10 | PASS |  |
| vwap_mean_reversion_band | 20 | 25 | FAIL | Expected 20, got 25. Data: test_vwap_mean_reversion_band_data.csv |
| support_resistance_band | 60 | 65 | FAIL | Expected 60, got 65. Data: test_support_resistance_band_data.csv |
| order_block_zone | NEGATIVE | -1 | PASS |  |
| vwap_mean_reversion_band | NEGATIVE | -1 | PASS |  |
| support_resistance_band | NEGATIVE | -1 | PASS |  |
| vwap_mean_reversion_band | 20 | 20 | PASS |  |
| support_resistance_band | 60 | 60 | PASS |  |
| order_block_zone | NEGATIVE | -1 | PASS |  |
| vwap_mean_reversion_band | NEGATIVE | -1 | PASS |  |
| support_resistance_band | NEGATIVE | -1 | PASS |  |
| order_block_zone | 10 | -1 | FAIL | Expected 10, got -1. Data: test_order_block_zone_data.csv |
| vwap_mean_reversion_band | 20 | 20 | PASS |  |
| support_resistance_band | 60 | 60 | PASS |  |
| order_block_zone | NEGATIVE | -1 | PASS |  |
| vwap_mean_reversion_band | NEGATIVE | -1 | PASS |  |
| support_resistance_band | NEGATIVE | -1 | PASS |  |
| order_block_zone | 10 | -1 | FAIL | Expected 10, got -1. Data: test_order_block_zone_data.csv |
| vwap_mean_reversion_band | 20 | 20 | PASS |  |
| support_resistance_band | 60 | 60 | PASS |  |
| imbalance_memory_zone | 15 | -1 | FAIL | Expected 15, got -1. Data: test_imbalance_memory_zone_data.csv |
| order_block_zone | NEGATIVE | -1 | PASS |  |
| vwap_mean_reversion_band | NEGATIVE | -1 | PASS |  |
| support_resistance_band | NEGATIVE | -1 | PASS |  |
| imbalance_memory_zone | NEGATIVE | -1 | PASS |  |
| order_block_zone | 10 | -1 | FAIL | Expected 10, got -1. Data: test_order_block_zone_data.csv |
| vwap_mean_reversion_band | 20 | 20 | PASS |  |
| support_resistance_band | 60 | 60 | PASS |  |
| imbalance_memory_zone | 15 | -1 | FAIL | Expected 15, got -1. Data: test_imbalance_memory_zone_data.csv |
| order_block_zone | 10 | -1 | FAIL | Expected 10, got -1. Data: test_order_block_zone_data.csv |
| vwap_mean_reversion_band | 20 | 20 | PASS |  |
| support_resistance_band | 60 | 60 | PASS |  |
| imbalance_memory_zone | 15 | 20 | FAIL | Expected 15, got 20. Data: test_imbalance_memory_zone_data.csv |
| order_block_zone | 10 | -1 | FAIL | Expected 10, got -1. Data: test_order_block_zone_data.csv |
| vwap_mean_reversion_band | 20 | 20 | PASS |  |
| support_resistance_band | 60 | 60 | PASS |  |
| imbalance_memory_zone | 15 | 20 | FAIL | Expected 15, got 20. Data: test_imbalance_memory_zone_data.csv |
| order_block_zone | 10 | -1 | FAIL | Expected 10, got -1. Data: test_order_block_zone_data.csv |
| vwap_mean_reversion_band | 20 | 20 | PASS |  |
| support_resistance_band | 60 | 60 | PASS |  |
| imbalance_memory_zone | 15 | 15 | PASS |  |
| order_block_zone | 10 | -1 | FAIL | Expected 10, got -1. Data: test_order_block_zone_data.csv |
| vwap_mean_reversion_band | 20 | 20 | PASS |  |
| support_resistance_band | 60 | 60 | PASS |  |
| imbalance_memory_zone | 15 | 15 | PASS |  |
| order_block_zone | 10 | 11 | FAIL | Expected 10, got 11. Data: test_order_block_zone_data.csv |
| vwap_mean_reversion_band | 20 | 20 | PASS |  |
| support_resistance_band | 60 | 60 | PASS |  |
| imbalance_memory_zone | 15 | 15 | PASS |  |
| order_block_zone | 10 | 10 | PASS |  |
| vwap_mean_reversion_band | 20 | 20 | PASS |  |
| support_resistance_band | 60 | 60 | PASS |  |
| imbalance_memory_zone | 15 | 15 | PASS |  |
| order_block_zone | 10 | 10 | PASS |  |
| vwap_mean_reversion_band | 20 | 20 | PASS |  |
| support_resistance_band | 60 | 60 | PASS |  |
| imbalance_memory_zone | 15 | 15 | PASS |  |
| fvg | 7 | -1 | FAIL | Expected 7, got -1. Data: test_fvg_data.csv |
| engulfing | 6 | 1 | FAIL | Expected 6, got 1. Data: test_engulfing_data.csv |
| hammer | 5 | -1 | FAIL | Expected 5, got -1. Data: test_hammer_data.csv |
| double_wick | 5 | 0 | FAIL | Expected 5, got 0. Data: test_double_wick_data.csv |
| ii_bars | 6 | -1 | FAIL | Expected 6, got -1. Data: test_ii_bars_data.csv |
| doji | 5 | 0 | FAIL | Expected 5, got 0. Data: test_doji_data.csv |
| weak_body | 5 | 0 | FAIL | Expected 5, got 0. Data: test_weak_body_data.csv |
| marubozu | 5 | 0 | FAIL | Expected 5, got 0. Data: test_marubozu_data.csv |
| spinning_top | 5 | -1 | FAIL | Expected 5, got -1. Data: test_spinning_top_data.csv |
| order_block_zone | 10 | 10 | PASS |  |
| vwap_mean_reversion_band | 20 | 20 | PASS |  |
| support_resistance_band | 60 | 60 | PASS |  |
| imbalance_memory_zone | 15 | 15 | PASS |  |
| fvg | 7 | -1 | FAIL | Expected 7, got -1. Data: test_fvg_data.csv |
| engulfing | 6 | 1 | FAIL | Expected 6, got 1. Data: test_engulfing_data.csv |
| hammer | 5 | -1 | FAIL | Expected 5, got -1. Data: test_hammer_data.csv |
| double_wick | 5 | 0 | FAIL | Expected 5, got 0. Data: test_double_wick_data.csv |
| ii_bars | 6 | -1 | FAIL | Expected 6, got -1. Data: test_ii_bars_data.csv |
| doji | 5 | 0 | FAIL | Expected 5, got 0. Data: test_doji_data.csv |
| weak_body | 5 | 0 | FAIL | Expected 5, got 0. Data: test_weak_body_data.csv |
| marubozu | 5 | 0 | FAIL | Expected 5, got 0. Data: test_marubozu_data.csv |
| spinning_top | 5 | -1 | FAIL | Expected 5, got -1. Data: test_spinning_top_data.csv |
| vwap_mean_reversion_band | 20 | 20 | PASS |  |
| imbalance_memory_zone | 15 | 15 | PASS |  |
| fvg | 7 | 6 | FAIL | Expected 7, got 6. Data: test_fvg_data.csv |
| engulfing | 6 | 1 | FAIL | Expected 6, got 1. Data: test_engulfing_data.csv |
| hammer | 5 | -1 | FAIL | Expected 5, got -1. Data: test_hammer_data.csv |
| double_wick | 5 | 0 | FAIL | Expected 5, got 0. Data: test_double_wick_data.csv |
| ii_bars | 6 | -1 | FAIL | Expected 6, got -1. Data: test_ii_bars_data.csv |
| doji | 5 | 0 | FAIL | Expected 5, got 0. Data: test_doji_data.csv |
| weak_body | 5 | 0 | FAIL | Expected 5, got 0. Data: test_weak_body_data.csv |
| marubozu | 5 | 0 | FAIL | Expected 5, got 0. Data: test_marubozu_data.csv |
| spinning_top | 5 | -1 | FAIL | Expected 5, got -1. Data: test_spinning_top_data.csv |
| vwap_mean_reversion_band | NEGATIVE | -1 | PASS |  |
| imbalance_memory_zone | NEGATIVE | -1 | PASS |  |
| fvg | NEGATIVE | -1 | PASS |  |
| engulfing | NEGATIVE | -1 | PASS |  |
| hammer | NEGATIVE | -1 | PASS |  |
| double_wick | NEGATIVE | 4 | FAIL | Expected no detection, got 4. Data: test_double_wick_negative_data.csv |
| ii_bars | NEGATIVE | -1 | PASS |  |
| doji | NEGATIVE | -1 | PASS |  |
| weak_body | NEGATIVE | 4 | FAIL | Expected no detection, got 4. Data: test_weak_body_negative_data.csv |
| marubozu | NEGATIVE | -1 | PASS |  |
| spinning_top | NEGATIVE | 0 | FAIL | Expected no detection, got 0. Data: test_spinning_top_negative_data.csv |
