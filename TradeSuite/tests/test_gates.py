import pytest
from tests.data_factory import create_dataset, order_block_dataset
from tests.run_headless import run_headless_test

gate_cases = [
    ("fvg", create_dataset("fvg"), 6),
    ("order_block_gate", order_block_dataset("bullish"), 3),
    ("order_block_gate", order_block_dataset("bearish"), 3),
]

@pytest.mark.parametrize("gate, datafile, expected_idx", gate_cases)
def test_gate_detection(gate, datafile, expected_idx):
    strategy = {"name": f"Test_{gate}", "actions": [{"name": gate, "pattern": {"params": {}}}]}
    idx, _ = run_headless_test(strategy, datafile)
    assert idx == expected_idx, f"{gate}: expected {expected_idx}, got {idx}"

def test_order_block_minimal():
    from core.order_block_gate import OrderBlockGate
    # Create synthetic bars: up-bar followed by a large down-bar (bearish OB)
    bars = [
        {'open': 100, 'high': 101, 'low': 99, 'close': 102},  # up-bar
        {'open': 102, 'high': 103, 'low': 95, 'close': 96},   # large down impulse
        {'open': 96, 'high': 97, 'low': 95, 'close': 96},
        {'open': 96, 'high': 97, 'low': 95, 'close': 96},
        {'open': 96, 'high': 97, 'low': 95, 'close': 96},
    ]
    zones = OrderBlockGate.detect_zones(bars)
    print("[TEST] Detected OB zones:", zones)
    assert any(z['zone_type'] == 'OrderBlock' for z in zones), "No OrderBlock zone detected!" 