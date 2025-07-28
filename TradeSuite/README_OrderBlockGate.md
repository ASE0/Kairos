Order Block Gate
===============

1. Add OrderBlockGate to your strategy via gate: order_block_gate in config.
2. Params: buffer_pts, max_impulse_bars, min_impulse_body, gamma, tau, enable_comb, N, sigma.
3. Use detect_zones(bars, **params) to get OB zones.
4. Run tests: pytest -q tests/test_gates.py
5. Expected: All tests pass for FVG and Order Block gates.
6. See core/order_block_gate.py for implementation details.
7. See tests/data_factory.py for synthetic OB dataset.
8. See tests/test_gates.py for test usage. 