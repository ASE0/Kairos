#!/usr/bin/env python3
"""
Debug script for headless mode testing
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Create a simple FVG dataset
data = []
base_time = datetime(2024, 3, 7, 16, 15, 0)

# Pre-FVG bars (3 bars)
for i in range(3):
    data.append({
        'datetime': base_time + timedelta(minutes=i),
        'open': 100.0 + i * 0.5,
        'high': 100.5 + i * 0.5,
        'low': 99.5 + i * 0.5,
        'close': 100.2 + i * 0.5,
        'volume': 1000 + i * 100
    })

# FVG creation bar (index 3)
data.append({
    'datetime': base_time + timedelta(minutes=3),
    'open': 101.0,
    'high': 102.0,  # High above previous close
    'low': 100.5,   # Low below previous close
    'close': 101.5,
    'volume': 1500
})

# Post-FVG bars (3 bars)
for i in range(3):
    data.append({
        'datetime': base_time + timedelta(minutes=4+i),
        'open': 101.5 + i * 0.3,
        'high': 102.0 + i * 0.3,
        'low': 101.0 + i * 0.3,
        'close': 101.8 + i * 0.3,
        'volume': 1200 + i * 100
    })

df = pd.DataFrame(data)
csv_path = Path("debug_fvg.csv")
df.to_csv(csv_path, index=False)

# Create strategy config
strategy_config = {
    "name": "Single_Fvg",
    "type": "pattern",
    "actions": [
        {
            "name": "fvg",
            "pattern": {
                "type": "FVGPattern",
                "timeframes": ["1min"],
                "min_gap_size": 0.1,
                "max_touches": 3
            }
        }
    ],
    "combination_logic": "OR",
    "gates_and_logic": {
        "location_gate": False,
        "volatility_gate": False,
        "regime_gate": False
    }
}

strategy_path = Path("debug_strategy.json")
with open(strategy_path, 'w') as f:
    json.dump(strategy_config, f, indent=2)

print(f"Created test files:")
print(f"  CSV: {csv_path}")
print(f"  Strategy: {strategy_path}")

# Test the headless mode
import subprocess
cmd = [
    "python", "main.py",
    "--headless",
    "--strategy", str(strategy_path),
    "--data", str(csv_path),
    "--output", "debug_output.json"
]

print(f"\nRunning command: {' '.join(cmd)}")
result = subprocess.run(cmd, capture_output=True, text=True)

print(f"Return code: {result.returncode}")
print(f"STDOUT: {result.stdout}")
print(f"STDERR: {result.stderr}")

# Check if output file was created
output_path = Path("debug_output.json")
if output_path.exists():
    with open(output_path, 'r') as f:
        results = json.load(f)
    print(f"\nResults: {json.dumps(results, indent=2)}")
else:
    print(f"\nOutput file not found: {output_path}")

# Cleanup
csv_path.unlink(missing_ok=True)
strategy_path.unlink(missing_ok=True)
output_path.unlink(missing_ok=True) 