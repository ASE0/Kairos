#!/usr/bin/env python3
"""
Test Order Block Zone detection
"""

from tests.run_headless import run_headless_test
from tests.data_factory import create_dataset
import json

# Create order block zone test data
data_path = create_dataset('order_block_zone')
print(f"Created test data at: {data_path}")

# Create strategy with order block zone
strategy = {
    'name': 'Test_order_block_zone', 
    'actions': [
        {
            'name': 'order_block_zone', 
            'pattern': {'params': {}}
        }
    ]
}

# Run headless test
result = run_headless_test(strategy, data_path)
print(f"Order Block Zone Test Result: {result}") 