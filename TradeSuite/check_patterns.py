#!/usr/bin/env python3
"""Check available patterns"""

import sys
sys.path.append('.')

from core.pattern_registry import registry

print("Available patterns:", registry.get_pattern_names())
print("Available filters:", registry.get_filter_types())
print("Available gates:", registry.get_gate_types()) 