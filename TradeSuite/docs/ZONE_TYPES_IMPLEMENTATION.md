# Zone Types Implementation Guide

## Overview

The TradeSuite strategy builder now supports all 5 zone types as specified in the cheat-sheet, each with their own editable parameters including **zone-specific decay controls**. **Zone decay parameters are configured in the strategy builder only** - they are not available in the backtest engine, ensuring proper separation of concerns.

## Zone Types Available

### 1. FVG (Fair Value Gap)
**UI Name**: "FVG (Fair Value Gap)"

**Description**: Detects price gaps between candles and creates zones with micro-comb peaks.

**Detection Parameters**:
- **ε (Buffer Points)**: Zone buffer points [1, 5], default 2
- **N (Peak Count)**: Number of Gaussian peaks [1, 10], default 3
- **σ (Peak Width)**: Std-dev of Gaussian peaks [0.01, 0.5], default 0.1
- **β₁ (Base Weight)**: Flat base weight [0.6, 0.8], default 0.7
- **β₂ (Comb Weight)**: Micro-comb weight [0.2, 0.4], default 0.3
- **φ (Momentum Warp)**: Momentum warp factor [0, 0.5], default 0.2
- **λ (Directional Skew)**: Directional skew slope [-2, 2], default 0.0

**Decay Parameters**:
- **γ (Decay per Bar)**: Exponential decay per bar [0.8, 0.99], default 0.95
- **τ (Hard Purge Bars)**: Hard purge after τ bars [5, 200], default 50
- **Drop Threshold**: Minimum strength before early purge [0.001, 0.1], default 0.01

**Detection Logic**:
- Bullish FVG: H_{t-1} < L_{t+1}
- Bearish FVG: L_{t-1} > H_{t+1}
- Creates zones with micro-comb peaks using Gaussian mixture

### 2. Order Block
**UI Name**: "Order Block"

**Description**: Detects the last opposing candle before sharp impulse moves.

**Detection Parameters**:
- **Impulse Threshold**: Minimum impulse move [0.01, 0.1], default 0.02
- **Lookback Period**: Lookback for impulse detection [5, 50], default 10

**Decay Parameters**:
- **γ (Decay per Bar)**: Exponential decay per bar [0.8, 0.99], default 0.95
- **τ (Hard Purge Bars)**: Hard purge after τ bars [5, 200], default 80 (longer-lived)
- **Drop Threshold**: Minimum strength before early purge [0.001, 0.1], default 0.01

**Detection Logic**:
- Bearish OB: Last up candle before sharp down impulse
- Bullish OB: Last down candle before sharp up impulse
- Uses μ_block = (x_start + x_end)/2 for alignment

### 3. VWAP Mean-Reversion Band
**UI Name**: "VWAP Mean-Reversion Band"

**Description**: Creates zones around the Volume Weighted Average Price.

**Detection Parameters**:
- **k (Stdev Multiplier)**: Multiplier for VWAP stdev band [0.5, 2.0], default 1.0
- **VWAP Lookback**: VWAP calculation lookback [10, 50], default 20

**Decay Parameters**:
- **γ (Decay per Bar)**: Exponential decay per bar [0.8, 0.99], default 0.95
- **τ (Hard Purge Bars)**: Hard purge after τ bars [5, 200], default 15 (shorter-lived)
- **Drop Threshold**: Minimum strength before early purge [0.001, 0.1], default 0.01

**Detection Logic**:
- Center: μ = current VWAP
- Vertical edges: μ ± k · σ_VWAP
- Height: 2 k σ_VWAP

### 4. Support/Resistance Band
**UI Name**: "Support/Resistance Band"

**Description**: Market-Maker Reversion zones based on rolling high/low.

**Detection Parameters**:
- **W (Window Bars)**: Window W bars for rolling high/low [10, 100], default 20
- **σ_r (Spatial)**: Spatial std dev for MMRS [1, 10 pts], default 5
- **σ_t (Temporal)**: Temporal std dev for MMRS [1, 10 bars], default 3

**Decay Parameters**:
- **γ (Decay per Bar)**: Exponential decay per bar [0.8, 0.99], default 0.95
- **τ (Hard Purge Bars)**: Hard purge after τ bars [5, 200], default 60 (medium-lived)
- **Drop Threshold**: Minimum strength before early purge [0.001, 0.1], default 0.01

**Detection Logic**:
- R_sup(t) = max_{i=t−W … t} H_i
- R_inf(t) = min_{i=t−W … t} L_i
- MMRS(t) = exp[−(L_t − R_inf)²/(2σ_r²)] × exp[−ε²/(2σ_t²)]

### 5. Imbalance Memory Zone
**UI Name**: "Imbalance Memory Zone"

**Description**: Detects and stores significant price moves with decay.

**Detection Parameters**:
- **τ_imbalance (Threshold)**: Threshold points to register imbalance [10, 500], default 100
- **γ_mem (Memory Decay)**: Decay factor for imbalance memory [0.001, 0.1], default 0.01
- **σ_rev (Revisit Width)**: Width of Gaussian influence for revisit [5, 50 pts], default 20

**Decay Parameters**:
- **γ (Decay per Bar)**: Exponential decay per bar [0.8, 0.99], default 0.95
- **τ (Hard Purge Bars)**: Hard purge after τ bars [5, 200], default 100 (longer-lived)
- **Drop Threshold**: Minimum strength before early purge [0.001, 0.1], default 0.01

**Detection Logic**:
- When price move > τ_imbalance pts: store (p_start, p_end, t_i)
- Re-entry score: R_imbalance(p,t) = Σ w_i exp[−(p − p_i)²/(2σ_rev²)] e^{−γ_mem (t − t_i)}

## Global Settings

**Global Tab**:
- **Bar Interval (min)**: Minutes per bar for calendar conversion [1, 1440], default 1

## Zone Decay System

Each zone type now has its own decay parameters that control how the zone strength diminishes over time:

### Decay Formula
```
strength_n = initial_strength × γⁿ
```
Where:
- `strength_n` = zone strength after n bars
- `initial_strength` = initial zone strength (0.1 to 1.0)
- `γ` = zone-specific decay factor per bar
- `n` = number of bars since zone creation

### Zone Lifecycle
1. **Creation**: Zone is created with initial strength
2. **Exponential Decay**: Strength decreases by factor γ each bar
3. **Early Purge**: Zone is removed if strength < drop_threshold
4. **Hard Purge**: Zone is removed after τ bars regardless of strength

### Zone-Specific Defaults
| Zone Type | γ (Decay) | τ (Bars) | Typical Life | Use Case |
|-----------|-----------|----------|--------------|----------|
| FVG | 0.95 | 50 | ~3.5 days | Short-term gaps |
| Order Block | 0.95 | 80 | ~5.6 days | Medium-term structure |
| VWAP | 0.95 | 15 | ~1.0 day | Intraday mean reversion |
| Support/Resistance | 0.95 | 60 | ~4.2 days | Medium-term levels |
| Imbalance | 0.95 | 100 | ~6.9 days | Long-term imbalances |

## UI Organization

The strategy builder UI is organized into tabs:

1. **FVG Tab**: All FVG detection and decay parameters
2. **Order Block Tab**: All Order Block detection and decay parameters  
3. **VWAP Tab**: All VWAP detection and decay parameters
4. **Support/Resistance Tab**: All S/R detection and decay parameters
5. **Imbalance Tab**: All Imbalance detection and decay parameters
6. **Global Tab**: Global settings like bar interval

Each tab is divided into two sections:
- **Detection Parameters**: Controls how zones are identified
- **Decay Parameters**: Controls how zones decay over time

## Usage Example

1. **Select Zone Type**: Choose from the dropdown in the strategy builder
2. **Configure Detection**: Adjust detection parameters for sensitivity
3. **Set Decay Behavior**: Configure how long zones should remain active
4. **Test Strategy**: Use the test button to see zones in action
5. **Fine-tune**: Adjust parameters based on backtest results

## Technical Implementation

### Parameter Storage
Zone-specific parameters are stored in the strategy's `location_gate_params` dictionary with prefixes:
- `fvg_*` for FVG parameters
- `ob_*` for Order Block parameters  
- `vwap_*` for VWAP parameters
- `sr_*` for Support/Resistance parameters
- `imbalance_*` for Imbalance parameters

### Zone Creation
When zones are detected, they inherit the decay parameters specific to their type:
```python
if zone_type == 'FVG':
    gamma = params.get('fvg_gamma', 0.95)
    tau_bars = params.get('fvg_tau_bars', 50)
    drop_threshold = params.get('fvg_drop_threshold', 0.01)
```

### Decay Calculation
Zone strength is calculated using zone-specific parameters:
```python
def calculate_zone_strength(self, bars_since_creation, initial_strength, zone_type):
    gamma = self.get_zone_specific_gamma(zone_type)
    return initial_strength * (gamma ** bars_since_creation)
```

## Benefits of Zone-Specific Decay

1. **Realistic Behavior**: Different zone types have different natural lifespans
2. **Flexible Control**: Fine-tune decay behavior for each zone type
3. **Performance Optimization**: Remove zones that are no longer relevant
4. **Strategy Customization**: Adapt zone behavior to different market conditions

## Testing

Use the test script `test_zone_specific_decay.py` to verify that:
- Zone-specific parameters are correctly applied
- Decay calculations work as expected
- Zone lifecycle follows the specified parameters
- Different zone types have different decay characteristics

## Mathematical Implementation

All zone types follow the mathematical specifications from the cheat-sheet:

- **Penetration depth**: d = (clip(candle,[x₀,x₁]) − x₀) / (x₁ − x₀)
- **Kernel weight**: K_i(d) = [2/ω_i] φ((d−ξ_i)/ω_i) Φ(α_i (d−ξ_i)/ω_i)
- **Per-zone strength**: S_{t,i} = A_pattern · K_i(d_imp) · L_momentum_total(x)
- **Stacked zones**: L_stacked(x) = Σ γ_z L_z(x)

## Performance Considerations

- Zone detection is optimized for real-time use
- Maximum 10 zones processed per bar for performance
- Zone decay system prevents memory bloat
- Parameters are cached for efficiency

## Future Enhancements

Potential improvements:

1. **Zone Confluence**: Detect when multiple zone types overlap
2. **Dynamic Parameters**: Auto-adjust parameters based on market conditions
3. **Zone Scoring**: Advanced scoring algorithms for zone quality
4. **Multi-timeframe Zones**: Zones that span multiple timeframes
5. **Zone Templates**: Pre-configured parameter sets for different market conditions 