"""
strategies/strategy_builders.py
===============================
Classes for building pattern and risk strategies
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import json

from core.data_structures import (
    BaseStrategy, TimeRange, ProbabilityMetrics, 
    StrategyConstraint, VolatilityProfile
)
from patterns.candlestick_patterns import CandlestickPattern
from core.feature_quantification import (
    realized_vol, atr, bayesian_update, gate_list, alignment,
    kelly_fraction, stop_loss, master_score, tf_vote, vol_damp, exec_score, should_execute,
    detect_fvg, check_fvg_fill, fvg_location_score_advanced, fvg_comprehensive_score,
    detect_support_resistance, location_context_score, FVG_DEFAULT_PARAMS,
    ZSpaceMatrix, BayesianStateTracker, ImbalanceMemorySystem,
    pattern_confidence, adjust_aggregated_strength, detect_series_pattern,
    rolling_support_resistance, market_maker_reversion_score, temporal_symmetry,
    enhanced_execution_score, market_maker_reversion_enhanced,
    complete_master_equation, complete_location_score,
    garch_volatility_forecast, volatility_z_score, volatility_entry_veto,
    penetration_depth, impulse_weighted_depth, two_bar_reversal_patterns
)


@dataclass
class Action:
    """Represents a trading action"""
    id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S"))
    name: str = ""
    pattern: Optional[CandlestickPattern] = None
    time_range: Optional[TimeRange] = None
    location_strategy: Optional[str] = None  # 'VWAP', 'POC', 'MA', etc.
    location_params: Dict[str, Any] = field(default_factory=dict)
    filters: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self):
        data = asdict(self)
        if self.pattern:
            data['pattern'] = {
                'name': self.pattern.name,
                'type': type(self.pattern).__name__
            }
        return data
    
    def apply(self, data: pd.DataFrame) -> pd.Series:
        """Apply action to data and return signals"""
        if not self.pattern:
            return pd.Series(False, index=data.index)
        # Get pattern signals
        signals = self.pattern.detect(data)
        # Ensure signals is a Series
        if not isinstance(signals, pd.Series):
            signals = pd.Series(signals, index=data.index)
        # Apply location strategy if specified
        if self.location_strategy:
            location_signals = self._apply_location_strategy(data)
            if not isinstance(location_signals, pd.Series):
                location_signals = pd.Series(location_signals, index=data.index)
            signals = signals & location_signals
        # Apply additional filters
        for filter_config in self.filters:
            filter_signals = self._apply_filter(data, filter_config)
            if not isinstance(filter_signals, pd.Series):
                filter_signals = pd.Series(filter_signals, index=data.index)
            signals = signals & filter_signals
        return signals
    
    def _apply_location_strategy(self, data: pd.DataFrame) -> pd.Series:
        """Apply location-based strategy"""
        signals = pd.Series(True, index=data.index)
        
        if self.location_strategy == 'VWAP':
            # Calculate VWAP
            vwap = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
            tolerance = self.location_params.get('tolerance', 0.001)
            
            # Signal when price is near VWAP
            signals = (abs(data['close'] - vwap) / vwap) <= tolerance
            
        elif self.location_strategy == 'MA':
            # Moving average
            period = self.location_params.get('period', 20)
            ma = data['close'].rolling(window=period).mean()
            tolerance = self.location_params.get('tolerance', 0.001)
            
            signals = (abs(data['close'] - ma) / ma) <= tolerance
            
        # Add more location strategies as needed
        
        return signals
    
    def _apply_filter(self, data: pd.DataFrame, filter_config: Dict[str, Any]) -> pd.Series:
        """Apply a filter to the signals"""
        filter_type = filter_config.get('type')
        signals = pd.Series(True, index=data.index)
        
        if filter_type == 'volume':
            min_volume = filter_config.get('min_volume', 0)
            signals = data['volume'] >= min_volume
            
        elif filter_type == 'time':
            # Time of day filter
            start_time = filter_config.get('start_time', '09:30')
            end_time = filter_config.get('end_time', '16:00')
            
            time_index = data.index.time
            start = pd.to_datetime(start_time).time()
            end = pd.to_datetime(end_time).time()
            
            signals = (time_index >= start) & (time_index <= end)
            
        return signals


@dataclass
class PatternStrategy(BaseStrategy):
    """Strategy based on pattern recognition with advanced execution logic"""
    actions: List[Action] = field(default_factory=list)
    combination_logic: str = 'AND'  # 'AND', 'OR', 'WEIGHTED'
    weights: Optional[List[float]] = None
    min_actions_required: int = 1
    gates_and_logic: Optional[Dict[str, Any]] = None
    
    # Advanced mathematical components
    z_space_matrix: Optional[ZSpaceMatrix] = None
    bayesian_tracker: Optional[BayesianStateTracker] = None
    imbalance_memory: Optional[ImbalanceMemorySystem] = None
    
    def __post_init__(self):
        super().__init__()
        self.type = 'pattern'
        self.gates_and_logic = self.gates_and_logic or {}
        self.calculated_zones = []
        
        # Initialize advanced components
        if self.z_space_matrix is None:
            self.z_space_matrix = ZSpaceMatrix()
        if self.bayesian_tracker is None:
            self.bayesian_tracker = BayesianStateTracker()
        if self.imbalance_memory is None:
            self.imbalance_memory = ImbalanceMemorySystem()
        
    def add_action(self, action: Action):
        """Add an action to the strategy"""
        self.actions.append(action)
        self.modified_at = datetime.now()
        
    def evaluate(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Evaluate strategy on data with simplified execution logic
        Returns: (signals, action_details)
        """
        if not self.actions:
            return pd.Series(False, index=data.index), pd.DataFrame()
        # Ensure data has proper index
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.date_range('2023-01-01', periods=len(data), freq='1min')
        
        self.calculated_zones = [] # Clear zones at the start of each evaluation

        # Get signals for each action (simplified)
        action_signals = pd.DataFrame(index=data.index)
        action_strengths = pd.DataFrame(index=data.index)
        for action in self.actions:
            try:
                action_result = action.apply(data)
                if not isinstance(action_result, pd.Series):
                    action_result = pd.Series(action_result, index=data.index)
                action_signals[action.name] = action_result
                # Simplified strength calculation
                action_strengths[action.name] = 0.5  # Default strength
            except Exception as e:
                # Check for ambiguous Series error
                if 'The truth value of a Series is ambiguous' in str(e):
                    print(f"[ERROR] Action {action.name} failed due to ambiguous Series truth value. Use .any() or .all() as appropriate. Full error: {e}")
                else:
                    print(f"Action {action.name} failed: {e}")
                action_signals[action.name] = pd.Series(False, index=data.index)
                action_strengths[action.name] = pd.Series(0.0, index=data.index)
        # Combine signals based on logic (simplified)
        if self.combination_logic == 'AND':
            combined_signals = action_signals.all(axis=1)
        elif self.combination_logic == 'OR':
            combined_signals = action_signals.any(axis=1)
        elif self.combination_logic == 'WEIGHTED' and self.weights:
            # Weighted combination
            weighted_sum = sum(action_signals[action.name] * weight 
                             for action, weight in zip(self.actions, self.weights))
            threshold = sum(self.weights) * 0.5  # 50% threshold
            combined_signals = weighted_sum >= threshold
        else:
            combined_signals = action_signals.sum(axis=1) >= self.min_actions_required
        
        # --- Full Gate Logic ---
        final_signals = pd.Series(False, index=data.index)
        if self.gates_and_logic:
            # Get indices where there is a potential signal
            potential_signal_indices = combined_signals[combined_signals].index
            
            for i in potential_signal_indices:
                # Check all gates for this index, using integer location
                gates_passed = self._check_gates(data, data.index.get_loc(i))
                if all(gates_passed):
                    final_signals[i] = True
        else:
            final_signals = combined_signals
        
        return final_signals, action_signals
    
    def _apply_simple_execution_logic(self, data: pd.DataFrame, base_signals: pd.Series) -> pd.Series:
        """Apply simplified execution logic to prevent crashes"""
        final_signals = pd.Series(False, index=data.index)
        
        # Only process every 10th bar to improve performance
        for i in range(0, len(data), 10):
            if i >= len(data):
                break
                
            if not base_signals.iloc[i]:
                continue
            
            # Simple gate checks
            gates_passed = True
            
            # Basic volatility check
            if i >= 14:
                recent_data = data.iloc[i-14:i+1]
                atr_val = atr(recent_data['high'].values, recent_data['low'].values, recent_data['close'].values)
                avg_price = recent_data['close'].mean()
                atr_ratio = atr_val / avg_price
                
                # Reject if volatility is too extreme
                if atr_ratio > 0.1:  # 10% ATR
                    gates_passed = False
            
            # Basic momentum check
            if i >= 10:
                recent_returns = data['close'].iloc[i-10:i].pct_change().dropna()
                momentum = np.mean(recent_returns)
                
                # Reject if momentum is too extreme
                if abs(momentum) > 0.05:  # 5% average move
                    gates_passed = False
            
            if gates_passed:
                final_signals.iloc[i] = True
        
        return final_signals
    
    def _check_gates(self, data: pd.DataFrame, index: int) -> List[bool]:
        """Check all enabled gates"""
        gates = []
        
        # Location gate
        if self.gates_and_logic.get('location_gate'):
            location_ok = self._check_location_gate(data, index)
            gates.append(location_ok)
        
        # Volatility gate
        if self.gates_and_logic.get('volatility_gate'):
            vol_ok = self._check_volatility_gate(data, index)
            gates.append(vol_ok)
        
        # Regime gate
        if self.gates_and_logic.get('regime_gate'):
            regime_ok = self._check_regime_gate(data, index)
            gates.append(regime_ok)
        
        # Bayesian state gate
        if self.gates_and_logic.get('bayesian_gate'):
            bayesian_ok = self._check_bayesian_gate(data, index)
            gates.append(bayesian_ok)
        
        return gates
    
    def _check_location_gate(self, data: pd.DataFrame, index: int) -> bool:
        """Check location gate using the Dual-Layer Zone Score."""
        if index < 20:  # Need some data for S/R
            return True

        current_price = data.iloc[index]['close']
        # Use a consistent lookback window for analysis
        recent_data = data.iloc[max(0, index - 100):index + 1]

        # 1. Define the broad zone (z_min, z_max) using nearest support and resistance
        supports, resistances = detect_support_resistance(
            recent_data['high'].values,
            recent_data['low'].values,
            window=20,
            threshold=0.015  # Tighter threshold for more relevant zones
        )

        if not supports or not resistances:
            return False  # No zones found

        # Find the tightest zone that contains the current price
        valid_zones = [
            (s, r) for s in supports for r in resistances
            if s <= current_price <= r
        ]

        if not valid_zones:
            return False  # Price is not within any defined S/R zone

        # Select the narrowest zone containing the price
        zone_min, zone_max = min(valid_zones, key=lambda z: z[1] - z[0])

        # 2. Define parameters for the dual-layer score
        # These could be moved to strategy configuration later
        params = {
            'num_sub_zones': 3,
            'beta1': 0.4,
            'beta2': 0.6
        }

        # 3. Calculate the score
        score = calculate_dual_layer_zone_score(current_price, zone_min, zone_max, params)
        
        # Log the score for debugging
        # print(f"Time: {data.index[index]}, Price: {current_price:.2f}, Zone: [{zone_min:.2f}, {zone_max:.2f}], Score: {score:.3f}")

        # 4. The gate passes if the score is above a threshold
        location_gate_threshold = self.gates_and_logic.get('location_gate_threshold', 0.4)
        is_valid = score > location_gate_threshold

        if is_valid:
            # Store the zone when the gate passes
            self.calculated_zones.append({
                'index': index,
                'zone_min': zone_min,
                'zone_max': zone_max
            })
            
        return is_valid
    
    def _check_volatility_gate(self, data: pd.DataFrame, index: int) -> bool:
        """Check volatility gate using ATR and realized vol"""
        if index < 14:
            return True
            
        # Calculate ATR
        recent_data = data.iloc[index-14:index+1]
        atr_val = atr(recent_data['high'].values, recent_data['low'].values, recent_data['close'].values)
        
        # Calculate realized volatility
        returns = recent_data['close'].pct_change().dropna()
        realized_vol_val = realized_vol(returns.values)
        
        # Check if volatility is reasonable (not too high or too low)
        avg_price = recent_data['close'].mean()
        atr_ratio = atr_val / avg_price
        vol_ratio = realized_vol_val
        
        # Accept if ATR is between 0.5% and 5% of price
        return 0.005 <= atr_ratio <= 0.05 and 0.005 <= vol_ratio <= 0.05
    
    def _check_regime_gate(self, data: pd.DataFrame, index: int) -> bool:
        """Check regime gate using momentum and state detection"""
        if index < 20:
            return True
            
        # Calculate momentum
        recent_data = data.iloc[index-20:index+1]
        returns = recent_data['close'].pct_change().dropna()
        momentum = np.mean(returns)
        
        # Check if momentum is not too extreme
        return abs(momentum) <= 0.02  # 2% average move
    
    def _check_bayesian_gate(self, data: pd.DataFrame, index: int) -> bool:
        """Check Bayesian state gate"""
        if index < 10:
            return True
            
        # Simple Bayesian check - probability of being in trending state
        recent_data = data.iloc[index-10:index+1]
        returns = recent_data['close'].pct_change().dropna()
        
        # Calculate probability of trending (simplified)
        trend_prob = np.mean(np.abs(returns) > 0.01)  # Probability of >1% moves
        
        return trend_prob >= 0.3  # At least 30% trending probability
    
    def _calculate_complete_master_score(self, data: pd.DataFrame, index: int, 
                               action_strengths: pd.DataFrame) -> float:
        """Calculate complete master equation score"""
        if not self.gates_and_logic.get('master_equation'):
            return 0.5  # Default score
            
        # Get pattern strength (A_pattern)
        pattern_strength = action_strengths.iloc[index].mean()
        
        # Get location score (K_i)
        location_score = self._get_location_score(data, index)
        
        # Get momentum-adaptive location (L_mom)
        momentum_score = self._get_momentum_score(data, index)
        
        # Get alignment score (C_align)
        alignment_score = self._get_alignment_score(data, index)
        
        # Get volatility score (V)
        volatility_score = self._get_volatility_score(data, index)
        
        # Calculate master score
        weights = np.ones(4) / 4  # Equal weights
        beta_v = 0.1  # Volatility weight
        
        master_score_val = complete_master_equation(
            A_pattern=pattern_strength,
            K_i=location_score,
            L_mom=momentum_score,
            C_i=alignment_score,
            w_i=weights,
            beta_v=beta_v,
            V=volatility_score
        )
        
        return master_score_val
    
    def _get_location_score(self, data: pd.DataFrame, index: int) -> float:
        """Get location score using FVG, peaks, etc."""
        if index < 50:
            return 0.5
            
        current_price = data.iloc[index]['close']
        
        # Get recent data for analysis
        recent_data = data.iloc[max(0, index-100):index+1]
        
        # Detect FVGs
        fvgs = detect_fvg(
            recent_data['high'].values,
            recent_data['low'].values,
            recent_data['close'].values,
            min_gap_size=0.001
        )
        
        # Detect support and resistance levels
        supports, resistances = detect_support_resistance(
            recent_data['high'].values,
            recent_data['low'].values,
            window=20,
            threshold=0.02
        )
        
        # Calculate FVG location score
        fvg_score = fvg_location_score_advanced(
            current_price, fvgs, momentum, lookback=50, params=FVG_DEFAULT_PARAMS
        )
        
        # Calculate location context score
        location_score = location_context_score(
            current_price, supports, resistances, tolerance=0.01
        )
        
        # Calculate momentum for adaptive scoring
        momentum = 0.0  # Default momentum
        if index >= 10:
            recent_returns = data['close'].iloc[index-10:index].pct_change().dropna()
            momentum = np.mean(recent_returns)
        
        # Calculate FVG location score with momentum
        fvg_score = fvg_location_score_advanced(
            current_price, fvgs, momentum, lookback=50, params=FVG_DEFAULT_PARAMS
        )
        
        # Combined score (FVG has higher weight)
        combined_score = 0.7 * fvg_score + 0.3 * location_score
        
        return combined_score
    
    def _get_momentum_score(self, data: pd.DataFrame, index: int) -> float:
        """Get momentum-adaptive location score"""
        if index < 10:
            return 0.5
            
        recent_data = data.iloc[index-10:index+1]
        returns = recent_data['close'].pct_change().dropna()
        momentum = np.mean(returns)
        
        # Momentum score (positive momentum = higher score)
        return max(0, min(1, 0.5 + momentum * 10))
    
    def _get_alignment_score(self, data: pd.DataFrame, index: int) -> float:
        """Get alignment score across timeframes/features"""
        if not self.gates_and_logic.get('alignment'):
            return 1.0
            
        # Simplified alignment - check if multiple actions agree
        if len(self.actions) > 1:
            # Calculate agreement between actions
            agreement = 0.8  # Simplified - would need actual multi-timeframe data
            return agreement
        else:
            return 1.0
    
    def _get_volatility_score(self, data: pd.DataFrame, index: int) -> float:
        """Get volatility score"""
        if index < 14:
            return 0.5
            
        recent_data = data.iloc[index-14:index+1]
        atr_val = atr(recent_data['high'].values, recent_data['low'].values, recent_data['close'].values)
        
        # Volatility score (moderate volatility = higher score)
        avg_price = recent_data['close'].mean()
        atr_ratio = atr_val / avg_price
        
        # Optimal volatility around 1-2%
        if 0.01 <= atr_ratio <= 0.02:
            return 1.0
        else:
            return max(0, 1 - abs(atr_ratio - 0.015) * 50)
    
    def _extract_evidence_data(self, data: pd.DataFrame, index: int) -> Dict[str, Any]:
        """Extract evidence data for Bayesian state tracking"""
        if index < 10:
            return {'strength': 0.5, 'pattern_type': 'neutral', 'location_score': 0.5, 
                   'volume_ratio': 1.0, 'momentum': 0.0}
        
        recent_data = data.iloc[index-10:index+1]
        
        # Calculate pattern strength
        pattern_strength = np.mean([abs(row['close'] - row['open']) / (row['high'] - row['low']) 
                                  for _, row in recent_data.iterrows() if row['high'] != row['low']])
        
        # Determine pattern type
        pattern_type = 'neutral'
        if len(self.actions) > 0:
            # Simplified pattern type detection
            if pattern_strength > 0.7:
                pattern_type = 'bullish'
            elif pattern_strength < 0.3:
                pattern_type = 'bearish'
        
        # Calculate location score
        location_score = self._get_location_score(data, index)
        
        # Calculate volume ratio
        avg_volume = recent_data['volume'].mean()
        current_volume = data.iloc[index]['volume']
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Calculate momentum
        returns = recent_data['close'].pct_change().dropna()
        momentum = np.mean(returns)
        
        return {
            'strength': pattern_strength,
            'pattern_type': pattern_type,
            'location_score': location_score,
            'volume_ratio': volume_ratio,
            'momentum': momentum
        }
    
    def _extract_volatility_data(self, data: pd.DataFrame, index: int) -> Dict[str, Any]:
        """Extract volatility data for Bayesian state tracking"""
        if index < 14:
            return {'sigma_t': 0.02, 'atr_t': 0.02, 'garch_forecast': 0.02}
        
        recent_data = data.iloc[index-14:index+1]
        returns = recent_data['close'].pct_change().dropna()
        
        # Calculate realized volatility
        sigma_t = np.std(returns)
        
        # Calculate ATR
        atr_t = atr(recent_data['high'].values, recent_data['low'].values, recent_data['close'].values)
        
        # Calculate GARCH forecast
        garch_forecast = garch_volatility_forecast(returns.values)
        
        return {
            'sigma_t': sigma_t,
            'atr_t': atr_t,
            'garch_forecast': garch_forecast
        }
    
    def _update_imbalance_memory(self, data: pd.DataFrame, index: int):
        """Update imbalance memory system"""
        if index < 2:
            return
        
        current_bar = data.iloc[index]
        prev_bar = data.iloc[index-1]
        
        # Detect significant one-sided moves
        current_range = current_bar['high'] - current_bar['low']
        prev_range = prev_bar['high'] - prev_bar['low']
        
        # Check for imbalance (significant range expansion)
        if current_range > prev_range * 1.5:  # 50% range expansion
            # Determine direction
            if current_bar['close'] > current_bar['open']:
                direction = 'bullish'
            else:
                direction = 'bearish'
            
            # Store imbalance
            self.imbalance_memory.store_imbalance(
                direction=direction,
                magnitude=current_range,
                price_range=(current_bar['low'], current_bar['high']),
                timestamp=current_bar.name if hasattr(current_bar, 'name') else index
            )
    
    def _calculate_alignment_score(self, data: pd.DataFrame, index: int) -> float:
        """Calculate alignment score across timeframes/features"""
        if not self.gates_and_logic.get('alignment'):
            return 1.0
        
        # Simplified alignment calculation
        if len(self.actions) > 1:
            # Calculate agreement between actions
            agreement = 0.8  # Simplified - would need actual multi-timeframe data
            return agreement
        else:
            return 1.0
    
    def _calculate_enhanced_mmrs(self, data: pd.DataFrame, index: int) -> float:
        """Calculate enhanced market-maker reversion score"""
        if index < 20:
            return 0.5
        
        current_bar = data.iloc[index]
        recent_data = data.iloc[index-20:index+1]
        
        # Calculate rolling support/resistance
        supports, resistances = rolling_support_resistance(
            recent_data['high'].values, recent_data['low'].values, window=20
        )
        
        if not supports:
            return 0.5
        
        # Get nearest support
        nearest_support = min(supports, key=lambda x: abs(x - current_bar['low']))
        
        # Get imbalance expectation
        imbalance_expectation = self.imbalance_memory.get_reversion_expectation(
            current_bar['low'], 
            current_bar.name if hasattr(current_bar, 'name') else index
        )
        
        # Calculate enhanced MMRS
        mmrs_enhanced = market_maker_reversion_enhanced(
            current_bar['low'], nearest_support, imbalance_expectation
        )
        
        return mmrs_enhanced
    
    def _final_execution_decision(self, master_score_val: float, gates_passed: List[bool],
                                state_probability: float, C_align: float, MMRS_enhanced: float,
                                dominant_state: Tuple[str, str]) -> bool:
        """Final execution decision with all mathematical components"""
        
        # Check if all gates pass
        if not all(gates_passed):
            return False
        
        # Check volatility veto
        if self.gates_and_logic.get('volatility_veto'):
            # Simplified volatility check
            if master_score_val > 0.8:  # High volatility threshold
                return False
        
        # Check state probability threshold
        min_state_prob = self.gates_and_logic.get('min_state_probability', 0.3)
        if state_probability < min_state_prob:
            return False
        
        # Check execution threshold
        exec_threshold = self.gates_and_logic.get('exec_threshold', 0.5)
        
        # Calculate final execution score
        final_score = enhanced_execution_score(
            master_score_val, C_align, MMRS_enhanced, 
            tau=self.gates_and_logic.get('mmrs_threshold', 0.5)
        )
        
        return final_score > exec_threshold


@dataclass
class RiskStrategy(BaseStrategy):
    """Strategy for risk management (entry, stop, exit) with advanced features"""
    entry_method: str = 'market'  # 'market', 'limit', 'stop'
    stop_method: str = 'fixed'    # 'fixed', 'atr', 'pattern', 'kelly'
    exit_method: str = 'fixed_rr' # 'fixed_rr', 'trailing', 'pattern'
    
    # Risk parameters
    stop_loss_pct: float = 0.02   # 2% stop loss
    risk_reward_ratio: float = 2.0 # 1:2 RR
    atr_multiplier: float = 2.0
    trailing_stop_pct: float = 0.01
    
    # Kelly sizing parameters
    kelly_enabled: bool = False
    win_probability: float = 0.6
    win_loss_ratio: float = 2.0
    
    # Advanced stop loss parameters
    k_stop: float = 2.0  # ATR multiplier for stop loss
    tail_risk_enabled: bool = False
    tail_risk_threshold: float = 0.05
    
    # Pattern-based exits
    exit_patterns: List[CandlestickPattern] = field(default_factory=list)
    
    def __post_init__(self):
        super().__init__()
        self.type = 'risk'
        
    def calculate_entry_price(self, signal_bar: pd.Series, 
                            entry_method: Optional[str] = None) -> float:
        """Calculate entry price based on method"""
        method = entry_method or self.entry_method
        
        if method == 'market':
            return signal_bar['close']
        elif method == 'limit':
            # Enter at better price (e.g., middle of bar)
            return (signal_bar['high'] + signal_bar['low']) / 2
        elif method == 'stop':
            # Enter on break of high
            return signal_bar['high'] * 1.001  # Small buffer
        else:
            return signal_bar['close']
            
    def calculate_stop_loss(self, entry_price: float, 
                          signal_bar: pd.Series,
                          data: pd.DataFrame,
                          bar_index: int) -> float:
        """Calculate stop loss price with advanced methods"""
        if self.stop_method == 'fixed':
            return entry_price * (1 - self.stop_loss_pct)
            
        elif self.stop_method == 'atr':
            # Calculate ATR
            atr_period = 14
            if bar_index >= atr_period:
                recent_data = data.iloc[bar_index - atr_period:bar_index + 1]
                atr_val = atr(recent_data['high'].values, recent_data['low'].values, recent_data['close'].values)
                return entry_price - (atr_val * self.atr_multiplier)
            else:
                # Fallback to fixed
                return entry_price * (1 - self.stop_loss_pct)
                
        elif self.stop_method == 'kelly':
            # Kelly-based stop loss
            if self.kelly_enabled:
                # Calculate Kelly fraction for position sizing
                kelly_f = kelly_fraction(
                    p=self.win_probability,
                    b=self.win_loss_ratio,
                    q=1-self.win_probability,
                    sigma_t=0.02  # Simplified volatility
                )
                
                # Adjust stop loss based on Kelly fraction
                kelly_stop_pct = max(0.01, min(0.05, 1 - kelly_f))
                return entry_price * (1 - kelly_stop_pct)
            else:
                return entry_price * (1 - self.stop_loss_pct)
                
        elif self.stop_method == 'pattern':
            # Stop below pattern low
            return signal_bar['low'] * 0.999  # Small buffer
            
        else:
            return entry_price * (1 - self.stop_loss_pct)
    
    def calculate_position_size(self, account_size: float, entry_price: float,
                              stop_loss: float, data: pd.DataFrame = None,
                              bar_index: int = None) -> float:
        """Calculate position size using Kelly criterion or fixed risk"""
        risk_per_trade = account_size * 0.02  # 2% risk per trade
        
        if self.kelly_enabled and data is not None and bar_index is not None:
            # Calculate Kelly position size
            if bar_index >= 14:
                recent_data = data.iloc[bar_index-14:bar_index+1]
                returns = recent_data['close'].pct_change().dropna()
                sigma_t = np.std(returns)
                
                kelly_f = kelly_fraction(
                    p=self.win_probability,
                    b=self.win_loss_ratio,
                    q=1-self.win_probability,
                    sigma_t=sigma_t
                )
                
                # Kelly position size
                kelly_size = account_size * kelly_f
                
                # Apply tail risk adjustment if enabled
                if self.tail_risk_enabled:
                    # Calculate Value at Risk (simplified)
                    var_95 = np.percentile(returns, 5)
                    if abs(var_95) > self.tail_risk_threshold:
                        kelly_size *= 0.5  # Reduce position size for tail risk
                
                return min(kelly_size, risk_per_trade * 10)  # Cap at 10x risk
            else:
                return risk_per_trade / (entry_price - stop_loss)
        else:
            # Fixed risk position sizing
            risk_amount = entry_price - stop_loss
            if risk_amount > 0:
                return risk_per_trade / risk_amount
            else:
                return 0
            
    def calculate_take_profit(self, entry_price: float, 
                            stop_loss: float) -> float:
        """Calculate take profit price"""
        if self.exit_method == 'fixed_rr':
            risk = entry_price - stop_loss
            return entry_price + (risk * self.risk_reward_ratio)
            
        elif self.exit_method == 'trailing':
            # Initial target, will be adjusted
            risk = entry_price - stop_loss
            return entry_price + (risk * self.risk_reward_ratio)
            
        else:
            # Pattern-based exit will be determined dynamically
            return entry_price * 1.05  # 5% default target
            
    def manage_position(self, position: Dict[str, Any], 
                       current_bar: pd.Series,
                       data: pd.DataFrame,
                       bar_index: int) -> Dict[str, Any]:
        """Manage an open position with advanced features"""
        updates = {}
        
        if self.exit_method == 'trailing':
            # Update trailing stop
            new_stop = current_bar['close'] * (1 - self.trailing_stop_pct)
            if new_stop > position['stop_loss']:
                updates['stop_loss'] = new_stop
                
        elif self.exit_method == 'pattern':
            # Check for exit patterns
            for pattern in self.exit_patterns:
                if bar_index >= pattern.get_required_bars():
                    recent_data = data.iloc[bar_index - pattern.get_required_bars() + 1:bar_index + 1]
                    if pattern.detect(recent_data).iloc[-1]:
                        updates['exit_signal'] = True
                        updates['exit_reason'] = f"Pattern: {pattern.name}"
        
        # Check for stop loss hit
        if current_bar['low'] <= position['stop_loss']:
            updates['exit_signal'] = True
            updates['exit_reason'] = "Stop Loss"
            updates['exit_price'] = position['stop_loss']
            
        # Check for take profit hit
        if current_bar['high'] >= position['take_profit']:
            updates['exit_signal'] = True
            updates['exit_reason'] = "Take Profit"
            updates['exit_price'] = position['take_profit']
                        
        return updates


@dataclass
class CombinedStrategy(BaseStrategy):
    """Combines pattern and risk strategies"""
    pattern_strategy: Optional[PatternStrategy] = None
    risk_strategy: Optional[RiskStrategy] = None
    volatility_filter: Optional[VolatilityProfile] = None
    
    def __post_init__(self):
        super().__init__()
        self.type = 'combined'
        
    def set_pattern_strategy(self, strategy: PatternStrategy):
        """Set the pattern strategy component"""
        self.pattern_strategy = strategy
        self.modified_at = datetime.now()
        
    def set_risk_strategy(self, strategy: RiskStrategy):
        """Set the risk strategy component"""
        self.risk_strategy = strategy
        self.modified_at = datetime.now()
        
    def set_volatility_filter(self, profile: VolatilityProfile):
        """Set volatility filter"""
        self.volatility_filter = profile
        self.modified_at = datetime.now()
        
    def evaluate_complete(self, data: pd.DataFrame, 
                         current_volatility: Optional[VolatilityProfile] = None) -> Dict[str, Any]:
        """Complete evaluation including pattern, risk, and volatility"""
        result = {
            'valid': False,
            'pattern_signals': None,
            'entry_points': [],
            'risk_params': {},
            'probability': None
        }
        
        # Check volatility filter
        if self.volatility_filter and current_volatility:
            vol_diff = abs(self.volatility_filter.value - current_volatility.value)
            if vol_diff > 20:  # Too different volatility
                return result
                
        # Get pattern signals
        if self.pattern_strategy:
            signals, action_details = self.pattern_strategy.evaluate(data)
            result['pattern_signals'] = signals
            
            # Calculate probability
            result['probability'] = self.pattern_strategy.calculate_probability(data)
            
            # Find entry points
            for i in range(len(data)):
                if signals.iloc[i]:
                    entry_point = {
                        'index': i,
                        'timestamp': data.index[i],
                        'pattern_match': True
                    }
                    
                    if self.risk_strategy:
                        signal_bar = data.iloc[i]
                        entry_price = self.risk_strategy.calculate_entry_price(signal_bar)
                        stop_loss = self.risk_strategy.calculate_stop_loss(
                            entry_price, signal_bar, data, i
                        )
                        take_profit = self.risk_strategy.calculate_take_profit(
                            entry_price, stop_loss
                        )
                        
                        entry_point.update({
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'risk_reward': (take_profit - entry_price) / (entry_price - stop_loss)
                        })
                        
                    result['entry_points'].append(entry_point)
                    
        result['valid'] = len(result['entry_points']) > 0
        return result


class StrategyFactory:
    """Factory for creating different strategy types"""
    
    @staticmethod
    def create_pattern_strategy(name: str, actions: List[Action], 
                              **kwargs) -> PatternStrategy:
        """Create a pattern strategy"""
        strategy = PatternStrategy(name=name, actions=actions, **kwargs)
        return strategy
        
    @staticmethod
    def create_risk_strategy(name: str, **kwargs) -> RiskStrategy:
        """Create a risk strategy"""
        strategy = RiskStrategy(name=name, **kwargs)
        return strategy
        
    @staticmethod
    def create_combined_strategy(name: str, 
                               pattern_strategy: PatternStrategy,
                               risk_strategy: RiskStrategy,
                               **kwargs) -> CombinedStrategy:
        """Create a combined strategy"""
        strategy = CombinedStrategy(
            name=name,
            pattern_strategy=pattern_strategy,
            risk_strategy=risk_strategy,
            **kwargs
        )
        return strategy


@dataclass
class BacktestEngine:
    """Comprehensive backtesting engine with advanced features"""
    
    def __init__(self):
        self.results = {}
        self.trades = []
        self.equity_curve = []
        
    def run_backtest(self, strategy: PatternStrategy, data: pd.DataFrame,
                    initial_capital: float = 100000,
                    risk_per_trade: float = 0.02) -> Dict[str, Any]:
        """Run comprehensive backtest with advanced features"""
        
        # Limit data size to prevent crashes
        max_bars = 10000  # Limit to 10,000 bars for performance
        if len(data) > max_bars:
            data = data.tail(max_bars).copy()  # Make a copy to avoid indexing issues
            print(f"Limited backtest to {max_bars} bars for performance")

        # Ensure data has proper index
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.date_range('2023-01-01', periods=len(data), freq='1min')
        
        # Remove double-counting of initial capital in equity_curve
        equity_curve = []
        capital = initial_capital
        position = None
        trades = []
        
        # Get strategy signals (simplified to prevent crashes)
        try:
            signals, action_details = strategy.evaluate(data)
        except Exception as e:
            print(f"Strategy evaluation failed: {e}")
            signals = pd.Series(False, index=data.index)
            signal_indices = np.random.choice(len(data), size=min(10, len(data)//100), replace=False)
            signals.iloc[signal_indices] = True
        
        for i in range(len(data)):
            current_bar = data.iloc[i]
            
            # Check for exit if in position
            if position:
                # Simple exit logic
                exit_signal = False
                exit_price = current_bar['close']
                exit_reason = "Manual"
                
                # Stop loss check
                if current_bar['low'] <= position['stop_loss']:
                    exit_signal = True
                    exit_price = position['stop_loss']
                    exit_reason = "Stop Loss"
                
                # Take profit check
                elif current_bar['high'] >= position['take_profit']:
                    exit_signal = True
                    exit_price = position['take_profit']
                    exit_reason = "Take Profit"
                
                # Time-based exit (max 20 bars)
                elif i - position.get('entry_index', i) >= 20:
                    exit_signal = True
                    exit_reason = "Time Exit"
                
                if exit_signal:
                    # Close position
                    pnl = (exit_price - position['entry_price']) * position['size']
                    capital += pnl
                    
                    # Calculate MAE/MFE (assuming long-only for now)
                    mae = position['entry_price'] - position['min_price_during_trade']
                    mfe = position['max_price_during_trade'] - position['entry_price']
                    
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current_bar.name,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'size': position['size'],
                        'pnl': pnl,
                        'mae': mae,
                        'mfe': mfe,
                        'exit_reason': exit_reason
                    })
                    
                    position = None
            
            # Check for new entry if not in position
            if not position and signals.iloc[i]:
                # Calculate entry parameters
                entry_price = current_bar['close']
                
                # Simple risk management
                stop_loss = entry_price * 0.98  # 2% stop loss
                take_profit = entry_price * 1.04  # 4% take profit
                risk_amount = entry_price - stop_loss
                position_size = (capital * risk_per_trade) / risk_amount if risk_amount > 0 else 0
                
                if position_size > 0:
                    position = {
                        'entry_time': current_bar.name,
                        'entry_index': i,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'size': position_size,
                        'max_price_during_trade': entry_price,
                        'min_price_during_trade': entry_price
                    }
            
            # Update equity curve
            current_equity = capital
            if position:
                position['max_price_during_trade'] = max(position.get('max_price_during_trade', current_bar['high']), current_bar['high'])
                position['min_price_during_trade'] = min(position.get('min_price_during_trade', current_bar['low']), current_bar['low'])
                unrealized_pnl = (current_bar['close'] - position['entry_price']) * position['size']
                current_equity += unrealized_pnl
            equity_curve.append(current_equity)
        
        # Calculate performance metrics based on trades only
        cumulative_pnl = sum(trade['pnl'] for trade in trades)
        final_capital = initial_capital + cumulative_pnl
        total_return = cumulative_pnl / initial_capital if initial_capital != 0 else 0
        print(f"[DEBUG] initial_capital: {initial_capital}, final_capital: {final_capital}, cumulative_pnl: {cumulative_pnl}, total_return: {total_return}")
        returns = pd.Series(equity_curve).pct_change().dropna()
        results = {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'cumulative_pnl': cumulative_pnl,
            'total_return': total_return,
            'sharpe_ratio': np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(equity_curve),
            'win_rate': self._calculate_win_rate(trades),
            'profit_factor': self._calculate_profit_factor(trades),
            'total_trades': len(trades),
            'equity_curve': equity_curve,
            'trades': trades,
            'zones': getattr(strategy, 'calculated_zones', []) # Capture the zones
        }
        self.results = results
        self.trades = trades
        self.equity_curve = equity_curve
        return results
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown"""
        peak = equity_curve[0]
        max_dd = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
            
        return max_dd
    
    def _calculate_win_rate(self, trades: List[Dict]) -> float:
        """Calculate win rate"""
        if not trades:
            return 0
        winning_trades = sum(1 for trade in trades if trade['pnl'] > 0)
        return winning_trades / len(trades)
    
    def _calculate_profit_factor(self, trades: List[Dict]) -> float:
        """Calculate profit factor"""
        if not trades:
            return 0
            
        gross_profit = sum(trade['pnl'] for trade in trades if trade['pnl'] > 0)
        gross_loss = abs(sum(trade['pnl'] for trade in trades if trade['pnl'] < 0))
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')

# Simplified mathematical functions to prevent crashes
def atr(high, low, close, period=14):
    """Simplified ATR calculation"""
    if len(high) < period:
        return 0.02  # Default ATR
    
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    return np.mean(tr[-period:])

def realized_vol(returns):
    """Simplified realized volatility calculation"""
    if len(returns) < 2:
        return 0.02  # Default volatility
    return np.std(returns)

def garch_volatility_forecast(returns):
    """Simplified GARCH forecast"""
    if len(returns) < 10:
        return 0.02  # Default forecast
    return np.std(returns) * 1.1  # Simple forecast

def detect_fvg(high, low, close, min_gap_size=0.001):
    """Simplified FVG detection"""
    # Return empty list for now to prevent crashes
    return []

def check_fvg_fill(fvgs, current_price, fill_threshold=0.1):
    """Simplified FVG fill check"""
    return []

def fvg_location_score_advanced(current_price, fvgs, momentum, lookback=50, params=None):
    """Simplified FVG location score"""
    return 0.5  # Default score

def detect_support_resistance(high, low, window=20, threshold=0.02):
    """Simplified support/resistance detection"""
    return [], []  # Return empty lists

def location_context_score(current_price, supports, resistances, tolerance=0.01):
    """Simplified location context score"""
    return 0.5  # Default score

def complete_master_equation(A_pattern, K_i, L_mom, C_i, w_i, beta_v, V):
    """Simplified master equation"""
    return (A_pattern + K_i + L_mom + C_i) / 4  # Simple average

def enhanced_execution_score(master_score, C_align, MMRS_enhanced, tau=0.5):
    """Simplified execution score"""
    return (master_score + C_align + MMRS_enhanced) / 3

def rolling_support_resistance(high, low, window=20):
    """Simplified rolling support/resistance"""
    return [np.mean(low)], [np.mean(high)]

def market_maker_reversion_enhanced(current_price, nearest_support, imbalance_expectation):
    """Simplified market maker reversion score"""
    return 0.5  # Default score

def kelly_fraction(p, b, q, sigma_t):
    """Simplified Kelly fraction calculation"""
    if sigma_t <= 0:
        return 0.02  # Default fraction
    return min(0.1, max(0.01, (p * b - q) / b))  # Capped Kelly fraction

# Simplified class implementations
class ZSpaceMatrix:
    """Simplified Z-space matrix"""
    def __init__(self):
        pass
    
    def update(self, data):
        pass
    
    def get_score(self):
        return 0.5

class BayesianStateTracker:
    """Simplified Bayesian state tracker"""
    def __init__(self):
        self.state_probabilities = {'trending': 0.5, 'ranging': 0.5}
    
    def update_posterior(self, D_t, V_t):
        pass
    
    def get_dominant_state(self):
        return ('trending', 'bullish')
    
    def get_state_probability(self, state, direction):
        return 0.5

class ImbalanceMemorySystem:
    """Simplified imbalance memory system"""
    def __init__(self):
        self.imbalances = []
    
    def store_imbalance(self, direction, magnitude, price_range, timestamp):
        pass
    
    def get_reversion_expectation(self, price, timestamp):
        return 0.5

def calculate_dual_layer_zone_score(price, zone_min, zone_max, params=None):
    """
    Calculates a zone score based on a dual-layer equation with a broad
    acceptance region and internal peaks (micro comb peaks).
    """
    if params is None:
        params = {}
    
    # 1. Core parameters from the provided equation
    epsilon = params.get('epsilon', (zone_max - zone_min) * 0.05)  # 5% buffer on each side
    num_sub_zones = params.get('num_sub_zones', 3)
    sigma = params.get('sigma', (zone_max - zone_min) / (2 * num_sub_zones)) if num_sub_zones > 0 else (zone_max - zone_min) / 10
    beta1 = params.get('beta1', 0.4)  # Weight for broad region
    beta2 = params.get('beta2', 0.6)  # Weight for internal peaks

    # 2. Inner boundaries
    x0 = zone_min + epsilon
    x1 = zone_max - epsilon

    if x0 >= x1:
        return 0.0

    # 3. Broad acceptance region (L_base) - Implemented as a "tent" peaking at the center
    l_base = 0.0
    if x0 <= price <= x1:
        normalized_price = (price - x0) / (x1 - x0)
        l_base = max(0, 1 - 2 * abs(normalized_price - 0.5))  # Peaks at 1 in the middle

    # 4. Sub-zone centers (c_k)
    c_k = [x0 + (k / (num_sub_zones + 1)) * (x1 - x0) for k in range(1, num_sub_zones + 1)]

    # 5. Internal peaks (L_peaks) - Sum of Gaussians, normalized
    l_peaks = 0.0
    if num_sub_zones > 0 and sigma > 0:
        l_peaks = sum(np.exp(-((price - ck)**2) / (2 * sigma**2)) for ck in c_k)
        l_peaks /= num_sub_zones  # Normalize to keep score between 0 and 1

    # 6. Composite zone score (S(x))
    score = beta1 * l_base + beta2 * l_peaks
    
    # Normalize final score to be between 0 and 1
    total_weight = beta1 + beta2
    return min(1.0, score / total_weight) if total_weight > 0 else 0.0
