"""
Microstructure Trading Strategies
===============================
Implementation of the three strategies from Index Strat11.2.txt:
1. Order Flow Momentum (OFM)
2. Microstructure Mean Reversion (MMR)
3. Liquidity Vacuum Breakout (LVB)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

from core.microstructure_analysis import (
    TickData, MarketState, MicrostructureMetrics,
    MarketEnvironmentClassifier, OrderFlowAnalyzer,
    SweepDetector, ConsolidationDetector, TickDataValidator
)

logger = logging.getLogger(__name__)


@dataclass
class PositionSizing:
    """Tick-based position sizing from Index Strat11.2.txt"""
    account_value: float
    risk_per_trade_pct: float = 0.01  # 1% default
    tick_size: float = 0.01
    tick_value: float = 1.0
    
    def calculate_position_size(self, entry_price: float, stop_price: float, 
                              tick_volatility: float = 1.0) -> int:
        """
        Position Sizing from Index Strat11.2.txt:
        
        # Tick-based volatility
        tick_changes_per_minute = count_tick_changes(60_seconds)
        volatility_scalar = min(1.0, 20 / tick_changes_per_minute)
        
        Risk_Per_Trade = Account_Value * 0.01 * volatility_scalar
        Tick_Risk = abs(entry - stop) / tick_size
        Contracts = floor(Risk_Per_Trade / (Tick_Risk * tick_value))
        """
        # Volatility scalar from tick changes
        volatility_scalar = min(1.0, 20 / max(1, tick_volatility))
        
        # Risk per trade adjusted for volatility
        risk_per_trade = self.account_value * self.risk_per_trade_pct * volatility_scalar
        
        # Calculate tick risk
        tick_risk = abs(entry_price - stop_price) / self.tick_size
        
        # Calculate contracts
        if tick_risk > 0:
            contracts = int(risk_per_trade / (tick_risk * self.tick_value))
            return max(1, contracts)  # Minimum 1 contract
        
        return 1


@dataclass
class TradeSignal:
    """Trading signal with execution details"""
    strategy_name: str
    direction: str  # 'LONG', 'SHORT'
    entry_price: float
    stop_price: float
    target_price: Optional[float] = None
    confidence: float = 0.0
    size: int = 1
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class OrderFlowMomentumStrategy:
    """
    Strategy 1: Order Flow Momentum (OFM)
    Best for: Institutional accumulation/distribution phases
    
    Parameters from Index Strat11.2.txt:
    - CVD_Period = 1000 ticks
    - Imbalance_Threshold = 1500 contracts (net delta)
    - Large_Trade_Size = 10 contracts minimum
    - Absorption_Ratio = 400 (volume per tick movement)
    - Trail_Ticks = 3
    """
    
    def __init__(self, 
                 cvd_period: int = 1000,
                 imbalance_threshold: float = 1500,
                 large_trade_size: int = 10,
                 absorption_ratio: float = 400,
                 trail_ticks: int = 3):
        
        self.cvd_period = cvd_period
        self.imbalance_threshold = imbalance_threshold
        self.large_trade_size = large_trade_size
        self.absorption_ratio = absorption_ratio
        self.trail_ticks = trail_ticks
        
        # Components
        self.order_flow_analyzer = OrderFlowAnalyzer(cvd_period, large_trade_size)
        self.position_sizer = PositionSizing(account_value=100000)  # Default 100k
        
        # State tracking
        self.current_position = None
        self.trailing_stop = None
        
    def generate_signals(self, ticks: List[TickData]) -> List[TradeSignal]:
        """
        Generate OFM signals based on order flow analysis
        
        LONG Entry (all conditions must be true):
        1. cvd > Imbalance_Threshold
        2. large_ratio > 0.35  # Institutional involvement
        3. absorption < Absorption_Ratio  # Not hitting resistance
        4. bid_pulling == True  # Market makers bullish
        5. spread <= 1 tick
        """
        signals = []
        
        if len(ticks) < self.cvd_period:
            return signals
            
        # Analyze order flow
        metrics = self.order_flow_analyzer.analyze_order_flow(ticks)
        current_tick = ticks[-1]
        
        # Calculate tick-based volatility for position sizing
        tick_changes = self._count_tick_changes(ticks[-60:])  # Last 60 ticks (~1 minute)
        
        # LONG Entry Conditions
        long_conditions = [
            metrics.cvd > self.imbalance_threshold,
            metrics.large_trade_ratio > 0.35,
            metrics.absorption_ratio < self.absorption_ratio,
            metrics.bid_pulling,
            current_tick.spread <= 0.01  # 1 tick spread
        ]
        
        # SHORT Entry Conditions (opposite)
        short_conditions = [
            metrics.cvd < -self.imbalance_threshold,
            metrics.large_trade_ratio > 0.35,
            metrics.absorption_ratio < self.absorption_ratio,
            metrics.ask_pulling,
            current_tick.spread <= 0.01
        ]
        
        if all(long_conditions):
            # Find stop price (largest bid cluster below entry - 1 tick)
            stop_price = self._find_support_level(ticks) - 0.01
            
            # Position sizing
            size = self.position_sizer.calculate_position_size(
                current_tick.price, stop_price, tick_changes
            )
            
            signal = TradeSignal(
                strategy_name="OrderFlowMomentum",
                direction="LONG",
                entry_price=current_tick.bid_price + 0.01,  # Aggressive post
                stop_price=stop_price,
                confidence=0.8,
                size=size,
                timestamp=current_tick.timestamp,
                metadata={
                    "cvd": metrics.cvd,
                    "large_ratio": metrics.large_trade_ratio,
                    "absorption": metrics.absorption_ratio,
                    "bid_pulling": metrics.bid_pulling
                }
            )
            signals.append(signal)
            
        elif all(short_conditions):
            # Find stop price (largest ask cluster above entry + 1 tick)
            stop_price = self._find_resistance_level(ticks) + 0.01
            
            # Position sizing
            size = self.position_sizer.calculate_position_size(
                current_tick.price, stop_price, tick_changes
            )
            
            signal = TradeSignal(
                strategy_name="OrderFlowMomentum",
                direction="SHORT",
                entry_price=current_tick.ask_price - 0.01,  # Aggressive post
                stop_price=stop_price,
                confidence=0.8,
                size=size,
                timestamp=current_tick.timestamp,
                metadata={
                    "cvd": metrics.cvd,
                    "large_ratio": metrics.large_trade_ratio,
                    "absorption": metrics.absorption_ratio,
                    "ask_pulling": metrics.ask_pulling
                }
            )
            signals.append(signal)
        
        return signals
    
    def _count_tick_changes(self, ticks: List[TickData]) -> float:
        """Count tick changes per minute for volatility adjustment"""
        if len(ticks) < 2:
            return 1.0
            
        changes = 0
        for i in range(1, len(ticks)):
            if abs(ticks[i].price - ticks[i-1].price) >= 0.01:  # 1 tick
                changes += 1
                
        # Convert to per-minute rate
        time_span_minutes = (ticks[-1].timestamp - ticks[0].timestamp).total_seconds() / 60
        if time_span_minutes > 0:
            return changes / time_span_minutes
        return 1.0
    
    def _find_support_level(self, ticks: List[TickData]) -> float:
        """Find support level (simplified - largest bid cluster)"""
        # In real implementation, this would analyze order book levels
        # For now, use recent lows as proxy
        recent_prices = [t.price for t in ticks[-100:]]
        return min(recent_prices) if recent_prices else ticks[-1].price
    
    def _find_resistance_level(self, ticks: List[TickData]) -> float:
        """Find resistance level (simplified - largest ask cluster)"""
        # In real implementation, this would analyze order book levels
        # For now, use recent highs as proxy
        recent_prices = [t.price for t in ticks[-100:]]
        return max(recent_prices) if recent_prices else ticks[-1].price


class MicrostructureMeanReversionStrategy:
    """
    Strategy 2: Microstructure Mean Reversion (MMR)
    Best for: Sweep exhaustion and liquidity gaps
    
    Parameters from Index Strat11.2.txt:
    - Sweep_Threshold = 75 contracts (single aggressive order)
    - Book_Imbalance = 3.0 (bid/ask ratio)
    - Quiet_Period = 200 ticks
    - Reversion_Percent = 0.6
    - Max_Heat = 4 ticks
    """
    
    def __init__(self,
                 sweep_threshold: int = 75,
                 book_imbalance: float = 3.0,
                 quiet_period: int = 200,
                 reversion_percent: float = 0.6,
                 max_heat: int = 4):
        
        self.sweep_threshold = sweep_threshold
        self.book_imbalance = book_imbalance
        self.quiet_period = quiet_period
        self.reversion_percent = reversion_percent
        self.max_heat = max_heat
        
        # Components
        self.sweep_detector = SweepDetector(sweep_threshold)
        self.position_sizer = PositionSizing(account_value=100000)
        
    def generate_signals(self, ticks: List[TickData]) -> List[TradeSignal]:
        """
        Generate MMR signals based on sweep detection and mean reversion
        
        Setup from Index Strat11.2.txt:
        1. Detect aggressive sweep
        2. Monitor post-sweep activity for quiet period
        3. Check book imbalance for entry
        """
        signals = []
        
        if len(ticks) < self.quiet_period:
            return signals
            
        # Detect recent sweep
        recent_ticks = ticks[-50:]  # Look at recent ticks for sweep
        sweep = self.sweep_detector.detect_sweep(recent_ticks)
        
        if sweep:
            # Check if quiet period has passed
            ticks_since_sweep = self._ticks_since_sweep(ticks, sweep)
            
            if ticks_since_sweep >= self.quiet_period:
                # Check if volume has dried up
                if self._is_volume_dried_up(ticks, sweep):
                    # Analyze book support
                    book_support_ratio = self._analyze_book_support(ticks[-1])
                    
                    current_tick = ticks[-1]
                    
                    # LONG Entry (fade down sweep)
                    if (sweep['direction'] == 'SELL' and 
                        book_support_ratio > self.book_imbalance):
                        
                        entry_price = sweep['price'] + 0.01  # 1 tick above sweep
                        stop_price = sweep['price'] - (sweep['levels'] * 0.01)
                        target_price = entry_price + (self.reversion_percent * 
                                                    (entry_price - sweep['price']))
                        
                        size = self.position_sizer.calculate_position_size(
                            entry_price, stop_price
                        )
                        
                        signal = TradeSignal(
                            strategy_name="MicrostructureMeanReversion",
                            direction="LONG",
                            entry_price=entry_price,
                            stop_price=stop_price,
                            target_price=target_price,
                            confidence=0.75,
                            size=size,
                            timestamp=current_tick.timestamp,
                            metadata={
                                "sweep_price": sweep['price'],
                                "sweep_size": sweep['size'],
                                "book_support": book_support_ratio,
                                "ticks_since_sweep": ticks_since_sweep
                            }
                        )
                        signals.append(signal)
                    
                    # SHORT Entry (fade up sweep)
                    elif (sweep['direction'] == 'BUY' and 
                          1/book_support_ratio > self.book_imbalance):
                        
                        entry_price = sweep['price'] - 0.01  # 1 tick below sweep
                        stop_price = sweep['price'] + (sweep['levels'] * 0.01)
                        target_price = entry_price - (self.reversion_percent * 
                                                    (sweep['price'] - entry_price))
                        
                        size = self.position_sizer.calculate_position_size(
                            entry_price, stop_price
                        )
                        
                        signal = TradeSignal(
                            strategy_name="MicrostructureMeanReversion",
                            direction="SHORT",
                            entry_price=entry_price,
                            stop_price=stop_price,
                            target_price=target_price,
                            confidence=0.75,
                            size=size,
                            timestamp=current_tick.timestamp,
                            metadata={
                                "sweep_price": sweep['price'],
                                "sweep_size": sweep['size'],
                                "book_resistance": 1/book_support_ratio,
                                "ticks_since_sweep": ticks_since_sweep
                            }
                        )
                        signals.append(signal)
        
        return signals
    
    def _ticks_since_sweep(self, ticks: List[TickData], sweep: Dict) -> int:
        """Count ticks since sweep occurred"""
        sweep_time = sweep['timestamp']
        count = 0
        
        for tick in reversed(ticks):
            if tick.timestamp <= sweep_time:
                break
            count += 1
            
        return count
    
    def _is_volume_dried_up(self, ticks: List[TickData], sweep: Dict) -> bool:
        """
        Check if volume has dried up after sweep
        Condition: avg_volume < sweep.size * 0.2
        """
        ticks_since_sweep = self._ticks_since_sweep(ticks, sweep)
        
        if ticks_since_sweep < self.quiet_period:
            return False
            
        # Get post-sweep ticks
        post_sweep_ticks = ticks[-ticks_since_sweep:]
        
        if not post_sweep_ticks:
            return False
            
        avg_volume = np.mean([t.volume for t in post_sweep_ticks])
        return avg_volume < sweep['size'] * 0.2
    
    def _analyze_book_support(self, tick: TickData) -> float:
        """
        Analyze book support/resistance
        From Index Strat11.2.txt: sum of bid/ask sizes in top 5 levels
        Simplified to use current bid/ask sizes
        """
        if tick.ask_size == 0:
            return float('inf')
        return tick.bid_size / tick.ask_size


class LiquidityVacuumBreakoutStrategy:
    """
    Strategy 3: Liquidity Vacuum Breakout (LVB)
    Best for: Pre-breakout consolidation with dried up volume
    
    Parameters from Index Strat11.2.txt:
    - Consolidation_Ticks = 500
    - Volume_Reduction = 0.3 (vs average)
    - Range_Ticks = 5 (max during consolidation)
    - Breakout_Volume = 100 contracts
    - Target_Multiple = 2.5
    """
    
    def __init__(self,
                 consolidation_ticks: int = 500,
                 volume_reduction: float = 0.3,
                 range_ticks: int = 5,
                 breakout_volume: int = 100,
                 target_multiple: float = 2.5):
        
        self.consolidation_ticks = consolidation_ticks
        self.volume_reduction = volume_reduction
        self.range_ticks = range_ticks
        self.breakout_volume = breakout_volume
        self.target_multiple = target_multiple
        
        # Components
        self.consolidation_detector = ConsolidationDetector(consolidation_ticks, range_ticks)
        self.position_sizer = PositionSizing(account_value=100000)
        
    def generate_signals(self, ticks: List[TickData]) -> List[TradeSignal]:
        """
        Generate LVB signals based on consolidation detection and breakout
        
        Logic from Index Strat11.2.txt:
        1. Detect consolidation (low range, reduced volume)
        2. Monitor for volume surge breakout
        3. Confirm with next tick
        """
        signals = []
        
        if len(ticks) < self.consolidation_ticks * 2:
            return signals
            
        # Detect consolidation
        consolidation = self.consolidation_detector.detect_consolidation(ticks)
        
        if consolidation:
            # Monitor for breakout
            current_tick = ticks[-1]
            
            # Check for volume surge
            volume_surge = current_tick.volume >= self.breakout_volume
            
            if volume_surge:
                # Calculate CVD for breakout direction
                tick_cvd = self._calculate_tick_cvd(current_tick)
                
                # LONG Breakout
                if (current_tick.price > consolidation['high'] and tick_cvd > 0):
                    # Confirm with next tick (would need next tick in real implementation)
                    # For now, assume confirmation
                    
                    entry_price = current_tick.price  # Market order
                    stop_price = consolidation['low'] - 0.01
                    target_price = entry_price + (self.target_multiple * 
                                                 (entry_price - stop_price))
                    
                    size = self.position_sizer.calculate_position_size(
                        entry_price, stop_price
                    )
                    
                    signal = TradeSignal(
                        strategy_name="LiquidityVacuumBreakout",
                        direction="LONG",
                        entry_price=entry_price,
                        stop_price=stop_price,
                        target_price=target_price,
                        confidence=0.85,
                        size=size,
                        timestamp=current_tick.timestamp,
                        metadata={
                            "consolidation_high": consolidation['high'],
                            "consolidation_low": consolidation['low'],
                            "breakout_volume": current_tick.volume,
                            "cvd": tick_cvd
                        }
                    )
                    signals.append(signal)
                
                # SHORT Breakout
                elif (current_tick.price < consolidation['low'] and tick_cvd < 0):
                    entry_price = current_tick.price  # Market order
                    stop_price = consolidation['high'] + 0.01
                    target_price = entry_price - (self.target_multiple * 
                                                 (stop_price - entry_price))
                    
                    size = self.position_sizer.calculate_position_size(
                        entry_price, stop_price
                    )
                    
                    signal = TradeSignal(
                        strategy_name="LiquidityVacuumBreakout",
                        direction="SHORT",
                        entry_price=entry_price,
                        stop_price=stop_price,
                        target_price=target_price,
                        confidence=0.85,
                        size=size,
                        timestamp=current_tick.timestamp,
                        metadata={
                            "consolidation_high": consolidation['high'],
                            "consolidation_low": consolidation['low'],
                            "breakout_volume": current_tick.volume,
                            "cvd": tick_cvd
                        }
                    )
                    signals.append(signal)
        
        return signals
    
    def _calculate_tick_cvd(self, tick: TickData) -> float:
        """Calculate single tick CVD (simplified)"""
        if tick.aggressor == 'BUY':
            return tick.volume
        elif tick.aggressor == 'SELL':
            return -tick.volume
        else:
            # Use bid/ask imbalance as proxy
            if tick.bid_size > tick.ask_size:
                return tick.volume
            else:
                return -tick.volume


class MasterControlLayer:
    """
    Master Control Layer (Tick-Based) from Index Strat11.2.txt
    Coordinates all three strategies and provides risk management
    """
    
    def __init__(self, account_value: float = 100000):
        # Strategy instances
        self.ofm_strategy = OrderFlowMomentumStrategy()
        self.mmr_strategy = MicrostructureMeanReversionStrategy()
        self.lvb_strategy = LiquidityVacuumBreakoutStrategy()
        
        # Market environment classifier
        self.market_classifier = MarketEnvironmentClassifier()
        
        # Risk management parameters from Index Strat11.2.txt
        self.MAX_TICKS_PER_SECOND = 50
        self.MIN_BOOK_DEPTH = 100
        self.MAX_SPREAD = 2  # ticks
        
        # State
        self.trading_disabled_until = None
        
    def process_ticks(self, ticks: List[TickData]) -> List[TradeSignal]:
        """
        Main tick processing pipeline
        
        From Index Strat11.2.txt:
        1. Validate tick data
        2. Classify market environment
        3. Select appropriate strategy
        4. Apply risk management
        """
        signals = []
        
        if not ticks:
            return signals
            
        current_tick = ticks[-1]
        
        # 1. Validate tick data
        if not TickDataValidator.validate_tick_data(current_tick):
            return signals
            
        if not TickDataValidator.validate_tick_frequency(ticks[-10:], self.MAX_TICKS_PER_SECOND):
            # Disable trading for 60 seconds
            self.trading_disabled_until = current_tick.timestamp + timedelta(seconds=60)
            return signals
        
        # Check if trading is disabled
        if (self.trading_disabled_until and 
            current_tick.timestamp < self.trading_disabled_until):
            return signals
        
        # 2. Classify market environment
        market_state = self.market_classifier.classify_market(ticks)
        
        # 3. Strategy selection based on market state
        selected_strategy = self._select_strategy(market_state, ticks)
        
        if selected_strategy:
            # 4. Generate signals from selected strategy
            if selected_strategy == "OFM":
                signals = self.ofm_strategy.generate_signals(ticks)
            elif selected_strategy == "MMR":
                signals = self.mmr_strategy.generate_signals(ticks)
            elif selected_strategy == "LVB":
                signals = self.lvb_strategy.generate_signals(ticks)
        
        # 5. Apply additional risk checks
        signals = self._apply_risk_management(signals, ticks)
        
        return signals
    
    def _select_strategy(self, market_state: MarketState, ticks: List[TickData]) -> Optional[str]:
        """
        Strategy Selection from Index Strat11.2.txt:
        
        # Toxic market - no trading
        if market_state == "TOXIC" or not validate_tick_data():
            return None
        
        # Clear institutional flow
        if abs(cvd_5000) > 5000 and large_trade_ratio > 0.4:
            return "OFM"
        
        # Post-sweep opportunity
        if recent_sweep_detected and ticks_since_sweep > 100:
            return "MMR"
        
        # Consolidation breakout setup
        if volatility_contraction and volume_decline:
            return "LVB"
        """
        # Toxic market - no trading
        if market_state.state == "TOXIC":
            return None
            
        # Clear institutional flow -> OFM
        if (abs(market_state.order_flow_efficiency) > 0.3 and 
            market_state.large_trade_ratio > 0.4):
            return "OFM"
        
        # Check for recent sweep -> MMR
        sweep_detector = SweepDetector()
        recent_sweep = sweep_detector.detect_sweep(ticks[-100:])
        if recent_sweep:
            ticks_since_sweep = len(ticks) - next(
                (i for i, t in enumerate(reversed(ticks)) 
                 if t.timestamp <= recent_sweep['timestamp']), 
                0
            )
            if ticks_since_sweep > 100:
                return "MMR"
        
        # Consolidation setup -> LVB
        consolidation_detector = ConsolidationDetector()
        consolidation = consolidation_detector.detect_consolidation(ticks)
        if consolidation and market_state.spread_volatility < 1.0:
            return "LVB"
            
        return None  # No clear opportunity
    
    def _apply_risk_management(self, signals: List[TradeSignal], 
                             ticks: List[TickData]) -> List[TradeSignal]:
        """
        Apply risk management filters from Index Strat11.2.txt
        
        # Pre-trade checks
        if ticks_per_second > MAX_TICKS_PER_SECOND:
            disable_trading(60_seconds)
        
        if book_depth < MIN_BOOK_DEPTH:
            reduce_size_by_half()
        
        if spread > MAX_SPREAD:
            cancel_all_orders()
            flatten_positions()
        """
        if not ticks or not signals:
            return signals
            
        current_tick = ticks[-1]
        filtered_signals = []
        
        for signal in signals:
            # Check book depth
            book_depth = current_tick.bid_size + current_tick.ask_size
            if book_depth < self.MIN_BOOK_DEPTH:
                signal.size = max(1, signal.size // 2)  # Reduce size by half
            
            # Check spread
            spread_ticks = current_tick.spread / 0.01  # Convert to ticks
            if spread_ticks > self.MAX_SPREAD:
                continue  # Skip signal (equivalent to cancel orders)
            
            filtered_signals.append(signal)
        
        return filtered_signals 