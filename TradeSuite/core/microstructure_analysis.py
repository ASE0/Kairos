"""
Microstructure Analysis Module
=============================
Tick-based analysis components for high-frequency strategy validation
Implements the algorithms from Index Strat11.2.txt
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class TickData:
    """Represents a single tick with microstructure information"""
    timestamp: datetime
    price: float
    volume: int
    bid_price: float = 0.0
    ask_price: float = 0.0
    bid_size: int = 0
    ask_size: int = 0
    aggressor: Optional[str] = None  # 'BUY', 'SELL', None
    trade_size: int = 0
    
    @property
    def spread(self) -> float:
        """Bid-ask spread"""
        if self.bid_price > 0 and self.ask_price > 0:
            return self.ask_price - self.bid_price
        return 0.0
    
    @property
    def mid_price(self) -> float:
        """Mid price"""
        if self.bid_price > 0 and self.ask_price > 0:
            return (self.bid_price + self.ask_price) / 2
        return self.price


@dataclass
class MarketState:
    """Market environment classification"""
    state: str  # TRENDING, RANGING, VOLATILE, TOXIC
    order_flow_efficiency: float = 0.0
    large_trade_ratio: float = 0.0
    spread_volatility: float = 0.0
    confidence: float = 0.0
    
    
@dataclass
class MicrostructureMetrics:
    """Comprehensive microstructure metrics"""
    # Order flow metrics
    cvd: float = 0.0  # Cumulative Volume Delta
    large_trade_ratio: float = 0.0
    absorption_ratio: float = 0.0
    
    # Book metrics
    bid_pulling: bool = False
    ask_pulling: bool = False
    book_imbalance: float = 0.0
    
    # Quality metrics
    spread: float = 0.0
    ticks_per_second: float = 0.0
    book_depth: int = 0
    
    # Flow metrics
    aggressive_buying: float = 0.0
    aggressive_selling: float = 0.0
    sweep_detected: bool = False


class MarketEnvironmentClassifier:
    """
    Market Environment Classification (Tick-Based)
    Implements the system from Index Strat11.2.txt
    """
    
    def __init__(self, tick_window: int = 5000):
        self.tick_window = tick_window
        self.history: List[TickData] = []
        
    def classify_market(self, ticks: List[TickData]) -> MarketState:
        """
        Classify market environment every 5000 ticks
        
        Market States:
        - TRENDING: Order Flow Efficiency > 0.3 AND Large Trade Ratio > 0.4
        - RANGING: Bid/Ask Balance within 45-55% AND Spread stable
        - VOLATILE: Spread Volatility > 2.0 OR Sweep frequency > normal
        - TOXIC: Spread > 2 ticks OR Quote changes > 100/second
        """
        if len(ticks) < self.tick_window:
            return MarketState("UNKNOWN", confidence=0.0)
        
        # Calculate metrics
        order_flow_efficiency = self._calculate_order_flow_efficiency(ticks)
        large_trade_ratio = self._calculate_large_trade_ratio(ticks)
        spread_volatility = self._calculate_spread_volatility(ticks)
        
        # Additional metrics for classification
        bid_ask_balance = self._calculate_bid_ask_balance(ticks)
        quote_changes_per_second = self._calculate_quote_frequency(ticks)
        avg_spread = np.mean([t.spread for t in ticks if t.spread > 0])
        normal_spread = self._estimate_normal_spread(ticks)
        
        # Classification logic
        if avg_spread > 2 * normal_spread or quote_changes_per_second > 100:
            state = "TOXIC"
            confidence = 0.9
        elif spread_volatility > 2.0:
            state = "VOLATILE"
            confidence = 0.8
        elif order_flow_efficiency > 0.3 and large_trade_ratio > 0.4:
            state = "TRENDING"
            confidence = 0.85
        elif 0.45 <= bid_ask_balance <= 0.55 and spread_volatility < 1.0:
            state = "RANGING"
            confidence = 0.75
        else:
            state = "UNCERTAIN"
            confidence = 0.5
            
        return MarketState(
            state=state,
            order_flow_efficiency=order_flow_efficiency,
            large_trade_ratio=large_trade_ratio,
            spread_volatility=spread_volatility,
            confidence=confidence
        )
    
    def _calculate_order_flow_efficiency(self, ticks: List[TickData]) -> float:
        """Calculate Order Flow Efficiency = |CVD[0] - CVD[5000]| / Sum(|Volume_i|)"""
        cvd_start = 0.0
        cvd_end = 0.0
        total_volume = 0.0
        
        for i, tick in enumerate(ticks):
            volume_delta = 0.0
            if tick.aggressor == 'BUY':
                volume_delta = tick.volume
            elif tick.aggressor == 'SELL':
                volume_delta = -tick.volume
                
            if i == 0:
                cvd_start = volume_delta
            if i == len(ticks) - 1:
                cvd_end = volume_delta
                
            total_volume += abs(tick.volume)
        
        if total_volume > 0:
            return abs(cvd_end - cvd_start) / total_volume
        return 0.0
    
    def _calculate_large_trade_ratio(self, ticks: List[TickData], large_threshold: int = 10) -> float:
        """Calculate Trade Size Distribution = Large_Trades / Total_Trades"""
        large_trades = sum(1 for t in ticks if t.trade_size >= large_threshold)
        total_trades = len([t for t in ticks if t.trade_size > 0])
        
        if total_trades > 0:
            return large_trades / total_trades
        return 0.0
    
    def _calculate_spread_volatility(self, ticks: List[TickData]) -> float:
        """Calculate Spread Volatility = StdDev(Bid_Ask_Spread) / Mean(Spread)"""
        spreads = [t.spread for t in ticks if t.spread > 0]
        
        if len(spreads) > 1:
            mean_spread = np.mean(spreads)
            std_spread = np.std(spreads)
            if mean_spread > 0:
                return std_spread / mean_spread
        return 0.0
    
    def _calculate_bid_ask_balance(self, ticks: List[TickData]) -> float:
        """Calculate bid/ask balance"""
        bid_volume = sum(t.bid_size for t in ticks if t.bid_size > 0)
        ask_volume = sum(t.ask_size for t in ticks if t.ask_size > 0)
        total_volume = bid_volume + ask_volume
        
        if total_volume > 0:
            return bid_volume / total_volume
        return 0.5
    
    def _calculate_quote_frequency(self, ticks: List[TickData]) -> float:
        """Calculate quote changes per second"""
        if len(ticks) < 2:
            return 0.0
            
        time_span = (ticks[-1].timestamp - ticks[0].timestamp).total_seconds()
        if time_span > 0:
            return len(ticks) / time_span
        return 0.0
    
    def _estimate_normal_spread(self, ticks: List[TickData]) -> float:
        """Estimate normal spread (median or mode)"""
        spreads = [t.spread for t in ticks if t.spread > 0]
        if spreads:
            return np.median(spreads)
        return 0.01  # Default tick size


class OrderFlowAnalyzer:
    """
    Order Flow Analysis for Strategy 1: Order Flow Momentum (OFM)
    Implements the algorithms from Index Strat11.2.txt
    """
    
    def __init__(self, cvd_period: int = 1000, large_trade_size: int = 10):
        self.cvd_period = cvd_period
        self.large_trade_size = large_trade_size
        self.cvd_history: List[float] = []
        
    def analyze_order_flow(self, ticks: List[TickData]) -> MicrostructureMetrics:
        """
        Analyze order flow for OFM strategy
        
        Parameters from Index Strat11.2.txt:
        - CVD_Period = 1000 ticks
        - Imbalance_Threshold = 1500 contracts (net delta)
        - Large_Trade_Size = 10 contracts minimum
        - Absorption_Ratio = 400 (volume per tick movement)
        """
        if len(ticks) < self.cvd_period:
            return MicrostructureMetrics()
        
        # Calculate CVD over period
        cvd = self._calculate_cvd(ticks[-self.cvd_period:])
        
        # Calculate large trade ratio
        large_ratio = self._calculate_large_trade_ratio(ticks[-self.cvd_period:])
        
        # Calculate absorption ratio
        absorption = self._calculate_absorption_ratio(ticks[-100:])  # Last 100 ticks
        
        # Detect microstructure signals
        bid_pulling, ask_pulling = self._detect_market_maker_signals(ticks[-10:])
        
        # Calculate book imbalance
        book_imbalance = self._calculate_book_imbalance(ticks[-1:])
        
        # Calculate spread
        current_spread = ticks[-1].spread if ticks else 0.0
        
        return MicrostructureMetrics(
            cvd=cvd,
            large_trade_ratio=large_ratio,
            absorption_ratio=absorption,
            bid_pulling=bid_pulling,
            ask_pulling=ask_pulling,
            book_imbalance=book_imbalance,
            spread=current_spread
        )
    
    def _calculate_cvd(self, ticks: List[TickData]) -> float:
        """Calculate cumulative volume delta"""
        cvd = 0.0
        for tick in ticks:
            if tick.aggressor == 'BUY':
                cvd += tick.volume
            elif tick.aggressor == 'SELL':
                cvd -= tick.volume
        return cvd
    
    def _calculate_large_trade_ratio(self, ticks: List[TickData]) -> float:
        """Calculate ratio of large trades"""
        large_trades = sum(1 for t in ticks if t.trade_size >= self.large_trade_size)
        total_trades = len([t for t in ticks if t.trade_size > 0])
        
        if total_trades > 0:
            return large_trades / total_trades
        return 0.0
    
    def _calculate_absorption_ratio(self, ticks: List[TickData]) -> float:
        """Calculate absorption = volume_last_100_ticks / price_range_ticks"""
        if len(ticks) < 2:
            return 0.0
            
        total_volume = sum(t.volume for t in ticks)
        price_range = max(t.price for t in ticks) - min(t.price for t in ticks)
        
        # Assuming tick size of 0.01 (can be parameterized)
        tick_size = 0.01
        price_range_ticks = price_range / tick_size
        
        if price_range_ticks > 0:
            return total_volume / price_range_ticks
        return float('inf')  # No price movement
    
    def _detect_market_maker_signals(self, ticks: List[TickData]) -> Tuple[bool, bool]:
        """
        Detect market maker signals:
        - bid_pulling = bid_size_decreased > ask_size_decreased  # Bullish
        - ask_pulling = ask_size_decreased > bid_size_decreased  # Bearish
        """
        if len(ticks) < 2:
            return False, False
            
        bid_size_change = ticks[-1].bid_size - ticks[0].bid_size
        ask_size_change = ticks[-1].ask_size - ticks[0].ask_size
        
        bid_pulling = bid_size_change < ask_size_change  # Bid decreased more
        ask_pulling = ask_size_change < bid_size_change  # Ask decreased more
        
        return bid_pulling, ask_pulling
    
    def _calculate_book_imbalance(self, ticks: List[TickData]) -> float:
        """Calculate bid/ask book imbalance"""
        if not ticks or ticks[-1].ask_size == 0:
            return 1.0
            
        return ticks[-1].bid_size / ticks[-1].ask_size


class SweepDetector:
    """
    Aggressive Sweep Detection for Strategy 2: Microstructure Mean Reversion (MMR)
    """
    
    def __init__(self, sweep_threshold: int = 75):
        self.sweep_threshold = sweep_threshold
        self.recent_sweeps: List[Dict] = []
        
    def detect_sweep(self, ticks: List[TickData]) -> Optional[Dict]:
        """
        Detect aggressive sweep
        
        Parameters from Index Strat11.2.txt:
        - Sweep_Threshold = 75 contracts (single aggressive order)
        """
        for tick in ticks:
            if (tick.trade_size >= self.sweep_threshold and 
                tick.aggressor is not None):
                
                # Count price levels taken (simplified)
                levels_taken = self._count_levels_taken(tick, ticks)
                
                if levels_taken >= 3:  # Swept multiple levels
                    sweep = {
                        'price': tick.price,
                        'direction': tick.aggressor,
                        'size': tick.trade_size,
                        'levels': levels_taken,
                        'timestamp': tick.timestamp
                    }
                    self.recent_sweeps.append(sweep)
                    return sweep
        
        return None
    
    def _count_levels_taken(self, sweep_tick: TickData, ticks: List[TickData]) -> int:
        """Count how many price levels were swept (simplified estimation)"""
        # In real implementation, this would analyze the order book
        # For now, estimate based on trade size relative to typical size
        typical_size = np.mean([t.trade_size for t in ticks if t.trade_size > 0])
        if typical_size > 0:
            return min(10, int(sweep_tick.trade_size / typical_size))
        return 1


class ConsolidationDetector:
    """
    Consolidation Detection for Strategy 3: Liquidity Vacuum Breakout (LVB)
    """
    
    def __init__(self, consolidation_ticks: int = 500, range_ticks: int = 5):
        self.consolidation_ticks = consolidation_ticks
        self.range_ticks = range_ticks
        
    def detect_consolidation(self, ticks: List[TickData]) -> Optional[Dict]:
        """
        Detect consolidation pattern
        
        Parameters from Index Strat11.2.txt:
        - Consolidation_Ticks = 500
        - Volume_Reduction = 0.3 (vs average)
        - Range_Ticks = 5 (max during consolidation)
        """
        if len(ticks) < self.consolidation_ticks * 2:
            return None
            
        # Analyze current window
        current_window = ticks[-self.consolidation_ticks:]
        
        # Calculate range in ticks
        high_price = max(t.price for t in current_window)
        low_price = min(t.price for t in current_window)
        tick_size = 0.01  # Parameterizable
        range_in_ticks = (high_price - low_price) / tick_size
        
        # Check if range is small enough
        if range_in_ticks <= self.range_ticks:
            # Calculate volume reduction
            current_volume = sum(t.volume for t in current_window)
            
            # Compare to previous windows
            if len(ticks) >= self.consolidation_ticks * 11:  # Need 10 previous windows
                previous_volumes = []
                for i in range(10):
                    start_idx = -(self.consolidation_ticks * (i + 2))
                    end_idx = -(self.consolidation_ticks * (i + 1))
                    window_volume = sum(t.volume for t in ticks[start_idx:end_idx])
                    previous_volumes.append(window_volume)
                
                avg_previous_volume = np.mean(previous_volumes)
                
                if current_volume < avg_previous_volume * 0.3:  # Volume reduction threshold
                    return {
                        'high': high_price,
                        'low': low_price,
                        'mid': (high_price + low_price) / 2,
                        'start_tick': len(ticks) - self.consolidation_ticks,
                        'volume_ratio': current_volume / avg_previous_volume
                    }
        
        return None


class TickDataValidator:
    """
    Tick Data Quality Checks from Index Strat11.2.txt
    """
    
    @staticmethod
    def validate_tick_data(tick: TickData) -> bool:
        """
        Validate tick data quality
        
        Checks from Index Strat11.2.txt:
        - bid > ask: False (crossed market)
        - spread > 5 * normal_spread: False (wide spread)
        - time_since_last_tick > 1000ms: False (stale data)
        - bid_size == 0 or ask_size == 0: False (no liquidity)
        """
        # Check for crossed market
        if tick.bid_price > tick.ask_price:
            return False
            
        # Check for wide spread (need normal spread reference)
        normal_spread = 0.01  # This should be calculated dynamically
        if tick.spread > 5 * normal_spread:
            return False
            
        # Check for no liquidity
        if tick.bid_size == 0 or tick.ask_size == 0:
            return False
            
        return True
    
    @staticmethod
    def validate_tick_frequency(ticks: List[TickData], max_ticks_per_second: float = 50) -> bool:
        """Check if tick frequency is within normal range"""
        if len(ticks) < 2:
            return True
            
        time_span = (ticks[-1].timestamp - ticks[0].timestamp).total_seconds()
        if time_span > 0:
            ticks_per_second = len(ticks) / time_span
            return ticks_per_second <= max_ticks_per_second
        return True


# Utility functions for converting OHLCV to tick data (for testing)
def ohlcv_to_synthetic_ticks(ohlcv_data: pd.DataFrame, ticks_per_bar: int = 10) -> List[TickData]:
    """
    Convert OHLCV data to synthetic tick data for testing
    This is a simplified conversion - real tick data would be much more complex
    """
    ticks = []
    
    for idx, row in ohlcv_data.iterrows():
        # Create synthetic ticks within each bar
        for i in range(ticks_per_bar):
            # Linear interpolation between OHLC prices
            if i == 0:
                price = row['open']
            elif i == ticks_per_bar - 1:
                price = row['close']
            else:
                # Simulate price movement within the bar
                progress = i / (ticks_per_bar - 1)
                if progress < 0.3:
                    price = row['open'] + (row['high'] - row['open']) * (progress / 0.3)
                elif progress < 0.7:
                    price = row['high'] + (row['low'] - row['high']) * ((progress - 0.3) / 0.4)
                else:
                    price = row['low'] + (row['close'] - row['low']) * ((progress - 0.7) / 0.3)
            
            # Synthetic timestamp
            if isinstance(idx, pd.Timestamp):
                timestamp = idx + pd.Timedelta(seconds=i * 6)  # 6 seconds between ticks
            else:
                timestamp = datetime.now() + timedelta(seconds=i * 6)
            
            # Synthetic bid/ask (spread = 1 tick)
            tick_size = 0.01
            bid_price = price - tick_size / 2
            ask_price = price + tick_size / 2
            
            # Synthetic volume and aggressor
            volume = int(row['volume'] / ticks_per_bar)
            aggressor = 'BUY' if i % 2 == 0 else 'SELL'
            
            tick = TickData(
                timestamp=timestamp,
                price=price,
                volume=volume,
                bid_price=bid_price,
                ask_price=ask_price,
                bid_size=volume,
                ask_size=volume,
                aggressor=aggressor,
                trade_size=volume
            )
            
            ticks.append(tick)
    
    return ticks


# News time handling from Index Strat11.2.txt
class NewsTimeHandler:
    """
    News Time Handling from Index Strat11.2.txt
    
    News Windows (ET):
    - 8:30 AM ± 10 min: Major economic data
    - 10:00 AM ± 10 min: Secondary data
    - 2:00 PM ± 5 min: Fed minutes (when applicable)
    """
    
    def __init__(self):
        self.news_windows = [
            {"time": "08:30", "window_minutes": 10, "type": "major"},
            {"time": "10:00", "window_minutes": 10, "type": "secondary"},
            {"time": "14:00", "window_minutes": 5, "type": "fed"},
        ]
    
    def is_news_time(self, timestamp: datetime) -> Dict[str, Any]:
        """
        Check if current time is within a news window
        
        Returns:
        - in_news_window: bool
        - window_type: str
        - minutes_to_news: int (negative if after)
        """
        # Convert to ET (simplified - assumes timestamp is already in ET)
        current_time = timestamp.time()
        
        for window in self.news_windows:
            news_hour, news_minute = map(int, window["time"].split(":"))
            news_time = current_time.replace(hour=news_hour, minute=news_minute)
            
            # Calculate minutes difference
            current_minutes = current_time.hour * 60 + current_time.minute
            news_minutes = news_time.hour * 60 + news_time.minute
            minutes_diff = current_minutes - news_minutes
            
            if abs(minutes_diff) <= window["window_minutes"]:
                return {
                    "in_news_window": True,
                    "window_type": window["type"],
                    "minutes_to_news": -minutes_diff
                }
        
        return {
            "in_news_window": False,
            "window_type": None,
            "minutes_to_news": None
        }
    
    def get_news_adjustments(self, news_info: Dict[str, Any]) -> Dict[str, float]:
        """
        Get trading adjustments during news windows
        
        From Index Strat11.2.txt:
        - Increase minimum trade size filter by 2x
        - Require 2x normal edge to enter
        - Exit all positions 30 seconds before
        """
        if not news_info["in_news_window"]:
            return {"size_multiplier": 1.0, "edge_multiplier": 1.0, "exit_before_seconds": 0}
        
        return {
            "size_multiplier": 2.0,
            "edge_multiplier": 2.0,
            "exit_before_seconds": 30 if news_info["minutes_to_news"] > 0 else 0
        } 