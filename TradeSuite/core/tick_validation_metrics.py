"""
Tick-Based Strategy Validation Metrics
=====================================
Implementation of validation framework from Mathematical Framework for STRAT VALIDATION.txt
Includes edge quantification, regime detection, risk metrics, and execution quality analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from scipy import stats
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import logging

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a completed trade with execution details"""
    entry_time: datetime
    exit_time: datetime
    entry_bid: float
    entry_ask: float
    exit_bid: float
    exit_ask: float
    fill_price: float
    limit_price: float
    size: float
    commission: float
    pnl: float
    direction: str  # 'LONG', 'SHORT'
    
    @property
    def entry_mid(self) -> float:
        """Entry mid price"""
        return (self.entry_bid + self.entry_ask) / 2
    
    @property
    def exit_mid(self) -> float:
        """Exit mid price"""
        return (self.exit_bid + self.exit_ask) / 2
    
    @property
    def tick_return(self) -> float:
        """Tick-level return"""
        return (self.exit_mid - self.entry_mid) / self.entry_mid
    
    @property
    def price_improvement(self) -> float:
        """Price improvement in ticks"""
        tick_size = 0.01
        return (self.fill_price - self.limit_price) / tick_size
    
    @property
    def execution_alpha(self) -> float:
        """Execution alpha from price improvement"""
        tick_value = 1.0
        return self.price_improvement * tick_value


@dataclass
class RegimeState:
    """Market regime state with transition probabilities"""
    current_state: str  # 'Trending', 'Ranging', 'Volatile'
    probability: float
    duration: int  # bars in current state
    transition_matrix: np.ndarray = field(default_factory=lambda: np.eye(3))


@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics"""
    # Edge metrics
    net_edge_ticks: float = 0.0
    expected_value_ticks: float = 0.0
    win_rate: float = 0.0
    avg_win_ticks: float = 0.0
    avg_loss_ticks: float = 0.0
    
    # Risk metrics
    sharpe_ratio: float = 0.0
    modified_sharpe: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    risk_of_ruin: float = 0.0
    
    # Execution metrics
    fill_rate: float = 0.0
    adverse_selection: float = 0.0
    implementation_shortfall: float = 0.0
    slippage_distribution: List[float] = field(default_factory=list)
    
    # Regime metrics
    regime_accuracy: float = 0.0
    regime_f1_scores: Dict[str, float] = field(default_factory=dict)
    
    # Statistical metrics
    hasbrouck_information_share: float = 0.0
    walk_forward_efficiency: float = 0.0


class TickBasedEdgeQuantification:
    """
    Edge Quantification (Tick-Based) from Mathematical Framework
    """
    
    def __init__(self, tick_size: float = 0.01, tick_value: float = 1.0):
        self.tick_size = tick_size
        self.tick_value = tick_value
    
    def calculate_edge(self, trades: List[Trade]) -> Dict[str, float]:
        """
        Calculate edge metrics from Mathematical Framework:
        
        # Expected Value per Trade (in ticks)
        E[Ticks] = win_rate * avg_win_ticks - loss_rate * avg_loss_ticks
        
        # Adjust for market conditions
        spread_cost = avg_spread / 2  # Half spread per side
        tick_slippage = 0.3  # Empirical slippage
        commission_ticks = commission / tick_value
        
        # Net Edge
        Net_Edge = E[Ticks] - spread_cost - tick_slippage - commission_ticks
        Required: Net_Edge > 0.5 ticks
        """
        if not trades:
            return {}
        
        # Convert PnL to ticks
        tick_outcomes = []
        for trade in trades:
            tick_pnl = trade.pnl / self.tick_value
            tick_outcomes.append(tick_pnl)
        
        # Calculate win/loss statistics
        winning_trades = [t for t in tick_outcomes if t > 0]
        losing_trades = [t for t in tick_outcomes if t < 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0
        loss_rate = 1 - win_rate
        avg_win_ticks = np.mean(winning_trades) if winning_trades else 0
        avg_loss_ticks = abs(np.mean(losing_trades)) if losing_trades else 0
        
        # Expected value per trade
        expected_value = win_rate * avg_win_ticks - loss_rate * avg_loss_ticks
        
        # Calculate costs
        avg_spread = np.mean([(t.entry_ask - t.entry_bid) for t in trades])
        spread_cost = (avg_spread / 2) / self.tick_size  # Half spread in ticks
        tick_slippage = 0.3  # Empirical slippage
        avg_commission = np.mean([t.commission for t in trades])
        commission_ticks = avg_commission / self.tick_value
        
        # Net edge
        net_edge = expected_value - spread_cost - tick_slippage - commission_ticks
        
        # Confidence bounds using bootstrap
        bootstrap_means = []
        for _ in range(1000):
            sample = np.random.choice(tick_outcomes, size=len(tick_outcomes), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        confidence_interval = np.percentile(bootstrap_means, [2.5, 97.5])
        
        return {
            'expected_value_ticks': expected_value,
            'net_edge_ticks': net_edge,
            'win_rate': win_rate,
            'avg_win_ticks': avg_win_ticks,
            'avg_loss_ticks': avg_loss_ticks,
            'spread_cost_ticks': spread_cost,
            'slippage_ticks': tick_slippage,
            'commission_ticks': commission_ticks,
            'confidence_lower': confidence_interval[0],
            'confidence_upper': confidence_interval[1],
            'edge_acceptable': net_edge > 0.5
        }
    
    def test_microstructure_noise(self, trades: List[Trade]) -> float:
        """
        Test for microstructure noise using Hasbrouck Information Share
        
        # Test for microstructure noise
        Hasbrouck_Information_Share = var(efficient_price) / var(transaction_price)
        if Hasbrouck_IS < 0.8:
            warning("High microstructure noise")
        """
        if len(trades) < 10:
            return 0.0
        
        # Use mid prices as proxy for efficient prices
        efficient_prices = [t.entry_mid for t in trades]
        transaction_prices = [t.fill_price for t in trades]
        
        var_efficient = np.var(efficient_prices)
        var_transaction = np.var(transaction_prices)
        
        if var_transaction == 0:
            return 1.0
        
        hasbrouck_is = var_efficient / var_transaction
        return hasbrouck_is


class MarkovRegimeSwitchingModel:
    """
    Markov Regime Switching Model from Mathematical Framework
    """
    
    def __init__(self, n_states: int = 3):
        self.n_states = n_states
        self.states = ['Trending', 'Ranging', 'Volatile']
        self.transition_matrix = np.eye(n_states) * 0.8 + 0.1  # Initial guess
        self.current_state = 0
        self.state_history = []
        
    def estimate_transition_matrix(self, regime_sequence: List[int]) -> np.ndarray:
        """
        Estimate transition probability matrix
        
        # Estimate from data:
        p_ij = count(transitions from i to j) / count(state i)
        """
        if len(regime_sequence) < 2:
            return self.transition_matrix
        
        # Count transitions
        transition_counts = np.zeros((self.n_states, self.n_states))
        state_counts = np.zeros(self.n_states)
        
        for i in range(len(regime_sequence) - 1):
            current_state = regime_sequence[i]
            next_state = regime_sequence[i + 1]
            transition_counts[current_state, next_state] += 1
            state_counts[current_state] += 1
        
        # Calculate probabilities
        for i in range(self.n_states):
            if state_counts[i] > 0:
                self.transition_matrix[i, :] = transition_counts[i, :] / state_counts[i]
        
        return self.transition_matrix
    
    def classify_regime(self, returns: np.ndarray, window: int = 20) -> Tuple[int, float]:
        """
        Classify current market regime based on returns
        
        States:
        0: Trending - high momentum, directional moves
        1: Ranging - low momentum, mean-reverting
        2: Volatile - high volatility, erratic moves
        """
        if len(returns) < window:
            return 0, 0.33
        
        recent_returns = returns[-window:]
        
        # Calculate regime indicators
        momentum = np.abs(np.mean(recent_returns))
        volatility = np.std(recent_returns)
        autocorr = np.corrcoef(recent_returns[:-1], recent_returns[1:])[0, 1] if len(recent_returns) > 1 else 0
        
        # Classify based on thresholds
        if momentum > 0.01 and autocorr > 0.2:  # Strong trend
            regime = 0  # Trending
            confidence = min(0.95, momentum * 50 + autocorr)
        elif volatility < 0.01 and abs(autocorr) < 0.1:  # Low vol, no trend
            regime = 1  # Ranging
            confidence = min(0.95, 1 - volatility * 100)
        else:  # High volatility
            regime = 2  # Volatile
            confidence = min(0.95, volatility * 50)
        
        return regime, confidence
    
    def calculate_regime_performance(self, returns: np.ndarray, 
                                   regimes: List[int]) -> Dict[str, Dict[str, float]]:
        """
        Calculate strategy performance by regime
        
        # Strategy performance by regime
        E[R|S] = expected return given state S
        σ[R|S] = volatility given state S
        
        # Optimal strategy selection
        π*(S) = argmax_strategy E[R|S] / σ[R|S]  # Maximum Sharpe by regime
        """
        regime_performance = {}
        
        for regime_id, regime_name in enumerate(self.states):
            regime_mask = np.array(regimes) == regime_id
            if np.any(regime_mask):
                regime_returns = returns[regime_mask]
                
                expected_return = np.mean(regime_returns)
                volatility = np.std(regime_returns)
                sharpe = expected_return / volatility if volatility > 0 else 0
                
                regime_performance[regime_name] = {
                    'expected_return': expected_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe,
                    'count': np.sum(regime_mask)
                }
        
        return regime_performance


class RegimeDetectionAccuracy:
    """
    Regime Detection Accuracy from Mathematical Framework
    """
    
    @staticmethod
    def calculate_accuracy_metrics(true_regimes: List[int], 
                                 predicted_regimes: List[int]) -> Dict[str, float]:
        """
        Calculate regime detection accuracy metrics
        
        # Confusion matrix for regime classification
        Accuracy = (TP + TN) / Total
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        
        # F1 Score for regime detection
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        Target: F1 > 0.7 for each regime
        """
        if len(true_regimes) != len(predicted_regimes):
            return {}
        
        # Overall accuracy
        accuracy = accuracy_score(true_regimes, predicted_regimes)
        
        # Per-class metrics
        precision_scores = precision_score(true_regimes, predicted_regimes, average=None, zero_division=0)
        recall_scores = recall_score(true_regimes, predicted_regimes, average=None, zero_division=0)
        f1_scores = f1_score(true_regimes, predicted_regimes, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(true_regimes, predicted_regimes)
        
        return {
            'accuracy': accuracy,
            'precision_trending': precision_scores[0] if len(precision_scores) > 0 else 0,
            'precision_ranging': precision_scores[1] if len(precision_scores) > 1 else 0,
            'precision_volatile': precision_scores[2] if len(precision_scores) > 2 else 0,
            'recall_trending': recall_scores[0] if len(recall_scores) > 0 else 0,
            'recall_ranging': recall_scores[1] if len(recall_scores) > 1 else 0,
            'recall_volatile': recall_scores[2] if len(recall_scores) > 2 else 0,
            'f1_trending': f1_scores[0] if len(f1_scores) > 0 else 0,
            'f1_ranging': f1_scores[1] if len(f1_scores) > 1 else 0,
            'f1_volatile': f1_scores[2] if len(f1_scores) > 2 else 0,
            'confusion_matrix': cm.tolist()
        }


class RiskAdjustedPerformanceMetrics:
    """
    Risk-Adjusted Performance Metrics from Mathematical Framework
    """
    
    @staticmethod
    def calculate_modified_sharpe_ratio(returns: np.ndarray) -> float:
        """
        Modified Sharpe Ratio (for non-normal returns)
        
        # Cornish-Fisher adjustment
        SR_modified = SR_standard * (1 + S/6 * SR_standard - (K-3)/24 * SR_standard²)
        """
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Standard Sharpe ratio
        sr_standard = mean_return / std_return
        
        # Skewness and kurtosis
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Cornish-Fisher adjustment
        sr_modified = sr_standard * (1 + skewness/6 * sr_standard - (kurtosis-3)/24 * sr_standard**2)
        
        return sr_modified
    
    @staticmethod
    def calculate_probabilistic_sharpe_ratio(observed_sr: float, benchmark_sr: float,
                                           n_observations: int, skewness: float,
                                           kurtosis: float) -> float:
        """
        Probabilistic Sharpe Ratio
        
        # Probabilistic Sharpe Ratio
        PSR = Φ((SR_observed - SR_benchmark) * √n / √(1 - S*SR + (K-3)/4 * SR²))
        where Φ = cumulative standard normal
        """
        if n_observations <= 0:
            return 0.0
        
        numerator = (observed_sr - benchmark_sr) * np.sqrt(n_observations)
        denominator = np.sqrt(1 - skewness * observed_sr + (kurtosis - 3) / 4 * observed_sr**2)
        
        if denominator == 0:
            return 0.0
        
        z_score = numerator / denominator
        psr = stats.norm.cdf(z_score)
        
        return psr
    
    @staticmethod
    def calculate_maximum_drawdown_distribution(returns: np.ndarray) -> Dict[str, float]:
        """
        Maximum Drawdown Distribution
        
        # For strategy validation, calculate:
        MDD_expected = σ * √(2 * log(T))  # Theoretical maximum
        MDD_observed = max peak-to-trough decline
        
        # Calmar Ratio
        Calmar = Annual_Return / MDD_observed
        Target: Calmar > 1.0
        """
        if len(returns) < 2:
            return {}
        
        # Calculate drawdown
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        
        mdd_observed = np.min(drawdown)
        
        # Theoretical maximum drawdown
        sigma = np.std(returns)
        T = len(returns)
        mdd_expected = sigma * np.sqrt(2 * np.log(T)) if T > 1 else 0
        
        # Calmar ratio
        annual_return = np.mean(returns) * 252  # Assuming daily returns
        calmar = abs(annual_return / mdd_observed) if mdd_observed != 0 else 0
        
        return {
            'mdd_observed': mdd_observed,
            'mdd_expected': mdd_expected,
            'calmar_ratio': calmar,
            'calmar_acceptable': calmar > 1.0
        }
    
    @staticmethod
    def calculate_risk_of_ruin(win_prob: float, win_loss_ratio: float,
                              initial_capital: float, kelly_fraction: float) -> float:
        """
        Risk of Ruin calculation
        
        # Kelly Criterion for position sizing
        f* = (p * b - q) / b
        where:
          p = win probability
          b = win/loss ratio
          q = 1 - p
        
        # Probability of 50% drawdown
        P(ruin) = ((1-f*p)/(1+f*b))^(C/2)
        where C = initial capital in units
        Target: P(ruin) < 0.01
        """
        if win_prob <= 0 or win_prob >= 1 or win_loss_ratio <= 0:
            return 1.0
        
        p = win_prob
        q = 1 - p
        b = win_loss_ratio
        
        # Kelly fraction
        f_star = (p * b - q) / b
        
        # Risk of ruin for 50% drawdown
        if f_star <= 0:
            return 1.0
        
        # Simplified risk of ruin formula
        capital_units = initial_capital / 1000  # Assume $1000 per unit
        
        base = (1 - f_star * p) / (1 + f_star * b)
        if base <= 0:
            return 0.0
        
        risk_of_ruin = base ** (capital_units / 2)
        
        return min(1.0, risk_of_ruin)


class ExecutionQualityMetrics:
    """
    Execution Quality metrics from Mathematical Framework
    """
    
    @staticmethod
    def calculate_slippage_analysis(trades: List[Trade]) -> Dict[str, float]:
        """
        Slippage analysis from Mathematical Framework
        
        # Slippage analysis
        Slippage_Distribution = histogram(fill_price - intended_price)
        Expected_Slippage = mean(slippage)
        Slippage_95th = percentile(slippage, 95)
        """
        if not trades:
            return {}
        
        slippages = []
        for trade in trades:
            # Calculate slippage (difference between fill and limit price)
            slippage = trade.fill_price - trade.limit_price
            slippages.append(slippage)
        
        expected_slippage = np.mean(slippages)
        slippage_95th = np.percentile(slippages, 95)
        slippage_std = np.std(slippages)
        
        return {
            'expected_slippage': expected_slippage,
            'slippage_95th': slippage_95th,
            'slippage_std': slippage_std,
            'slippage_distribution': slippages,
            'expected_slippage_acceptable': abs(expected_slippage) < 0.5  # < 0.5 ticks
        }
    
    @staticmethod
    def calculate_market_impact(trades: List[Trade]) -> Dict[str, float]:
        """
        Market Impact analysis from Mathematical Framework
        
        # Temporary impact
        Temp_Impact = (price_at_fill - price_at_decision) / order_size
        # Permanent impact  
        Perm_Impact = (price_10min_later - price_at_decision) / order_size
        
        # Model: Impact = α√size + β*participation_rate
        """
        if len(trades) < 10:
            return {}
        
        # Calculate temporary impact (simplified)
        temp_impacts = []
        sizes = []
        
        for trade in trades:
            # Temporary impact approximation
            temp_impact = (trade.fill_price - trade.entry_mid) / trade.size
            temp_impacts.append(temp_impact)
            sizes.append(trade.size)
        
        # Fit linear model: impact ~ sqrt(size)
        sqrt_sizes = np.sqrt(sizes)
        if len(sqrt_sizes) > 1 and np.std(sqrt_sizes) > 0:
            correlation = np.corrcoef(temp_impacts, sqrt_sizes)[0, 1]
            
            # Linear regression coefficients (simplified)
            alpha = np.mean(temp_impacts) / np.mean(sqrt_sizes) if np.mean(sqrt_sizes) != 0 else 0
        else:
            correlation = 0
            alpha = 0
        
        return {
            'temp_impact_mean': np.mean(temp_impacts),
            'temp_impact_std': np.std(temp_impacts),
            'impact_size_correlation': correlation,
            'alpha_coefficient': alpha
        }


class ComprehensiveValidator:
    """
    Comprehensive strategy validator implementing all metrics from Mathematical Framework
    """
    
    def __init__(self):
        self.edge_quantifier = TickBasedEdgeQuantification()
        self.regime_model = MarkovRegimeSwitchingModel()
        self.regime_accuracy = RegimeDetectionAccuracy()
        self.risk_metrics = RiskAdjustedPerformanceMetrics()
        self.execution_metrics = ExecutionQualityMetrics()
    
    def validate_strategy(self, trades: List[Trade], 
                         returns: np.ndarray,
                         regimes_true: Optional[List[int]] = None,
                         regimes_predicted: Optional[List[int]] = None) -> ValidationMetrics:
        """
        Comprehensive strategy validation
        
        Minimum Statistical Requirements from Mathematical Framework:
        1. Sample_Size > 100 trades
        2. Sharpe_Ratio > 1.0
        3. Profit_Factor > 1.5
        4. Win_Rate * Avg_Win > 1.2 * Loss_Rate * Avg_Loss
        5. Max_Drawdown < 15%
        6. P(profit) > 0.95 (Monte Carlo)
        7. WFE > 0.5
        8. All parameters stable (±20%)
        9. Positive in 2/3 market regimes
        10. Transaction costs < 30% of gross profit
        """
        metrics = ValidationMetrics()
        
        # 1. Edge quantification
        if trades:
            edge_metrics = self.edge_quantifier.calculate_edge(trades)
            metrics.net_edge_ticks = edge_metrics.get('net_edge_ticks', 0)
            metrics.expected_value_ticks = edge_metrics.get('expected_value_ticks', 0)
            metrics.win_rate = edge_metrics.get('win_rate', 0)
            metrics.avg_win_ticks = edge_metrics.get('avg_win_ticks', 0)
            metrics.avg_loss_ticks = edge_metrics.get('avg_loss_ticks', 0)
            
            # Hasbrouck Information Share
            metrics.hasbrouck_information_share = self.edge_quantifier.test_microstructure_noise(trades)
        
        # 2. Risk metrics
        if len(returns) > 0:
            metrics.sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            metrics.modified_sharpe = self.risk_metrics.calculate_modified_sharpe_ratio(returns)
            
            # Drawdown metrics
            dd_metrics = self.risk_metrics.calculate_maximum_drawdown_distribution(returns)
            metrics.max_drawdown = dd_metrics.get('mdd_observed', 0)
            metrics.calmar_ratio = dd_metrics.get('calmar_ratio', 0)
            
            # Risk of ruin
            if metrics.win_rate > 0 and metrics.avg_win_ticks > 0:
                win_loss_ratio = metrics.avg_win_ticks / max(metrics.avg_loss_ticks, 0.1)
                metrics.risk_of_ruin = self.risk_metrics.calculate_risk_of_ruin(
                    metrics.win_rate, win_loss_ratio, 100000, 0.1
                )
        
        # 3. Execution metrics
        if trades:
            metrics.fill_rate = len([t for t in trades if t.fill_price > 0]) / len(trades)
            
            # Adverse selection
            adverse_selections = []
            for trade in trades:
                # Simplified: use mid price change as proxy
                price_change = trade.exit_mid - trade.entry_mid
                adverse_selections.append(abs(price_change))
            
            metrics.adverse_selection = np.mean(adverse_selections) if adverse_selections else 0
            
            # Implementation shortfall
            implementation_shortfalls = []
            for trade in trades:
                shortfall = (trade.fill_price - trade.entry_mid) / trade.entry_mid
                implementation_shortfalls.append(abs(shortfall))
            
            metrics.implementation_shortfall = np.mean(implementation_shortfalls) if implementation_shortfalls else 0
            
            # Slippage
            slippage_metrics = self.execution_metrics.calculate_slippage_analysis(trades)
            metrics.slippage_distribution = slippage_metrics.get('slippage_distribution', [])
        
        # 4. Regime accuracy
        if regimes_true and regimes_predicted:
            accuracy_metrics = self.regime_accuracy.calculate_accuracy_metrics(
                regimes_true, regimes_predicted
            )
            metrics.regime_accuracy = accuracy_metrics.get('accuracy', 0)
            metrics.regime_f1_scores = {
                'trending': accuracy_metrics.get('f1_trending', 0),
                'ranging': accuracy_metrics.get('f1_ranging', 0),
                'volatile': accuracy_metrics.get('f1_volatile', 0)
            }
        
        return metrics
    
    def check_acceptance_criteria(self, metrics: ValidationMetrics, 
                                trades: List[Trade]) -> Dict[str, bool]:
        """
        Check minimum acceptance criteria from Mathematical Framework
        """
        criteria = {}
        
        # 1. Sample size
        criteria['sample_size'] = len(trades) > 100
        
        # 2. Sharpe ratio
        criteria['sharpe_ratio'] = metrics.sharpe_ratio > 1.0
        
        # 3. Profit factor (simplified as win/loss ratio)
        if metrics.avg_loss_ticks > 0:
            profit_factor = (metrics.win_rate * metrics.avg_win_ticks) / ((1 - metrics.win_rate) * metrics.avg_loss_ticks)
            criteria['profit_factor'] = profit_factor > 1.5
        else:
            criteria['profit_factor'] = False
        
        # 4. Risk-reward ratio
        if metrics.avg_loss_ticks > 0:
            criteria['risk_reward'] = (metrics.win_rate * metrics.avg_win_ticks) > (1.2 * (1 - metrics.win_rate) * metrics.avg_loss_ticks)
        else:
            criteria['risk_reward'] = False
        
        # 5. Maximum drawdown
        criteria['max_drawdown'] = abs(metrics.max_drawdown) < 0.15  # 15%
        
        # 6. Fill rate
        criteria['fill_rate'] = metrics.fill_rate > 0.85
        
        # 7. Adverse selection
        criteria['adverse_selection'] = metrics.adverse_selection < 0.5 * metrics.avg_win_ticks
        
        # 8. Implementation shortfall
        criteria['implementation_shortfall'] = metrics.implementation_shortfall < 0.001  # 0.1%
        
        # 9. Risk of ruin
        criteria['risk_of_ruin'] = metrics.risk_of_ruin < 0.01
        
        # 10. Regime F1 scores
        criteria['regime_detection'] = all(score > 0.7 for score in metrics.regime_f1_scores.values())
        
        return criteria 