"""
test_statistics_analyzer.py
==========================
Test the statistics analyzer functionality
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gui.main_hub import TradingStrategyHub
from gui.statistics_window import StatisticsWindow
from core.data_structures import BaseStrategy, ProbabilityMetrics
from strategies.strategy_builders import PatternStrategy, Action
from core.data_structures import TimeRange


def create_test_data():
    """Create test data for the statistics analyzer"""
    # Create test dataset
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    data = pd.DataFrame({
        'Open': np.random.uniform(100, 200, 1000),
        'High': np.random.uniform(100, 200, 1000),
        'Low': np.random.uniform(100, 200, 1000),
        'Close': np.random.uniform(100, 200, 1000),
        'Volume': np.random.uniform(1000, 10000, 1000)
    }, index=dates)
    
    # Create test pattern using a concrete implementation
    timeframes = [TimeRange(value=1, unit='minute')]
    
    # Create test action
    action = Action(
        name="Test Action",
        pattern=pattern,
        time_range={"value": 5, "unit": "minutes"}
    )
    
    # Create test strategy
    strategy = PatternStrategy(
        name="Test Strategy",
        actions=[action],
        min_actions_required=1,
        gates_and_logic={
            'location_gate': True,
            'volatility_gate': False,
            'regime_gate': False,
            'bayesian_gate': False,
            'exec_gates': False,
            'alignment': False,
            'master_equation': False,
            'kelly_sizing': False,
            'stop_loss': True,
            'k_stop': 2,
            'tail_risk': False,
            'rolling_support_resistance': False,
            'sr_window': 20,
            'market_maker_reversion': False,
            'sigma_r': 1.0,
            'sigma_t': 1.0,
            'epsilon': 0.1,
            'mmrs_threshold': 0.5,
            'pattern_confidence': False,
            'kappa_conf': 0.8,
            'tau_conf': 0.6,
            'imbalance_memory': False,
            'gamma_mem': 0.9,
            'sigma_rev': 1.0,
            'bayesian_tracking': False,
            'min_state_probability': 0.5,
            'exec_threshold': 0.7,
        }
    )
    
    # Add probability metrics
    metrics = ProbabilityMetrics()
    metrics.probability = 0.65
    metrics.confidence_interval = (0.60, 0.70)
    metrics.sample_size_adequate = True
    strategy.update_probability(metrics)
    
    return data, strategy


def test_statistics_analyzer():
    """Test the statistics analyzer"""
    print("Testing Statistics Analyzer...")
    
    # Create QApplication
    app = QApplication(sys.argv)
    
    # Create main hub
    hub = TradingStrategyHub()
    
    # Create test data
    test_data, test_strategy = create_test_data()
    
    # Add test data to hub
    hub.datasets['Test Dataset'] = {
        'data': test_data,
        'metadata': {'name': 'Test Dataset', 'source': 'test'}
    }
    
    # Add test strategy to hub
    hub.strategies['pattern'][test_strategy.id] = test_strategy
    
    # Create mock backtest results
    backtest_results = {
        'strategy_name': 'Test Strategy',
        'total_return': 0.15,
        'sharpe_ratio': 1.2,
        'max_drawdown': -0.08,
        'win_rate': 0.65,
        'profit_factor': 1.8,
        'total_trades': 45,
        'avg_win': 0.02,
        'avg_loss': -0.015,
        'expectancy': 0.008,
        'equity_curve': [100000, 101000, 102500, 101800, 103200],
        'trades': [],
        'status': 'completed'
    }
    
    # Add backtest results
    result_id = datetime.now().strftime("%Y%m%d%H%M%S")
    hub.results[result_id] = backtest_results
    
    # Create statistics window
    stats_window = StatisticsWindow(hub)
    stats_window.show()
    
    # Test data population
    print("Testing data population...")
    stats_window.refresh_data()
    
    # Check if strategies are populated
    strategy_count = stats_window.strategy_list.count()
    print(f"Strategies in list: {strategy_count}")
    assert strategy_count > 0, "No strategies found in list"
    
    # Check if datasets are populated
    dataset_count = stats_window.dataset_combo.count()
    print(f"Datasets in combo: {dataset_count}")
    assert dataset_count > 1, "No datasets found in combo"  # > 1 because of "-- Select Dataset --"
    
    # Test probability analysis
    print("Testing probability analysis...")
    if stats_window.strategy_list.count() > 0:
        # Select first strategy
        stats_window.strategy_list.item(0).setSelected(True)
        
        # Run probability analysis
        stats_window._run_probability_analysis()
        
        # Check if results are displayed
        prob_text = stats_window.prob_results.toPlainText()
        print(f"Probability analysis results: {len(prob_text)} characters")
        assert len(prob_text) > 0, "No probability analysis results"
    
    # Test validation tests
    print("Testing validation tests...")
    if stats_window.strategy_list.count() > 0:
        # Run validation tests
        stats_window._run_validation_tests()
        
        # Check if results are displayed
        validation_text = stats_window.validation_results.toPlainText()
        print(f"Validation results: {len(validation_text)} characters")
        assert len(validation_text) > 0, "No validation results"
    
    # Test acceptance scoring
    print("Testing acceptance scoring...")
    if stats_window.strategy_list.count() > 0:
        # Run acceptance scoring
        stats_window._run_acceptance_scoring()
        
        # Check if results are displayed
        acceptance_text = stats_window.acceptance_results.toPlainText()
        print(f"Acceptance results: {len(acceptance_text)} characters")
        assert len(acceptance_text) > 0, "No acceptance results"
    
    # Test correlation analysis
    print("Testing correlation analysis...")
    if stats_window.strategy_list.count() >= 2:
        # Select multiple strategies
        for i in range(min(2, stats_window.strategy_list.count())):
            stats_window.strategy_list.item(i).setSelected(True)
        
        # Run correlation analysis
        stats_window._run_correlation_analysis()
        
        # Check if correlation table is populated
        table_rows = stats_window.corr_table.rowCount()
        print(f"Correlation table rows: {table_rows}")
        assert table_rows > 0, "Correlation table not populated"
    
    print("All statistics analyzer tests passed!")
    
    # Close window after a delay
    QTimer.singleShot(2000, stats_window.close)
    QTimer.singleShot(2500, app.quit)
    
    # Run the application
    app.exec()


if __name__ == "__main__":
    test_statistics_analyzer() 