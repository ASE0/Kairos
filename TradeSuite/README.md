# Trading Strategy Hub - Professional Edition

## Overview

Trading Strategy Hub is a comprehensive GUI application for building, testing, and combining advanced trading strategies with sophisticated quantification and execution logic. This professional-grade platform integrates mathematical candlestick quantification, Bayesian state tracking, volatility integration, and master equation scoring for optimal trading decisions.

## 🚀 Key Features

### Core Components
- **Pattern Builder**: Create custom candlestick patterns with advanced quantification
- **Strategy Builder**: Build complex strategies with execution gates and logic
- **Strategy Combiner**: Combine multiple strategies for enhanced performance
- **Risk Manager**: Advanced risk management with Kelly sizing and tail risk
- **Backtest Engine**: Comprehensive backtesting with performance metrics
- **Workspace Manager**: Save/load/export complete workspaces

### Advanced Quantification Features
- **Body Size & Wick Analysis**: Mathematical quantification of candle components
- **Doji-ness Scoring**: Gaussian-based doji detection with configurable parameters
- **Two-Bar Strength**: Pattern strength across multiple bars
- **Location Context**: Fair Value Gaps (FVG), peaks, and skew models
- **Momentum Integration**: Adaptive location scoring based on recent momentum
- **Volatility Integration**: ATR, realized volatility, and GARCH models

### Execution Logic & Gates
- **Location Gate**: Ensures trades at optimal price locations
- **Volatility Gate**: Filters based on market volatility conditions
- **Regime Gate**: Adapts to different market regimes (trending/ranging)
- **Bayesian State Gate**: Probability-based state tracking
- **Master Equation**: Ultimate scoring system combining all features
- **Kelly Sizing**: Optimal position sizing based on edge and volatility

## 🏗️ Architecture

### Core Modules
```
core/
├── feature_quantification.py    # Advanced quantification functions
├── data_structures.py          # Core data structures
├── dataset_manager.py          # Dataset management
└── workspace_manager.py        # Workspace persistence

patterns/
├── candlestick_patterns.py     # Pattern definitions and detection
└── enhanced_candlestick_patterns.py

strategies/
└── strategy_builders.py        # Strategy construction and backtesting

gui/
├── main_hub.py                 # Main application window
├── pattern_builder_window.py   # Pattern creation interface
├── strategy_builder_window.py  # Strategy building interface
├── strategy_combiner_window.py # Strategy combination interface
└── workspace_manager_dialog.py # Workspace management
```

## 📊 Mathematical Framework

### 1. Candlestick Quantification
- **Body Size**: `Bt = |C-O|`
- **Upper Wick**: `Wu = H - max(O,C)`
- **Lower Wick**: `Wl = min(O,C) - L`
- **Doji-ness**: `Dt = exp[-(Bt/Range)²/(2σ_b²)] · exp[-(Ŵu-Ŵl)²/(2σ_w²)]`
- **Two-Bar Strength**: `A₂bar = β_pat·(Body₂/Body₁)`

### 2. Location Models
- **Flat Plateau**: `L_base(x) = flat plateau inside gap [x0, x1]`
- **Micro Peaks**: `C_peaks(x) = sum_k exp[-(x-x_k)²/(2σ²)]`
- **Dual Layer**: `L_total = β₁ L_base + β₂ C_peaks`
- **Momentum Boost**: `L_mom = L_total·(1+κ_m|M|)`

### 3. Volatility Integration
- **Realized Volatility**: `σ_t = sqrt[(1/N)Σ(r_i-r̄)²]`
- **ATR**: `ATR_t = (1/n)Σ TR_i`
- **Composite Vol**: `V(x,y) = w₁ σ_t + w₂ ATR_t`
- **GARCH**: `σ_t² = ω + α ε²_{t-1} + β σ²_{t-1}`

### 4. Master Equation
- **Pattern Score**: `St,i = A_pattern · K_i(d_imp) · L_mom(x) · C_i`
- **Final Score**: `S(t,x,y) = Σ w_i St,i + β_v V`
- **Execution**: `S_exec = S_adj·C_align·1_{MMRS_enhanced>τ}`

### 5. Risk Management
- **Kelly Sizing**: `f* = (p·b - q)/b · 1/σ_t`
- **Stop Loss**: `Stop = k_stop·ATR_t·√h`
- **Tail Risk**: Fat-tail adjustment using GPD/ES

## 🎯 Usage Guide

### 1. Creating Patterns
1. Open **Pattern Builder**
2. Select base pattern type or create custom
3. Configure OHLC ratios and custom formulas
4. Enable advanced features (body size, doji-ness, etc.)
5. Set location and momentum context parameters
6. Save pattern to workspace

### 2. Building Strategies
1. Open **Strategy Builder**
2. Add patterns as actions with time ranges
3. Configure location strategies (VWAP, POC, etc.)
4. Set filters (volume, volatility, time)
5. Enable execution gates and logic
6. Configure master equation parameters
7. Set Kelly sizing and risk management
8. Test strategy on datasets
9. Save strategy to workspace

### 3. Combining Strategies
1. Open **Strategy Combiner**
2. Select multiple saved strategies
3. Configure combination logic (AND, OR, weighted)
4. Set probability thresholds
5. Run combination analysis
6. Save combined strategy

### 4. Workspace Management
1. Use **Workspace Manager** to:
   - View all saved components
   - Load/delete/rename items
   - Export/import complete workspaces
   - Manage configurations

## 🔧 Installation

### Prerequisites
```bash
Python 3.8+
PyQt6
pandas
numpy
scipy
pyqtgraph
```

### Setup
```bash
# Clone repository
git clone <repository-url>
cd TradeSuite

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
python main.py
```

## 📈 Performance Metrics

The backtesting engine provides comprehensive metrics:
- **Total Return**: Overall strategy performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Kelly Score**: Optimal position sizing

## 🎨 Advanced Features

### Quantification Integration
- All patterns automatically use advanced quantification when enabled
- Real-time strength calculation based on multiple factors
- Context-aware pattern detection

### Execution Logic
- Multi-gate filtering system
- Bayesian state tracking
- Volatility-adaptive execution
- Master equation scoring

### Risk Management
- Kelly criterion position sizing
- ATR-based stop losses
- Tail risk adjustments
- Dynamic position management

### Workspace Persistence
- Automatic saving of all components
- Export/import functionality
- Version control for strategies
- Configuration management

## 🔮 Future Enhancements

- **Machine Learning Integration**: Neural networks for pattern recognition
- **Real-time Data**: Live market data integration
- **Multi-Asset Support**: Forex, crypto, commodities
- **Advanced Analytics**: Monte Carlo simulations, stress testing
- **API Integration**: Broker connectivity for live trading
- **Cloud Deployment**: Web-based interface

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📞 Support

For support and questions, please open an issue in the repository or contact the development team.

---

**Trading Strategy Hub - Professional Edition v2.0**  
*Advanced Trading Strategy Development Platform with Complete Integration* 