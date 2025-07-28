import sys
import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel,c
    QLineEdit, QPushButton, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox
)
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
from matplotlib.patches import Rectangle


# simple streaming CUSUM detector
class CUSUMDetector:
    def __init__(self, threshold, drift=0.0):
        self.threshold = threshold
        self.drift = drift
        self.pos_sum = 0.0
        self.neg_sum = 0.0

    def update(self, x):
        self.pos_sum = max(0.0, self.pos_sum + x - self.drift)
        self.neg_sum = min(0.0, self.neg_sum + x + self.drift)
        if self.pos_sum > self.threshold or abs(self.neg_sum) > self.threshold:
            self.pos_sum = 0.0
            self.neg_sum = 0.0
            
            return True
        return False


# identify fair value gaps
def detect_fvg_gaps(df, lookback, multiplier):
    fvg = []
    total = len(df)


    for i in range(2, total):
        ph = float(df['High'].iloc[i-2])
        pl = float(df['Low'].iloc[i-2])
        o  = float(df['Open'].iloc[i-1])
        c  = float(df['Close'].iloc[i-1])
        low = float(df['Low'].iloc[i])
        high = float(df['High'].iloc[i])


        start = max(0, i - lookback - 1)
        opens = df['Open'].iloc[start:i-1].astype(float).values
        closes = df['Close'].iloc[start:i-1].astype(float).values
        bodies = np.abs(closes - opens)
        avg_body = float(np.mean(bodies)) if bodies.size else 1e-6

        size = abs(c - o)
        if low > ph and size > avg_body * multiplier:
            fvg.append({'type':'bullish','start':ph,'end':low,'bar':i})
        elif high < pl and size > avg_body * multiplier:
            fvg.append({'type':'bearish','start':pl,'end':high,'bar':i})
    return fvg


# identify 5-bar fractal pivots
def detect_fractal_pivots(df, atr, mult):
    pivots = []
    total = len(df)


    for i in range(2, total - 2):
        win_h = df['High'].iloc[i-2:i+3]
        win_l = df['Low'].iloc[i-2:i+3]
        o = float(df['Open'].iloc[i])
        c = float(df['Close'].iloc[i])

        body = abs(c - o)
        atr_v = float(atr.iloc[i])
        hi = float(df['High'].iloc[i])
        lo = float(df['Low'].iloc[i])

        if hi == float(win_h.max()) and body >= mult * atr_v:
            pivots.append({'type':'high','price':hi,'bar':i})
        if lo == float(win_l.min()) and body >= mult * atr_v:
            pivots.append({'type':'low','price':lo,'bar':i})
    return pivots


class StockChart(QWidget):

    def __init__(self):
        super().__init__()
        self.symbol = "AAPL"
        self.timeframe = "1d"
        self.period_map = {
            '1m':'7d','5m':'1mo','15m':'60d',
            '1h':'6mo','1d':'1y','1wk':'5y'
        }
        self.period = self.period_map[self.timeframe]
        self.cusum_k = 4.0

        # UI setup
        main_layout = QVBoxLayout(self)
        controls = QHBoxLayout()

        controls.addWidget(QLabel("Ticker:"))
        self.ticker_input = QLineEdit(self.symbol)
        controls.addWidget(self.ticker_input)

        controls.addWidget(QLabel("Interval:"))
        self.interval_combo = QComboBox()
        for tf in ['1wk','1d','1h','15m','5m','1m']:
            self.interval_combo.addItem(tf)
        self.interval_combo.setCurrentText(self.timeframe)
        controls.addWidget(self.interval_combo)



        controls.addWidget(QLabel("Enable Regime"))
        self.regime_cb = QCheckBox()
        self.regime_cb.setChecked(True)
        controls.addWidget(self.regime_cb)

        controls.addWidget(QLabel("Dynamic Params"))
        self.dynamic_cb = QCheckBox("Dynamic")
        controls.addWidget(self.dynamic_cb)

        controls.addWidget(QLabel("Lookback:"))
        self.lookback_spin = QSpinBox()
        self.lookback_spin.setRange(5,50)
        self.lookback_spin.setValue(10)
        controls.addWidget(self.lookback_spin)

        controls.addWidget(QLabel("Body ×:"))
        self.mult_spin = QDoubleSpinBox()
        self.mult_spin.setRange(1.0,5.0)
        self.mult_spin.setSingleStep(0.1)
        self.mult_spin.setValue(1.5)
        controls.addWidget(self.mult_spin)

        controls.addWidget(QLabel("Extension:"))
        self.ext_spin = QSpinBox()
        self.ext_spin.setRange(3,30)
        self.ext_spin.setValue(5)
        controls.addWidget(self.ext_spin)

        controls.addWidget(QLabel("Fractal ATR×:"))
        self.fractal_spin = QDoubleSpinBox()
        self.fractal_spin.setRange(0.00001,3.0)
        self.fractal_spin.setSingleStep(0.00001)
        self.fractal_spin.setValue(0.08)
        controls.addWidget(self.fractal_spin)

        controls.addWidget(QLabel("Uptrend ≥"))
        self.up_spin = QDoubleSpinBox()
        self.up_spin.setRange(-1,1)
        self.up_spin.setSingleStep(0.001)
        self.up_spin.setDecimals(4)
        self.up_spin.setValue(0.002)
        controls.addWidget(self.up_spin)

        controls.addWidget(QLabel("Downtrend ≤"))
        self.down_spin = QDoubleSpinBox()
        self.down_spin.setRange(-1,1)
        self.down_spin.setSingleStep(0.001)
        self.down_spin.setDecimals(4)
        self.down_spin.setValue(-0.002)
        controls.addWidget(self.down_spin)


        controls.addWidget(QLabel("Calib k:"))
        self.k_spin = QDoubleSpinBox()
        self.k_spin.setRange(0.0,5.0)
        self.k_spin.setSingleStep(0.1)
        self.k_spin.setDecimals(2)
        self.k_spin.setValue(0.50)
        controls.addWidget(self.k_spin)

        controls.addWidget(QLabel("Days:"))
        self.days_spin = QSpinBox()
        self.days_spin.setRange(1,365)
        self.days_spin.setValue(7)
        controls.addWidget(self.days_spin)

        load_btn = QPushButton("Load")
        controls.addWidget(load_btn)

        main_layout.addLayout(controls)


        self.figure, self.axis = plt.subplots(figsize=(10,5))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas)

        # signals
        load_btn.clicked.connect(self.redraw_chart)
        self.interval_combo.currentTextChanged.connect(self.on_interval_change)

        # initial draw
        self.redraw_chart()

    def on_interval_change(self, new_interval):
        self.timeframe = new_interval
        self.period = self.period_map[new_interval]

    def compute_adaptive_parameters(self, df):
        recent_high = float(df['High'].iloc[-1])
        recent_low = float(df['Low'].iloc[-1])
        prev_close = float(df['Close'].iloc[-2])
        rv = ((recent_high - recent_low) / prev_close)**2

        returns = df['Close'].pct_change().dropna() * 100
        max_lb = min(50, len(returns))

        mse = {}
        for L in range(5, max_lb + 1):
            r_slice = returns.iloc[-L:]
            model = arch_model(r_slice, mean='Zero', vol='GARCH', p=1, q=1, dist='normal')
            res = model.fit(disp='off')
            fc_var = res.forecast(horizon=1).variance.values[-1,0] / 100**2
            mse[L] = (rv - float(fc_var))**2

        best_lb = min(mse, key=mse.get)

        atr = df['High'].sub(df['Low']).ewm(span=14, adjust=False).mean()
        scaled = np.abs((df['Close'] - df['Open']) / atr)
        best_mult = float(np.quantile(scaled.iloc[-best_lb:], 0.90))

        return int(best_lb), best_mult, atr

    def redraw_chart(self):
        # fetch data
        df = yf.download(
            self.ticker_input.text().upper().strip() or "AAPL",
            period=self.period,
            interval=self.timeframe,
            auto_adjust=False,
            progress=False
        ).dropna()

        days = self.days_spin.value()
        if self.timeframe in ['1m','5m','15m','30m','1h']:
            df = df.last(f"{days}d")

        else:
            df = df.tail(days)

        df['bar'] = range(len(df))

        if self.dynamic_cb.isChecked():
            lookback, multiplier, atr = self.compute_adaptive_parameters(df)
        else:
            lookback = self.lookback_spin.value()
            multiplier = self.mult_spin.value()
            atr = df['High'].sub(df['Low']).ewm(span=14, adjust=False).mean()

        self.axis.clear()


        candlestick_ohlc(self.axis, df[['bar','Open','High','Low','Close']].values,
                         width=0.6, colorup='green', colordown='red', alpha=0.8)

        # regime logic
        if self.regime_cb.isChecked():
            df['ret'] = df['Close'].pct_change().fillna(0)
            sigma = float(df['ret'].std())
            thresh = self.cusum_k * sigma
            detector = CUSUMDetector(thresh)

            changes = []
            for idx, r in zip(df['bar'], df['ret']):
                if detector.update(r):
                    changes.append(idx)

            pivots = detect_fractal_pivots(df, atr, self.fractal_spin.value())

            # compute zones
            bounds = [0] + changes + [df['bar'].iloc[-1] + 1]
            zones = []
            for i in range(len(bounds) - 1):
                s, e = bounds[i], bounds[i+1]
                p0 = float(df['Close'].iloc[s])
                p1 = float(df['Close'].iloc[e-1])
                pct = (p1 - p0) / p0
                rate = pct / (e - s) if (e - s) > 0 else 0.0
                zones.append((s, e, rate))

            # auto-calibrate thresholds

            rates = np.array([r for (_s,_e,r) in zones])
            if rates.size:
                med, std = np.median(rates), rates.std()
                k = float(self.k_spin.value())
                auto_up = med + k * std
                auto_down = med - k * std
                self.up_spin.setValue(auto_up)
                self.down_spin.setValue(auto_down)
            else:
                auto_up = float(self.up_spin.value())
                auto_down = float(self.down_spin.value())

            # highlight zones

            for s, e, rate in zones:
                if rate >= auto_up:
                    clr = 'green'
                elif rate <= auto_down:
                    clr = 'red'
                else:
                    continue
                self.axis.axvspan(s, e, color=clr, alpha=0.2)

            # regime lines & pivots
            for c in changes:
                self.axis.axvline(x=c, color='purple', linestyle='--', linewidth=1)
            for p in pivots:
                mark = '^' if p['type']=='low' else 'v'
                self.axis.scatter(p['bar'], p['price'], marker=mark, color='blue', s=50)

        # fair value gaps
        gaps = detect_fvg_gaps(df, lookback, multiplier)
        for g in gaps:

            i = g['bar']
            x0 = i - 0.3
            if self.dynamic_cb.isChecked():
                h = abs(g['end'] - g['start'])
                rate = 0.35 * h / float(atr.iloc[-1])
                ext = int(np.clip(np.log(2)/rate, 3, 30))
            else:
                ext = self.ext_spin.value()
            self.axis.add_patch(
                Rectangle(
                    (x0, min(g['start'],g['end'])),
                    ext,
                    abs(g['end']-g['start']),
                    facecolor='green' if g['type']=='bullish' else 'red',
                    alpha=0.3
                )
            )

        # x-axis formatting
        ticks = df['bar'][::max(len(df)//10,1)]
        labels = [d.strftime('%Y-%m-%d\n%H:%M') for
                   d in df.index[::max(len(df)//10,1)]]
        self.axis.set_xticks(ticks)
        self.axis.set_xticklabels(labels, rotation=45, ha='right')

        self.axis.set_title(f"{self.ticker_input.text().upper()} — {self.timeframe}/{self.period}")
        self.axis.set_ylabel("Price")
        self.canvas.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock Viewer with Regime Zones, FVGs & Pivots")
        self.setCentralWidget(StockChart())
        self.resize(1000, 600)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())
