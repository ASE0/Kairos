import csv
from dataclasses import dataclass, asdict
from typing import List, Optional, Sequence, Tuple, Dict, Any

# --- Bar and OBZone definitions (from spec) ---
@dataclass(slots=True)
class Bar:
    dt: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    @property
    def body(self) -> float:
        return abs(self.close - self.open)
    @property
    def up(self) -> bool:
        return self.close > self.open
    @property
    def down(self) -> bool:
        return self.close < self.open
    @property
    def wick_up(self) -> float:
        return self.high - max(self.open, self.close)
    @property
    def wick_dn(self) -> float:
        return min(self.open, self.close) - self.low
    @property
    def range(self) -> float:
        return self.high - self.low

@dataclass(slots=True)
class OBZone:
    kind: str  # 'bull' or 'bear'
    high: float
    low: float
    created_index: int
    expires_index: int
    mu: float
    strength: float = 1.0
    meta: Dict[str, Any] | None = None
    def asdict(self) -> Dict[str, Any]:
        return asdict(self)

# --- Rolling average helper ---
def _sma(values: Sequence[float], window: int, default: float = 0.0) -> List[float]:
    out: List[float] = [default] * len(values)
    if not values:
        return out
    run_sum = 0.0
    w = max(1, window)
    for i, v in enumerate(values):
        run_sum += v
        if i >= w:
            run_sum -= values[i - w]
        if i >= w - 1:
            out[i] = run_sum / w
        else:
            out[i] = run_sum / (i + 1)
    return out

def _impulse_score(bar: Bar, avg_range: float, gamma_imp: float, delta_imp: float) -> float:
    eps = 1e-9
    range_ratio = (bar.range / avg_range) if avg_range > 0 else 0.0
    wick_body_ratio = ((bar.wick_up + bar.wick_dn) / (bar.body + eps)) if bar.body > 0 else 0.0
    return (range_ratio ** gamma_imp) * (wick_body_ratio ** delta_imp)

def _impulse_over_window(bars: Sequence[Bar], start: int, max_bars: int,
                         avg_ranges: Sequence[float],
                         min_body_mult: float,
                         min_score: float,
                         gamma_imp: float, delta_imp: float,
                         avg_bodies: Sequence[float]) -> Optional[Tuple[int, str, float]]:
    n = len(bars)
    base_bar = bars[start]
    best: Tuple[int, str, float] | None = None
    for j in range(start + 1, min(n, start + 1 + max_bars)):
        b = bars[j]
        dir_ = 'up' if b.close > base_bar.close else 'down'
        body_avg = avg_bodies[j]
        body_ok = b.body >= min_body_mult * body_avg
        avg_rng = avg_ranges[j]
        score = _impulse_score(b, avg_rng if avg_rng > 0 else b.range, gamma_imp, delta_imp)
        score_ok = score >= min_score
        if body_ok and score_ok:
            if best is None or score > best[2]:
                best = (j, dir_, score)
    return best

def _find_block_candle(bars: Sequence[Bar], impulse_idx: int, direction: str,
                       max_lookback: int, min_body_frac: float,
                       avg_bodies: Sequence[float]) -> Optional[int]:
    need_down = direction == 'up'
    need_up = direction == 'down'
    for i in range(impulse_idx - 1, max(-1, impulse_idx - 1 - max_lookback), -1):
        b = bars[i]
        if need_down and not b.down:
            continue
        if need_up and not b.up:
            continue
        avgB = avg_bodies[i]
        if avgB <= 0:
            avgB = b.body
        if b.body < min_body_frac * avgB:
            continue
        return i
    return None

def _construct_zone_from_block(b: Bar, kind: str, epsilon: float,
                               created_idx: int, tau_bars: int,
                               gamma_decay: float) -> Optional[OBZone]:
    if kind == 'bull':
        zone_high = b.high
        zone_low = b.open
    else:
        zone_high = b.open
        zone_low = b.low
    hi = max(zone_high, zone_low) - epsilon
    lo = min(zone_high, zone_low) + epsilon
    if hi <= lo:
        return None
    mu = (hi + lo) / 2.0
    return OBZone(kind=kind, high=hi, low=lo, created_index=created_idx,
                  expires_index=created_idx + tau_bars, mu=mu, strength=1.0,
                  meta={'gamma_decay': gamma_decay, 'tau_bars': tau_bars, 'epsilon': epsilon})

def detect_order_blocks(
    bars: Sequence[Bar],
    *,
    epsilon_pts: float = 0.1,
    max_impulse_bars: int = 3,
    min_impulse_score: float = 1.0,
    min_impulse_body_mult: float = 1.5,
    max_block_lookback: int = 3,
    min_block_body_frac: float = 0.25,
    gamma_imp: float = 2.0,
    delta_imp: float = 1.5,
    gamma_decay: float = 0.95,
    tau_bars: int = 50,
) -> List[OBZone]:
    n = len(bars)
    if n == 0:
        return []
    ranges = [b.range for b in bars]
    bodies = [b.body for b in bars]
    avg_ranges = _sma(ranges, window=20, default=ranges[0])
    avg_bodies = _sma(bodies, window=20, default=bodies[0])
    zones: List[OBZone] = []
    for i in range(n):
        impulse = _impulse_over_window(
            bars, i, max_impulse_bars,
            avg_ranges=avg_ranges,
            min_body_mult=min_impulse_body_mult,
            min_score=min_impulse_score,
            gamma_imp=gamma_imp,
            delta_imp=delta_imp,
            avg_bodies=avg_bodies
        )
        if impulse is None:
            continue
        impulse_idx, direction, score = impulse
        block_idx = _find_block_candle(
            bars,
            impulse_idx=impulse_idx,
            direction=direction,
            max_lookback=max_block_lookback,
            min_body_frac=min_block_body_frac,
            avg_bodies=avg_bodies,
        )
        if block_idx is None:
            continue
        block_bar = bars[block_idx]
        kind = 'bull' if direction == 'up' else 'bear'
        z = _construct_zone_from_block(block_bar, kind, epsilon_pts, block_idx, tau_bars, gamma_decay)
        if z is None:
            continue
        if z.meta is None:
            z.meta = {}
        z.meta.update({'impulse_index': impulse_idx, 'impulse_score': score, 'impulse_dir': direction})
        zones.append(z)
    return zones

def load_csv(filepath: str) -> List[Bar]:
    bars = []
    with open(filepath, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith('#') or row[0] == 'datetime':
                continue
            dt, o, h, l, c, v, *_ = row
            bars.append(Bar(dt=dt, open=float(o), high=float(h), low=float(l), close=float(c), volume=float(v)))
    return bars

def main():
    bars = load_csv('workspaces/datasets/dataset.csv')
    zones = detect_order_blocks(bars)
    print(f"Found {len(zones)} order block(s)")
    for z in zones:
        print(f"Zone: kind={z.kind}, created_idx={z.created_index}, high={z.high:.2f}, low={z.low:.2f}, mu={z.mu:.2f}, expires={z.expires_index}, impulse_idx={z.meta.get('impulse_index')}, impulse_score={z.meta.get('impulse_score'):.3f}")
        print(f"  Block candle: dt={bars[z.created_index].dt}, open={bars[z.created_index].open}, high={bars[z.created_index].high}, low={bars[z.created_index].low}, close={bars[z.created_index].close}")
        print(f"  Impulse bar: dt={bars[z.meta.get('impulse_index')].dt if z.meta.get('impulse_index') is not None else 'N/A'}")

if __name__ == '__main__':
    main() 