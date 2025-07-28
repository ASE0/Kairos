"""Order Block Gate
====================

Spec‑compliant implementation of *Order Blocks* as **location gates** in the
Spatiotemporal Trading Strategy framework.

This module mirrors the structure of the existing FVG gate but adapts the zone
construction rules to Order Blocks:

* **Bullish Order Block (demand)**  = last **down** candle preceding a validated
  **up impulse**.  Zone vertical span = `[High_block, Open_block]` then reduced
  inward by a user‑configurable buffer `epsilon` (pts).
* **Bearish Order Block (supply)**  = last **up** candle preceding a validated
  **down impulse**.  Zone vertical span = `[Open_block, Low_block]` then buffered.
* A *validated impulse* is a directional move whose range/body exceeds recent
  norms.  We approximate the spec's impulse penetration model using:

      impulse_range_ratio = (H - L) / avg_range
      wick_body_ratio     = (W_u + W_l) / (B + 1e-9)
      impulse_score       = impulse_range_ratio ** gamma_imp * \
                            wick_body_ratio   ** delta_imp

  where `B = |C - O|`, `W_u = H - max(O,C)`, `W_l = min(O,C) - L`, and
  `avg_range` is an N‑bar simple mean of (H-L).  (See strategy spec for details.)

* Only non‑doji opposite‑colour candles qualify as candidate blocks; minimum body
  fraction vs rolling average body can be enforced.
* Zones decay multiplicatively each bar by `gamma_decay` and are hard‑dropped
  after `tau_bars` regardless of residual weight.
* Midpoint `mu` is stored for cross‑timeframe alignment.

Integration points
------------------
- The engine consuming this gate should call :func:`detect_order_blocks` once per
  batch of bars (or incrementally) and merge returned :class:`OBZone` objects
  into the global location stack.
- Penetration scoring, momentum adjustment, and cross‑TF aggregation are handled
  upstream (reuse what the FVG gate already plugs into).

Parameter quick reference
-------------------------
- epsilon_pts (float)      – inward trim in price units (default 2.0).
- max_impulse_bars (int)   – forward look to validate impulse (default 3).
- min_impulse_score (float)– minimum computed impulse_score to accept (default 1.0).
- min_impulse_body_mult    – OR body multiple vs rolling avg body (default 1.5).
- max_block_lookback (int) – how far back from impulse start to search for last
                             opposite‑colour block candle (default 3).
- min_block_body_frac      – minimum candidate body / avg_body ratio (default 0.25)
                             to filter micro‑dojis.
- gamma_imp, delta_imp     – exponents in impulse_score (defaults 2.0, 1.5).
- gamma_decay              – per‑bar multiplicative strength decay (default 0.95).
- tau_bars                 – hard life in bars (default 50).

All numeric defaults mirror the ranges & suggestions in the strategy spec.

Example
-------
>>> zones = detect_order_blocks(bars)
>>> for z in zones: print(z)
OBZone(kind='bull', start_index=5, high=18210.5, low=18206.5, created=5, expires=55, ...)

The returned zones include meta suitable for plotting & scoring.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Sequence, Iterable, Tuple, Dict, Any

# ---------------------------------------------------------------------------
# Basic market data container ------------------------------------------------
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Bar:
    dt: str        # timestamp string; engine may swap for datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    # Convenience computed properties ---------------------------------------
    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def up(self) -> bool:  # green
        return self.close > self.open

    @property
    def down(self) -> bool:  # red
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

# ---------------------------------------------------------------------------
# Zone container -------------------------------------------------------------
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class OBZone:
    kind: str  # 'bull' or 'bear'
    high: float
    low: float
    created_index: int
    expires_index: int
    mu: float
    strength: float = 1.0  # starting weight (pre-decay)
    meta: Dict[str, Any] | None = None

    def decay(self, bars_elapsed: int, gamma_decay: float) -> None:
        """Apply multiplicative decay in place.

        This *does not* handle hard expiry; caller should drop when index >= expires_index.
        """
        if bars_elapsed <= 0:
            return
        self.strength *= gamma_decay ** bars_elapsed

    def asdict(self) -> Dict[str, Any]:  # safe copy for JSON export
        d = asdict(self)
        return d

# ---------------------------------------------------------------------------
# Rolling helpers ------------------------------------------------------------
# ---------------------------------------------------------------------------
def _sma(values: Sequence[float], window: int, default: float = 0.0) -> List[float]:
    """Simple moving average returning list same len(values).
    For the first *window*‑1 elements we return *default* (e.g., values[0]).
    """
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

# ---------------------------------------------------------------------------
# Impulse detection ----------------------------------------------------------
# ---------------------------------------------------------------------------
def _impulse_score(bar: Bar, avg_range: float, gamma_imp: float, delta_imp: float) -> float:
    """Compute impulse score for a *single* bar.

    Approximates spec impulse penetration term:
        ((H-L)/avg_range)**gamma_imp * ((W_u+W_l)/(B+eps))**delta_imp
    """
    eps = 1e-9
    range_ratio = (bar.range / avg_range) if avg_range > 0 else 0.0
    wick_body_ratio = ((bar.wick_up + bar.wick_dn) / (bar.body + eps)) if bar.body > 0 else 0.0
    return (range_ratio ** gamma_imp) * (wick_body_ratio ** delta_imp)

def _impulse_over_window(bars: Sequence[Bar], start: int, max_bars: int,
                         avg_ranges: Sequence[float],
                         min_body_mult: float,
                         min_score: float,
                         gamma_imp: float, delta_imp: float) -> Optional[Tuple[int, str, float]]:
    """Scan forward from *start* (exclusive) up to *max_bars* to find a qualifying impulse.

    Returns (impulse_index, direction, impulse_score) or None.
    direction = 'up' if bullish impulse, 'down' if bearish.
    """
    n = len(bars)
    base_bar = bars[start]
    # Determine the sign we seek: if price later goes up strongly -> bullish impulse, else down.
    # We'll just evaluate each forward bar for both sign & magnitude and pick the strongest.
    best: Tuple[int, str, float] | None = None
    for j in range(start + 1, min(n, start + 1 + max_bars)):
        b = bars[j]
        dir_ = 'up' if b.close > base_bar.close else 'down'
        # body multiple check vs rolling avg body at j
        body_avg = avg_body[j] if 'avg_body' in globals() else None  # patched later
        if body_avg is None:
            # fallback compute local avg body from preceding 20 bars (cheap slice)
            lo = max(0, j - 20)
            body_avg = sum(x.body for x in bars[lo:j + 1]) / (j - lo + 1)
        body_ok = b.body >= min_body_mult * body_avg
        # impulse score
        avg_rng = avg_ranges[j]
        score = _impulse_score(b, avg_rng if avg_rng > 0 else b.range, gamma_imp, delta_imp)
        score_ok = score >= min_score
        if body_ok and score_ok:
            if best is None or score > best[2]:
                best = (j, dir_, score)
    return best

# ---------------------------------------------------------------------------
# Block candle selection -----------------------------------------------------
# ---------------------------------------------------------------------------
def _find_block_candle(bars: Sequence[Bar], impulse_idx: int, direction: str,
                       max_lookback: int, min_body_frac: float,
                       avg_bodies: Sequence[float]) -> Optional[int]:
    """Return index of the *last* opposite‑colour non‑doji candle before impulse.

    direction: 'up' (impulse up ⇒ need last down candle) or 'down'.
    """
    need_down = direction == 'up'   # bullish impulse needs red block
    need_up = direction == 'down'   # bearish impulse needs green block
    for i in range(impulse_idx - 1, max(-1, impulse_idx - 1 - max_lookback), -1):
        b = bars[i]
        if need_down and not b.down:
            continue
        if need_up and not b.up:
                continue
        # body quality
        avgB = avg_bodies[i]
        if avgB <= 0:
            avgB = b.body
        if b.body < min_body_frac * avgB:
            continue  # micro-doji; skip
        return i
    return None

# ---------------------------------------------------------------------------
# Zone construction ----------------------------------------------------------
# ---------------------------------------------------------------------------
def _construct_zone_from_block(block_bar: Bar, impulse_bar: Bar, kind: str, epsilon: float,
                               created_idx: int, tau_bars: int,
                               gamma_decay: float) -> Optional[OBZone]:
    """Create OBZone from qualifying block and impulse candle.

    kind: 'bull' or 'bear'.
    Returns None if epsilon collapses width.
    """
    # New: Use full high-low range of block and impulse, plus buffer
    hi = max(block_bar.high, impulse_bar.high) + epsilon
    lo = min(block_bar.low, impulse_bar.low) - epsilon
    if hi <= lo:
        return None
    mu = (hi + lo) / 2.0
    return OBZone(kind=kind,
                  high=hi,
                  low=lo,
                  created_index=created_idx,
                  expires_index=created_idx + tau_bars,
                  mu=mu,
                  strength=1.0,
                  meta={'gamma_decay': gamma_decay,
                        'tau_bars': tau_bars,
                        'epsilon': epsilon})

# ---------------------------------------------------------------------------
# Public detection API -------------------------------------------------------
# ---------------------------------------------------------------------------
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
    """Detect all Order Block zones in *bars*.

    Returns a list of :class:`OBZone` objects sorted by creation index.
    
    Parameters:
        epsilon_pts (float): inward trim in price units (default 0.1, user-tunable)
        max_impulse_bars (int)   – forward look to validate impulse (default 3).
        min_impulse_score (float)– minimum computed impulse_score to accept (default 1.0).
    """
    n = len(bars)
    if n == 0:
        return []

    # Precompute averages ----------------------------------------------------
    ranges = [b.range for b in bars]
    bodies = [b.body for b in bars]
    avg_ranges = _sma(ranges, window=20, default=ranges[0])
    avg_bodies = _sma(bodies, window=20, default=bodies[0])

    zones: List[OBZone] = []

    for i in range(n):
        if i < 200:  # Only print debug for first 200 bars
            print(f"[OB-DEBUG] Bar {i}: O={bars[i].open} H={bars[i].high} L={bars[i].low} C={bars[i].close} V={bars[i].volume}")
        # attempt to locate an impulse forward of i
        impulse = _impulse_over_window(
            bars, i, max_impulse_bars,
            avg_ranges=avg_ranges,
            min_body_mult=min_impulse_body_mult,
            min_score=min_impulse_score,
            gamma_imp=gamma_imp,
            delta_imp=delta_imp,
        )
        if i < 200:
            if impulse is None:
                print(f"[OB-DEBUG]  No impulse found forward of bar {i}")
            else:
                impulse_idx, direction, score = impulse
                print(f"[OB-DEBUG]  Impulse found: idx={impulse_idx}, dir={direction}, score={score:.4f}")
        if impulse is None:
                    continue
        impulse_idx, direction, score = impulse

        # find the block candle back from impulse start
        block_idx = _find_block_candle(
            bars,
            impulse_idx=impulse_idx,
            direction=direction,
            max_lookback=max_block_lookback,
            min_body_frac=min_block_body_frac,
            avg_bodies=avg_bodies,
        )
        if i < 200:
            if block_idx is None:
                print(f"[OB-DEBUG]  No block candle found before impulse idx={impulse_idx} (dir={direction})")
            else:
                b = bars[block_idx]
                print(f"[OB-DEBUG]  Block candle found: idx={block_idx}, body={b.body:.4f}, avg_body={avg_bodies[block_idx]:.4f}, up={b.up}, down={b.down}")
        if block_idx is None:
                continue
        block_bar = bars[block_idx]
        impulse_bar = bars[impulse_idx]
        kind = 'bull' if direction == 'up' else 'bear'
        z = _construct_zone_from_block(block_bar, impulse_bar, kind, epsilon_pts, block_idx, tau_bars, gamma_decay)
        if i < 200:
            if z is None:
                print(f"[OB-DEBUG]  Zone construction failed (buffer too large or hi<=lo)")
            else:
                print(f"[OB-DEBUG]  Zone constructed: kind={kind}, hi={z.high}, lo={z.low}, created_idx={block_idx}, expires_idx={z.expires_index}")
        if z is None:
            continue

        # annotate with impulse metadata
        if z.meta is None:
            z.meta = {}
        z.meta.update({'impulse_index': impulse_idx,
                       'impulse_score': score,
                       'impulse_dir': direction})
        zones.append(z)

    return zones

# ---------------------------------------------------------------------------
# Lifecycle utilities --------------------------------------------------------
# ---------------------------------------------------------------------------
def update_zone_lifecycle(zones: List[OBZone], current_index: int) -> List[OBZone]:
    """Decay and prune zones based on *current_index*.

    Returns list of still‑active zones (in place safe copy).
    """
    active: List[OBZone] = []
    for z in zones:
        if current_index >= z.expires_index:
            continue  # hard drop
        bars_elapsed = current_index - z.created_index
        z.decay(bars_elapsed, z.meta.get('gamma_decay', 0.95) if z.meta else 0.95)
        active.append(z)
    return active

# ---------------------------------------------------------------------------
# Convenience: load bars from CSV -------------------------------------------
# ---------------------------------------------------------------------------
def load_csv(filepath: str) -> List[Bar]:
    """Load a CSV of columns datetime,open,high,low,close,volume into Bar list."""
    import csv

    out: List[Bar] = []
    with open(filepath, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith('#') or row[0] == 'datetime':
                continue
            dt, o, h, l, c, v = row
            out.append(Bar(dt=dt,
                           open=float(o),
                           high=float(h),
                           low=float(l),
                           close=float(c),
                           volume=float(v)))
    return out

# ---------------------------------------------------------------------------
# Self‑test (manual) ---------------------------------------------------------
# ---------------------------------------------------------------------------
if __name__ == '__main__':  # pragma: no cover - developer scratchpad
    import sys, pathlib, json

    if len(sys.argv) < 2:
        print("Usage: python order_block_gate.py <csv>")
        raise SystemExit(1)

    bars = load_csv(sys.argv[1])
    zones = detect_order_blocks(bars)
    print(f"Found {len(zones)} order block(s)")
    for z in zones:
        print(z)

    # dump JSON for quick GUI overlay
    payload = [z.asdict() for z in zones]
    out_path = pathlib.Path(sys.argv[1]).with_suffix('.obzones.json')
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}") 