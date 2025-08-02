"""
core/feature_quantification.py
=============================
Reusable quantification and scoring features for patterns and strategies
"""
import numpy as np
try:
    from scipy.signal import argrelextrema
except ImportError:
    argrelextrema = None

# SECTION 2: Core Candlestick Quantification

def body_size(open_, close_):
    """Bt = |C-O|"""
    return abs(close_ - open_)

def upper_wick(open_, close_, high):
    """Vectorized upper wick calculation"""
    return high - np.maximum(open_, close_)

def lower_wick(open_, close_, low):
    """Vectorized lower wick calculation"""
    return np.minimum(open_, close_) - low

def wick_ratios(high, low, open_, close_):
    """Ŵu = Wu/(H-L), Ŵl = Wl/(H-L)"""
    rng = high - low
    wu = upper_wick(high, open_, close_)
    wl = lower_wick(low, open_, close_)
    return wu / rng if rng else 0, wl / rng if rng else 0

def doji_ness(Bt, rng, wu_ratio, wl_ratio, sigma_b=0.05, sigma_w=0.10):
    """
    Doji-ness score per spec: Dₜ = exp[−(Bₜ/(Hₜ−Lₜ))²/(2σ_b²)] × exp[−(Ẇᵘₜ−Ẇˡₜ)²/(2σ_w²)]
    
    Args:
        Bt: Real body size |Cₜ - Oₜ|
        rng: Total range Hₜ - Lₜ
        wu_ratio: Normalized upper wick Ẇᵘₜ = Wᵘₜ/(Hₜ-Lₜ)
        wl_ratio: Normalized lower wick Ẇˡₜ = Wˡₜ/(Hₜ-Lₜ)
        sigma_b: Body size sensitivity [0.01, 0.1], default 0.05
        sigma_w: Wick symmetry requirement [0.05, 0.2], default 0.10
    
    Returns:
        Doji-ness score ∈ [0, 1]
    """
    if rng == 0:
        return 0.0
    
    # Clamp parameters to spec ranges
    sigma_b = max(0.01, min(0.1, sigma_b))
    sigma_w = max(0.05, min(0.2, sigma_w))
    
    # Body size term: exp[−(Bₜ/(Hₜ−Lₜ))²/(2σ_b²)]
    body_ratio = Bt / rng
    term1 = np.exp(-(body_ratio ** 2) / (2 * sigma_b ** 2))
    
    # Wick symmetry term: exp[−(Ẇᵘₜ−Ẇˡₜ)²/(2σ_w²)]
    wick_diff = wu_ratio - wl_ratio
    term2 = np.exp(-(wick_diff ** 2) / (2 * sigma_w ** 2))
    
    return term1 * term2

def two_bar_strength(body2, body1, beta_pat=1.0):
    """A2bar = β_pat·(Body₂/Body₁)"""
    return beta_pat * (body2 / body1) if body1 != 0 else 0

# Add more functions for other sections as needed (location, momentum, etc.) 

# SECTION 4: Location Distribution Models

def detect_fvg(highs, lows, closes, min_gap_size=0.001):
    """
    Detect Fair Value Gaps (FVG) in price data
    FVG occurs when there's a gap between bars that hasn't been filled
    
    Returns: List of FVG dictionaries with gap boundaries and properties
    """
    fvgs = []
    
    for i in range(1, len(highs) - 1):
        # Bullish FVG: Current low > Previous high
        if lows[i] > highs[i-1]:
            gap_size = lows[i] - highs[i-1]
            if gap_size >= min_gap_size * highs[i-1]:  # Make min_gap_size relative to price
                strength = gap_size / highs[i-1]  # Relative gap size
                fvgs.append({
                    'type': 'bullish',
                    'zone_type': 'FVG',
                    'start_idx': i-1,
                    'end_idx': i,
                    'gap_low': highs[i-1],
                    'gap_high': lows[i],
                    'gap_size': gap_size,
                    'strength': strength,
                    'zone_min': highs[i-1],
                    'zone_max': lows[i],
                    'zone_direction': 'bullish',
                    'creation_index': i-1,
                    'initial_strength': strength,
                    'gamma': 0.95,  # Decay rate for zone strength
                    'tau_bars': 20,  # Time constant for strength decay
                    'drop_threshold': 0.2  # Threshold for removing weak zones
                })
        
        # Bearish FVG: Current high < Previous low
        if highs[i] < lows[i-1]:
            gap_size = lows[i-1] - highs[i]
            if gap_size >= min_gap_size * lows[i-1]:  # Make min_gap_size relative to price
                strength = gap_size / lows[i-1]  # Relative gap size
                fvgs.append({
                    'type': 'bearish',
                    'zone_type': 'FVG',
                    'start_idx': i-1,
                    'end_idx': i,
                    'gap_low': highs[i],
                    'gap_high': lows[i-1],
                    'gap_size': gap_size,
                    'strength': strength,
                    'zone_min': highs[i],
                    'zone_max': lows[i-1],
                    'zone_direction': 'bearish',
                    'creation_index': i-1,
                    'initial_strength': strength,
                    'gamma': 0.95,  # Decay rate for zone strength
                    'tau_bars': 20,  # Time constant for strength decay
                    'drop_threshold': 0.2  # Threshold for removing weak zones
                })
        
        # Bearish FVG: Current high < Previous low
        elif highs[i] < lows[i-1]:
            gap_size = lows[i-1] - highs[i]
            if gap_size >= min_gap_size * lows[i-1]:  # Make min_gap_size relative to price
                fvgs.append({
                    'type': 'bearish',
                    'zone_type': 'FVG',
                    'start_idx': i-1,
                    'end_idx': i,
                    'gap_low': highs[i],
                    'gap_high': lows[i-1],
                    'gap_size': gap_size,
                    'strength': gap_size / lows[i-1],  # Relative gap size
                    'zone_min': highs[i],
                    'zone_max': lows[i-1],
                    'zone_direction': 'bearish',
                    'creation_index': i-1,
                    'initial_strength': gap_size / lows[i-1],
                    'gamma': 0.95,
                    'tau_bars': 50,
                    'drop_threshold': 0.01,
                    'bar_interval_minutes': 1,
                    'zone_days_valid': 1
                })
    
    return fvgs

def check_fvg_fill(fvgs, current_price, fill_threshold=0.1):
    """
    Check if current price is filling any FVG
    Returns: List of FVGs being filled and fill percentage
    """
    active_fills = []
    
    for fvg in fvgs:
        if fvg['type'] == 'bullish':
            # Price is moving up into bullish FVG
            if current_price >= fvg['gap_low'] and current_price <= fvg['gap_high']:
                fill_pct = (current_price - fvg['gap_low']) / fvg['gap_size']
                active_fills.append({
                    'fvg': fvg,
                    'fill_percentage': fill_pct,
                    'is_filling': fill_pct > fill_threshold
                })
        elif fvg['type'] == 'bearish':
            # Price is moving down into bearish FVG
            if current_price >= fvg['gap_low'] and current_price <= fvg['gap_high']:
                fill_pct = (fvg['gap_high'] - current_price) / fvg['gap_size']
                active_fills.append({
                    'fvg': fvg,
                    'fill_percentage': fill_pct,
                    'is_filling': fill_pct > fill_threshold
                })
    
    return active_fills

def flat_plateau(x, x0, x1):
    """L_base(x) = flat plateau inside gap [x0, x1]"""
    return 1.0 if x0 <= x <= x1 else 0.0

def micro_peaks(x, x_ks, sigma):
    """C_peaks(x) = sum_k exp[-(x-x_k)^2/(2σ^2)]"""
    return sum(np.exp(-((x - xk) ** 2) / (2 * sigma ** 2)) for xk in x_ks)

def dual_layer_location(x, x0, x1, x_ks, beta1=1.0, beta2=1.0, sigma=1.0):
    """L_total = β₁ L_base + β₂ C_peaks"""
    return beta1 * flat_plateau(x, x0, x1) + beta2 * micro_peaks(x, x_ks, sigma)

def skew_location(x, x0, lmbda, base):
    """L_skew = L_base·[1+λ(x-x₀)]"""
    return base * (1 + lmbda * (x - x0))

def skew_normal_kernel(d, xi, omega, alpha):
    """K_i(d_imp) = 2/ω·φ((d-ξ)/ω)·Φ[α((d-ξ)/ω)]"""
    from scipy.stats import norm
    z = (d - xi) / omega
    phi = norm.pdf(z)
    Phi = norm.cdf(alpha * z)
    return 2 / omega * phi * Phi

# PROPER DUAL-LAYER FVG ARCHITECTURE (Section 4.1 Guidelines)

def fvg_base_distribution(x, x0, x1, epsilon=0.002):
    """
    L_base(x) = 0 outside the gap; inside, it is a constant 1/(x₁–x₀)
    A minimum penetration buffer ε (1–5 pts) is required before any score is granted.
    
    Args:
        x: Current price
        x0: Gap low boundary
        x1: Gap high boundary  
        epsilon: Minimum penetration buffer (default 0.2%)
    """
    if x < x0 + epsilon or x > x1 - epsilon:
        return 0.0  # Outside gap or insufficient penetration
    
    # Uniform score inside gap
    return 1.0 / (x1 - x0)

def fvg_micro_comb(x, x0, x1, N=3, sigma=0.1):
    """
    C(x) = Σ exp[−(x − x_k)² / 2σ²], where the peak centers x_k are evenly spaced
    
    Args:
        x: Current price
        x0: Gap low boundary
        x1: Gap high boundary
        N: Number of peaks (default 3: start, middle, end)
        sigma: Peak width (default 0.1)
    """
    # Evenly spaced peak centers
    x_ks = [x0 + i * (x1 - x0) / (N - 1) for i in range(N)]
    
    # Sum of Gaussian peaks
    comb_score = sum(np.exp(-((x - xk) ** 2) / (2 * sigma ** 2)) for xk in x_ks)
    return comb_score

def fvg_dual_layer_score(x, x0, x1, beta1=0.7, beta2=0.3, N=3, sigma=0.1, epsilon=0.002):
    """
    Final static location score: L_total(x) = β₁ L_base(x) + β₂ C(x)
    
    Args:
        x: Current price
        x0: Gap low boundary
        x1: Gap high boundary
        beta1: Base distribution weight (default 0.7)
        beta2: Micro-comb weight (default 0.3)
        N: Number of peaks (default 3)
        sigma: Peak width (default 0.1)
        epsilon: Penetration buffer (default 0.2%)
    """
    base_score = fvg_base_distribution(x, x0, x1, epsilon)
    comb_score = fvg_micro_comb(x, x0, x1, N, sigma)
    
    return beta1 * base_score + beta2 * comb_score

def fvg_directional_skew(x, x0, x1, lmbda=0.0, epsilon=0.002):
    """
    Directional skew: L_skew(x) = L_base(x)·[1 + λ(x − x₀)]
    λ∈[−2, 2] sets the bias toward deeper side of gap
    
    Args:
        x: Current price
        x0: Gap low boundary
        x1: Gap high boundary
        lmbda: Skew parameter (-2 to 2, default 0 = no skew)
        epsilon: Penetration buffer
    """
    base_score = fvg_base_distribution(x, x0, x1, epsilon)
    skew_factor = 1 + lmbda * (x - x0)
    return base_score * skew_factor

def fvg_momentum_expansion(x0, x1, momentum, phi=0.5):
    """
    Momentum-adaptive expansion: x₀ → x₀ − φ·|M|·(x₁-x₀)
    When momentum is high, expand both gap boundaries
    
    Args:
        x0: Original gap low
        x1: Original gap high
        momentum: Current momentum
        phi: Expansion factor (default 0.5)
    """
    expansion = phi * abs(momentum) * (x1 - x0)
    x0_expanded = x0 - expansion
    x1_expanded = x1 + expansion
    return x0_expanded, x1_expanded

def fvg_comprehensive_score(x, fvg, momentum=0.0, params=None):
    """
    Comprehensive FVG scoring with all features from guidelines
    
    Args:
        x: Current price
        fvg: FVG dictionary
        momentum: Current momentum
        params: FVG parameters dictionary
    """
    # Default parameters from guidelines
    if params is None:
        params = {
            'beta1': 0.7,      # Base distribution weight
            'beta2': 0.3,      # Micro-comb weight
            'N': 3,            # Number of peaks
            'sigma': 0.1,      # Peak width
            'epsilon': 0.002,  # Penetration buffer (0.2%)
            'lmbda': 0.0,      # Skew parameter
            'phi': 0.5,        # Momentum expansion factor
            'kappa_m': 0.5     # Momentum boost factor
        }
    
    x0, x1 = fvg['gap_low'], fvg['gap_high']
    
    # Apply momentum expansion
    x0_exp, x1_exp = fvg_momentum_expansion(x0, x1, momentum, params['phi'])
    
    # Calculate dual-layer score
    dual_score = fvg_dual_layer_score(
        x, x0_exp, x1_exp, 
        params['beta1'], params['beta2'], 
        params['N'], params['sigma'], params['epsilon']
    )
    
    # Apply directional skew
    if params['lmbda'] != 0:
        skew_score = fvg_directional_skew(
            x, x0_exp, x1_exp, 
            params['lmbda'], params['epsilon']
        )
        # Combine skew with dual-layer
        final_score = 0.8 * dual_score + 0.2 * skew_score
    else:
        final_score = dual_score
    
    # Apply momentum boost
    momentum_boost = 1 + params['kappa_m'] * abs(momentum)
    final_score *= momentum_boost
    
    return min(1.0, final_score)  # Cap at 1.0

def fvg_location_score_advanced(current_price, fvgs, momentum=0.0, lookback=50, params=None):
    """
    Advanced FVG location score using comprehensive dual-layer architecture
    
    Args:
        current_price: Current price
        fvgs: List of FVG dictionaries
        momentum: Current momentum
        lookback: Number of recent FVGs to consider
        params: FVG parameters dictionary
    """
    if not fvgs:
        return 0.5  # Neutral score if no FVGs
    
    # Get recent FVGs
    recent_fvgs = [fvg for fvg in fvgs if fvg['end_idx'] >= len(fvgs) - lookback]
    
    if not recent_fvgs:
        return 0.5
    
    scores = []
    for fvg in recent_fvgs:
        # Calculate comprehensive FVG score
        fvg_score = fvg_comprehensive_score(current_price, fvg, momentum, params)
        scores.append(fvg_score)
    
    return np.mean(scores) if scores else 0.5

# FVG Parameter Ranges (Section 12 Guidelines)
FVG_PARAMETER_RANGES = {
    'epsilon': (0.001, 0.005),    # Penetration buffer: 0.1% to 0.5%
    'N': (2, 5),                  # Number of peaks: 2 to 5
    'sigma': (0.05, 0.2),         # Peak width: 0.05 to 0.2
    'lmbda': (-2.0, 2.0),         # Skew parameter: -2 to 2
    'beta1': (0.6, 0.8),          # Base weight: 0.6 to 0.8
    'beta2': (0.2, 0.4),          # Comb weight: 0.2 to 0.4
    'phi': (0.3, 0.7),            # Expansion factor: 0.3 to 0.7
    'kappa_m': (0.3, 0.7)         # Momentum boost: 0.3 to 0.7
}

FVG_DEFAULT_PARAMS = {
    'epsilon': 0.002,  # 0.2% penetration buffer
    'N': 3,           # 3 peaks (start, middle, end)
    'sigma': 0.1,     # Peak width
    'lmbda': 0.0,     # No skew
    'beta1': 0.7,     # 70% base weight
    'beta2': 0.3,     # 30% comb weight
    'phi': 0.5,       # Expansion factor
    'kappa_m': 0.5    # Momentum boost
}

# REMOVE: detect_support_resistance and rolling_support_resistance and all S/R logic

def location_context_score(current_price, supports, resistances, tolerance=0.01):
    """
    Calculate location context score based on proximity to S/R levels
    """
    if not supports and not resistances:
        return 0.5
    
    scores = []
    
    # Check proximity to supports
    for support in supports:
        distance = abs(current_price - support) / support
        if distance <= tolerance:
            # Closer to support = higher score for bullish setups
            proximity_score = 1 - (distance / tolerance)
            scores.append(proximity_score)
    
    # Check proximity to resistances
    for resistance in resistances:
        distance = abs(current_price - resistance) / resistance
        if distance <= tolerance:
            # Closer to resistance = higher score for bearish setups
            proximity_score = 1 - (distance / tolerance)
            scores.append(proximity_score)
    
    return np.mean(scores) if scores else 0.5

# SECTION 5: Momentum-Adaptive Location

def mean_recent_returns(returns):
    """M(t,y) = mean_recent_returns"""
    return np.mean(returns)

def gap_expand(x0, x1, phi, M):
    """x₀,₁_adj = x₀,₁ ± φ·|M|·(x₁-x₀)"""
    delta = phi * abs(M) * (x1 - x0)
    return x0 - delta, x1 + delta

def momentum_boost(L_total, kappa_m, M):
    """L_mom = L_total·(1+κ_m|M|)"""
    return L_total * (1 + kappa_m * abs(M))

# SECTION 7: Volatility Integration

def realized_vol(returns):
    """σ_t = sqrt[(1/N)Σ(r_i-r̄)^2]"""
    return np.std(returns)

def atr(highs, lows, closes, n=14):
    """ATR_t = (1/n)Σ TR_i"""
    trs = [max(h-l, abs(h-c), abs(l-c)) for h, l, c in zip(highs, lows, closes)]
    return np.mean(trs[-n:]) if len(trs) >= n else np.mean(trs)

def composite_vol(sigma_t, atr_t, w1=0.5, w2=0.5):
    """V(x,y) = w₁ σ_t + w₂ ATR_t"""
    return w1 * sigma_t + w2 * atr_t

def garch_vol(prev_sigma2, prev_eps2, omega, alpha, beta):
    """σ_t² = ω + α ε²_{t-1} + β σ²_{t-1}"""
    return omega + alpha * prev_eps2 + beta * prev_sigma2

def z_vol(sigma_t, mu_sigma, sigma_sigma):
    """Z_vol = (σ_t-μ_σ)/σ_σ"""
    return (sigma_t - mu_sigma) / sigma_sigma if sigma_sigma != 0 else 0

# SECTION 6: Bayesian State Tracking (simplified)
def bayesian_update(prior, likelihood):
    """Posterior ∝ Likelihood · Prior (normalized)"""
    posterior = prior * likelihood
    return posterior / np.sum(posterior) if np.sum(posterior) != 0 else posterior

# SECTION 11: Execution Logic & Signal Synthesis
def gate_list(*gates):
    """Return True if all gates pass (all are True)"""
    return all(gates)

def alignment(mu_ys, mu_bar, sigma_ys):
    """C_align = exp[-(1/|T|)Σ_y ((μ_y-μ̄)/σ_y)^2]"""
    T = len(mu_ys)
    if T == 0:
        return 0
    return np.exp(-np.mean([((mu_y - mu_bar) / sigma_y) ** 2 if sigma_y != 0 else 0 for mu_y, sigma_y in zip(mu_ys, sigma_ys)]))

# SECTION 12: Position Sizing & Risk
def kelly_fraction(p, b, q, sigma_t):
    """f* = (p·b - q)/b · 1/σ_t"""
    return ((p * b - q) / b) * (1 / sigma_t) if b != 0 and sigma_t != 0 else 0

def stop_loss(k_stop, atr_t, h):
    """Stop = k_stop·ATR_t·√h"""
    return k_stop * atr_t * np.sqrt(h)

def fat_tail_scale(ES, tail_prob, thresh):
    """Scale pos if tail prob > thresh via GPD ES"""
    return ES if tail_prob > thresh else 1

# SECTION 13: Master Equation Bundle
def master_score(A_pattern, K_i, L_mom, C_i, w_i, beta_v, V):
    """St,i = A_pattern · K_i(d_imp) · L_mom(x) · C_i
    S(t,x,y) = Σ w_i St,i + β_v V"""
    St_i = A_pattern * K_i * L_mom * C_i
    return np.sum(w_i * St_i) + beta_v * V

def tf_vote(S, alpha_y):
    """S_net = Σ α_y S"""
    return np.sum(alpha_y * S)

def vol_damp(S_net, kappa, V):
    """S_adj = S_net / (1+κ·V)"""
    return S_net / (1 + kappa * V)

def exec_score(S_adj, C_align, MMRS_enhanced, tau):
    """S_exec = S_adj·C_align·1_{MMRS_enhanced>τ}"""
    return S_adj * C_align if MMRS_enhanced > tau else 0

def should_execute(S_exec, theta_exec, gates):
    """Execute = 1_{S_exec>θ_exec} ∧ all gates above"""
    return S_exec > theta_exec and all(gates) 

# SECTION 3: Z-Space Matrix Architecture

class ZSpaceMatrix:
    """
    Z-Space Matrix: S(t,x,y) where t=time, x=price, y=timeframe
    Implements the core strength equation and temporal updates
    """
    
    def __init__(self, timeframes=None, price_resolution=0.001):
        self.timeframes = timeframes or ['1m', '5m', '15m', '1h', '4h', '1d']
        self.price_resolution = price_resolution
        self.matrix = {}  # (t, x, y) -> S value
        self.gamma = 0.95  # Memory decay factor
        self.eta = 0.1     # Opposite pattern penalty
        
    def update_strength(self, t, x, y, A_pattern, K_i, L_mom, C_i, w_i, beta_v, V):
        """
        Core strength equation: S(t,x,y) = Σ w_i A_i(x,y) L_i(x,y) + β_v V(x,y)
        """
        # Calculate pattern contribution
        pattern_contribution = sum(w * A * L * C for w, A, L, C in zip(w_i, A_pattern, K_i, L_mom, C_i))
        
        # Add volatility component
        total_strength = pattern_contribution + beta_v * V
        
        # Store in matrix
        self.matrix[(t, x, y)] = total_strength
        return total_strength
    
    def temporal_update(self, t, x, y, A_new, L_new, C_y, q_opposite=0):
        """
        Temporal update: S(t+Δt) = γ S(t)(1-η q_opposite) + (1-γ)[A_new L + C(y)1_series]
        """
        old_strength = self.matrix.get((t, x, y), 0)
        
        # Memory decay with opposite pattern penalty
        memory_component = self.gamma * old_strength * (1 - self.eta * q_opposite)
        
        # New information component
        new_component = (1 - self.gamma) * (A_new * L_new + C_y)
        
        new_strength = memory_component + new_component
        self.matrix[(t, x, y)] = new_strength
        return new_strength
    
    def cross_timeframe_sum(self, t, x_current, alpha_y):
        """
        Cross-timeframe roll-up: S_net(t) = Σ α_y S(t, x_current, y)
        """
        net_strength = 0
        for y, alpha in zip(self.timeframes, alpha_y):
            strength = self.matrix.get((t, x_current, y), 0)
            net_strength += alpha * strength
        return net_strength

# SECTION 6: Bayesian State Tracking (9-state system)

class BayesianStateTracker:
    """
    Bayesian state tracking with 9 states:
    States = {Bull, Bear, Neutral} × {Explosive, Normal, Exhausted}
    """
    
    def __init__(self):
        # 9 states: (direction, velocity)
        self.states = [
            ('Bull', 'Explosive'), ('Bull', 'Normal'), ('Bull', 'Exhausted'),
            ('Bear', 'Explosive'), ('Bear', 'Normal'), ('Bear', 'Exhausted'),
            ('Neutral', 'Explosive'), ('Neutral', 'Normal'), ('Neutral', 'Exhausted')
        ]
        
        # Initialize uniform prior
        self.prior = np.ones(9) / 9
        self.posterior = self.prior.copy()
        
        # State transition matrix (simplified)
        self.transition_matrix = self._create_transition_matrix()
        
    def _create_transition_matrix(self):
        """Create state transition matrix"""
        # Simplified transition matrix - states tend to persist
        matrix = np.eye(9) * 0.8  # 80% chance to stay in same state
        
        # Add some transitions between related states
        for i in range(9):
            for j in range(9):
                if i != j:
                    # Small probability of transitioning to other states
                    matrix[i, j] = 0.02
        
        # Normalize rows
        matrix = matrix / matrix.sum(axis=1, keepdims=True)
        return matrix
    
    def update_posterior(self, D_t, V_t):
        """
        Update posterior: P(H_t^d,v | D_1:t) = P(D_t | H_t^d,v, V_t) P(H_t-1^d,v) / normalization
        """
        # Calculate likelihood for each state
        likelihood = np.zeros(9)
        
        for i, (direction, velocity) in enumerate(self.states):
            likelihood[i] = self._calculate_likelihood(D_t, V_t, direction, velocity)
        
        # Update posterior
        self.posterior = likelihood * self.prior
        self.posterior = self.posterior / np.sum(self.posterior)  # Normalize
        
        # Update prior for next iteration
        self.prior = self.transition_matrix @ self.posterior
        
        return self.posterior
    
    def _calculate_likelihood(self, D_t, V_t, direction, velocity):
        """
        Calculate likelihood P(D_t | H_t^d,v, V_t) using composite likelihood
        """
        # Composite likelihood: P(D_t | H, V) = ∏ P(e | H, V)^w_e
        
        # Evidence sources with weights
        evidence_weights = {
            'strength': 0.4,
            'pattern': 0.3,
            'location': 0.2,
            'volume': 0.1,
            'momentum': 0.15
        }
        
        # Normalize weights
        total_weight = sum(evidence_weights.values())
        evidence_weights = {k: v/total_weight for k, v in evidence_weights.items()}
        
        # Calculate individual evidence likelihoods (simplified)
        strength_likelihood = self._strength_likelihood(D_t, direction, velocity)
        pattern_likelihood = self._pattern_likelihood(D_t, direction, velocity)
        location_likelihood = self._location_likelihood(D_t, direction, velocity)
        volume_likelihood = self._volume_likelihood(D_t, direction, velocity)
        momentum_likelihood = self._momentum_likelihood(D_t, direction, velocity)
        
        # Composite likelihood
        composite_likelihood = (
            strength_likelihood ** evidence_weights['strength'] *
            pattern_likelihood ** evidence_weights['pattern'] *
            location_likelihood ** evidence_weights['location'] *
            volume_likelihood ** evidence_weights['volume'] *
            momentum_likelihood ** evidence_weights['momentum']
        )
        
        return composite_likelihood
    
    def _strength_likelihood(self, D_t, direction, velocity):
        """Likelihood based on pattern strength"""
        # Simplified - higher strength favors explosive states
        strength = D_t.get('strength', 0.5)
        if velocity == 'Explosive':
            return strength
        elif velocity == 'Normal':
            return 0.5
        else:  # Exhausted
            return 1 - strength
    
    def _pattern_likelihood(self, D_t, direction, velocity):
        """Likelihood based on pattern type"""
        pattern_type = D_t.get('pattern_type', 'neutral')
        
        if direction == 'Bull' and pattern_type in ['bullish', 'engulfing_bullish']:
            return 0.8
        elif direction == 'Bear' and pattern_type in ['bearish', 'engulfing_bearish']:
            return 0.8
        elif direction == 'Neutral' and pattern_type in ['doji', 'neutral']:
            return 0.8
        else:
            return 0.2
    
    def _location_likelihood(self, D_t, direction, velocity):
        """Likelihood based on location context"""
        location_score = D_t.get('location_score', 0.5)
        return location_score
    
    def _volume_likelihood(self, D_t, direction, velocity):
        """Likelihood based on volume"""
        volume_ratio = D_t.get('volume_ratio', 1.0)
        
        if velocity == 'Explosive':
            return min(1.0, volume_ratio)
        elif velocity == 'Normal':
            return 0.5
        else:  # Exhausted
            return max(0.1, 1 - volume_ratio)
    
    def _momentum_likelihood(self, D_t, direction, velocity):
        """Likelihood based on momentum"""
        momentum = D_t.get('momentum', 0.0)
        
        if direction == 'Bull' and momentum > 0:
            return 0.8
        elif direction == 'Bear' and momentum < 0:
            return 0.8
        elif direction == 'Neutral' and abs(momentum) < 0.01:
            return 0.8
        else:
            return 0.2
    
    def get_state_probability(self, direction, velocity):
        """Get probability of specific state"""
        try:
            state_index = self.states.index((direction, velocity))
            return self.posterior[state_index]
        except ValueError:
            return 0.0
    
    def get_dominant_state(self):
        """Get the most probable state"""
        max_index = np.argmax(self.posterior)
        return self.states[max_index]

# SECTION 8: Pattern Recognition & Dynamic Adjustments

def pattern_confidence(candle_data, template, kappa=2.0, tau=0.7):
    """
    Pattern confidence: q_T = σ[κ(Corr_T - τ)]
    Corr_T = y_T · m / (||y_T|| ||m||)
    """
    # Normalize candle data and template
    y_T = np.array(candle_data)
    m = np.array(template)
    
    # Calculate correlation
    dot_product = np.dot(y_T, m)
    norm_y = np.linalg.norm(y_T)
    norm_m = np.linalg.norm(m)
    
    if norm_y == 0 or norm_m == 0:
        return 0.0
    
    corr_T = dot_product / (norm_y * norm_m)
    
    # Apply sigmoid function
    confidence = 1 / (1 + np.exp(-kappa * (corr_T - tau)))
    return confidence

def adjust_aggregated_strength(S_agg, q_pattern, pattern_direction='bullish'):
    """
    Adjust aggregated strength based on pattern agreement
    S_adj = S_agg (1 - η q_bear) if pattern disagrees
    """
    eta = 0.3  # Pattern disagreement penalty
    
    if pattern_direction == 'bullish':
        # If pattern is bullish, reduce strength if q_pattern is low
        S_adj = S_agg * (1 - eta * (1 - q_pattern))
    else:
        # If pattern is bearish, reduce strength if q_pattern is low
        S_adj = S_agg * (1 - eta * (1 - q_pattern))
    
    return max(0, S_adj)

def detect_series_pattern(data, min_bars=3, similarity_threshold=0.8):
    """
    Detect series of similar bars and add cluster bonus C(y)
    """
    if len(data) < min_bars:
        return 0.0
    
    # Calculate similarity between consecutive bars
    similarities = []
    for i in range(1, len(data)):
        # Simplified similarity metric
        bar1 = data[i-1]
        bar2 = data[i]
        
        # Compare body size, wick ratios, etc.
        body_similarity = 1 - abs(bar1['body_size'] - bar2['body_size']) / max(bar1['body_size'], bar2['body_size'])
        wick_similarity = 1 - abs(bar1['wick_ratio'] - bar2['wick_ratio'])
        
        similarity = (body_similarity + wick_similarity) / 2
        similarities.append(similarity)
    
    # Check if we have a series of similar bars
    if len(similarities) >= min_bars - 1:
        avg_similarity = np.mean(similarities)
        if avg_similarity >= similarity_threshold:
            # Series bonus increases with length and similarity
            series_bonus = min(1.0, avg_similarity * len(similarities) / 5)
            return series_bonus
    
    return 0.0

# SECTION 9: Market-Maker Reversion Models

def market_maker_reversion_score(low, resistance_inf, sigma_r=0.02, sigma_t=0.01, epsilon=0.001):
    """
    Market-maker reversion score: M_t = exp[-(L_t - R_t^inf)^2/(2σ_r^2)] exp[-ε^2/(2σ_t^2)]
    """
    # Distance from support
    distance_from_support = low - resistance_inf
    
    # First exponential term
    term1 = np.exp(-(distance_from_support ** 2) / (2 * sigma_r ** 2))
    
    # Second exponential term (temporal symmetry)
    term2 = np.exp(-(epsilon ** 2) / (2 * sigma_t ** 2))
    
    return term1 * term2

def temporal_symmetry(rise_time, fall_time):
    """
    Temporal symmetry: R_MM = 1 - |Δt_rise - Δt_fall| / (Δt_rise + Δt_fall)
    """
    if rise_time + fall_time == 0:
        return 0.0
    
    symmetry = 1 - abs(rise_time - fall_time) / (rise_time + fall_time)
    return max(0, symmetry)

# SECTION 10: Imbalance Memory System

class ImbalanceMemorySystem:
    """
    Imbalance memory system for storing and recalling one-sided moves
    """
    
    def __init__(self, gamma_mem=0.1, sigma_rev=0.02):
        self.imbalances = []  # List of {direction, magnitude, range, timestamp}
        self.gamma_mem = gamma_mem  # Memory decay rate
        self.sigma_rev = sigma_rev  # Reversion expectation width
        
    def store_imbalance(self, direction, magnitude, price_range, timestamp):
        """Store a new imbalance"""
        imbalance = {
            'direction': direction,
            'magnitude': magnitude,
            'range': price_range,
            'timestamp': timestamp,
            'weight': 1.0  # Initial weight
        }
        self.imbalances.append(imbalance)
        
        # Keep only recent imbalances (last 100)
        if len(self.imbalances) > 100:
            self.imbalances = self.imbalances[-100:]
    
    def get_reversion_expectation(self, price, current_time):
        """
        Reversion expectation: R_imbalance(p,t) = Σ w_i exp[-(p-p_i)^2/(2σ_rev^2)] e^(-γ_mem(t-t_i))
        """
        expectation = 0.0
        
        for imbalance in self.imbalances:
            # Calculate time decay
            time_diff = current_time - imbalance['timestamp']
            time_decay = np.exp(-self.gamma_mem * time_diff.total_seconds() / 3600)  # Hours
            
            # Calculate spatial decay
            price_diff = abs(price - imbalance['range'])
            spatial_decay = np.exp(-(price_diff ** 2) / (2 * self.sigma_rev ** 2))
            
            # Weighted contribution
            contribution = imbalance['weight'] * spatial_decay * time_decay
            expectation += contribution
        
        return expectation
    
    def update_weights(self, current_time):
        """Update weights based on time decay"""
        for imbalance in self.imbalances:
            time_diff = current_time - imbalance['timestamp']
            imbalance['weight'] = np.exp(-self.gamma_mem * time_diff.total_seconds() / 3600)
        
        # Remove very old imbalances
        self.imbalances = [imb for imb in self.imbalances if imb['weight'] > 0.01]

# SECTION 11: Enhanced Execution Logic

def enhanced_execution_score(S_adj, C_align, MMRS_enhanced, tau=0.5):
    """
    Enhanced execution score: S_exec(t) = S_adj(t) C_align(t) 1_{MMRS_enhanced(t)>τ}
    """
    if MMRS_enhanced <= tau:
        return 0.0
    
    return S_adj * C_align

def market_maker_reversion_enhanced(low, resistance_inf, imbalance_expectation, 
                                  sigma_r=0.02, sigma_t=0.01, epsilon=0.001):
    """
    Enhanced market-maker reversion score with imbalance memory
    """
    # Base MMRS
    base_mmrs = market_maker_reversion_score(low, resistance_inf, sigma_r, sigma_t, epsilon)
    
    # Enhance with imbalance expectation
    enhanced_mmrs = base_mmrs * (1 + imbalance_expectation)
    
    return min(1.0, enhanced_mmrs)

# SECTION 13: Master Equations Integration

def complete_master_equation(A_pattern, K_i, L_mom, C_i, w_i, beta_v, V, 
                           C_align=1.0, MMRS_enhanced=0.5, tau=0.5):
    """
    Complete master equation integrating all components:
    S_t,i = A_pattern K_i(d_t,i^imp) [β₁ L_skew + β₂ C](1 + κ_m|M|) C_i
    S_exec = S_adj C_align 1_{MMRS_enhanced>τ}
    """
    # Per-zone strength
    zone_strength = master_score(A_pattern, K_i, L_mom, C_i, w_i, beta_v, V)
    
    # Volatility adjustment
    S_adj = vol_damp(zone_strength, kappa=0.1, V=V)
    
    # Final execution score
    S_exec = enhanced_execution_score(S_adj, C_align, MMRS_enhanced, tau)
    
    return S_exec

def complete_location_score(x, y, L_z, gamma_z, delta_y, kappa_m, M):
    """
    Complete location score: L_final(x,y) = [Σ γ_z L_z(x)](1 + δ_y)(1 + κ_m|M|)
    """
    # Stacked location score
    L_stacked = sum(gamma * L for gamma, L in zip(gamma_z, L_z))
    
    # Timeframe adjustment
    L_adjusted = L_stacked * (1 + delta_y)
    
    # Momentum boost
    L_final = L_adjusted * (1 + kappa_m * abs(M))
    
    return L_final

# SECTION 7: Enhanced Volatility Integration

def garch_volatility_forecast(returns, omega=0.0001, alpha=0.1, beta=0.8):
    """
    GARCH(1,1) volatility forecast: σ_t² = ω + α ε_{t-1}² + β σ_{t-1}²
    """
    if len(returns) < 2:
        return np.std(returns) if len(returns) > 0 else 0.01
    
    # Initialize
    sigma2 = np.var(returns)
    forecast = omega + alpha * returns[-1]**2 + beta * sigma2
    
    return np.sqrt(forecast)

def volatility_z_score(sigma_t, mu_sigma, sigma_sigma):
    """
    Volatility z-score: Z_vol = (σ_t - μ_σ) / σ_σ
    """
    if sigma_sigma == 0:
        return 0.0
    return (sigma_t - mu_sigma) / sigma_sigma

def volatility_entry_veto(Z_vol, epsilon_vol=2.5):
    """
    Volatility-based entry veto: Reject trade if |Z_vol| > ε_vol
    """
    return abs(Z_vol) <= epsilon_vol

# SECTION 2: Enhanced Candlestick Quantification

def penetration_depth(open_, close_, high, low, zone_high, zone_low):
    """
    Basic penetration depth: d_t,i = (min(max{O_t,C_t,H_t,L_t}, H_i) - L_i) / (H_i - L_i)
    """
    if zone_high <= zone_low:
        return 0.0
    
    # Get maximum price reached
    max_price = max(open_, close_, high, low)
    
    # Calculate penetration
    if max_price < zone_low:
        return 0.0
    elif max_price > zone_high:
        return 1.0
    else:
        return (max_price - zone_low) / (zone_high - zone_low)

def impulse_weighted_depth(open_, close_, high, low, zone_high, zone_low, 
                         avg_range, gamma=1.0, delta=0.5, epsilon=0.001):
    """
    Impulse-weighted depth: d_t,i^imp = (H_t - L_t / R̄)^γ (W_t^u + W_t^ℓ / B_t + ε)^δ d_t,i
    """
    # Basic penetration depth
    basic_depth = penetration_depth(open_, close_, high, low, zone_high, zone_low)
    
    if basic_depth == 0:
        return 0.0
    
    # Range boost
    current_range = high - low
    range_boost = (current_range / avg_range) ** gamma if avg_range > 0 else 1.0
    
    # Wick boost
    body_size = abs(close_ - open_)
    upper_wick = high - max(open_, close_)
    lower_wick = min(open_, close_) - low
    wick_boost = ((upper_wick + lower_wick) / (body_size + epsilon)) ** delta
    
    return basic_depth * range_boost * wick_boost

def two_bar_reversal_patterns(bar1, bar2, pattern_type='bullish_engulfing'):
    """Detect two-bar reversal patterns"""
    # Implementation for two-bar patterns
    return True  # Placeholder


# Additional functions needed by strategy_builders.py

def compute_penetration_depth(open_, close_, high, low, zone_high, zone_low):
    """Compute penetration depth into a zone"""
    if zone_high <= zone_low:
        return 0.0
    
    # Calculate how much the candle penetrates the zone
    candle_low = min(open_, close_)
    candle_high = max(open_, close_)
    
    # Check if candle overlaps with zone
    if candle_high < zone_low or candle_low > zone_high:
        return 0.0
    
    # Calculate penetration
    penetration_low = max(candle_low, zone_low)
    penetration_high = min(candle_high, zone_high)
    penetration = penetration_high - penetration_low
    
    # Normalize by zone size
    zone_size = zone_high - zone_low
    return penetration / zone_size if zone_size > 0 else 0.0


def compute_impulse_penetration(open_, close_, high, low, zone_high, zone_low, 
                               avg_range, gamma=1.0, delta=0.5, epsilon=0.001):
    """Compute impulse-weighted penetration depth"""
    base_penetration = compute_penetration_depth(open_, close_, high, low, zone_high, zone_low)
    
    if base_penetration == 0:
        return 0.0
    
    # Calculate impulse factor
    body_size = abs(close_ - open_)
    total_range = high - low
    
    if total_range == 0:
        return base_penetration
    
    # Impulse ratio: body size relative to total range
    impulse_ratio = body_size / total_range
    
    # Apply impulse weighting
    impulse_factor = (impulse_ratio ** gamma) * (1 + delta * impulse_ratio)
    
    return base_penetration * impulse_factor


def per_zone_strength(A_pattern, d_imp, kernel_params, C_i, kappa_m, M_t_y):
    """Calculate per-zone strength score"""
    xi, omega, alpha = kernel_params
    
    # Kernel function
    kernel = skew_normal_kernel(d_imp, xi, omega, alpha)
    
    # Momentum adjustment
    momentum_factor = 1 + kappa_m * M_t_y
    
    # Final strength
    S_ti = A_pattern * kernel * C_i * momentum_factor
    
    return max(0.0, min(1.0, S_ti))  # Clamp to [0, 1]


def flat_fvg_base(current_price, x0, x1):
    """Flat FVG base distribution"""
    return fvg_base_distribution(current_price, x0, x1)


def micro_comb_peaks(current_price, x0, x1, N, sigma):
    """Micro combination peaks"""
    return fvg_micro_comb(current_price, x0, x1, N, sigma)


def combined_location_strength(current_price, x0, x1, beta1, beta2, N, sigma):
    """Combined location strength"""
    L_base = flat_fvg_base(current_price, x0, x1)
    C_peaks = micro_comb_peaks(current_price, x0, x1, N, sigma)
    return beta1 * L_base + beta2 * C_peaks


def directional_skew(current_price, x0, L_base, lambda_skew):
    """Directional skew adjustment"""
    return L_base * (1 + lambda_skew * (current_price - x0))


def z_space_aggregate(S_ti_list, w_list, beta_v, V_xy):
    """Z-space aggregation of zone strengths"""
    if not S_ti_list or not w_list:
        return 0.0
    
    # Weighted sum
    weighted_sum = sum(S_ti * w for S_ti, w in zip(S_ti_list, w_list))
    total_weight = sum(w_list)
    
    if total_weight == 0:
        return 0.0
    
    # Volatility adjustment
    S_net = weighted_sum / total_weight
    S_adj = S_net / (1 + beta_v * V_xy)
    
    return max(0.0, min(1.0, S_adj))


def momentum_weighted_location(L_total, kappa_m, M_t_y):
    """Momentum-weighted location score"""
    momentum_factor = 1 + kappa_m * M_t_y
    return L_total * momentum_factor


def market_maker_reversion_enhanced(low, resistance_inf, imbalance_expectation, 
                                  sigma_r=0.02, sigma_t=0.01, epsilon=0.001):
    """Enhanced market maker reversion score"""
    # Base MMRS
    base_mmrs = market_maker_reversion_score(low, resistance_inf, sigma_r, sigma_t, epsilon)
    
    # Add imbalance expectation
    enhanced_mmrs = base_mmrs + imbalance_expectation
    
    return max(0.0, min(1.0, enhanced_mmrs))


def enhanced_execution_score(S_adj, C_align, MMRS_enhanced, tau=0.5):
    """Enhanced execution score"""
    # Combine all components
    execution_score = S_adj * C_align * MMRS_enhanced
    
    # Apply threshold
    return execution_score > tau


# SECTION: Technical Indicators for Advanced Multi-Timeframe Strategies
# =====================================================================

def calculate_ema(data, period=21):
    """
    Calculate Exponential Moving Average
    
    Args:
        data: Price series (close prices) - can be pandas Series or numpy array
        period: EMA period (default 21)
    
    Returns:
        EMA values - pandas Series if input is Series, numpy array otherwise
    """
    import pandas as pd
    import numpy as np
    
    if isinstance(data, pd.Series):
        return data.ewm(span=period, adjust=False).mean()
    else:
        # Convert to pandas Series for calculation, then back to numpy
        data_series = pd.Series(data)
        ema_series = data_series.ewm(span=period, adjust=False).mean()
        return ema_series.values


def calculate_keltner_channels(high, low, close, ema_period=21, atr_period=14, multiplier=1.0):
    """
    Calculate Keltner Channels
    
    Args:
        high: High prices
        low: Low prices  
        close: Close prices
        ema_period: EMA period (default 21)
        atr_period: ATR period (default 14)
        multiplier: ATR multiplier (default 1.0)
    
    Returns:
        Dictionary with 'upper', 'middle', 'lower' channels
    """
    import pandas as pd
    import numpy as np
    
    # Calculate EMA (middle line)
    ema = calculate_ema(close, ema_period)
    
    # Calculate ATR using existing function
    atr_values = atr(high, low, close, atr_period)
    
    # Calculate bands
    if isinstance(close, pd.Series):
        upper = ema + (multiplier * atr_values)
        lower = ema - (multiplier * atr_values)
    else:
        upper = ema + (multiplier * atr_values)
        lower = ema - (multiplier * atr_values)
    
    return {
        'upper': upper,
        'middle': ema,
        'lower': lower
    }


def calculate_atr_ratio(atr1, atr2):
    """
    Calculate ATR ratio between two timeframes
    
    Args:
        atr1: ATR from first timeframe
        atr2: ATR from second timeframe
    
    Returns:
        ATR ratio (atr1/atr2)
    """
    import numpy as np
    
    # Handle division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = atr1 / atr2
        ratio = np.where(atr2 == 0, 0, ratio)
    
    return ratio


def detect_market_regime(atr_ratio, mean_reversion_threshold=1.35, expansion_threshold=1.9):
    """
    Detect market regime based on ATR ratios
    
    Args:
        atr_ratio: ATR ratio values
        mean_reversion_threshold: Threshold for mean reversion (default 1.35)
        expansion_threshold: Threshold for expansion (default 1.9)
    
    Returns:
        Regime labels: 0=mean_reverting, 1=neutral, 2=expanding
    """
    import numpy as np
    
    regime = np.ones_like(atr_ratio)  # Default to neutral (1)
    regime[atr_ratio < mean_reversion_threshold] = 0  # mean_reverting
    regime[atr_ratio > expansion_threshold] = 2  # expanding
    
    return regime


def check_keltner_ema_alignment(keltner_bands, ema, tolerance=0.001):
    """
    Check if Keltner bands align with EMA
    
    Args:
        keltner_bands: Dictionary with 'upper', 'middle', 'lower' bands
        ema: EMA values to check alignment with
        tolerance: Alignment tolerance (default 0.001)
    
    Returns:
        Boolean array indicating alignment
    """
    import numpy as np
    
    upper_aligned = np.abs(keltner_bands['upper'] - ema) <= tolerance
    lower_aligned = np.abs(keltner_bands['lower'] - ema) <= tolerance
    
    return upper_aligned | lower_aligned


def detect_location_density(keltner_bands_15m, keltner_bands_5m, 
                          keltner_bands_2000t, keltner_bands_200t, tolerance=0.002):
    """
    Detect when all Keltner bands align within a location density
    
    Args:
        keltner_bands_*: Keltner bands for different timeframes
        tolerance: Alignment tolerance
    
    Returns:
        Boolean array indicating location density alignment
    """
    import numpy as np
    
    # Get middle bands (EMA) for each timeframe
    ema_15m = keltner_bands_15m['middle']
    ema_5m = keltner_bands_5m['middle'] 
    ema_2000t = keltner_bands_2000t['middle']
    ema_200t = keltner_bands_200t['middle']
    
    # Check if all EMAs are within tolerance of each other
    ema_15m_5m_aligned = np.abs(ema_15m - ema_5m) <= tolerance
    ema_5m_2000t_aligned = np.abs(ema_5m - ema_2000t) <= tolerance
    ema_2000t_200t_aligned = np.abs(ema_2000t - ema_200t) <= tolerance
    
    return ema_15m_5m_aligned & ema_5m_2000t_aligned & ema_2000t_200t_aligned


def calculate_vwap(close, volume):
    """
    Calculate Volume Weighted Average Price
    
    Args:
        close: Close prices
        volume: Volume data
    
    Returns:
        VWAP values
    """
    import pandas as pd
    import numpy as np
    
    if isinstance(close, pd.Series):
        return (close * volume).cumsum() / volume.cumsum()
    else:
        # Convert to pandas for cumsum operations
        close_series = pd.Series(close)
        volume_series = pd.Series(volume)
        vwap_series = (close_series * volume_series).cumsum() / volume_series.cumsum()
        return vwap_series.values 