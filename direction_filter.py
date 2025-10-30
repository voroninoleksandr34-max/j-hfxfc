# direction_filter.py
# Ensemble BTC direction filter with 2D Kalman, Supertrend, and HMM.

from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd

# ---------- 2D Kalman Filter (price + slope) ----------
class KalmanTrend2D:
    def __init__(self, q_price: float = 1e-4, q_slope: float = 5e-4, r_meas: float = 1e-2,
                 init_var_price: float = 1.0, init_var_slope: float = 1.0):
        self.q_price, self.q_slope, self.r_meas = q_price, q_slope, r_meas
        self.init_var_price, self.init_var_slope = init_var_price, init_var_slope
        self.x, self.P = None, None

    def filter_series(self, y: np.ndarray):
        n = len(y)
        price_est, slope_est = np.zeros(n), np.zeros(n)
        if n == 0:
            return price_est, slope_est

        F = np.array([[1., 1.], [0., 1.]])
        H = np.array([[1., 0.]])
        Q = np.diag([self.q_price, self.q_slope])
        R = np.array([[self.r_meas]])
        I = np.eye(2)
        
        # Инициализация с первым значением
        self.x = np.array([y[0], 0.0])
        self.P = np.diag([self.init_var_price, self.init_var_slope])

        for i, z in enumerate(y):
            # Predict
            self.x = F @ self.x
            self.P = F @ self.P @ F.T + Q
            
            # Update
            z_pred = (H @ self.x).item()
            S = (H @ self.P @ H.T + R).item()
            K = (self.P @ H.T) / S
            self.x = self.x + (K.flatten() * (z - z_pred))
            self.P = (I - K @ H) @ self.P
            
            price_est[i] = self.x[0]
            slope_est[i] = self.x[1]
            
        return price_est, slope_est

# ---------- Supertrend (canonical TradingView-like) ----------
def _rma(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1/period, adjust=False).mean()

def supertrend(df: pd.DataFrame, atr_period: int = 10, multiplier: float = 3.0):
    high, low, close = df['high'], df['low'], df['close']
    hl2 = (high + low) / 2.0
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = _rma(tr, atr_period)
    
    basic_ub, basic_lb = hl2 + multiplier * atr, hl2 - multiplier * atr
    final_ub, final_lb = basic_ub.copy(), basic_lb.copy()

    for i in range(1, len(df)):
        final_ub.iat[i] = basic_ub.iat[i] if (basic_ub.iat[i] < final_ub.iat[i-1]) or (close.iat[i-1] > final_ub.iat[i-1]) else final_ub.iat[i-1]
        final_lb.iat[i] = basic_lb.iat[i] if (basic_lb.iat[i] > final_lb.iat[i-1]) or (close.iat[i-1] < final_lb.iat[i-1]) else final_lb.iat[i-1]

    st = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i > 0 and st.iat[i-1] == final_ub.iat[i-1] and close.iat[i] <= final_ub.iat[i]:
            st.iat[i] = final_ub.iat[i]
        elif i > 0 and st.iat[i-1] == final_ub.iat[i-1] and close.iat[i] > final_ub.iat[i]:
            st.iat[i] = final_lb.iat[i]
        elif i > 0 and st.iat[i-1] == final_lb.iat[i-1] and close.iat[i] >= final_lb.iat[i]:
            st.iat[i] = final_lb.iat[i]
        else:
            st.iat[i] = final_lb.iat[i] if i == 0 or close.iat[i] >= final_lb.iat[i] else final_ub.iat[i]
            
    st_dir = pd.Series(np.where(close >= st, 1, -1), index=df.index)
    return st, st_dir, atr

# ---------- Utilities and Config ----------
def _zscore(x: np.ndarray, eps: float = 1e-9):
    mu, sd = np.nanmean(x), np.nanstd(x)
    return np.where(np.isfinite(sd) & (sd > eps), (x - mu) / sd, np.zeros_like(x))

def _sigmoid(x: np.ndarray):
    return 1.0 / (1.0 + np.exp(-x))

@dataclass
class EnsembleConfig:
    q_price: float = 1e-4; q_slope: float = 5e-4; r_meas: float = 1e-2
    atr_period: int = 10; atr_mult: float = 3.0
    hmm_ema_alpha: float = 0.25; hmm_trend_prob_threshold: float = 0.55
    w_k: float = 0.5; w_st: float = 0.3; w_hmm: float = 0.2
    long_thresh: float = 0.55; short_thresh: float = 0.45
    require_two_of_three: bool = True; min_bars_confirm: int = 2
    use_atr_norm_for_slope: bool = True; slope_z_clip: float = 3.0

# ---------- Main Filter Class ----------
class BTCDirectionFilter:
    def __init__(self, cfg: EnsembleConfig = EnsembleConfig()):
        self.cfg = cfg
        self.kalman = KalmanTrend2D(q_price=cfg.q_price, q_slope=cfg.q_slope, r_meas=cfg.r_meas)

    def _smooth_hmm_prob(self, prob: pd.Series) -> pd.Series:
        p = prob.clip(0.0, 1.0).fillna(0.5)
        return p.ewm(alpha=self.cfg.hmm_ema_alpha, adjust=False).mean().clip(0.1, 0.9)

    def _debounce_direction(self, raw_dir: np.ndarray) -> np.ndarray:
        N = self.cfg.min_bars_confirm
        out = np.zeros_like(raw_dir)
        current = 0; pending = 0; pending_dir = 0
        for i, sig in enumerate(raw_dir):
            if sig == 0 or sig == current:
                pending = 0; pending_dir = 0
            else:
                if pending_dir != sig:
                    pending_dir = sig; pending = 1
                else:
                    pending += 1
                if pending >= N:
                    current = pending_dir
            out[i] = current
        return out

    def run(self, df: pd.DataFrame, hmm_trend_prob_series: pd.Series) -> pd.DataFrame:
        price = df['close'].astype(float).values
        k_price, k_slope = self.kalman.filter_series(price)
        st_line, st_dir, atr = supertrend(df, atr_period=self.cfg.atr_period, multiplier=self.cfg.atr_mult)
        atr_np = atr.fillna(method='bfill').fillna(1e-9).values
        
        slope_norm = np.divide(k_slope, atr_np) if self.cfg.use_atr_norm_for_slope else k_slope
        slope_z = np.clip(_zscore(slope_norm), -self.cfg.slope_z_clip, self.cfg.slope_z_clip)
        
        hmm_prob = self._smooth_hmm_prob(hmm_trend_prob_series.reindex(df.index)).values
        
        score = (self.cfg.w_k * slope_z) + \
                (self.cfg.w_st * st_dir.values) + \
                (self.cfg.w_hmm * (2.0 * hmm_prob - 1.0))
        prob_long = _sigmoid(score)
        
        raw_dir = np.where(prob_long > self.cfg.long_thresh, 1, np.where(prob_long < self.cfg.short_thresh, -1, 0))
        
        gated_dir = raw_dir
        if self.cfg.require_two_of_three:
            slope_vote, hmm_vote = np.sign(slope_z), np.where(hmm_prob >= self.cfg.hmm_trend_prob_threshold, 1, -1)
            vote_dir = np.sign(slope_vote + st_dir.values + hmm_vote)
            gated_dir = np.where(vote_dir == raw_dir, raw_dir, 0)
        
        direction = self._debounce_direction(gated_dir)
        
        out = pd.DataFrame(index=df.index)
        out['prob_long'], out['direction'] = prob_long, direction
        return out

    def latest_signal(self, df: pd.DataFrame, hmm_trend_prob_series: pd.Series) -> Tuple[int, float]:
        out = self.run(df, hmm_trend_prob_series)
        return int(out['direction'].iloc[-1]), float(out['prob_long'].iloc[-1])