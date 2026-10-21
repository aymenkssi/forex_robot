"""Strategie « recul dans la tendance » (etape 2).

Regles fixees a priori (pas d'optimisation fine, pour limiter le sur-ajustement) :
- Tendance H4 : cloture > EMA200 et EMA50 > EMA200 (achat), inverse (vente).
- Recul H1 : sur les 5 dernieres bougies H1, le prix a touche l'EMA20 H1.
- Declencheur : la bougie H1 cloture du bon cote de l'EMA20 et au-dela du
  plus haut (achat) / plus bas (vente) de la bougie precedente.
- SL : sous le plus bas (au-dessus du plus haut) des 5 bougies, moins
  0,2 ATR H1 ; distance comprise entre 0,8 et 3 ATR, sinon pas de trade.
- TP : multiple fixe du risque (tp_r).

Fonctions pures sur des colonnes numpy (time/open/high/low/close) : utilisees
par le backtest et, une fois validees, par le robot en direct.
"""
from typing import Optional

import numpy as np

H4_SECONDS = 4 * 3600
H1_SECONDS = 3600
LOOKBACK = 5


def ema_series(values: np.ndarray, period: int) -> np.ndarray:
    """EMA initialisee par la moyenne simple des `period` premieres valeurs."""
    out = np.full(len(values), np.nan)
    if len(values) < period:
        return out
    k = 2.0 / (period + 1)
    out[period - 1] = float(np.mean(values[:period]))
    for i in range(period, len(values)):
        out[i] = values[i] * k + out[i - 1] * (1 - k)
    return out


def atr_series(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """ATR de Wilder (lissage exponentiel 1/period), comme dans MT5."""
    tr = np.full(len(close), np.nan)
    tr[1:] = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    out = np.full(len(close), np.nan)
    if len(close) <= period:
        return out
    out[period] = np.mean(tr[1:period + 1])
    for i in range(period + 1, len(close)):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


def compute_indicators(h4: dict, h1: dict) -> dict:
    return {
        "h4_ema50": ema_series(h4["close"], 50),
        "h4_ema200": ema_series(h4["close"], 200),
        "h1_ema20": ema_series(h1["close"], 20),
        "h1_atr": atr_series(h1["high"], h1["low"], h1["close"], 14),
    }


def h4_trend(h4: dict, ind: dict, k: int) -> int:
    """+1 haussier, -1 baissier, 0 sinon, sur la bougie H4 cloturee d'indice k."""
    if k < 0:
        return 0
    c, e50, e200 = h4["close"][k], ind["h4_ema50"][k], ind["h4_ema200"][k]
    if np.isnan(e200) or np.isnan(e50):
        return 0
    if c > e200 and e50 > e200:
        return 1
    if c < e200 and e50 < e200:
        return -1
    return 0


def pullback_signal(h4: dict, h1: dict, ind: dict, i: int, k: int) -> Optional[dict]:
    """Signal a la cloture de la bougie H1 i (bougie H4 cloturee k).

    Retourne {"dir", "stop", "atr"} : le stop est un prix absolu, la distance
    reelle depend du prix d'entree (bougie suivante, spread compris).
    """
    if i < LOOKBACK + 1:
        return None
    trend = h4_trend(h4, ind, k)
    if trend == 0:
        return None
    ema20 = ind["h1_ema20"]
    atr = ind["h1_atr"][i]
    if np.isnan(atr) or np.isnan(ema20[i - LOOKBACK]) or atr <= 0:
        return None
    lo, hi = i - LOOKBACK + 1, i + 1
    close, high, low = h1["close"], h1["high"], h1["low"]

    if trend > 0:
        touched = np.any(low[lo:hi] <= ema20[lo:hi])
        trigger = close[i] > ema20[i] and close[i] > high[i - 1]
        if touched and trigger:
            return {"dir": 1, "stop": float(np.min(low[lo:hi]) - 0.2 * atr), "atr": float(atr)}
    else:
        touched = np.any(high[lo:hi] >= ema20[lo:hi])
        trigger = close[i] < ema20[i] and close[i] < low[i - 1]
        if touched and trigger:
            return {"dir": -1, "stop": float(np.max(high[lo:hi]) + 0.2 * atr), "atr": float(atr)}
    return None


def stop_distance_ok(entry: float, stop: float, atr: float, direction: int,
                     min_atr: float = 0.8, max_atr: float = 3.0) -> bool:
    dist = (entry - stop) * direction
    return min_atr * atr <= dist <= max_atr * atr
