"""Strategie « suivi de tendance journalier » (cassure de canal de Donchian).

Regles classiques, fixees a priori :
- Achat quand la cloture D1 depasse le plus haut des `entry_n` jours precedents,
  vente quand elle passe sous le plus bas des `entry_n` jours precedents.
- Entree a l'ouverture du jour suivant.
- SL initial a `sl_atr` x ATR(20) journalier.
- Sortie quand la cloture repasse sous le plus bas (au-dessus du plus haut) des
  `exit_n` jours precedents, a l'ouverture suivante.

Fonctions pures sur des colonnes numpy, reutilisables par le robot en direct.
"""
import numpy as np

from strategy_pullback import atr_series

ATR_PERIOD = 20


def channel(values: np.ndarray, n: int, highest: bool) -> np.ndarray:
    """Plus haut (ou plus bas) des `n` bougies PRECEDENTES (bougie courante exclue)."""
    out = np.full(len(values), np.nan)
    for i in range(n, len(values)):
        window = values[i - n:i]
        out[i] = window.max() if highest else window.min()
    return out


def compute_indicators(d1: dict, entry_n: int, exit_n: int) -> dict:
    return {
        "entry_high": channel(d1["high"], entry_n, True),
        "entry_low": channel(d1["low"], entry_n, False),
        "exit_high": channel(d1["high"], exit_n, True),
        "exit_low": channel(d1["low"], exit_n, False),
        "atr": atr_series(d1["high"], d1["low"], d1["close"], ATR_PERIOD),
    }


def entry_signal(d1: dict, ind: dict, i: int) -> int:
    """+1 cassure haussiere, -1 baissiere, 0 sinon (a la cloture du jour i)."""
    c = d1["close"][i]
    if np.isnan(ind["entry_high"][i]) or np.isnan(ind["atr"][i]):
        return 0
    if c > ind["entry_high"][i]:
        return 1
    if c < ind["entry_low"][i]:
        return -1
    return 0


def exit_signal(d1: dict, ind: dict, i: int, direction: int) -> bool:
    c = d1["close"][i]
    if direction > 0:
        return not np.isnan(ind["exit_low"][i]) and c < ind["exit_low"][i]
    return not np.isnan(ind["exit_high"][i]) and c > ind["exit_high"][i]
