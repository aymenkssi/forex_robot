#!/usr/bin/env python3
"""Backtest du robot sur l'historique MT5 (a lancer sur le VPS, bridge actif).

Rejoue la logique d'entree/sortie du robot sur les bougies H1/M15/M5 :
score technique H1 (meme fonction que le robot), filtres MTF, spread, session
et cooldown, SL/TP ATR, break-even, trailing, fermeture sur inversion, puis
contraintes de portefeuille (positions max, exposition devise, perte du jour).

Non simule : validation Mistral et calendrier economique (pas d'historique).
Le spread vient du champ "spread" des bougies MT5 (spread minimal de la
bougie, donc optimiste) : --extra-spread ajoute une marge en points.

Aucun ordre n'est envoye. N'appelle jamais mt5.shutdown().

Usage :
    cd /opt/apps/iamrobot
    .venv/bin/python tools/backtest.py                 # 6 mois, variantes par defaut
    .venv/bin/python tools/backtest.py --months 12 --extra-spread 3
"""
import argparse
import csv
import hashlib
import inspect
import multiprocessing
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "robot"))
import forex_robot as fr  # noqa: E402  (client mt5linux + fonctions du robot)

DATA_DIR = ROOT / "backtest_data"
RESULTS_DIR = ROOT / "backtest_results"
TF_SECONDS = {"M5": 300, "M15": 900, "H1": 3600, "H4": 14400}
TREND = {"bullish": 1, "bearish": -1, "neutral": 0}


# ============================================================ PARAMETRES
@dataclass(frozen=True)
class Params:
    name: str
    min_score: int
    mtf: str                    # strict | m15 | pullback | none
    max_spread_points: float
    max_spread_sl_pct: float
    atr_tf: str
    atr_period: int
    atr_sl: float
    atr_tp: float
    be_r: float
    trail_r: float
    trail_atr: float
    reversal: bool
    reversal_score: int
    reversal_hold_min: int
    cooldown_min: int
    block_rollover: bool
    max_positions: int
    max_currency_exposure: int
    risk_pct: float
    daily_loss_pct: float


def baseline_params() -> Params:
    """Parametres actuels du robot (config.json du VPS)."""
    c = fr.CONFIG
    return Params(
        name="actuel",
        min_score=max(int(c["min_score"]), int(c.get("ai_validation_min_score", 0))),
        mtf="strict",
        max_spread_points=float(c["max_spread_points"]),
        max_spread_sl_pct=float(c.get("max_spread_sl_pct", 15.0)),
        atr_tf=str(c.get("atr_timeframe", "M15")).upper(),
        atr_period=int(c["atr_period"]),
        atr_sl=float(c["atr_sl_multiplier"]),
        atr_tp=float(c["atr_tp_multiplier"]),
        be_r=float(c["breakeven_trigger_r"]),
        trail_r=float(c["trailing_trigger_r"]),
        trail_atr=float(c["trailing_atr_multiplier"]),
        reversal=bool(c.get("close_on_reversal", True)),
        reversal_score=int(c.get("reversal_min_score", 4)),
        reversal_hold_min=int(c.get("reversal_min_hold_minutes", 30)),
        cooldown_min=int(c["trade_cooldown_minutes"]),
        block_rollover=bool(c.get("block_rollover", True)),
        max_positions=min(int(c["max_positions"]), int(c["hard_max_positions"])),
        max_currency_exposure=int(c["max_currency_exposure"]),
        risk_pct=float(c.get("risk_per_trade_pct", 0.5)),
        daily_loss_pct=float(c.get("daily_loss_limit_pct", 3.0)),
    )


def default_variants(base: Params) -> list:
    return [
        base,
        replace(base, name="score 3", min_score=3),
        replace(base, name="score 5", min_score=5),
        replace(base, name="MTF M15 seul", mtf="m15"),
        replace(base, name="MTF repli", mtf="pullback"),
        replace(base, name="sans MTF", mtf="none"),
        replace(base, name="spread 20%", max_spread_sl_pct=20.0),
        replace(base, name="spread 10%", max_spread_sl_pct=10.0),
        replace(base, name="ATR M15", atr_tf="M15"),
        replace(base, name="TP 3 ATR", atr_tp=3.0),
        replace(base, name="sans inversion", reversal=False),
    ]


# ============================================================ DONNEES MT5
def _to_columns(raw) -> dict:
    """Bougies MT5 (locales ou proxy RPyC) -> colonnes numpy locales."""
    try:
        from rpyc.core.netref import BaseNetref
        from rpyc.utils.classic import obtain
        if isinstance(raw, BaseNetref):
            # tolist() cote Wine puis transfert en tuples Python purs
            # (evite les incompatibilites de pickle numpy entre Windows et Linux).
            names = list(obtain(raw.dtype.names))
            rows = obtain(raw.tolist())
        else:
            names, rows = list(raw.dtype.names), raw.tolist()
    except ImportError:
        names, rows = list(raw.dtype.names), raw.tolist()
    cols = list(zip(*rows)) if rows else [[] for _ in names]
    data = {n: np.asarray(c) for n, c in zip(names, cols)}
    return {
        "time": data["time"].astype(np.int64),
        "open": data["open"].astype(float),
        "high": data["high"].astype(float),
        "low": data["low"].astype(float),
        "close": data["close"].astype(float),
        "spread": data["spread"].astype(float),
    }


def fetch_bars(symbol: str, tf: str, months: int, use_cache: bool) -> dict | None:
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"{symbol}_{tf}_{months}m.npz"
    if use_cache and path.exists() and time.time() - path.stat().st_mtime < 12 * 3600:
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    timeframe = {"M5": fr.mt5.TIMEFRAME_M5, "M15": fr.mt5.TIMEFRAME_M15,
                 "H1": fr.mt5.TIMEFRAME_H1, "H4": fr.mt5.TIMEFRAME_H4}[tf]
    count = int((months + 1) * 31 * 86400 / TF_SECONDS[tf])
    fr.mt5.symbol_select(symbol, True)
    raw = fr.mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if raw is None or len(raw) == 0:
        return None
    bars = _to_columns(raw)
    np.savez_compressed(path, **bars)
    return bars


# ============================================================ PRE-CALCULS
def _score_worker(args):
    symbol, h1_time, h1_close, start = args
    scores = np.zeros(len(h1_time), dtype=np.int16)
    month = 30 * 86400
    lo = 0
    for i in range(start, len(h1_time)):
        t_close = h1_time[i] + 3600
        while h1_time[lo] <= t_close - month:
            lo += 1
        scores[i] = int(fr.score_from_prices(symbol, h1_close[lo:i + 1]).get("score", 0))
    return symbol, scores


def _score_cache_key() -> str:
    src = inspect.getsource(fr.score_from_prices) + inspect.getsource(fr.calculate_trend)
    src += inspect.getsource(fr.find_support_resistance) + inspect.getsource(fr.calculate_macd)
    params = [fr.CONFIG[k] for k in ("rsi_period", "rsi_oversold", "rsi_overbought", "sma_fast", "sma_slow")]
    return hashlib.sha1((src + repr(params)).encode()).hexdigest()[:10]


def atr_series(bars: dict, period: int) -> np.ndarray:
    """ATR identique a calculate_atr_from_rates : moyenne simple des `period` derniers TR."""
    h, l, c = bars["high"], bars["low"], bars["close"]
    tr = np.full(len(c), np.nan)
    tr[1:] = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    out = np.full(len(c), np.nan)
    if len(c) > period:
        csum = np.cumsum(np.nan_to_num(tr))
        out[period:] = (csum[period:] - csum[:-period]) / period
    return out


class TrendCache:
    """trend_from_closes (fonction du robot) calcule a la demande et memorise."""

    def __init__(self, closes: np.ndarray, window: int):
        self.closes, self.window = closes, window
        self.values = np.full(len(closes), 9, dtype=np.int8)

    def at(self, idx: int) -> int:
        if idx < self.window - 1:
            return 0
        if self.values[idx] == 9:
            self.values[idx] = TREND[fr.trend_from_closes(self.closes[idx - self.window + 1:idx + 1])]
        return int(self.values[idx])


@dataclass
class SymbolData:
    symbol: str
    point: float
    h1: dict
    m15: dict
    m5: dict
    atr_bars: dict
    scores: np.ndarray
    m15_trend: TrendCache
    m5_trend: TrendCache
    test_start: int


# ============================================================ SIMULATION
def blocked_session(t: int) -> bool:
    d = datetime.fromtimestamp(int(t), tz=timezone.utc)
    return d.hour in (23, 0) or (d.weekday() == 4 and d.hour >= 21)


def simulate_symbol(sd: SymbolData, p: Params, extra_spread: float) -> list:
    """Trades d'une paire (une position a la fois), resultat en R."""
    trades = []
    h1t, m5 = sd.h1["time"], sd.m5
    m5t, m15t = m5["time"], sd.m15["time"]
    atr_arr = sd.atr_bars[(p.atr_tf, p.atr_period)]
    atr_t = sd.atr_bars[(p.atr_tf, "time")]
    atr_secs = TF_SECONDS[p.atr_tf]
    pos = None
    last_entry = -10 ** 12

    def atr_at(t):
        k = np.searchsorted(atr_t + atr_secs, t, side="right") - 1
        return float(atr_arr[k]) if k >= 0 and not np.isnan(atr_arr[k]) else 0.0

    def close_pos(t, price, reason):
        nonlocal pos
        r = (price - pos["entry"]) * pos["dir"] / pos["risk"]
        trades.append({"symbol": sd.symbol, "dir": pos["dir"], "entry_time": pos["t"],
                       "exit_time": int(t), "r": r, "reason": reason})
        pos = None

    for i in range(sd.test_start, len(h1t)):
        t_h1 = int(h1t[i]) + 3600
        score = int(sd.scores[i])
        j0 = int(np.searchsorted(m5t, t_h1))
        j1 = int(np.searchsorted(m5t, t_h1 + 3600))
        if j0 >= len(m5t):
            break

        if pos is not None and p.reversal and (t_h1 - pos["t"]) >= p.reversal_hold_min * 60:
            if score * pos["dir"] <= -p.reversal_score:
                spread = (m5["spread"][j0] + extra_spread) * sd.point
                price = m5["open"][j0] if pos["dir"] > 0 else m5["open"][j0] + spread
                close_pos(m5t[j0], price, "inversion")

        for j in range(j0, j1):
            spread = (m5["spread"][j] + extra_spread) * sd.point
            t_close = int(m5t[j]) + 300
            if pos is not None:
                d = pos["dir"]
                # Prix vus par la position : bid pour un BUY, ask pour un SELL.
                o, h, l, c = (m5["open"][j], m5["high"][j], m5["low"][j], m5["close"][j])
                if d < 0:
                    o, h, l, c = o + spread, h + spread, l + spread, c + spread
                adverse_hit = (l <= pos["sl"]) if d > 0 else (h >= pos["sl"])
                target_hit = (h >= pos["tp"]) if d > 0 else (l <= pos["tp"])
                gap_through = (o <= pos["sl"]) if d > 0 else (o >= pos["sl"])
                if gap_through:
                    close_pos(m5t[j], o, "gap SL")
                elif adverse_hit:
                    close_pos(t_close, pos["sl"], "SL" if pos["sl"] * d < pos["entry"] * d else "BE/trailing")
                elif target_hit:
                    close_pos(t_close, pos["tp"], "TP")
                else:
                    atr_now = atr_at(t_close)
                    if atr_now > 0:
                        favorable = (c - pos["entry"]) * d
                        r_mult = favorable / pos["risk"]
                        if r_mult >= p.trail_r:
                            cand = c - d * atr_now * p.trail_atr
                            if (cand - pos["sl"]) * d > 0 and (c - cand) * d > 0:
                                pos["sl"] = cand
                        elif r_mult >= p.be_r and (pos["entry"] - pos["sl"]) * d > 0:
                            pos["sl"] = pos["entry"]
                continue

            if abs(score) < p.min_score:
                continue
            d = 1 if score > 0 else -1
            if t_close - last_entry < p.cooldown_min * 60:
                continue
            if p.block_rollover and blocked_session(t_close):
                continue
            atr = atr_at(t_close)
            if atr <= 0:
                continue
            sl_dist = max(atr * p.atr_sl, sd.point * 10)
            tp_dist = max(atr * p.atr_tp, sd.point * 12)
            spread_pts = m5["spread"][j] + extra_spread
            if spread_pts > p.max_spread_points:
                continue
            if p.max_spread_sl_pct > 0 and spread / (atr * p.atr_sl) * 100 > p.max_spread_sl_pct:
                continue
            if p.mtf != "none":
                k15 = int(np.searchsorted(m15t + 900, t_close, side="right")) - 1
                t15 = sd.m15_trend.at(k15) if k15 >= 0 else 0
                t5 = sd.m5_trend.at(j)
                if p.mtf == "strict" and not (t15 == d and t5 == d):
                    continue
                if p.mtf == "m15" and t15 != d:
                    continue
                if p.mtf == "pullback" and not (t15 != -d and t5 == d and sd.m5_trend.at(j - 1) != d):
                    continue
            bid = m5["close"][j]
            entry = bid + spread if d > 0 else bid
            pos = {"dir": d, "t": t_close, "entry": entry, "risk": sl_dist,
                   "sl": entry - d * sl_dist, "tp": entry + d * tp_dist}
            last_entry = t_close

    if pos is not None:
        close_pos(m5t[-1] + 300, m5["close"][-1] + (0 if pos["dir"] > 0 else m5["spread"][-1] * sd.point), "fin")
    return trades


def apply_portfolio(trades: list, p: Params) -> tuple[list, np.ndarray]:
    """Positions max, exposition par devise, perte du jour ; capitalisation du risque."""
    accepted, open_pos = [], []
    equity, peak, max_dd = 1.0, 1.0, 0.0
    day, day_r = None, 0.0
    curve = []
    for tr in sorted(trades, key=lambda x: x["entry_time"]):
        # Clotures survenues avant cette entree.
        for op in sorted([o for o in open_pos if o["exit_time"] <= tr["entry_time"]], key=lambda x: x["exit_time"]):
            open_pos.remove(op)
            equity *= 1 + op["r"] * op["risk_frac"]
            peak = max(peak, equity)
            max_dd = max(max_dd, (peak - equity) / peak)
            curve.append((op["exit_time"], equity))
            d_exit = datetime.fromtimestamp(op["exit_time"], tz=timezone.utc).date()
            if d_exit == day:
                day_r += op["r"]
        d_entry = datetime.fromtimestamp(tr["entry_time"], tz=timezone.utc).date()
        if d_entry != day:
            day, day_r = d_entry, 0.0
        if p.daily_loss_pct > 0 and -day_r * p.risk_pct >= p.daily_loss_pct:
            continue
        if len(open_pos) >= p.max_positions:
            continue
        base, quote = tr["symbol"][:3], tr["symbol"][3:6]
        expo = {}
        for o in open_pos:
            expo[o["symbol"][:3]] = expo.get(o["symbol"][:3], 0) + o["dir"]
            expo[o["symbol"][3:6]] = expo.get(o["symbol"][3:6], 0) - o["dir"]
        if abs(expo.get(base, 0) + tr["dir"]) > p.max_currency_exposure:
            continue
        if abs(expo.get(quote, 0) - tr["dir"]) > p.max_currency_exposure:
            continue
        item = dict(tr, risk_frac=p.risk_pct / 100.0)
        open_pos.append(item)
        accepted.append(item)
    for op in sorted(open_pos, key=lambda x: x["exit_time"]):
        equity *= 1 + op["r"] * op["risk_frac"]
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak)
        curve.append((op["exit_time"], equity))
    return accepted, np.array([equity, max_dd])


def summarize(name: str, trades: list, eq: np.ndarray, months_span: float) -> dict:
    r = np.array([t["r"] for t in trades]) if trades else np.array([])
    wins, losses = r[r > 0], r[r < 0]
    pf = wins.sum() / -losses.sum() if len(losses) and losses.sum() < 0 else float("inf") if len(wins) else 0.0
    return {
        "variante": name,
        "trades": len(r),
        "trades/mois": round(len(r) / months_span, 1) if months_span > 0 else 0,
        "gagnants %": round(100 * len(wins) / len(r), 1) if len(r) else 0,
        "gain moyen R": round(float(wins.mean()), 2) if len(wins) else 0,
        "perte moy. R": round(float(losses.mean()), 2) if len(losses) else 0,
        "esperance R": round(float(r.mean()), 3) if len(r) else 0,
        "profit factor": round(pf, 2) if pf != float("inf") else "inf",
        "total R": round(float(r.sum()), 1),
        "resultat %": round((eq[0] - 1) * 100, 1),
        "DD max %": round(eq[1] * 100, 1),
    }


def print_table(rows: list):
    cols = list(rows[0].keys())
    widths = [max(len(str(c)), *(len(str(r[c])) for r in rows)) for c in cols]
    line = " | ".join(str(c).ljust(w) for c, w in zip(cols, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print(" | ".join(str(r[c]).ljust(w) for c, w in zip(cols, widths)))


# ============================================================ MAIN
def main():
    ap = argparse.ArgumentParser(description="Backtest du robot sur l'historique MT5")
    ap.add_argument("--months", type=int, default=6, help="duree testee (defaut 6 mois)")
    ap.add_argument("--symbols", default="", help="liste separee par des virgules (defaut : toutes)")
    ap.add_argument("--extra-spread", type=float, default=0.0, help="points de spread ajoutes (defaut 0)")
    ap.add_argument("--workers", type=int, default=2, help="processus pour le calcul des scores")
    ap.add_argument("--no-cache", action="store_true", help="retelecharger l'historique MT5")
    args = ap.parse_args()

    if not fr.connect_mt5():
        sys.exit("MT5 injoignable via le bridge : verifier MT5 et le service mt5-bridge.")

    base = baseline_params()
    variants = default_variants(base)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or list(fr.CONFIG["symbols"])
    atr_needed = {(v.atr_tf, v.atr_period) for v in variants}

    print(f"Backtest {args.months} mois | {len(symbols)} paires | spread +{args.extra_spread} pts | "
          f"base: score>={base.min_score}, ATR {base.atr_tf}, SL {base.atr_sl}x, TP {base.atr_tp}x, "
          f"spread<={base.max_spread_sl_pct}% du SL")
    started = time.time()

    loaded, score_jobs = {}, []
    for sym in symbols:
        info = fr.mt5.symbol_info(sym)
        if info is None:
            print(f"  {sym}: indisponible chez le broker")
            continue
        bars = {tf: fetch_bars(sym, tf, args.months, not args.no_cache) for tf in ("H1", "M15", "M5")}
        for tf in {v.atr_tf for v in variants} - set(bars):
            bars[tf] = fetch_bars(sym, tf, args.months, not args.no_cache)
        if any(b is None or len(b["time"]) < 200 for b in bars.values()):
            print(f"  {sym}: historique insuffisant")
            continue
        # Fenetre testee : les `months` derniers mois de l'historique disponible.
        test_from = int(bars["M5"]["time"][-1]) - args.months * 30.5 * 86400
        start = max(int(np.searchsorted(bars["H1"]["time"], max(test_from, bars["M5"]["time"][100]))),
                    int(np.searchsorted(bars["H1"]["time"], bars["H1"]["time"][0] + 31 * 86400)))
        loaded[sym] = (float(info.point), bars, start)
        score_jobs.append((sym, bars["H1"]["time"], bars["H1"]["close"], start))
        print(f"  {sym}: H1={len(bars['H1']['time'])} M15={len(bars['M15']['time'])} M5={len(bars['M5']['time'])}")

    if not loaded:
        sys.exit("Aucune donnee exploitable.")

    key = _score_cache_key()
    scores, todo = {}, []
    for job in score_jobs:
        path = DATA_DIR / f"{job[0]}_scores_{args.months}m_{key}.npy"
        if not args.no_cache and path.exists() and path.stat().st_mtime >= (DATA_DIR / f"{job[0]}_H1_{args.months}m.npz").stat().st_mtime:
            scores[job[0]] = np.load(path)
        else:
            todo.append(job)
    if todo:
        print(f"\nCalcul des scores H1 ({len(todo)} paires, quelques minutes)...")
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(max(1, args.workers)) as pool:
            for sym, sc in pool.imap_unordered(_score_worker, todo):
                scores[sym] = sc
                np.save(DATA_DIR / f"{sym}_scores_{args.months}m_{key}.npy", sc)
                print(f"  scores {sym} OK")

    datasets = []
    for sym, (point, bars, start) in loaded.items():
        atr_bars = {}
        for tf, period in atr_needed:
            atr_bars[(tf, period)] = atr_series(bars[tf], period)
            atr_bars[(tf, "time")] = bars[tf]["time"]
        datasets.append(SymbolData(
            symbol=sym, point=point, h1=bars["H1"], m15=bars["M15"], m5=bars["M5"],
            atr_bars=atr_bars, scores=scores[sym],
            m15_trend=TrendCache(bars["M15"]["close"], 60),
            m5_trend=TrendCache(bars["M5"]["close"], 80),
            test_start=start,
        ))

    span_start = min(int(d.h1["time"][d.test_start]) for d in datasets)
    span_end = max(int(d.m5["time"][-1]) for d in datasets)
    months_span = (span_end - span_start) / (30.5 * 86400)
    print(f"\nPeriode testee : {datetime.fromtimestamp(span_start, tz=timezone.utc):%Y-%m-%d} -> "
          f"{datetime.fromtimestamp(span_end, tz=timezone.utc):%Y-%m-%d} ({months_span:.1f} mois)\n")

    RESULTS_DIR.mkdir(exist_ok=True)
    rows, base_trades = [], None
    for v in variants:
        raw = []
        for sd in datasets:
            raw.extend(simulate_symbol(sd, v, args.extra_spread))
        accepted, eq = apply_portfolio(raw, v)
        rows.append(summarize(v.name, accepted, eq, months_span))
        if v is base:
            base_trades = accepted
        print(f"  variante '{v.name}' : {len(accepted)} trades")

    print()
    print_table(rows)

    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    with open(RESULTS_DIR / f"resume-{stamp}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    if base_trades:
        with open(RESULTS_DIR / f"trades-actuel-{stamp}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["symbol", "dir", "entry_time", "exit_time", "r", "reason"])
            w.writeheader()
            for t in base_trades:
                w.writerow({k: t[k] for k in w.fieldnames})
        per_sym = {}
        for t in base_trades:
            per_sym[t["symbol"]] = per_sym.get(t["symbol"], 0.0) + t["r"]
        reasons = {}
        for t in base_trades:
            reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
        print("\nConfiguration actuelle - R par paire :",
              ", ".join(f"{s} {r:+.1f}" for s, r in sorted(per_sym.items(), key=lambda kv: kv[1])))
        print("Sorties :", ", ".join(f"{k} {v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])))
    print(f"\nResultats : {RESULTS_DIR}/resume-{stamp}.csv  ({time.time() - started:.0f} s)")


if __name__ == "__main__":
    main()
