"""Nifty candles + global cues from Yahoo Finance (free, no key)."""
import time

import pandas as pd

from .config import GLOBAL_CUES, IST
from .util import clip, log


def _retry(fn, tries=3, wait=3):
    last = None
    for i in range(tries):
        try:
            r = fn()
            if r is not None and len(r):
                return r
        except Exception as e:  # noqa: BLE001 - network flakiness is expected
            last = e
        time.sleep(wait * (i + 1))
    if last:
        log(f"yfinance failed: {last}")
    return None


def fetch_nifty():
    """Returns (1-minute candles for ~5 days, daily candles for ~1 year), both IST-indexed."""
    import yfinance as yf
    t = yf.Ticker("^NSEI")
    m1 = _retry(lambda: t.history(period="5d", interval="1m", auto_adjust=False))
    daily = _retry(lambda: t.history(period="1y", interval="1d", auto_adjust=False))
    if m1 is None or daily is None:
        return None, None
    m1.index = m1.index.tz_convert(IST) if m1.index.tz else m1.index.tz_localize(IST)
    daily.index = (daily.index.tz_convert(IST) if daily.index.tz else daily.index.tz_localize(IST)).normalize()
    return m1[["Open", "High", "Low", "Close", "Volume"]], daily[["Open", "High", "Low", "Close", "Volume"]]


def fetch_global(today):
    """Percent change of each global cue vs its previous close, plus a combined score in [-1, 1]."""
    import yfinance as yf
    out = []
    tickers = list(GLOBAL_CUES)
    df = _retry(lambda: yf.download(tickers, period="7d", interval="1d", group_by="ticker",
                                    progress=False, threads=False, auto_adjust=False))
    if df is None:
        return [], None
    num = den = 0.0
    for sym, (label, sgn, w, scale) in GLOBAL_CUES.items():
        try:
            close = df[sym]["Close"].dropna()
            if len(close) < 2:
                continue
            last, prev = float(close.iloc[-1]), float(close.iloc[-2])
            pct = (last / prev - 1) * 100
            fresh = close.index[-1].date() >= today   # is the latest bar from today?
            eff_w = w * (1.0 if fresh else 0.4)       # stale markets count less
            sig = clip(sgn * pct / scale)
            num += eff_w * sig
            den += eff_w
            out.append({"symbol": sym, "label": label, "price": last, "change_pct": pct,
                        "signal": sig, "stale": not fresh, "weight": eff_w})
        except Exception:  # noqa: BLE001
            continue
    return out, (clip(num / den) if den > 0 else None)
