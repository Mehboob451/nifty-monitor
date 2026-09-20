"""Prediction engine: combines signal families into a bias, then turns bias + volatility into
expected ranges for +15 / +30 / +60 minutes. All numbers come from code, never from the LLM."""
import datetime as dt
import math

import numpy as np
import pandas as pd

from . import gemini
from .config import (DIRECTION_THRESHOLD, DRIFT_K, HORIZONS, IST, MARKET_CLOSE, TRADING_MINUTES_PER_DAY,
                     WEIGHTS, Z68, Z90)
from .util import clip, sign


def realized_sigma5(m5):
    dates = pd.Series(m5.index.date, index=m5.index)
    same_day = dates == dates.shift(1)
    r = np.log(m5["Close"]).diff()[same_day].tail(60)
    return float(r.std()) if len(r) >= 10 and r.std() > 0 else 0.0007


def sigma_points(m5, vix, spot, h):
    """1-sigma move in index points over h minutes, blending realized and implied (VIX) volatility."""
    rv = realized_sigma5(m5) * math.sqrt(h / 5) * spot
    if vix:
        iv = (vix / 100) / math.sqrt(252) * math.sqrt(h / TRADING_MINUTES_PER_DAY) * spot
        return max(0.5 * rv + 0.5 * iv, 0.4 * iv)
    return rv


def combine(scores, h):
    w = WEIGHTS[h]
    avail = {k: v for k, v in scores.items() if v is not None}
    if not avail:
        return 0.0, 0.0, 0.0
    tw = sum(w[k] for k in avail)
    bias = sum(w[k] * v for k, v in avail.items()) / tw
    agree = sum(w[k] for k, v in avail.items() if sign(v, 0.05) == sign(bias) and sign(bias) != 0) / tw
    completeness = tw / sum(w.values())
    base = min(1.0, abs(bias) / 0.6)
    conf = 100 * base * (0.4 + 0.6 * agree) * (0.5 + 0.5 * completeness)
    return bias, min(conf, 90.0), agree


def direction(bias):
    return "UP" if bias >= DIRECTION_THRESHOLD else ("DOWN" if bias <= -DIRECTION_THRESHOLD else "SIDEWAYS")


def conf_label(c):
    return "High" if c >= 55 else ("Medium" if c >= 30 else "Low")


def build_prediction(anchor, spot, scores, m5, vix, calib):
    close_dt = dt.datetime.combine(anchor.date(), MARKET_CLOSE, tzinfo=IST)
    out = {}
    for h in HORIZONS:
        bias, conf, agree = combine(scores, h)
        sig_raw = sigma_points(m5, vix, spot, h)
        scale = calib.get(str(h), 1.0)
        sig = sig_raw * scale
        mid = spot + bias * DRIFT_K * sig
        target = anchor + dt.timedelta(minutes=h)
        out[str(h)] = {
            "bias": bias, "dir": direction(bias), "conf": conf, "conf_label": conf_label(conf), "agree": agree,
            "mid": mid, "low": mid - Z68 * sig, "high": mid + Z68 * sig,
            "low90": mid - Z90 * sig, "high90": mid + Z90 * sig,
            "sigma": sig, "sigma_raw": sig_raw, "scale": scale,
            "target_at": target.isoformat(), "scored": target <= close_dt,
            "actual": None, "move": None, "in68": None, "in90": None, "dir_hit": None, "err": None, "resolved_at": None,
        }
    return out


def rule_reasoning(scores, parts, pred, ind, an, glob, news):
    """Plain-English reasoning built purely from the computed numbers (used when Gemini is unavailable)."""
    p30 = pred["30"]
    pts = []
    names = {"technical": "Technicals", "options": "Options data", "global": "Global cues", "news": "News"}
    verb = {"technical": "are", "options": "are", "global": "are", "news": "is"}
    for k in ("technical", "options", "global", "news"):
        v = scores.get(k)
        if v is None:
            pts.append(f"{names[k]}: not available this run.")
            continue
        lean = "bullish" if v > 0.12 else "bearish" if v < -0.12 else "neutral"
        detail = ""
        if k == "technical" and parts.get("technical"):
            top = sorted(parts["technical"], key=lambda x: abs(x["value"] * x["weight"]), reverse=True)[:2]
            detail = "; ".join(f"{t['name']}: {t['note']}" for t in top)
        if k == "options" and an:
            detail = f"PCR {an['pcr']:.2f}, support {an['support']['strike']:.0f}, resistance {an['resistance']['strike']:.0f}" \
                if an.get("pcr") and an.get("support") and an.get("resistance") else ""
        if k == "global" and glob:
            top = sorted(glob, key=lambda x: abs(x["signal"] * x["weight"]), reverse=True)[:2]
            detail = "; ".join(f"{g['label']} {g['change_pct']:+.2f}%" for g in top)
        if k == "news" and news:
            detail = news.get("summary", "")[:160]
        pts.append(f"{names[k]} {verb[k]} {lean} ({v:+.2f}). {detail}".strip())
    risks = []
    if ind.get("rsi", 50) > 70 or ind.get("rsi", 50) < 30:
        risks.append("RSI is stretched, so a snap-back is possible.")
    if p30["agree"] < 0.6:
        risks.append("Signal families disagree, so conviction is low.")
    if an and an.get("dte") is not None and an["dte"] <= 1:
        risks.append("Expiry is near; option-driven whipsaws are more likely.")
    lead = {"UP": "Leaning up", "DOWN": "Leaning down", "SIDEWAYS": "No clear edge, likely range-bound"}[p30["dir"]]
    return {"headline": f"{lead} over the next 30 minutes ({p30['conf_label'].lower()} confidence).",
            "points": pts, "risks": risks or ["No specific extra risk flagged."],
            "invalidation": f"A sustained move outside {p30['low']:.0f}-{p30['high']:.0f} would mean this call is wrong.",
            "source": "rules"}


def gemini_reasoning(payload):
    if not gemini.available():
        return None
    prompt = ("You are explaining a Nifty 50 intraday forecast to a trader. All numbers below were computed by code; "
              "do NOT change or invent numbers. Explain WHY the model leans the way it does, mention what conflicts, "
              "and what would invalidate it. Be concrete and short.\n\nDATA:\n" + str(payload) +
              '\n\nReply with ONLY JSON: {"headline": "<=22 words", "points": ["3 to 5 short sentences"], '
              '"risks": ["2 to 3 short sentences"], "invalidation": "one sentence"}')
    obj, model = gemini.ask_json(prompt)
    if not obj:
        return None
    try:
        return {"headline": str(obj["headline"])[:220], "points": [str(x)[:260] for x in obj["points"]][:6],
                "risks": [str(x)[:220] for x in obj["risks"]][:4], "invalidation": str(obj.get("invalidation", ""))[:260],
                "source": model}
    except (KeyError, TypeError):
        return None
