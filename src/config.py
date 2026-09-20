"""All tunable settings in one place."""
import datetime as dt
import os
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
DATA_DIR = os.environ.get("DATA_DIR", "data")

MARKET_OPEN = dt.time(9, 15)
MARKET_CLOSE = dt.time(15, 30)
EOD_CUTOFF = dt.time(16, 30)      # after this, runs do nothing
HORIZONS = (15, 30, 60)           # minutes

# How much each signal family counts, per horizon.
WEIGHTS = {
    15: {"technical": 0.40, "options": 0.30, "global": 0.15, "news": 0.15},
    30: {"technical": 0.35, "options": 0.30, "global": 0.20, "news": 0.15},
    60: {"technical": 0.30, "options": 0.28, "global": 0.22, "news": 0.20},
}

DIRECTION_THRESHOLD = 0.12   # |bias| below this = SIDEWAYS call
DRIFT_K = 0.5                # bias=1 shifts the range centre by 0.5 sigma
Z68, Z90 = 1.0, 1.645        # range widths (in sigmas)
TRADING_MINUTES_PER_DAY = 375

# Global cues: yahoo symbol -> (label, sign, weight, pct-move that counts as "full" signal)
# sign +1 means "up is good for Nifty", -1 means "up is bad for Nifty"
GLOBAL_CUES = {
    "ES=F":      ("S&P 500 futures", +1, 0.22, 1.0),
    "NQ=F":      ("Nasdaq futures", +1, 0.12, 1.5),
    "^N225":     ("Nikkei 225", +1, 0.08, 1.5),
    "^HSI":      ("Hang Seng", +1, 0.10, 1.5),
    "^GDAXI":    ("DAX", +1, 0.08, 1.2),
    "^INDIAVIX": ("India VIX", -1, 0.20, 5.0),
    "BZ=F":      ("Brent crude", -1, 0.08, 2.0),
    "DX-Y.NYB":  ("US Dollar Index", -1, 0.05, 0.5),
    "INR=X":     ("USD/INR", -1, 0.05, 0.4),
    "^TNX":      ("US 10Y yield", -1, 0.05, 2.0),
}

# Alert thresholds
OI_MIN_Z = 4.0               # robust z-score of the 15-min OI change across strikes
OI_MIN_PCT = 0.20            # and at least +20% vs previous OI at that strike
OI_MIN_SHARE = 0.005         # and at least 0.5% of that side's total OI
PCR_SHIFT = 0.08             # PCR move between runs that triggers an alert
VIX_SPIKE_PCT = 4.0
PRICE_SHOCK_SIGMA = 1.6
ALERT_COOLDOWN_MIN = 45

GEMINI_MODELS = [m.strip() for m in os.environ.get(
    "GEMINI_MODELS",
    "gemini-3.5-flash,gemini-3.1-flash-lite,gemini-2.5-flash,gemini-2.5-flash-lite").split(",") if m.strip()]
USE_GROUNDED_SEARCH = os.environ.get("USE_GROUNDED_SEARCH", "1") == "1"

HISTORY_DAYS = 30            # detailed prediction records kept
