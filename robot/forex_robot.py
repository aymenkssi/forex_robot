import requests
import time
import threading
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List
from fastapi import FastAPI, APIRouter, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
import traceback
import logging
import json as json_module
import re
import os

# ============ MT5 BRIDGE (mt5linux / RPyC) ============
# Le terminal MetaTrader 5 est gere par le serveur bridge sous Wine.
# Le robot Linux ne doit PAS appeler mt5.initialize(), mt5.login() ou
# mt5.shutdown() : ces appels concernent le MT5 distant du bridge.
from mt5linux import MetaTrader5

MT5_BRIDGE_HOST = os.environ.get("MT5_BRIDGE_HOST", "127.0.0.1")
MT5_BRIDGE_PORT = int(os.environ.get("MT5_BRIDGE_PORT", "18812"))
MT5_BRIDGE_TIMEOUT = int(os.environ.get("MT5_BRIDGE_TIMEOUT", "60"))
mt5 = MetaTrader5(
    host=MT5_BRIDGE_HOST,
    port=MT5_BRIDGE_PORT,
    timeout=MT5_BRIDGE_TIMEOUT,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============ OPTIMISATION IA / OLLAMA ============
# Une seule generation Ollama a la fois pour eviter la concurrence CPU/RAM.
OLLAMA_GENERATION_LOCK = threading.Lock()
OLLAMA_STATUS_CACHE = {"timestamp": 0.0, "data": None}
OLLAMA_STATUS_CACHE_TTL = 15

# L'IA n'est appelee automatiquement que pour les signaux techniques forts.
AI_VALIDATION_MIN_SCORE = 4
AI_QUICK_NUM_PREDICT = 96
AI_QUICK_TIMEOUT = 45
AI_FULL_NUM_PREDICT = 180
AI_FULL_TIMEOUT = 60
AI_KEEP_ALIVE = "60s"

# Cooldown entre deux validations IA automatiques afin de limiter la charge
# CPU/RAM d'Ollama sur le VPS. Les analyses FULL de l'interface ne sont pas
# bloquees par ce cooldown, mais restent protegees par le verrou Ollama.
AI_AUTO_COOLDOWN = 120
LAST_AUTO_AI_VALIDATION = 0.0

# ============ RISK ENGINE / PROTECTION ============
# Protections independantes de CONFIG pour eviter une surexposition accidentelle.
HARD_MAX_POSITIONS = 5
MAX_CURRENCY_EXPOSURE = 2
TRADE_COOLDOWN_MINUTES = 15
MAX_SPREAD_POINTS = 40
MAX_SPREAD_SL_PCT = 15.0
MAX_DRAWDOWN_FOR_NEW_TRADES_PCT = 5.0
AUTO_STOP_DRAWDOWN_PCT = 8.0
M15_MIN_BARS = 60
M5_MIN_BARS = 80
ATR_PERIOD = 14
ATR_TIMEFRAME = "M15"
ATR_SL_MULTIPLIER = 1.5
ATR_TP_MULTIPLIER = 2.2
BREAKEVEN_TRIGGER_R = 1.0
TRAILING_TRIGGER_R = 1.5
TRAILING_ATR_MULTIPLIER = 1.0
BROKER_SL_REQUIRED = True
NEWS_BLACKOUT_MINUTES = 30
NEWS_AFTER_MINUTES = 15
RISK_PER_TRADE_PCT = 0.5
DAILY_LOSS_LIMIT_PCT = 3.0
CLOSE_ON_REVERSAL = True
REVERSAL_MIN_SCORE = 4
REVERSAL_MIN_HOLD_MINUTES = 30
BLOCK_ROLLOVER = True
ROBOT_MAGIC = 202401

# ============ CONFIGURATION ============
CONFIG = {
    # Le compte est celui deja connecte dans le terminal MT5 du bridge.
    # Aucun mot de passe MT5 n'est stocke dans ce fichier.
    "account": None,
    "password": None,
    "server": None,

    # Trading
    "lot_size": 0.1,
    "target_profit": 20.0,
    "stop_loss": 50.0,
    "max_positions": 1,
    "min_score": 3,

    # Tous les couples forex majeurs et mineurs
    "symbols": [
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
        "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD",
        "GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD", "GBPNZD",
        "AUDJPY", "CADJPY", "CHFJPY", "NZDJPY",
        "AUDCAD", "AUDCHF", "AUDNZD", "CADCHF", "NZDCAD", "NZDCHF"
    ],

    # Analyse technique
    "sma_fast": 10,
    "sma_slow": 30,
    "rsi_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30,

    # Niveaux d'impact calendrier
    "impacts": {"high": True, "medium": True, "low": False},

    # Intervalle de verification
    "check_interval": 30,

    # Parametres IA / Ollama
    "ollama_url": "http://localhost:11434",
    "ollama_model": "mistral",
    "use_ai_analysis": True,
    "ai_validation_min_score": 4,
    "ai_auto_cooldown": 120,
    "ai_quick_num_predict": 96,
    "ai_quick_timeout": 45,
    "ai_full_num_predict": 180,
    "ai_full_timeout": 60,
    "ai_keep_alive": "60s",

    # Risk Engine / protections
    "hard_max_positions": 5,
    "max_currency_exposure": 2,
    "trade_cooldown_minutes": 15,
    "max_spread_points": 40,
    # Le spread ne doit pas depasser ce % de la distance du SL (cout relatif a la volatilite).
    "max_spread_sl_pct": 15.0,
    # Unite de temps de l'ATR pour SL/TP/trailing (M15, H1 ou H4).
    "atr_timeframe": "M15",
    "max_drawdown_new_trades_pct": 5.0,
    "auto_stop_drawdown_pct": 8.0,
    "m15_min_bars": 60,
    "m5_min_bars": 80,
    "atr_period": 14,
    "atr_sl_multiplier": 1.5,
    "atr_tp_multiplier": 2.2,
    "breakeven_trigger_r": 1.0,
    "trailing_trigger_r": 1.5,
    "trailing_atr_multiplier": 1.0,
    "broker_sl_required": True,
    "news_blackout_minutes": 30,
    "news_after_minutes": 15,
    # Taille de position calculee pour risquer ce % du capital (lot_size = plafond).
    "risk_per_trade_pct": 0.5,
    # Perte max de la journee (equity) avant blocage des nouvelles entrees.
    "daily_loss_limit_pct": 3.0,
    # Fermeture d'une position du robot si le signal s'inverse fortement.
    "close_on_reversal": True,
    "reversal_min_score": 4,
    "reversal_min_hold_minutes": 30,
    # Pas de nouvelle entree autour du rollover et le vendredi soir (heure serveur).
    "block_rollover": True,

    # Telegram Alerts
    "telegram_bot_token": "",
    "telegram_chat_id": "",
    "telegram_enabled": False,
    "telegram_on_open": True,
    "telegram_on_close": True,
    "telegram_on_target": True,
    "telegram_on_stoploss": True,
    "telegram_on_analysis": False,

}

# ============ CONFIGURATION PERSISTANTE ============
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

PERSISTENT_CONFIG_KEYS = {
    "lot_size", "target_profit", "stop_loss", "max_positions", "min_score",
    "check_interval", "impacts", "rsi_period", "rsi_overbought", "rsi_oversold",
    "sma_fast", "sma_slow", "ollama_url", "ollama_model", "use_ai_analysis",
    "ai_validation_min_score", "ai_auto_cooldown", "ai_quick_num_predict",
    "ai_quick_timeout", "ai_full_num_predict", "ai_full_timeout", "ai_keep_alive",
    "hard_max_positions", "max_currency_exposure", "trade_cooldown_minutes",
    "max_spread_points", "max_spread_sl_pct", "max_drawdown_new_trades_pct", "auto_stop_drawdown_pct",
    "m15_min_bars", "m5_min_bars", "atr_period", "atr_sl_multiplier",
    "atr_tp_multiplier", "breakeven_trigger_r", "trailing_trigger_r",
    "trailing_atr_multiplier", "broker_sl_required", "news_blackout_minutes",
    "news_after_minutes", "risk_per_trade_pct", "daily_loss_limit_pct",
    "close_on_reversal", "reversal_min_score", "reversal_min_hold_minutes",
    "block_rollover", "recommended_version", "atr_timeframe",
    "telegram_bot_token", "telegram_chat_id", "telegram_enabled",
    "telegram_on_open", "telegram_on_close", "telegram_on_target",
    "telegram_on_stoploss", "telegram_on_analysis",
}

def _persistent_config_snapshot() -> dict:
    return {k: CONFIG[k] for k in PERSISTENT_CONFIG_KEYS if k in CONFIG}

def save_persistent_config() -> bool:
    try:
        payload = _persistent_config_snapshot()
        tmp_file = CONFIG_FILE + ".tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json_module.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp_file, CONFIG_FILE)
        try:
            os.chmod(CONFIG_FILE, 0o600)
        except OSError:
            pass
        logger.info("Configuration sauvegardee dans %s", CONFIG_FILE)
        return True
    except Exception as e:
        logger.exception("Erreur sauvegarde configuration: %s", e)
        return False

RECOMMENDED_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.recommended.json")
# Reglages propres a l'installation, jamais ecrases par la configuration recommandee.
RECOMMENDED_EXCLUDED_KEYS = {
    "telegram_bot_token", "telegram_chat_id", "telegram_enabled",
    "telegram_on_open", "telegram_on_close", "telegram_on_target",
    "telegram_on_stoploss", "telegram_on_analysis",
    "ollama_url", "ollama_model", "recommended_version",
}


def load_persistent_config() -> bool:
    loaded = False
    if not os.path.exists(CONFIG_FILE):
        logger.info("Aucun config.json: utilisation des valeurs par defaut")
    else:
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json_module.load(f)
            if not isinstance(saved, dict):
                logger.warning("config.json invalide: objet JSON attendu")
            else:
                _apply_config_values(saved)
                logger.info("Configuration chargee depuis %s", CONFIG_FILE)
                loaded = True
        except Exception as e:
            logger.exception("Erreur chargement configuration: %s", e)

    apply_recommended_config(force=False)
    return loaded


def load_recommended_config() -> Optional[dict]:
    if not os.path.exists(RECOMMENDED_CONFIG_FILE):
        return None
    try:
        with open(RECOMMENDED_CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json_module.load(f)
        if not isinstance(data, dict) or not isinstance(data.get("values"), dict):
            raise ValueError("objet {version, values} attendu")
        return data
    except Exception as e:
        logger.exception("config.recommended.json invalide: %s", e)
        return None


def apply_recommended_config(force: bool = False) -> dict:
    """Applique config.recommended.json si sa version est plus recente (ou si force).

    Les reglages modifies ensuite depuis l'interface restent en place jusqu'a
    la prochaine version du fichier recommande.
    """
    data = load_recommended_config()
    if data is None:
        return {"applied": False, "reason": "config.recommended.json absent ou invalide"}
    version = int(data.get("version", 0))
    current = int(CONFIG.get("recommended_version", 0) or 0)
    if not force and current >= version:
        return {"applied": False, "version": version, "reason": "deja appliquee"}

    values = {
        k: v for k, v in data["values"].items()
        if k in PERSISTENT_CONFIG_KEYS and k not in RECOMMENDED_EXCLUDED_KEYS
    }
    _apply_config_values(values)
    CONFIG["recommended_version"] = version
    saved = save_persistent_config()
    logger.info(
        "CONFIG RECOMMANDEE v%s APPLIQUEE (%s parametres, persistent=%s)",
        version, len(values), saved,
    )
    return {"applied": True, "version": version, "updated_fields": sorted(values), "persistent": saved}


def _apply_config_values(saved: dict) -> None:
    """Copie des valeurs (config.json ou recommandee) dans CONFIG avec typage et bornes."""
    for key in PERSISTENT_CONFIG_KEYS:
        if key not in saved:
            continue
        value = saved[key]

        if key == "impacts" and isinstance(value, dict):
            merged = CONFIG.get("impacts", {}).copy()
            merged.update(value)
            CONFIG[key] = merged
        elif key in {"lot_size", "target_profit", "stop_loss"}:
            CONFIG[key] = float(value)
        elif key in {
            "max_positions", "min_score", "check_interval",
            "rsi_period", "rsi_overbought", "rsi_oversold",
            "sma_fast", "sma_slow", "ai_validation_min_score",
            "ai_auto_cooldown", "ai_quick_num_predict", "ai_quick_timeout",
            "ai_full_num_predict", "ai_full_timeout", "hard_max_positions",
            "max_currency_exposure", "trade_cooldown_minutes", "max_spread_points",
            "m15_min_bars", "m5_min_bars", "atr_period",
            "news_after_minutes", "reversal_min_score", "reversal_min_hold_minutes"
        }:
            CONFIG[key] = int(value)
            if key == "hard_max_positions":
                CONFIG[key] = max(1, min(CONFIG[key], 20))
            elif key == "max_positions":
                CONFIG[key] = max(1, min(CONFIG[key], int(CONFIG.get("hard_max_positions", HARD_MAX_POSITIONS))))
        elif key in {
            "atr_sl_multiplier", "atr_tp_multiplier", "breakeven_trigger_r",
            "trailing_trigger_r", "trailing_atr_multiplier",
            "max_drawdown_new_trades_pct", "auto_stop_drawdown_pct",
            "risk_per_trade_pct", "daily_loss_limit_pct", "max_spread_sl_pct"
        }:
            CONFIG[key] = float(value)
        elif key in {
            "use_ai_analysis", "telegram_enabled",
            "broker_sl_required", "close_on_reversal", "block_rollover",
            "telegram_on_open", "telegram_on_close",
            "telegram_on_target", "telegram_on_stoploss",
            "telegram_on_analysis"
        }:
            CONFIG[key] = bool(value)
        else:
            CONFIG[key] = value

    # Normalisation finale apres chargement.
    CONFIG["hard_max_positions"] = max(1, min(int(CONFIG.get("hard_max_positions", HARD_MAX_POSITIONS)), 20))
    CONFIG["max_positions"] = max(1, min(int(CONFIG.get("max_positions", 1)), CONFIG["hard_max_positions"]))
    # Le SL broker n'est plus desactivable : sans SL, une seule position peut
    # effacer des dizaines de gains (cause des pertes constatees en demo).
    CONFIG["broker_sl_required"] = True

load_persistent_config()

# Lecture dynamique des reglages avances. Les constantes globales ci-dessus
# servent de valeurs par defaut/fallback afin de rester compatibles.
def runtime_cfg(name: str, default):
    return CONFIG.get(name, default)


# ============ AUTHENTIFICATION / ROLES ============
AUTH_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.json")
SESSION_TTL_SECONDS = 8 * 60 * 60
AUTH_USERS = {}
AUTH_SESSIONS = {}

def _hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    import hashlib, secrets
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200000)
    return salt.hex(), digest.hex()

def load_auth_users() -> bool:
    global AUTH_USERS
    if not os.path.exists(AUTH_FILE):
        AUTH_USERS = {}
        logger.warning("Aucun users.json: authentification a initialiser")
        return False
    try:
        with open(AUTH_FILE, "r", encoding="utf-8") as f:
            data = json_module.load(f)
        AUTH_USERS = data if isinstance(data, dict) else {}
        return bool(AUTH_USERS)
    except Exception as e:
        logger.exception("Erreur chargement users.json: %s", e)
        AUTH_USERS = {}
        return False

def save_auth_users() -> bool:
    try:
        tmp = AUTH_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json_module.dump(AUTH_USERS, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, AUTH_FILE)
        try:
            os.chmod(AUTH_FILE, 0o600)
        except OSError:
            pass
        return True
    except Exception as e:
        logger.exception("Erreur sauvegarde users.json: %s", e)
        return False

def verify_password(username: str, password: str) -> bool:
    import hashlib, hmac
    user = AUTH_USERS.get(username)
    if not isinstance(user, dict):
        return False
    try:
        salt = bytes.fromhex(user["salt"])
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200000).hex()
        return hmac.compare_digest(digest, user["password_hash"])
    except Exception:
        return False

def create_user(username: str, password: str, role: str) -> None:
    salt_hex, hash_hex = _hash_password(password)
    AUTH_USERS[username] = {
        "salt": salt_hex,
        "password_hash": hash_hex,
        "role": role,
    }

def create_session(username: str, role: str) -> str:
    import secrets
    token = secrets.token_urlsafe(32)
    AUTH_SESSIONS[token] = {
        "username": username,
        "role": role,
        "expires_at": time.time() + SESSION_TTL_SECONDS,
    }
    return token

def current_user(request: Request):
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    token = auth[7:].strip()
    session = AUTH_SESSIONS.get(token)
    if not session:
        return None
    if session["expires_at"] <= time.time():
        AUTH_SESSIONS.pop(token, None)
        return None
    return session

load_auth_users()

class LoginRequest(BaseModel):
    username: str
    password: str

# ============ ETAT GLOBAL ============
state = {
    "connected": False,
    "auto_trading": False,
    "daily_pnl": 0,
    "trades_today": 0,
    "analyses": {},
    "current_position": None,
    "auto_thread": None,
    "last_bridge_error": None,
    "last_trade_time": {},
    "risk_block_reason": None,
    "session_start_balance": None,
}

# ============ FASTAPI ============
app = FastAPI(title="Forex Robot Pro - Ubuntu Edition v7 Risk Engine")
api_router = APIRouter(prefix="/api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ API AUTH MIDDLEWARE ============
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    path = request.url.path

    public = {
        "/api/auth/login",
        "/api/auth/setup",
        "/api/auth/status",
    }

    if path.startswith("/api/") and path not in public:
        user = current_user(request)

        if path == "/api/auth/logout":
            if not user:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Authentification requise"},
                )
        else:
            if not user:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Authentification requise"},
                )

            if request.method not in {"GET", "HEAD", "OPTIONS"} and user.get("role") != "admin":
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Privileges administrateur requis"},
                )

    return await call_next(request)

class ConnectRequest(BaseModel):
    account_number: str
    password: str
    server: str

class TradeRequest(BaseModel):
    symbol: str
    type: str
    volume: float
    sl: Optional[float] = None
    tp: Optional[float] = None
    comment: str = ""

# ============ INDICATEURS TECHNIQUES ============
def calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
    if len(prices) < period:
        return np.array([])
    return np.convolve(prices, np.ones(period)/period, mode='valid')

def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    if len(prices) < period:
        return np.array([])
    ema = np.zeros(len(prices))
    multiplier = 2 / (period + 1)
    ema[period-1] = np.mean(prices[:period])
    for i in range(period, len(prices)):
        ema[i] = (prices[i] * multiplier) + (ema[i-1] * (1 - multiplier))
    return ema[period-1:]

def calculate_rsi(prices: np.ndarray, period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices: np.ndarray) -> Dict:
    if len(prices) < 26:
        return {"macd": 0, "signal": 0, "histogram": 0}
    ema12 = calculate_ema(prices, 12)
    ema26 = calculate_ema(prices, 26)
    min_len = min(len(ema12), len(ema26))
    ema12 = ema12[-min_len:]
    ema26 = ema26[-min_len:]
    macd_line = ema12 - ema26
    if len(macd_line) >= 9:
        signal_line = calculate_ema(macd_line, 9)
        histogram = macd_line[-1] - signal_line[-1] if len(signal_line) > 0 else 0
    else:
        signal_line = macd_line
        histogram = 0
    return {
        "macd": macd_line[-1] if len(macd_line) > 0 else 0,
        "signal": signal_line[-1] if len(signal_line) > 0 else 0,
        "histogram": histogram
    }

def calculate_trend(prices: np.ndarray) -> Dict:
    if len(prices) < 30:
        return {"short": "neutral", "medium": "neutral", "long": "neutral", "strength": 0, "long_change": 0}
    short_change = (prices[-1] - prices[-5]) / prices[-5] * 100
    medium_change = (prices[-1] - prices[-15]) / prices[-15] * 100
    long_change = (prices[-1] - prices[-30]) / prices[-30] * 100

    def get_trend(change):
        if change > 0.5:
            return "bullish"
        elif change < -0.5:
            return "bearish"
        return "neutral"

    strength = min(100, abs(long_change) * 20)
    return {
        "short": get_trend(short_change),
        "medium": get_trend(medium_change),
        "long": get_trend(long_change),
        "strength": strength,
        "long_change": round(long_change, 2)
    }

def find_support_resistance(prices: np.ndarray, window: int = 20) -> Dict:
    if len(prices) < window:
        return {"support": prices[-1] * 0.99, "resistance": prices[-1] * 1.01}
    supports = []
    resistances = []
    for i in range(window, len(prices) - window):
        if prices[i] == min(prices[i-window:i+window]):
            supports.append(prices[i])
        if prices[i] == max(prices[i-window:i+window]):
            resistances.append(prices[i])
    current_price = prices[-1]
    valid_supports = [s for s in supports if s < current_price]
    support = max(valid_supports) if valid_supports else current_price * 0.995
    valid_resistances = [r for r in resistances if r > current_price]
    resistance = min(valid_resistances) if valid_resistances else current_price * 1.005
    return {"support": support, "resistance": resistance}

def get_historical_rates(symbol: str, timeframe, bars: int = 200):
    """Retourne les bougies OHLC demandees depuis MT5."""
    if not state["connected"]:
        return None
    try:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return None
        if not symbol_info.visible:
            mt5.symbol_select(symbol, True)
            time.sleep(0.05)
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, int(bars))
        if rates is None or len(rates) < 2:
            return None
        return rates
    except Exception as e:
        logger.debug("Erreur OHLC %s timeframe=%s: %s", symbol, timeframe, e)
        return None


def calculate_atr_from_rates(rates, period: int = runtime_cfg("atr_period", ATR_PERIOD)) -> float:
    """Calcule l'ATR Wilder simplifie a partir des bougies OHLC."""
    if rates is None or len(rates) < period + 1:
        return 0.0
    highs = np.asarray([float(r["high"]) for r in rates], dtype=float)
    lows = np.asarray([float(r["low"]) for r in rates], dtype=float)
    closes = np.asarray([float(r["close"]) for r in rates], dtype=float)
    prev_close = closes[:-1]
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(np.abs(highs[1:] - prev_close), np.abs(lows[1:] - prev_close)),
    )
    if len(tr) < period:
        return 0.0
    return float(np.mean(tr[-period:]))


def get_symbol_atr(symbol: str) -> float:
    """ATR sur l'unite de temps configuree, base unique des SL, TP et trailing."""
    timeframes = {"M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4}
    timeframe = timeframes.get(str(runtime_cfg("atr_timeframe", ATR_TIMEFRAME)).upper(), mt5.TIMEFRAME_M15)
    period = int(runtime_cfg("atr_period", ATR_PERIOD))
    return calculate_atr_from_rates(get_historical_rates(symbol, timeframe, period + 40), period)


def trend_from_closes(prices: np.ndarray) -> str:
    """Direction simple basee sur EMA20/EMA50 + variation recente."""
    if len(prices) < 50:
        return "neutral"
    ema20 = calculate_ema(prices, 20)
    ema50 = calculate_ema(prices, 50)
    if len(ema20) == 0 or len(ema50) == 0:
        return "neutral"
    recent = (prices[-1] - prices[-10]) / prices[-10] * 100
    if ema20[-1] > ema50[-1] and recent > 0:
        return "bullish"
    if ema20[-1] < ema50[-1] and recent < 0:
        return "bearish"
    return "neutral"


def get_multitimeframe_filter(symbol: str, expected_action: str) -> Dict:
    """Valide la direction sur M15/M5 pour les seuls candidats proches d'une entree."""
    result = {
        "allowed": True,
        "m15": "neutral",
        "m5": "neutral",
        "atr": 0.0,
        "spread_points": None,
        "reason": "",
    }
    try:
        m15 = get_historical_rates(symbol, mt5.TIMEFRAME_M15, runtime_cfg("m15_min_bars", M15_MIN_BARS))
        m5 = get_historical_rates(symbol, mt5.TIMEFRAME_M5, runtime_cfg("m5_min_bars", M5_MIN_BARS))
        if m15 is None or len(m15) < runtime_cfg("m15_min_bars", M15_MIN_BARS) or m5 is None or len(m5) < runtime_cfg("m5_min_bars", M5_MIN_BARS):
            result["allowed"] = False
            result["reason"] = "Donnees M15/M5 insuffisantes"
            return result

        m15_close = np.asarray([float(r["close"]) for r in m15], dtype=float)
        m5_close = np.asarray([float(r["close"]) for r in m5], dtype=float)
        result["m15"] = trend_from_closes(m15_close)
        result["m5"] = trend_from_closes(m5_close)
        result["atr"] = get_symbol_atr(symbol)

        tick = mt5.symbol_info_tick(symbol)
        info = mt5.symbol_info(symbol)
        if tick is None or info is None:
            result["allowed"] = False
            result["reason"] = "Prix/spread indisponible"
            return result

        point = float(getattr(info, "point", 0) or 0)
        if point > 0:
            result["spread_points"] = round((float(tick.ask) - float(tick.bid)) / point, 2)
            if result["spread_points"] > runtime_cfg("max_spread_points", MAX_SPREAD_POINTS):
                result["allowed"] = False
                result["reason"] = f"Spread trop eleve ({result['spread_points']:.1f} > {runtime_cfg("max_spread_points", MAX_SPREAD_POINTS)})"
                return result
            # Un spread de 3 pips est acceptable avec un SL de 30 pips, pas avec un SL de 10.
            sl_distance = result["atr"] * runtime_cfg("atr_sl_multiplier", ATR_SL_MULTIPLIER)
            max_pct = float(runtime_cfg("max_spread_sl_pct", MAX_SPREAD_SL_PCT))
            if sl_distance > 0 and max_pct > 0:
                spread_pct = (float(tick.ask) - float(tick.bid)) / sl_distance * 100.0
                result["spread_sl_pct"] = round(spread_pct, 1)
                if spread_pct > max_pct:
                    result["allowed"] = False
                    result["reason"] = f"Spread trop cher ({spread_pct:.0f}% du SL > {max_pct:g}%)"
                    return result

        wanted = "bullish" if expected_action == "BUY" else "bearish"
        if result["m15"] != wanted or result["m5"] != wanted:
            result["allowed"] = False
            result["reason"] = f"MTF divergent M15={result['m15']} M5={result['m5']}"
            return result

        result["reason"] = "MTF + spread OK"
        return result
    except Exception as e:
        result["allowed"] = False
        result["reason"] = f"Erreur filtre MTF: {e}"
        return result


def calculate_atr_sl_tp(symbol_info, tick, action: str, volume: float, atr: float):
    """Construit des SL/TP broker reels a partir de l'ATR."""
    if atr <= 0:
        return None, None

    digits = int(getattr(symbol_info, "digits", 5))
    point = float(getattr(symbol_info, "point", 0) or 0)
    stops_level = float(getattr(symbol_info, "trade_stops_level", 0) or 0) * point
    min_distance = max(stops_level, point * 10)
    sl_distance = max(atr * runtime_cfg("atr_sl_multiplier", ATR_SL_MULTIPLIER), min_distance)
    tp_distance = max(atr * runtime_cfg("atr_tp_multiplier", ATR_TP_MULTIPLIER), min_distance * 1.2)

    if action == "BUY":
        entry = float(tick.ask)
        sl = entry - sl_distance
        tp = entry + tp_distance
    else:
        entry = float(tick.bid)
        sl = entry + sl_distance
        tp = entry - tp_distance

    return round(sl, digits), round(tp, digits)


def currency_exposure_allows(symbol: str, action: str, exposure: Dict) -> tuple[bool, str]:
    """Bloque une nouvelle entree lorsque l'exposition nette d'une devise devient excessive."""
    if len(symbol) < 6:
        return False, "Symbole invalide"
    base, quote = symbol[:3], symbol[3:6]
    direction = 1 if action == "BUY" else -1
    base_after = float(exposure.get(base, 0)) + direction
    quote_after = float(exposure.get(quote, 0)) - direction
    if abs(base_after) > runtime_cfg("max_currency_exposure", MAX_CURRENCY_EXPOSURE):
        return False, f"Exposition {base} limite ({base_after:g})"
    if abs(quote_after) > runtime_cfg("max_currency_exposure", MAX_CURRENCY_EXPOSURE):
        return False, f"Exposition {quote} limite ({quote_after:g})"
    return True, "Exposition devises OK"


def get_current_drawdown_pct(account) -> float:
    """Drawdown instantane par rapport au balance du compte."""
    if account is None or float(account.balance) <= 0:
        return 0.0
    return max(0.0, (float(account.balance) - float(account.equity)) / float(account.balance) * 100.0)


def trade_cooldown_allows(symbol: str) -> tuple[bool, str]:
    last = state.get("last_trade_time", {}).get(symbol)
    if last is None:
        return True, "Cooldown OK"
    elapsed = time.monotonic() - float(last)
    remaining = runtime_cfg("trade_cooldown_minutes", TRADE_COOLDOWN_MINUTES) * 60 - elapsed
    if remaining > 0:
        return False, f"Cooldown {symbol} ({int(remaining)}s restantes)"
    return True, "Cooldown OK"


def calculate_risk_volume(symbol_info, entry: float, sl: float, account) -> Optional[float]:
    """Volume pour risquer risk_per_trade_pct du capital entre entree et SL.

    Arrondi a l'inferieur sur le pas du broker afin de ne jamais depasser le
    risque vise, et plafonne par lot_size. None si meme le lot minimum
    risquerait davantage que prevu.
    """
    if account is None:
        return None
    capital = min(float(account.balance), float(account.equity))
    risk_money = capital * float(runtime_cfg("risk_per_trade_pct", RISK_PER_TRADE_PCT)) / 100.0
    tick_size = float(getattr(symbol_info, "trade_tick_size", 0) or 0)
    tick_value = float(getattr(symbol_info, "trade_tick_value", 0) or 0)
    distance = abs(float(entry) - float(sl))
    if risk_money <= 0 or tick_size <= 0 or tick_value <= 0 or distance <= 0:
        return None

    loss_per_lot = distance / tick_size * tick_value
    minimum = float(getattr(symbol_info, "volume_min", 0.01) or 0.01)
    maximum = float(getattr(symbol_info, "volume_max", 100.0) or 100.0)
    step = float(getattr(symbol_info, "volume_step", 0.01) or 0.01)
    cap = min(maximum, float(CONFIG.get("lot_size", maximum)))

    raw = min(risk_money / loss_per_lot, cap)
    volume = round(int(raw / step + 1e-9) * step, 8)
    if volume < minimum:
        logger.warning(
            "Volume refuse %s: lot min %.2f risquerait %.2f > %.2f (%.2f%%)",
            symbol_info.name, minimum, minimum * loss_per_lot, risk_money,
            float(runtime_cfg("risk_per_trade_pct", RISK_PER_TRADE_PCT)),
        )
        return None
    return volume


def server_time_now() -> Optional[datetime]:
    """Heure du serveur broker (les timestamps MT5 sont exprimes en heure serveur)."""
    try:
        tick = mt5.symbol_info_tick("EURUSD")
        if tick is None or not getattr(tick, "time", 0):
            return None
        return datetime.fromtimestamp(int(tick.time), tz=timezone.utc).replace(tzinfo=None)
    except Exception:
        return None


def trading_session_allows() -> tuple[bool, str]:
    """Bloque les entrees autour du rollover (spreads larges) et le vendredi soir."""
    if not runtime_cfg("block_rollover", BLOCK_ROLLOVER):
        return True, "Session OK"
    now = server_time_now()
    if now is None:
        return True, "Heure serveur indisponible"
    if now.hour in (23, 0):
        return False, f"Rollover ({now:%H:%M} serveur)"
    if now.weekday() == 4 and now.hour >= 21:
        return False, f"Vendredi soir ({now:%H:%M} serveur)"
    return True, "Session OK"


def daily_loss_allows(account) -> tuple[bool, str]:
    """Bloque les nouvelles entrees si la perte du jour depasse la limite."""
    limit = float(runtime_cfg("daily_loss_limit_pct", DAILY_LOSS_LIMIT_PCT))
    if limit <= 0 or account is None:
        return True, "Perte journaliere OK"
    today = (server_time_now() or datetime.now(timezone.utc)).date()
    if state.get("day") != today or not state.get("day_start_equity"):
        state["day"] = today
        state["day_start_equity"] = float(account.equity)
    start = float(state["day_start_equity"])
    loss_pct = (start - float(account.equity)) / start * 100.0 if start > 0 else 0.0
    if loss_pct >= limit:
        return False, f"Perte du jour {loss_pct:.2f}% >= {limit}%"
    return True, "Perte journaliere OK"


def emergency_close_all(reason: str) -> int:
    """Ferme toutes les positions du robot et arrete l'auto-trading."""
    closed = 0
    for pos in list(mt5.positions_get() or []):
        if int(getattr(pos, "magic", 0) or 0) != ROBOT_MAGIC:
            continue
        if close_position(int(pos.ticket)):
            closed += 1
    state["auto_trading"] = False
    state["risk_block_reason"] = f"ARRET D'URGENCE: {reason}"
    logger.error("ARRET D'URGENCE | %s | %s position(s) fermee(s)", reason, closed)
    send_telegram(f"<b>ARRET D'URGENCE</b>\n{reason}\n{closed} position(s) fermee(s), auto-trading arrete.")
    return closed


def modify_position_sl_tp(position, sl: Optional[float] = None, tp: Optional[float] = None) -> bool:
    """Modifie le SL/TP d'une position MT5 sans la fermer."""
    try:
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": position.symbol,
            "position": int(position.ticket),
            "sl": float(sl if sl is not None else getattr(position, "sl", 0) or 0),
            "tp": float(tp if tp is not None else getattr(position, "tp", 0) or 0),
            "magic": ROBOT_MAGIC,
        }
        result = mt5.order_send(request)
        if result is None:
            logger.warning("SLTP None ticket=%s error=%s", position.ticket, mt5.last_error())
            return False
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.warning("SLTP refuse ticket=%s retcode=%s comment=%s", position.ticket, result.retcode, result.comment)
            return False
        return True
    except Exception as e:
        logger.warning("Erreur modification SLTP ticket=%s: %s", getattr(position, "ticket", "?"), e)
        return False


def manage_position_protection(position):
    """Break-even + trailing stop apres entree, sans deplacer un SL dans le mauvais sens."""
    try:
        symbol = str(position.symbol)
        action = "BUY" if int(position.type) == 0 else "SELL"
        atr = get_symbol_atr(symbol)
        if atr <= 0:
            return

        entry = float(position.price_open)
        current_sl = float(getattr(position, "sl", 0) or 0)
        current_tp = float(getattr(position, "tp", 0) or 0)
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return
        current = float(tick.bid if action == "BUY" else tick.ask)
        initial_r = abs(entry - current_sl) if current_sl > 0 else atr * runtime_cfg("atr_sl_multiplier", ATR_SL_MULTIPLIER)
        if initial_r <= 0:
            return
        favorable = (current - entry) if action == "BUY" else (entry - current)
        r_multiple = favorable / initial_r

        new_sl = None
        if r_multiple >= runtime_cfg("trailing_trigger_r", TRAILING_TRIGGER_R):
            candidate = current - atr * runtime_cfg("trailing_atr_multiplier", TRAILING_ATR_MULTIPLIER) if action == "BUY" else current + atr * runtime_cfg("trailing_atr_multiplier", TRAILING_ATR_MULTIPLIER)
            if action == "BUY" and (current_sl <= 0 or candidate > current_sl) and candidate < current:
                new_sl = candidate
            elif action == "SELL" and (current_sl <= 0 or candidate < current_sl) and candidate > current:
                new_sl = candidate
        elif r_multiple >= runtime_cfg("breakeven_trigger_r", BREAKEVEN_TRIGGER_R):
            if action == "BUY" and (current_sl <= 0 or current_sl < entry):
                new_sl = entry
            elif action == "SELL" and (current_sl <= 0 or current_sl > entry):
                new_sl = entry

        if new_sl is not None:
            info = mt5.symbol_info(symbol)
            digits = int(getattr(info, "digits", 5)) if info else 5
            new_sl = round(new_sl, digits)
            if modify_position_sl_tp(position, sl=new_sl, tp=current_tp):
                logger.info("PROTECTION %s ticket=%s | R=%.2f | SL=%s", symbol, position.ticket, r_multiple, new_sl)
    except Exception as e:
        logger.debug("Protection position %s: %s", getattr(position, "ticket", "?"), e)

def analyze_symbol_score(symbol: str) -> Dict:
    prices = get_historical_prices(symbol, 30)

    if len(prices) < 30:
        return {"symbol": symbol, "score": 0, "action": "NONE", "confidence": 0, "error": "Pas assez de donnees"}

    current_price = prices[-1]
    score = 0
    reasons = []

    trend = calculate_trend(prices)
    if trend["long"] == "bullish" and trend["medium"] == "bullish":
        score += 3
        reasons.append(f"Forte tendance haussiere ({trend['long_change']}%)")
    elif trend["long"] == "bearish" and trend["medium"] == "bearish":
        score -= 3
        reasons.append(f"Forte tendance baissiere ({trend['long_change']}%)")
    elif trend["long"] == "bullish":
        score += 1
    elif trend["long"] == "bearish":
        score -= 1

    rsi = calculate_rsi(prices, CONFIG["rsi_period"])
    if rsi < 25:
        score += 3
        reasons.append(f"RSI tres survendu ({rsi:.1f})")
    elif rsi < CONFIG["rsi_oversold"]:
        score += 2
        reasons.append(f"RSI survendu ({rsi:.1f})")
    elif rsi > 75:
        score -= 3
        reasons.append(f"RSI tres surachete ({rsi:.1f})")
    elif rsi > CONFIG["rsi_overbought"]:
        score -= 2
        reasons.append(f"RSI surachete ({rsi:.1f})")

    macd = calculate_macd(prices)
    if macd["histogram"] > 0 and macd["macd"] > macd["signal"]:
        score += 2
        reasons.append("MACD haussier")
    elif macd["histogram"] < 0 and macd["macd"] < macd["signal"]:
        score -= 2
        reasons.append("MACD baissier")

    sma_fast = calculate_sma(prices, CONFIG["sma_fast"])
    sma_slow = calculate_sma(prices, CONFIG["sma_slow"])
    if len(sma_fast) > 0 and len(sma_slow) > 0:
        if sma_fast[-1] > sma_slow[-1]:
            score += 1
        else:
            score -= 1

    sr = find_support_resistance(prices)
    distance_to_support = (current_price - sr["support"]) / current_price * 100
    distance_to_resistance = (sr["resistance"] - current_price) / current_price * 100

    if distance_to_support < 0.5:
        score += 2
        reasons.append("Proche du support")
    if distance_to_resistance < 0.5:
        score -= 2
        reasons.append("Proche de la resistance")

    if score >= 4:
        action = "STRONG_BUY"
        confidence = min(95, 60 + abs(score) * 5)
    elif score >= 2:
        action = "BUY"
        confidence = min(80, 50 + abs(score) * 5)
    elif score <= -4:
        action = "STRONG_SELL"
        confidence = min(95, 60 + abs(score) * 5)
    elif score <= -2:
        action = "SELL"
        confidence = min(80, 50 + abs(score) * 5)
    else:
        action = "NEUTRAL"
        confidence = 30

    return {
        "symbol": symbol,
        "score": score,
        "abs_score": abs(score),
        "action": action,
        "confidence": confidence,
        "current_price": float(current_price),
        "rsi": float(rsi),
        "trend": trend["long"],
        "trend_change": trend["long_change"],
        "reasons": reasons,
    }

# ============ FONCTIONS MT5 (via mt5linux bridge) ============
def connect_mt5(account: int = None, password: str = None, server: str = None) -> bool:
    """
    Verifie la connexion au terminal MT5 via le bridge mt5linux.

    Les parametres account/password/server sont conserves pour compatibilite
    avec l'ancienne interface HTML. Le compte actif est celui du terminal MT5
    cote Wine.
    """
    global mt5

    try:
        # Initialisation du client mt5linux/RPyC.
        initialized = mt5.initialize()

        # Si le client RPyC est stale ou si l'initialisation echoue,
        # recreer le client et retenter une seule fois.
        if not initialized:
            logger.warning(
                "Initialisation MT5 echouee, reconnexion au bridge..."
            )

            mt5 = MetaTrader5(
                host=MT5_BRIDGE_HOST,
                port=MT5_BRIDGE_PORT,
                timeout=MT5_BRIDGE_TIMEOUT,
            )

            initialized = mt5.initialize()

            if not initialized:
                state["connected"] = False
                state["last_bridge_error"] = str(mt5.last_error())

                logger.error(
                    "Impossible d'initialiser MT5 via le bridge: %s",
                    mt5.last_error(),
                )
                return False

        version = mt5.version()

        if not version or version[0] == 0:
            state["connected"] = False
            state["last_bridge_error"] = str(mt5.last_error())

            logger.error(
                "Bridge MT5 connecte mais MT5 non initialise: %s",
                mt5.last_error(),
            )
            return False

        terminal = mt5.terminal_info()
        info = mt5.account_info()

        if terminal is None or info is None:
            state["connected"] = False
            state["last_bridge_error"] = str(mt5.last_error())

            logger.error(
                "Terminal/compte MT5 inaccessible: %s",
                mt5.last_error(),
            )
            return False

        state["connected"] = True
        state["last_bridge_error"] = None

        CONFIG["account"] = int(info.login)
        CONFIG["server"] = info.server

        logger.info(
            "MT5 bridge OK | version=%s | compte=%s | serveur=%s | "
            "balance=%.2f | equity=%.2f | trade_allowed=%s",
            version,
            info.login,
            info.server,
            info.balance,
            info.equity,
            terminal.trade_allowed,
        )

        return True

    except Exception as e:
        state["connected"] = False
        state["last_bridge_error"] = str(e)

        logger.exception("Erreur connexion MT5 via bridge")
        return False


def ensure_mt5_connection() -> bool:
    """Verifie rapidement que le terminal distant est toujours disponible."""
    try:
        if not state["connected"]:
            return connect_mt5()
        version = mt5.version()
        account = mt5.account_info()
        if not version or version[0] == 0 or account is None:
            state["connected"] = False
            return connect_mt5()
        return True
    except Exception as e:
        state["connected"] = False
        state["last_bridge_error"] = str(e)
        logger.warning("Connexion MT5 perdue: %s", e)
        return False


def normalize_volume(symbol_info, requested: float) -> float:
    """Adapte le volume aux contraintes du broker."""
    minimum = float(getattr(symbol_info, "volume_min", 0.01) or 0.01)
    maximum = float(getattr(symbol_info, "volume_max", requested) or requested)
    step = float(getattr(symbol_info, "volume_step", 0.01) or 0.01)
    volume = max(minimum, min(maximum, float(requested)))
    steps = round(volume / step)
    volume = steps * step
    return round(max(minimum, min(maximum, volume)), 8)


def get_historical_prices(symbol: str, days: int = 30) -> np.ndarray:
    if not state["connected"]:
        return np.array([])
    try:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return np.array([])

        if not symbol_info.visible:
            mt5.symbol_select(symbol, True)
            time.sleep(0.1)

        utc_to = datetime.now(timezone.utc)
        utc_from = utc_to - timedelta(days=days)

        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, utc_from, utc_to)

        if rates is None or len(rates) == 0:
            return np.array([])

        return np.array([float(rate['close']) for rate in rates])
    except Exception as e:
        logger.warning(f"Erreur prix historiques {symbol}: {e}")
        return np.array([])

def get_available_symbols() -> List[str]:
    available = []
    for symbol in CONFIG["symbols"]:
        try:
            info = mt5.symbol_info(symbol)
            if info is not None:
                available.append(symbol)
        except:
            continue
    return available

def get_economic_calendar() -> list:
    try:
        response = requests.get(
            "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
            timeout=10,
            verify=True
        )
        if response.status_code == 200:
            events = response.json()
            now = datetime.now(timezone.utc)
            upcoming = []
            for event in events:
                try:
                    event_time = datetime.strptime(event.get('date', ''), '%Y-%m-%dT%H:%M:%S%z')
                    time_diff = (event_time - now).total_seconds() / 60
                    if -runtime_cfg("news_after_minutes", NEWS_AFTER_MINUTES) <= time_diff <= 120:
                        upcoming.append({
                            "title": event.get('title', ''),
                            "country": event.get('country', ''),
                            "impact": event.get('impact', 'Low'),
                            "minutes_until": int(time_diff)
                        })
                except:
                    continue
            return upcoming
    except Exception as e:
        logger.warning(f"Erreur calendrier economique: {e}")
    return []

def fetch_full_calendar() -> list:
    # Source 1: faireconomy
    try:
        response = requests.get(
            "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
            timeout=10, verify=False
        )
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                logger.info(f"Calendrier charge: {len(data)} evenements")
                return data
    except Exception as e:
        logger.warning(f"Erreur faireconomy: {e}")

    # Source 2: JBlanked
    try:
        response = requests.get(
            "https://www.jblanked.com/news/api/forex/calendar/week/",
            timeout=10, verify=False
        )
        if response.status_code == 200:
            data = response.json()
            events = []
            for event in data:
                events.append({
                    "title": event.get("title", event.get("event", "Unknown")),
                    "country": event.get("country", ""),
                    "impact": event.get("impact", "Medium"),
                    "date": event.get("date", ""),
                    "forecast": event.get("forecast", ""),
                    "previous": event.get("previous", ""),
                })
            if events:
                return events
    except Exception as e:
        logger.warning(f"Erreur JBlanked: {e}")
    return []

# ============ TELEGRAM ALERTS ============
def send_telegram(message: str, silent: bool = False) -> bool:
    """Envoie un message Telegram"""
    if not CONFIG.get("telegram_enabled"):
        return False
    token = CONFIG.get("telegram_bot_token", "")
    chat_id = CONFIG.get("telegram_chat_id", "")
    if not token or not chat_id:
        return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML",
            "disable_notification": silent,
        }
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            logger.info("Telegram: message envoye")
            return True
        else:
            logger.warning(f"Telegram erreur: {resp.status_code} {resp.text}")
            return False
    except Exception as e:
        logger.warning(f"Telegram erreur: {e}")
        return False

def open_trade(symbol: str, action: str, comment: str = "", requested_volume: float = None, sl_price: Optional[float] = None, tp_price: Optional[float] = None, atr: float = 0.0) -> Optional[int]:
    if not ensure_mt5_connection():
        return None
    try:
        positions = mt5.positions_get()
        max_allowed = min(int(CONFIG.get("max_positions", 1)), runtime_cfg("hard_max_positions", HARD_MAX_POSITIONS))
        if positions and len(positions) >= max_allowed:
            logger.warning("Position deja ouverte - Max %s autorise", max_allowed)
            return None

        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            logger.warning("Symbole %s non trouve", symbol)
            return None

        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                logger.warning("Impossible de selectionner %s", symbol)
                return None
            time.sleep(0.1)

        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            logger.warning("Pas de prix pour %s", symbol)
            return None

        action = action.upper()
        if action == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
        elif action == "SELL":
            order_type = mt5.ORDER_TYPE_SELL
        else:
            logger.warning("Action invalide: %s", action)
            return None

        price = float(tick.ask if action == "BUY" else tick.bid)

        # SL broker TOUJOURS obligatoire, calcule sur l'ATR si absent.
        if atr <= 0:
            atr = get_symbol_atr(symbol)
        calc_sl, calc_tp = calculate_atr_sl_tp(symbol_info, tick, action, 0.0, atr)
        if sl_price is None:
            sl_price = calc_sl
        if tp_price is None:
            tp_price = calc_tp
        if sl_price is None or float(sl_price) <= 0:
            logger.error("OUVERTURE BLOQUEE %s: impossible de calculer un SL broker", symbol)
            return None

        # Volume : impose par l'appelant (trade manuel) ou calcule selon le risque.
        if requested_volume is not None:
            volume = normalize_volume(symbol_info, requested_volume)
        else:
            volume = calculate_risk_volume(symbol_info, price, float(sl_price), mt5.account_info())
            if volume is None:
                return None

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": ROBOT_MAGIC,
            "comment": comment or "ForexRobot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        if sl_price is not None:
            request["sl"] = float(sl_price)
        if tp_price is not None:
            request["tp"] = float(tp_price)

        result = mt5.order_send(request)
        if result is None:
            logger.error("order_send retourne None: %s", mt5.last_error())
            return None
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error("Erreur ouverture %s %s: retcode=%s comment=%s", symbol, action, result.retcode, result.comment)
            return None

        ticket = int(getattr(result, "order", 0) or getattr(result, "deal", 0))
        logger.info(
            "POSITION OUVERTE | %s %s @ %.5f | volume=%.2f | SL=%s | TP=%s | ticket=%s",
            action, symbol, price, volume, sl_price, tp_price, ticket
        )

        state["trades_today"] += 1
        state.setdefault("last_trade_time", {})[symbol] = time.monotonic()
        state["current_position"] = {
            "ticket": ticket,
            "symbol": symbol,
            "action": action,
            "open_price": price,
            "sl": sl_price,
            "tp": tp_price,
            "volume": volume,
        }

        if CONFIG.get("telegram_on_open"):
            send_telegram(
                f"<b>POSITION OUVERTE</b>\\n"
                f"{action} <b>{symbol}</b> @ {price:.5f}\\n"
                f"Lot: {volume} | TP: {tp_price if tp_price is not None else '-'} | SL: {sl_price if sl_price is not None else '-'}\\n"
                f"#{symbol} #{action}"
            )

        return ticket
    except Exception as e:
        logger.exception("Erreur ouverture trade: %s", e)
        return None


def close_position(ticket: int) -> bool:
    if not ensure_mt5_connection():
        return False
    try:
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return False

        pos = positions[0]
        tick = mt5.symbol_info_tick(pos.symbol)
        if not tick:
            return False

        if pos.type == mt5.ORDER_TYPE_BUY:
            price = tick.bid
            order_type = mt5.ORDER_TYPE_SELL
        else:
            price = tick.ask
            order_type = mt5.ORDER_TYPE_BUY

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": ROBOT_MAGIC,
            "comment": "ForexRobot close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            state["daily_pnl"] += float(pos.profit)
            state["current_position"] = None
            logger.info("POSITION FERMEE | %s | Profit: %.2f", pos.symbol, pos.profit)

            if CONFIG.get("telegram_on_close"):
                emoji = "+" if pos.profit >= 0 else ""
                send_telegram(
                    f"<b>POSITION FERMEE</b>\n"
                    f"{pos.symbol} | Profit: <b>{emoji}${pos.profit:.2f}</b>\n"
                    f"PnL jour: ${state['daily_pnl']:.2f}"
                )
            return True

        logger.error("Erreur fermeture ticket=%s: %s", ticket, getattr(result, "comment", mt5.last_error()))
        return False
    except Exception as e:
        logger.exception("Erreur fermeture: %s", e)
        return False


def find_best_opportunity() -> Optional[Dict]:
    print("\nAnalyse de TOUS les couples forex...")
    available_symbols = get_available_symbols()
    print(f"   {len(available_symbols)} couples disponibles")

    all_analyses = []
    for symbol in available_symbols:
        analysis = analyze_symbol_score(symbol)
        if "error" not in analysis and analysis["action"] != "NEUTRAL":
            all_analyses.append(analysis)

    if not all_analyses:
        return None

    all_analyses.sort(key=lambda x: x["abs_score"], reverse=True)
    best = all_analyses[0]

    print(f"\nTOP 5 OPPORTUNITES:")
    for i, a in enumerate(all_analyses[:5]):
        print(f"   {i+1}. {a['symbol']}: {a['action']} (Score: {a['score']}, Conf: {a['confidence']}%)")

    return best

def economic_news_allows_trade(events: list, symbol: str = "") -> tuple[bool, str]:
    """Bloque les entrees avant et juste apres une annonce HIGH sur une devise de la paire."""
    currencies = {symbol[:3], symbol[3:6]} if len(symbol) >= 6 else set()
    before = runtime_cfg("news_blackout_minutes", NEWS_BLACKOUT_MINUTES)
    after = runtime_cfg("news_after_minutes", NEWS_AFTER_MINUTES)
    for event in events or []:
        impact = str(event.get("impact", "")).lower()
        country = str(event.get("country", "")).upper()
        if currencies and country and country != "ALL" and country not in currencies:
            continue
        try:
            minutes = float(event.get("minutes_until", 9999))
        except Exception:
            minutes = 9999
        if impact == "high" and -after <= minutes <= before:
            when = f"dans {int(minutes)} min" if minutes >= 0 else f"il y a {int(-minutes)} min"
            return False, f"News HIGH {country} {when}: {event.get('title', 'evenement')}"
    return True, "News filter OK"


def evaluate_new_trade_risk(account, portfolio: Dict, opportunity: Dict, action: str, events: list | None = None) -> tuple[bool, str, Dict]:
    """Applique les protections avant toute ouverture."""
    drawdown = get_current_drawdown_pct(account)
    state["risk_block_reason"] = None

    if drawdown >= runtime_cfg("auto_stop_drawdown_pct", AUTO_STOP_DRAWDOWN_PCT):
        reason = f"Drawdown critique {drawdown:.2f}% >= {runtime_cfg("auto_stop_drawdown_pct", AUTO_STOP_DRAWDOWN_PCT)}%"
        state["risk_block_reason"] = reason
        return False, reason, {}

    if drawdown >= runtime_cfg("max_drawdown_new_trades_pct", MAX_DRAWDOWN_FOR_NEW_TRADES_PCT):
        reason = f"Nouvelles entrees bloquees: drawdown {drawdown:.2f}% >= {runtime_cfg("max_drawdown_new_trades_pct", MAX_DRAWDOWN_FOR_NEW_TRADES_PCT)}%"
        state["risk_block_reason"] = reason
        return False, reason, {}

    daily_ok, daily_reason = daily_loss_allows(account)
    if not daily_ok:
        state["risk_block_reason"] = daily_reason
        return False, daily_reason, {}

    session_ok, session_reason = trading_session_allows()
    if not session_ok:
        return False, session_reason, {}

    symbol = str(opportunity.get("symbol", ""))
    if not symbol:
        return False, "Symbole absent", {}

    news_ok, news_reason = economic_news_allows_trade(events or [], symbol)
    if not news_ok:
        return False, news_reason, {}

    cooldown_ok, cooldown_reason = trade_cooldown_allows(symbol)
    if not cooldown_ok:
        return False, cooldown_reason, {}

    # Position limitee globalement par le moteur, meme si CONFIG contient une valeur plus haute.
    hard_max = min(int(CONFIG.get("max_positions", 1)), runtime_cfg("hard_max_positions", HARD_MAX_POSITIONS))
    if int(portfolio.get("open_positions", 0)) >= hard_max:
        return False, f"Limite positions securisee: {hard_max}", {}

    exposure_ok, exposure_reason = currency_exposure_allows(
        symbol,
        action,
        portfolio.get("currency_exposure", {}),
    )
    if not exposure_ok:
        return False, exposure_reason, {}

    mtf = get_multitimeframe_filter(symbol, action)
    if not mtf["allowed"]:
        return False, mtf["reason"], mtf

    return True, "Risk + MTF + spread OK", mtf


def reversal_requires_close(pos) -> tuple[bool, str]:
    """Signal technique fortement oppose a une position du robot detenue assez longtemps."""
    if not runtime_cfg("close_on_reversal", CLOSE_ON_REVERSAL):
        return False, ""
    if int(getattr(pos, "magic", 0) or 0) != ROBOT_MAGIC:
        return False, ""
    now = server_time_now()
    if now is None:
        return False, ""
    held_minutes = (now - datetime.fromtimestamp(int(pos.time), tz=timezone.utc).replace(tzinfo=None)).total_seconds() / 60
    if held_minutes < runtime_cfg("reversal_min_hold_minutes", REVERSAL_MIN_HOLD_MINUTES):
        return False, ""
    analysis = analyze_symbol_score(str(pos.symbol))
    score = float(analysis.get("score", 0) or 0)
    threshold = runtime_cfg("reversal_min_score", REVERSAL_MIN_SCORE)
    is_buy = int(pos.type) == 0
    if (is_buy and score <= -threshold) or (not is_buy and score >= threshold):
        return True, f"signal inverse {analysis.get('action')} (score {score:g})"
    return False, ""


def manage_open_positions(positions: list, account=None):
    """Gere les objectifs, le stop logique, l'inversion et la protection BE/trailing."""
    # Un objectif $ inferieur au risque pris recree un ratio gain/perte perdant :
    # il n'est applique que s'il vaut au moins le risque par trade.
    risk_money = 0.0
    if account is not None:
        risk_money = float(account.balance) * float(runtime_cfg("risk_per_trade_pct", RISK_PER_TRADE_PCT)) / 100.0
    target = float(CONFIG.get("target_profit", 0) or 0)
    target_active = target > 0 and target >= risk_money
    for pos in positions:
        try:
            profit = float(getattr(pos, "profit", 0.0) or 0.0)
            if target_active and profit >= target:
                print(f"  TARGET {pos.symbol}: +${profit:.2f} -> fermeture")
                if CONFIG.get("telegram_on_target"):
                    send_telegram(f"<b>OBJECTIF ATTEINT!</b>\\n{pos.symbol} | Profit: <b>+${profit:.2f}</b>")
                close_position(pos.ticket)
                continue
            if (CONFIG.get("stop_loss") or 0) > 0 and profit <= -CONFIG["stop_loss"]:
                print(f"  STOP LOGIQUE {pos.symbol}: ${profit:.2f} -> fermeture")
                if CONFIG.get("telegram_on_stoploss"):
                    send_telegram(f"<b>STOP LOSS LOGIQUE</b>\\n{pos.symbol} | Perte: <b>${profit:.2f}</b>")
                close_position(pos.ticket)
                continue
            reverse, reverse_reason = reversal_requires_close(pos)
            if reverse:
                print(f"  INVERSION {pos.symbol}: {reverse_reason} -> fermeture (${profit:.2f})")
                if close_position(pos.ticket):
                    send_telegram(f"<b>FERMETURE SUR INVERSION</b>\n{pos.symbol} | {reverse_reason} | ${profit:.2f}")
                continue
            manage_position_protection(pos)
        except Exception as e:
            logger.warning("Erreur gestion position %s: %s", getattr(pos, "ticket", "?"), e)

# ============ AUTO TRADER ============
def auto_trader_loop():
    print("\n" + "="*60)
    print("AUTO TRADER PRO - MULTI-TIMEFRAME + RISK ENGINE + IA")
    print(f"   Couples: {len(CONFIG['symbols'])} | Lot: {CONFIG['lot_size']}")
    print(f"   Objectif: +${CONFIG['target_profit']} | Stop logique: ${CONFIG['stop_loss']}")
    print(f"   Max positions securise: {min(int(CONFIG['max_positions']), runtime_cfg("hard_max_positions", HARD_MAX_POSITIONS))} | Intervalle: {CONFIG['check_interval']}s")
    print(f"   IA cooldown: {runtime_cfg("ai_auto_cooldown", AI_AUTO_COOLDOWN)}s | Trade cooldown: {runtime_cfg("trade_cooldown_minutes", TRADE_COOLDOWN_MINUTES)}min")
    print(f"   SL broker: {'OBLIGATOIRE' if runtime_cfg("broker_sl_required", BROKER_SL_REQUIRED) else 'OPTIONNEL'} | ATR SL={runtime_cfg("atr_sl_multiplier", ATR_SL_MULTIPLIER)}x | ATR TP={runtime_cfg("atr_tp_multiplier", ATR_TP_MULTIPLIER)}x")
    print(f"   News blackout: {runtime_cfg("news_blackout_minutes", NEWS_BLACKOUT_MINUTES)} min | Max devise: {runtime_cfg("max_currency_exposure", MAX_CURRENCY_EXPOSURE)}")
    print("="*60 + "\n")

    while state["auto_trading"]:
        try:
            current_time = datetime.now().strftime("%H:%M:%S")
            if not ensure_mt5_connection():
                logger.warning("Bridge MT5 indisponible, nouvelle tentative dans 10s")
                time.sleep(10)
                continue

            account = mt5.account_info()
            positions = list(mt5.positions_get() or [])
            if account is None:
                state["connected"] = False
                time.sleep(10)
                continue

            if state.get("session_start_balance") is None:
                state["session_start_balance"] = float(account.balance)

            drawdown = get_current_drawdown_pct(account)
            print(f"\n[{current_time}] Balance: ${account.balance:.2f} | Equity: ${account.equity:.2f} | DD instantane: {drawdown:.2f}%")

            if drawdown >= runtime_cfg("auto_stop_drawdown_pct", AUTO_STOP_DRAWDOWN_PCT):
                emergency_close_all(
                    f"drawdown {drawdown:.2f}% >= {runtime_cfg('auto_stop_drawdown_pct', AUTO_STOP_DRAWDOWN_PCT)}%"
                )
                break

            # Gestion continue des positions existantes, meme si les nouvelles entrees sont bloquees.
            manage_open_positions(positions, account)

            positions = list(mt5.positions_get() or [])
            portfolio = get_ai_portfolio()
            hard_max = min(int(CONFIG.get("max_positions", 1)), runtime_cfg("hard_max_positions", HARD_MAX_POSITIONS))
            print(f"  Portfolio: {portfolio['open_positions']}/{hard_max} positions | slots={max(0, hard_max - portfolio['open_positions'])}")

            if drawdown >= runtime_cfg("auto_stop_drawdown_pct", AUTO_STOP_DRAWDOWN_PCT):
                print(f"  RISK STOP: drawdown {drawdown:.2f}% -> aucune nouvelle entree")
                state["risk_block_reason"] = f"Drawdown critique {drawdown:.2f}%"
            elif portfolio["open_positions"] >= hard_max:
                print("  Max Positions securise atteint - aucune nouvelle ouverture")
            else:
                events = get_economic_calendar()
                active_impacts = [k for k, enabled in CONFIG.get("impacts", {}).items() if enabled]
                for event in [e for e in events if e.get("impact", "").lower() in active_impacts][:3]:
                    print(f"  [{event['impact'].upper()}] {event['title']} dans {event['minutes_until']} min")

                opportunities = find_top_opportunities(limit=len(CONFIG["symbols"]))
                
                existing_symbols = {str(p.symbol) for p in positions}

                trade_opened = False

                for opportunity in opportunities:
                    symbol = opportunity.get("symbol")
                    score = float(opportunity.get("score", 0) or 0)

                    if not symbol:
                        continue

                    if symbol in existing_symbols:
                        print(f"  -> {symbol}: SKIP (position deja ouverte)")
                        continue

                    if abs(score) < int(CONFIG["min_score"]):
                        continue

                    ai_min_score = int(
                        runtime_cfg(
                            "ai_validation_min_score",
                            AI_VALIDATION_MIN_SCORE
                        )
                    )

                    if abs(score) < ai_min_score:
                        print(
                            f"  -> {symbol}: SKIP IA "
                            f"(score {score:g} < seuil IA {ai_min_score})"
                        )
                        continue

                    action = "BUY" if score > 0 else "SELL"

                    # Tester le Risk Engine sur CHAQUE candidat.
                    allowed_risk, risk_reason, mtf = evaluate_new_trade_risk(
                        account,
                        portfolio,
                        opportunity,
                        action,
                        events
                    )

                    if not allowed_risk:
                        print(
                            f"  -> {symbol} {action}: "
                            f"RISK REFUS | {risk_reason}"
                        )
                        continue

                    # Le candidat passe les protections techniques/risk.
                    opportunity["m15_trend"] = mtf.get("m15")
                    opportunity["m5_trend"] = mtf.get("m5")
                    opportunity["atr"] = mtf.get("atr", 0.0)
                    opportunity["spread_points"] = mtf.get("spread_points")

                    allowed_ai, ai_reason = ai_allows_trade(
                        opportunity,
                        portfolio
                    )

                    print(
                        f"  -> {symbol} {action} Score:{score:g} | "
                        f"MTF:{mtf.get('m15')}/{mtf.get('m5')} | "
                        f"spread:{mtf.get('spread_points')} | "
                        f"IA:{'OK' if allowed_ai else 'REFUS'} | "
                        f"{ai_reason}"
                    )

                    if not allowed_ai:
                        # Une validation IA refusee ne doit pas faire
                        # essayer plusieurs IA dans le meme cycle.
                        break

                    ticket = open_trade(
                        symbol,
                        action,
                        f"Score:{score:g}|AI|MTF",
                        atr=float(mtf.get("atr", 0.0) or 0.0),
                    )

                    if ticket:
                        trade_opened = True
                        print(
                            f"  TRADE VALIDE | {symbol} {action} | "
                            f"ticket={ticket}"
                        )
                        break

                if not trade_opened:
                    print(
                        "  Aucun candidat n'a ete valide "
                        "pour une ouverture ce cycle"
                    )

            time.sleep(CONFIG["check_interval"])

        except Exception as e:
            logger.exception("Erreur auto-trader: %s", e)
            time.sleep(10)

    print("AUTO TRADER ARRETE")

# ============ API ROUTES ============
@api_router.post("/mt5/connect")
async def api_connect(request: ConnectRequest):
    # Les champs de l'ancienne interface sont acceptes pour compatibilite,
    # mais le compte actif reste celui du terminal MT5 du bridge.
    if connect_mt5():
        account = mt5.account_info()
        return {
            "success": True,
            "account": {
                "login": account.login,
                "balance": account.balance,
                "equity": account.equity,
                "server": account.server,
            }
        }
    raise HTTPException(status_code=400, detail="Bridge MT5 non disponible ou terminal non initialise")

@api_router.post("/mt5/disconnect")
async def api_disconnect():
    # Ne jamais appeler mt5.shutdown() ici : cela arreterait l'IPC du MT5
    # cote serveur bridge. On desactive uniquement la connexion logique du robot.
    state["connected"] = False
    state["auto_trading"] = False
    state["auto_thread"] = None
    return {"success": True, "message": "Connexion logique du robot desactivee"}

@api_router.get("/mt5/status")
async def api_status():
    if not ensure_mt5_connection():
        return {"connected": False, "error": state.get("last_bridge_error")}
    try:
        account = mt5.account_info()
        positions = mt5.positions_get()
        position_info = None
        if positions and len(positions) > 0:
            pos = positions[0]
            position_info = {
                "symbol": pos.symbol,
                "type": "BUY" if pos.type == 0 else "SELL",
                "profit": float(pos.profit),
                "target": CONFIG["target_profit"],
                "progress": (pos.profit / CONFIG["target_profit"]) * 100
            }
        return {
            "connected": True,
            "balance": float(account.balance),
            "equity": float(account.equity),
            "auto_trading": state["auto_trading"],
            "position": position_info,
            "target_profit": CONFIG["target_profit"],
            "lot_size": CONFIG["lot_size"],
            "stop_loss": CONFIG["stop_loss"],
            "min_score": CONFIG["min_score"],
            "check_interval": CONFIG["check_interval"],
            "max_positions": min(int(CONFIG["max_positions"]), runtime_cfg("hard_max_positions", HARD_MAX_POSITIONS)),
            "impacts": CONFIG["impacts"],
        }
    except Exception as e:
        logger.error(f"Erreur status: {e}")
        return {"connected": False, "error": str(e)}

@api_router.get("/mt5/prices")
async def api_prices():
    prices = {}
    for symbol in CONFIG["symbols"][:10]:
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                prices[symbol] = {"bid": float(tick.bid), "ask": float(tick.ask)}
        except:
            continue
    return prices

@api_router.get("/analysis")
async def api_analyze_all():
    if not state["connected"]:
        raise HTTPException(status_code=400, detail="Non connecte")
    best = find_best_opportunity()
    return {"best_opportunity": best}

@api_router.post("/trading/open")
async def api_open_trade(trade: TradeRequest):
    if not ensure_mt5_connection():
        raise HTTPException(status_code=400, detail="MT5 indisponible")
    account = mt5.account_info()
    positions = list(mt5.positions_get() or [])
    if len(positions) >= min(int(CONFIG.get("max_positions", 1)), runtime_cfg("hard_max_positions", HARD_MAX_POSITIONS)):
        raise HTTPException(status_code=400, detail="Limite positions securisee atteinte")
    ticket = open_trade(trade.symbol, trade.type, trade.comment, trade.volume)
    if ticket:
        return {"success": True, "ticket": ticket}
    raise HTTPException(status_code=400, detail="Echec ouverture")

@api_router.post("/trading/close/{ticket}")
async def api_close_trade(ticket: int):
    if close_position(ticket):
        return {"success": True}
    raise HTTPException(status_code=400, detail="Echec fermeture")

@api_router.get("/trading/positions")
async def api_positions():
    try:
        positions = mt5.positions_get()
        if not positions:
            return []
        return [{
            "ticket": p.ticket,
            "symbol": p.symbol,
            "type": "BUY" if p.type == 0 else "SELL",
            "volume": float(p.volume),
            "profit": float(p.profit),
            "target": CONFIG["target_profit"],
            "progress": (p.profit / CONFIG["target_profit"]) * 100 if CONFIG["target_profit"] else 0,
            "open_price": float(getattr(p, "price_open", 0) or 0),
            "current_price": float(getattr(p, "price_current", 0) or 0),
            "sl": float(getattr(p, "sl", 0) or 0),
            "tp": float(getattr(p, "tp", 0) or 0),
        } for p in positions]
    except:
        return []

@api_router.post("/auto/start")
async def api_start_auto():
    if not ensure_mt5_connection():
        raise HTTPException(status_code=400, detail="Bridge MT5 non connecte")
    thread = state.get("auto_thread")
    if state["auto_trading"] and thread and thread.is_alive():
        return {"success": True, "message": "Auto trading deja actif"}

    state["auto_trading"] = True
    thread = threading.Thread(target=auto_trader_loop, daemon=True, name="forex-auto-trader")
    state["auto_thread"] = thread
    thread.start()
    return {"success": True, "message": "Auto trading demarre - Mode SNIPER"}

@api_router.post("/auto/stop")
async def api_stop_auto():
    state["auto_trading"] = False
    return {"success": True, "message": "Auto trading arrete"}

# ============ CALENDRIER ============
@api_router.get("/calendar/events")
async def api_calendar():
    return get_economic_calendar()

@api_router.get("/calendar/full")
async def api_calendar_full():
    try:
        events = fetch_full_calendar()
        return {"success": True, "events": events, "count": len(events)}
    except Exception as e:
        return {"success": False, "events": [], "error": str(e)}


# ============ HISTORIQUE DES TRADES ============
def _history_timestamp(value) -> str:
    """Convertit un timestamp MT5 en ISO lisible."""
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""

def get_trade_history(from_dt: datetime | None = None, to_dt: datetime | None = None) -> dict:
    """
    Retourne l'historique des positions clôturées à partir des deals MT5.
    Les deals OUT sont regroupés par position_id pour reconstituer un trade.
    """
    if not ensure_mt5_connection():
        return {"success": False, "error": "Bridge MT5 non connecté", "trades": []}

    try:
        if to_dt is None:
            to_dt = datetime.now(timezone.utc)
        if from_dt is None:
            # Depuis le début de l'historique accessible par le broker.
            from_dt = datetime(2000, 1, 1, tzinfo=timezone.utc)

        from_naive = from_dt.astimezone(timezone.utc).replace(tzinfo=None)
        to_naive = to_dt.astimezone(timezone.utc).replace(tzinfo=None)

        deals = mt5.history_deals_get(from_naive, to_naive)
        if deals is None:
            return {
                "success": False,
                "error": "MT5 n'a retourné aucun historique ou history_deals_get est indisponible",
                "trades": [],
            }

        grouped = {}

        for deal in deals:
            position_id = int(getattr(deal, "position_id", 0) or 0)
            if position_id <= 0:
                continue

            entry = int(getattr(deal, "entry", -1))
            # DEAL_ENTRY_IN=0, OUT=1, INOUT=2, OUT_BY=3
            if entry not in (1, 2, 3):
                continue

            symbol = str(getattr(deal, "symbol", "") or "")
            side_value = int(getattr(deal, "type", -1))
            side = "BUY" if side_value == 0 else "SELL" if side_value == 1 else str(side_value)

            item = grouped.setdefault(position_id, {
                "position_id": position_id,
                "symbol": symbol,
                "type": side,
                "open_time": None,
                "close_time": None,
                "volume": 0.0,
                "open_price": None,
                "close_price": None,
                "profit": 0.0,
                "commission": 0.0,
                "swap": 0.0,
                "fee": 0.0,
                "net_profit": 0.0,
                "deals": 0,
            })

            item["close_time"] = _history_timestamp(getattr(deal, "time", 0))
            item["close_price"] = float(getattr(deal, "price", 0.0) or 0.0)
            item["volume"] += float(getattr(deal, "volume", 0.0) or 0.0)
            item["profit"] += float(getattr(deal, "profit", 0.0) or 0.0)
            item["commission"] += float(getattr(deal, "commission", 0.0) or 0.0)
            item["swap"] += float(getattr(deal, "swap", 0.0) or 0.0)
            item["fee"] += float(getattr(deal, "fee", 0.0) or 0.0)
            item["net_profit"] = (
                item["profit"] + item["commission"] + item["swap"] + item["fee"]
            )
            item["deals"] += 1

        # Rechercher les deals d'entrée pour compléter open_time/open_price/side/volume.
        # On réutilise la même fenêtre, ce qui évite une deuxième requête globale.
        for deal in deals:
            position_id = int(getattr(deal, "position_id", 0) or 0)
            if position_id not in grouped:
                continue

            entry = int(getattr(deal, "entry", -1))
            if entry not in (0, 2):
                continue

            item = grouped[position_id]
            if not item["open_time"]:
                item["open_time"] = _history_timestamp(getattr(deal, "time", 0))
            if item["open_price"] is None:
                item["open_price"] = float(getattr(deal, "price", 0.0) or 0.0)

            side_value = int(getattr(deal, "type", -1))
            if side_value == 0:
                item["type"] = "BUY"
            elif side_value == 1:
                item["type"] = "SELL"

        trades = list(grouped.values())
        trades.sort(key=lambda x: x.get("close_time") or x.get("open_time") or "", reverse=True)

        total_profit = sum(float(t["net_profit"]) for t in trades)
        wins = sum(1 for t in trades if float(t["net_profit"]) > 0)
        losses = sum(1 for t in trades if float(t["net_profit"]) < 0)
        breakeven = len(trades) - wins - losses

        return {
            "success": True,
            "from": from_dt.isoformat(),
            "to": to_dt.isoformat(),
            "trades": trades,
            "count": len(trades),
            "wins": wins,
            "losses": losses,
            "breakeven": breakeven,
            "total_profit": total_profit,
        }
    except Exception as e:
        logger.exception("Erreur historique MT5: %s", e)
        return {"success": False, "error": str(e), "trades": []}


# ============ CONFIG DYNAMIQUE ============
@api_router.get("/trading/history")
async def api_trading_history(
    date_from: str = "",
    date_to: str = "",
    since_beginning: bool = False,
):
    """Historique des positions clôturées, filtrable par date."""
    try:
        if since_beginning:
            start = datetime(2000, 1, 1, tzinfo=timezone.utc)
        elif date_from:
            start = datetime.fromisoformat(date_from.replace("Z", "+00:00"))
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
        else:
            start = datetime.now(timezone.utc) - timedelta(days=7)

        if date_to:
            end = datetime.fromisoformat(date_to.replace("Z", "+00:00"))
            if end.tzinfo is None:
                end = end.replace(tzinfo=timezone.utc)
        else:
            end = datetime.now(timezone.utc)

        if end <= start:
            raise HTTPException(status_code=400, detail="date_to doit être après date_from")

        return get_trade_history(start, end)
    except ValueError:
        raise HTTPException(status_code=400, detail="Format de date invalide (YYYY-MM-DD ou ISO)")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Erreur route historique: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/auth/status")
async def api_auth_status(request: Request):
    user = current_user(request)
    return {
        "authenticated": bool(user),
        "username": user.get("username") if user else None,
        "role": user.get("role") if user else None,
        "initialized": bool(AUTH_USERS),
    }

@api_router.post("/auth/login")
async def api_auth_login(body: LoginRequest):
    username = body.username.strip()
    if username not in AUTH_USERS or not verify_password(username, body.password):
        raise HTTPException(status_code=401, detail="Utilisateur ou mot de passe invalide")
    role = str(AUTH_USERS[username].get("role", "guest"))
    token = create_session(username, role)
    return {
        "success": True,
        "token": token,
        "username": username,
        "role": role,
        "expires_in": SESSION_TTL_SECONDS,
    }

@api_router.post("/auth/logout")
async def api_auth_logout(request: Request):
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        AUTH_SESSIONS.pop(auth[7:].strip(), None)
    return {"success": True}

@api_router.post("/auth/setup")
async def api_auth_setup(body: dict):
    if AUTH_USERS:
        raise HTTPException(status_code=403, detail="Authentification deja initialisee")

    admin_username = str(body.get("admin_username", "")).strip()
    admin_password = str(body.get("admin_password", ""))
    guest_username = str(body.get("guest_username", "")).strip()
    guest_password = str(body.get("guest_password", ""))

    if not admin_username or not guest_username:
        raise HTTPException(status_code=400, detail="Deux usernames sont requis")
    if admin_username == guest_username:
        raise HTTPException(status_code=400, detail="Les usernames doivent etre differents")
    if len(admin_password) < 10:
        raise HTTPException(status_code=400, detail="Mot de passe admin: minimum 10 caracteres")
    if len(guest_password) < 8:
        raise HTTPException(status_code=400, detail="Mot de passe invite: minimum 8 caracteres")

    create_user(admin_username, admin_password, "admin")
    create_user(guest_username, guest_password, "guest")

    if not save_auth_users():
        AUTH_USERS.clear()
        raise HTTPException(status_code=500, detail="Impossible d'enregistrer users.json")

    return {
        "success": True,
        "users": [
            {"username": admin_username, "role": "admin"},
            {"username": guest_username, "role": "guest"},
        ],
    }

@api_router.get("/config")
async def api_get_config():
    return {
        "success": True,
        "persistent": True,
        "config_file": CONFIG_FILE,
        "config": {
            "lot_size": CONFIG["lot_size"],
            "target_profit": CONFIG["target_profit"],
            "stop_loss": CONFIG["stop_loss"],
            "min_score": CONFIG["min_score"],
            "check_interval": CONFIG["check_interval"],
            "max_positions": CONFIG["max_positions"],
            "impacts": CONFIG["impacts"],
            "rsi_period": CONFIG["rsi_period"],
            "rsi_oversold": CONFIG["rsi_oversold"],
            "rsi_overbought": CONFIG["rsi_overbought"],
            "sma_fast": CONFIG["sma_fast"],
            "sma_slow": CONFIG["sma_slow"],
            "ollama_url": CONFIG["ollama_url"],
            "ollama_model": CONFIG["ollama_model"],
            "use_ai_analysis": CONFIG["use_ai_analysis"],
            "ai_validation_min_score": CONFIG["ai_validation_min_score"],
            "ai_auto_cooldown": CONFIG["ai_auto_cooldown"],
            "ai_quick_num_predict": CONFIG["ai_quick_num_predict"],
            "ai_quick_timeout": CONFIG["ai_quick_timeout"],
            "ai_full_num_predict": CONFIG["ai_full_num_predict"],
            "ai_full_timeout": CONFIG["ai_full_timeout"],
            "ai_keep_alive": CONFIG["ai_keep_alive"],
            "hard_max_positions": CONFIG["hard_max_positions"],
            "max_currency_exposure": CONFIG["max_currency_exposure"],
            "trade_cooldown_minutes": CONFIG["trade_cooldown_minutes"],
            "max_spread_points": CONFIG["max_spread_points"],
            "max_spread_sl_pct": CONFIG["max_spread_sl_pct"],
            "atr_timeframe": CONFIG["atr_timeframe"],
            "max_drawdown_new_trades_pct": CONFIG["max_drawdown_new_trades_pct"],
            "auto_stop_drawdown_pct": CONFIG["auto_stop_drawdown_pct"],
            "m15_min_bars": CONFIG["m15_min_bars"],
            "m5_min_bars": CONFIG["m5_min_bars"],
            "atr_period": CONFIG["atr_period"],
            "atr_sl_multiplier": CONFIG["atr_sl_multiplier"],
            "atr_tp_multiplier": CONFIG["atr_tp_multiplier"],
            "breakeven_trigger_r": CONFIG["breakeven_trigger_r"],
            "trailing_trigger_r": CONFIG["trailing_trigger_r"],
            "trailing_atr_multiplier": CONFIG["trailing_atr_multiplier"],
            "broker_sl_required": CONFIG["broker_sl_required"],
            "news_blackout_minutes": CONFIG["news_blackout_minutes"],
            "news_after_minutes": CONFIG["news_after_minutes"],
            "risk_per_trade_pct": CONFIG["risk_per_trade_pct"],
            "daily_loss_limit_pct": CONFIG["daily_loss_limit_pct"],
            "close_on_reversal": CONFIG["close_on_reversal"],
            "reversal_min_score": CONFIG["reversal_min_score"],
            "reversal_min_hold_minutes": CONFIG["reversal_min_hold_minutes"],
            "block_rollover": CONFIG["block_rollover"],
        }
    }

@api_router.post("/config/recommended")
async def api_apply_recommended_config():
    """Reapplique la configuration recommandee (bouton de l'interface)."""
    result = apply_recommended_config(force=True)
    if not result.get("applied"):
        raise HTTPException(status_code=500, detail=result.get("reason", "Echec"))
    return {"success": True, **result}

@api_router.post("/config")
async def api_update_config(body: dict):
    global CONFIG
    updated_fields = []

    direct_keys = {
        "lot_size": "lot_size",
        "target_profit": "target_profit",
        "stop_loss": "stop_loss",
        "min_score": "min_score",
        "check_interval": "check_interval",
        "max_positions": "max_positions",
        "ai_validation_min_score": "ai_validation_min_score",
        "ai_auto_cooldown": "ai_auto_cooldown",
        "ai_quick_num_predict": "ai_quick_num_predict",
        "ai_quick_timeout": "ai_quick_timeout",
        "ai_full_num_predict": "ai_full_num_predict",
        "ai_full_timeout": "ai_full_timeout",
        "ai_keep_alive": "ai_keep_alive",
        "hard_max_positions": "hard_max_positions",
        "max_currency_exposure": "max_currency_exposure",
        "trade_cooldown_minutes": "trade_cooldown_minutes",
        "max_spread_points": "max_spread_points",
        "max_spread_sl_pct": "max_spread_sl_pct",
        "atr_timeframe": "atr_timeframe",
        "max_drawdown_new_trades_pct": "max_drawdown_new_trades_pct",
        "auto_stop_drawdown_pct": "auto_stop_drawdown_pct",
        "m15_min_bars": "m15_min_bars",
        "m5_min_bars": "m5_min_bars",
        "atr_period": "atr_period",
        "atr_sl_multiplier": "atr_sl_multiplier",
        "atr_tp_multiplier": "atr_tp_multiplier",
        "breakeven_trigger_r": "breakeven_trigger_r",
        "trailing_trigger_r": "trailing_trigger_r",
        "trailing_atr_multiplier": "trailing_atr_multiplier",
        "broker_sl_required": "broker_sl_required",
        "news_blackout_minutes": "news_blackout_minutes",
        "news_after_minutes": "news_after_minutes",
        "risk_per_trade_pct": "risk_per_trade_pct",
        "daily_loss_limit_pct": "daily_loss_limit_pct",
        "close_on_reversal": "close_on_reversal",
        "reversal_min_score": "reversal_min_score",
        "reversal_min_hold_minutes": "reversal_min_hold_minutes",
        "block_rollover": "block_rollover",
        "ollama_url": "ollama_url",
        "ollama_model": "ollama_model",
        "use_ai_analysis": "use_ai_analysis",
        "sma_fast": "sma_fast",
        "sma_slow": "sma_slow",
    }

    int_keys = {
        "min_score", "check_interval", "max_positions",
        "ai_validation_min_score", "ai_auto_cooldown",
        "ai_quick_num_predict", "ai_quick_timeout",
        "ai_full_num_predict", "ai_full_timeout",
        "hard_max_positions", "max_currency_exposure",
        "trade_cooldown_minutes", "max_spread_points",
        "m15_min_bars", "m5_min_bars", "atr_period",
        "sma_fast", "sma_slow", "news_blackout_minutes",
        "news_after_minutes", "reversal_min_score", "reversal_min_hold_minutes",
    }
    float_keys = {
        "lot_size", "target_profit", "stop_loss",
        "max_drawdown_new_trades_pct", "auto_stop_drawdown_pct",
        "atr_sl_multiplier", "atr_tp_multiplier",
        "breakeven_trigger_r", "trailing_trigger_r",
        "trailing_atr_multiplier", "risk_per_trade_pct", "daily_loss_limit_pct",
        "max_spread_sl_pct",
    }
    bool_keys = {"broker_sl_required", "use_ai_analysis", "close_on_reversal", "block_rollover"}

    for frontend_key, config_key in direct_keys.items():
        if frontend_key not in body:
            continue
        value = body[frontend_key]
        try:
            if config_key in int_keys:
                value = int(value)
            elif config_key in float_keys:
                value = float(value)
            elif config_key in bool_keys:
                value = bool(value)
            elif config_key == "ai_keep_alive":
                value = str(value).strip()
            elif config_key == "atr_timeframe":
                value = str(value).strip().upper()
                if value not in {"M15", "H1", "H4"}:
                    raise ValueError(value)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"Valeur invalide: {config_key}")
        CONFIG[config_key] = value
        updated_fields.append(config_key)

    # Bornes de securite / coherences
    CONFIG["lot_size"] = max(0.01, min(float(CONFIG["lot_size"]), 10.0))
    CONFIG["target_profit"] = max(0.0, min(float(CONFIG["target_profit"]), 100000.0))
    CONFIG["stop_loss"] = max(0.0, min(float(CONFIG["stop_loss"]), 100000.0))
    CONFIG["min_score"] = max(1, min(int(CONFIG["min_score"]), 10))
    CONFIG["check_interval"] = max(10, min(int(CONFIG["check_interval"]), 600))
    CONFIG["hard_max_positions"] = max(1, min(int(CONFIG["hard_max_positions"]), 20))
    CONFIG["max_positions"] = max(1, min(int(CONFIG["max_positions"]), CONFIG["hard_max_positions"]))
    CONFIG["max_currency_exposure"] = max(1, min(int(CONFIG["max_currency_exposure"]), 10))
    CONFIG["trade_cooldown_minutes"] = max(0, min(int(CONFIG["trade_cooldown_minutes"]), 1440))
    CONFIG["max_spread_points"] = max(0, min(int(CONFIG["max_spread_points"]), 1000))
    CONFIG["max_spread_sl_pct"] = max(0.0, min(float(CONFIG["max_spread_sl_pct"]), 100.0))
    CONFIG["max_drawdown_new_trades_pct"] = max(0.0, min(float(CONFIG["max_drawdown_new_trades_pct"]), 100.0))
    CONFIG["auto_stop_drawdown_pct"] = max(CONFIG["max_drawdown_new_trades_pct"], min(float(CONFIG["auto_stop_drawdown_pct"]), 100.0))
    CONFIG["m15_min_bars"] = max(50, min(int(CONFIG["m15_min_bars"]), 500))
    CONFIG["m5_min_bars"] = max(50, min(int(CONFIG["m5_min_bars"]), 500))
    CONFIG["atr_period"] = max(5, min(int(CONFIG["atr_period"]), 100))
    CONFIG["atr_sl_multiplier"] = max(0.1, min(float(CONFIG["atr_sl_multiplier"]), 10.0))
    CONFIG["atr_tp_multiplier"] = max(0.1, min(float(CONFIG["atr_tp_multiplier"]), 20.0))
    CONFIG["breakeven_trigger_r"] = max(0.1, min(float(CONFIG["breakeven_trigger_r"]), 10.0))
    CONFIG["trailing_trigger_r"] = max(CONFIG["breakeven_trigger_r"], min(float(CONFIG["trailing_trigger_r"]), 20.0))
    CONFIG["trailing_atr_multiplier"] = max(0.1, min(float(CONFIG["trailing_atr_multiplier"]), 10.0))
    CONFIG["ai_validation_min_score"] = max(1, min(int(CONFIG["ai_validation_min_score"]), 10))
    CONFIG["ai_auto_cooldown"] = max(0, min(int(CONFIG["ai_auto_cooldown"]), 3600))
    CONFIG["ai_quick_num_predict"] = max(16, min(int(CONFIG["ai_quick_num_predict"]), 512))
    CONFIG["ai_quick_timeout"] = max(5, min(int(CONFIG["ai_quick_timeout"]), 180))
    CONFIG["ai_full_num_predict"] = max(32, min(int(CONFIG["ai_full_num_predict"]), 1024))
    CONFIG["ai_full_timeout"] = max(10, min(int(CONFIG["ai_full_timeout"]), 300))
    CONFIG["news_blackout_minutes"] = max(0, min(int(CONFIG["news_blackout_minutes"]), 240))
    CONFIG["news_after_minutes"] = max(0, min(int(CONFIG["news_after_minutes"]), 240))
    CONFIG["risk_per_trade_pct"] = max(0.05, min(float(CONFIG["risk_per_trade_pct"]), 5.0))
    CONFIG["daily_loss_limit_pct"] = max(0.0, min(float(CONFIG["daily_loss_limit_pct"]), 50.0))
    CONFIG["reversal_min_score"] = max(1, min(int(CONFIG["reversal_min_score"]), 10))
    CONFIG["reversal_min_hold_minutes"] = max(0, min(int(CONFIG["reversal_min_hold_minutes"]), 1440))
    CONFIG["broker_sl_required"] = True
    CONFIG["rsi_period"] = max(5, min(int(CONFIG["rsi_period"]), 50))
    CONFIG["rsi_oversold"] = max(5, min(int(CONFIG["rsi_oversold"]), 49))
    CONFIG["rsi_overbought"] = max(CONFIG["rsi_oversold"] + 1, min(int(CONFIG["rsi_overbought"]), 95))
    CONFIG["sma_fast"] = max(2, min(int(CONFIG["sma_fast"]), 200))
    CONFIG["sma_slow"] = max(CONFIG["sma_fast"] + 1, min(int(CONFIG["sma_slow"]), 500))

    if "impacts" in body and isinstance(body["impacts"], dict):
        CONFIG["impacts"] = {
            "high": bool(body["impacts"].get("high", False)),
            "medium": bool(body["impacts"].get("medium", False)),
            "low": bool(body["impacts"].get("low", False)),
        }
        updated_fields.append("impacts")

    if "rsi" in body and isinstance(body["rsi"], dict):
        # Compatibilite avec l'ancien format de l'interface.
        if "period" in body["rsi"]:
            CONFIG["rsi_period"] = max(5, min(int(body["rsi"]["period"]), 50))
            updated_fields.append("rsi_period")
        if "oversold" in body["rsi"]:
            CONFIG["rsi_oversold"] = max(5, min(int(body["rsi"]["oversold"]), 49))
            updated_fields.append("rsi_oversold")
        if "overbought" in body["rsi"]:
            CONFIG["rsi_overbought"] = max(CONFIG["rsi_oversold"] + 1, min(int(body["rsi"]["overbought"]), 95))
            updated_fields.append("rsi_overbought")

    saved = save_persistent_config()
    logger.info(f"Config mise a jour: {updated_fields} | persistent={saved}")
    return {
        "success": True,
        "message": f"Configuration mise a jour ({len(updated_fields)} parametres)",
        "updated_fields": updated_fields,
        "persistent": saved,
        "current_config": {
            k: CONFIG[k] for k in [
                "lot_size", "target_profit", "stop_loss", "min_score",
                "check_interval", "max_positions", "impacts",
                "rsi_period", "rsi_oversold", "rsi_overbought",
                "sma_fast", "sma_slow", "ollama_url", "ollama_model",
                "use_ai_analysis", "ai_validation_min_score", "ai_auto_cooldown",
                "ai_quick_num_predict", "ai_quick_timeout", "ai_full_num_predict",
                "ai_full_timeout", "ai_keep_alive", "hard_max_positions",
                "max_currency_exposure", "trade_cooldown_minutes", "max_spread_points",
                "max_spread_sl_pct", "atr_timeframe", "max_drawdown_new_trades_pct", "auto_stop_drawdown_pct",
                "m15_min_bars", "m5_min_bars", "atr_period",
                "atr_sl_multiplier", "atr_tp_multiplier", "breakeven_trigger_r",
                "trailing_trigger_r", "trailing_atr_multiplier", "broker_sl_required",
                "news_blackout_minutes", "news_after_minutes", "risk_per_trade_pct",
                "daily_loss_limit_pct", "close_on_reversal", "reversal_min_score",
                "reversal_min_hold_minutes", "block_rollover"
            ]
        },
    }

@api_router.get("/risk/status")
async def api_risk_status():
    """Etat temps reel du Risk Engine pour l'interface."""
    account = None
    positions = []
    if ensure_mt5_connection():
        try:
            account = mt5.account_info()
            positions = list(mt5.positions_get() or [])
        except Exception:
            positions = []
    drawdown = get_current_drawdown_pct(account) if account else 0.0
    hard_max = min(int(CONFIG.get("max_positions", 1)), int(runtime_cfg("hard_max_positions", HARD_MAX_POSITIONS)))
    exposure = get_ai_portfolio().get("currency_exposure", {}) if state.get("connected") else {}
    risk_reason = state.get("risk_block_reason")
    return {
        "ok": True,
        "auto_trading": state.get("auto_trading", False),
        "drawdown_pct": round(drawdown, 2),
        "max_positions": hard_max,
        "open_positions": len(positions),
        "remaining_slots": max(0, hard_max - len(positions)),
        "max_currency_exposure": int(runtime_cfg("max_currency_exposure", MAX_CURRENCY_EXPOSURE)),
        "currency_exposure": exposure,
        "max_spread_points": int(runtime_cfg("max_spread_points", MAX_SPREAD_POINTS)),
        "max_spread_sl_pct": float(runtime_cfg("max_spread_sl_pct", MAX_SPREAD_SL_PCT)),
        "atr_timeframe": str(runtime_cfg("atr_timeframe", ATR_TIMEFRAME)),
        "trade_cooldown_minutes": int(runtime_cfg("trade_cooldown_minutes", TRADE_COOLDOWN_MINUTES)),
        "max_drawdown_new_trades_pct": float(runtime_cfg("max_drawdown_new_trades_pct", MAX_DRAWDOWN_FOR_NEW_TRADES_PCT)),
        "auto_stop_drawdown_pct": float(runtime_cfg("auto_stop_drawdown_pct", AUTO_STOP_DRAWDOWN_PCT)),
        "atr_period": int(runtime_cfg("atr_period", ATR_PERIOD)),
        "atr_sl_multiplier": float(runtime_cfg("atr_sl_multiplier", ATR_SL_MULTIPLIER)),
        "atr_tp_multiplier": float(runtime_cfg("atr_tp_multiplier", ATR_TP_MULTIPLIER)),
        "breakeven_trigger_r": float(runtime_cfg("breakeven_trigger_r", BREAKEVEN_TRIGGER_R)),
        "trailing_trigger_r": float(runtime_cfg("trailing_trigger_r", TRAILING_TRIGGER_R)),
        "trailing_atr_multiplier": float(runtime_cfg("trailing_atr_multiplier", TRAILING_ATR_MULTIPLIER)),
        "broker_sl_required": bool(runtime_cfg("broker_sl_required", BROKER_SL_REQUIRED)),
        "news_blackout_minutes": int(runtime_cfg("news_blackout_minutes", NEWS_BLACKOUT_MINUTES)),
        "m15_min_bars": int(runtime_cfg("m15_min_bars", M15_MIN_BARS)),
        "m5_min_bars": int(runtime_cfg("m5_min_bars", M5_MIN_BARS)),
        "ai_validation_min_score": int(runtime_cfg("ai_validation_min_score", AI_VALIDATION_MIN_SCORE)),
        "ai_auto_cooldown": int(runtime_cfg("ai_auto_cooldown", AI_AUTO_COOLDOWN)),
        "risk_per_trade_pct": float(runtime_cfg("risk_per_trade_pct", RISK_PER_TRADE_PCT)),
        "daily_loss_limit_pct": float(runtime_cfg("daily_loss_limit_pct", DAILY_LOSS_LIMIT_PCT)),
        "day_start_equity": state.get("day_start_equity"),
        "risk_block_reason": risk_reason,
        "session_start_balance": state.get("session_start_balance"),
    }

# ============ IA LOCALE (OLLAMA) ============
def analyze_news_with_ollama(
    headline: str,
    content: str = "",
    context: dict = None,
    quick: bool = False,
) -> dict:
    """Analyse Ollama avec deux profils:
    - quick=True : validation d'un trade, reponse courte et peu couteuse
    - quick=False : analyse IA complete pour l'interface/API
    """
    ollama_url = CONFIG.get("ollama_url", "http://localhost:11434")
    model = CONFIG.get("ollama_model", "mistral")

    market_context = ""
    if context:
        market_context = f"""
Contexte du marche actuel:
- Paires analysees: {context.get('pairs_count', 28)}
- Meilleure opportunite: {context.get('best_symbol', 'N/A')} ({context.get('best_action', 'N/A')}, Score: {context.get('best_score', 'N/A')})
- Evenements economiques proches: {context.get('upcoming_events', 'Aucun')}
- Symbole candidat: {context.get('symbol', 'N/A')}
- Action technique: {context.get('technical_action', 'N/A')}
- Score technique: {context.get('technical_score', 'N/A')}
- Confiance technique: {context.get('technical_confidence', 'N/A')}%
- RSI: {context.get('rsi', 'N/A')}
- Tendance: {context.get('trend', 'N/A')}
- Raisons techniques: {context.get('reasons', [])}
- Positions ouvertes: {context.get('open_positions', 'N/A')}/{context.get('max_positions', 'N/A')}
- Places restantes: {context.get('remaining_slots', 'N/A')}
"""

    if quick:
        prompt = f"""Tu es un filtre de validation forex.
Analyse uniquement le signal technique fourni et determine son biais directionnel.
Ne prends pas en compte des informations absentes du contexte.

Signal: {headline}
{market_context}

Reponds UNIQUEMENT en JSON compact:
{{
  "sentiment": "bullish" ou "bearish" ou "neutral",
  "score": nombre entre -1.0 et 1.0,
  "recommendation": "BUY" ou "SELL" ou "HOLD",
  "risk_level": "low" ou "medium" ou "high"
}}"""
        num_predict = runtime_cfg("ai_quick_num_predict", AI_QUICK_NUM_PREDICT)
        timeout = runtime_cfg("ai_quick_timeout", AI_QUICK_TIMEOUT)
    else:
        prompt = f"""Tu es un analyste forex expert. Analyse cette information et donne ton avis pour le trading forex.

News/Information: {headline}
{f"Details: {content}" if content else ""}
{market_context}

Reponds UNIQUEMENT en JSON avec ce format exact:
{{
    "sentiment": "bullish" ou "bearish" ou "neutral",
    "score": nombre entre -1.0 et 1.0,
    "currencies_affected": ["USD", "EUR"],
    "pairs_to_watch": ["EURUSD"],
    "analysis": "Explication en 2-3 phrases",
    "recommendation": "BUY/SELL/HOLD et sur quelles paires",
    "risk_level": "low" ou "medium" ou "high",
    "timeframe": "court terme" ou "moyen terme" ou "long terme"
}}"""
        num_predict = runtime_cfg("ai_full_num_predict", AI_FULL_NUM_PREDICT)
        timeout = runtime_cfg("ai_full_timeout", AI_FULL_TIMEOUT)

    # Evite plusieurs generations simultanees qui satureraient le VPS.
    if not OLLAMA_GENERATION_LOCK.acquire(timeout=2):
        return {"error": "IA Ollama deja occupee par une autre analyse."}

    started = time.monotonic()
    try:
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "keep_alive": runtime_cfg("ai_keep_alive", AI_KEEP_ALIVE),
                "options": {
                    "temperature": 0.2 if quick else 0.3,
                    "num_predict": num_predict,
                },
            },
            timeout=timeout,
        )

        elapsed = time.monotonic() - started
        logger.info(
            "Ollama %s | HTTP=%s | %.1fs | tokens_max=%s",
            "QUICK" if quick else "FULL",
            response.status_code,
            elapsed,
            num_predict,
        )

        if response.status_code == 200:
            result = response.json()
            ai_response = result.get("response", "")
            json_match = re.search(r'\{[\s\S]*\}', ai_response)
            if json_match:
                try:
                    parsed = json_module.loads(json_match.group())
                    parsed["model"] = model
                    parsed["source"] = "ollama_local"
                    parsed["quick"] = quick
                    return parsed
                except json_module.JSONDecodeError:
                    pass

            return {
                "sentiment": "neutral",
                "score": 0.0,
                "analysis": ai_response[:500],
                "model": model,
                "source": "ollama_local",
                "raw_response": True,
                "quick": quick,
            }

        return {"error": f"Ollama HTTP {response.status_code}"}

    except requests.exceptions.Timeout:
        return {"error": f"Ollama timeout apres {timeout}s"}
    except requests.exceptions.ConnectionError:
        return {"error": "Ollama non disponible. Verifiez que ollama est lance."}
    except Exception as e:
        return {"error": str(e)}
    finally:
        OLLAMA_GENERATION_LOCK.release()


def check_ollama_status() -> dict:
    """Retourne le statut Ollama avec cache court pour eviter des /api/tags repetitifs."""
    now = time.monotonic()
    cached = OLLAMA_STATUS_CACHE.get("data")
    if cached is not None and (now - OLLAMA_STATUS_CACHE.get("timestamp", 0.0)) < OLLAMA_STATUS_CACHE_TTL:
        return cached

    ollama_url = CONFIG.get("ollama_url", "http://localhost:11434")
    current_model = CONFIG.get("ollama_model", "mistral")
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=3)
        if response.status_code == 200:
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            result = {
                "online": True,
                "models": models,
                "current_model": current_model,
                "model_available": current_model in [m.split(":")[0] for m in models],
            }
        else:
            result = {
                "online": False,
                "models": [],
                "current_model": current_model,
                "model_available": False,
            }
    except Exception:
        result = {
            "online": False,
            "models": [],
            "current_model": current_model,
            "model_available": False,
        }

    OLLAMA_STATUS_CACHE["timestamp"] = now
    OLLAMA_STATUS_CACHE["data"] = result
    return result


def get_ai_portfolio() -> dict:
    """Retourne le contexte portefeuille reel utilise par l'IA et l'interface."""
    max_positions = min(int(CONFIG.get("max_positions", 1)), runtime_cfg("hard_max_positions", HARD_MAX_POSITIONS))
    positions = []
    try:
        if state.get("connected"):
            positions = mt5.positions_get() or []
    except Exception:
        logger.exception("Erreur récupération positions pour IA")
        positions = []

    normalized = []
    currency_exposure = {}
    for pos in positions:
        symbol = str(getattr(pos, "symbol", ""))
        ptype = int(getattr(pos, "type", 0))
        action = "BUY" if ptype == 0 else "SELL"
        normalized.append({
            "ticket": int(getattr(pos, "ticket", 0)),
            "symbol": symbol,
            "type": action,
            "volume": float(getattr(pos, "volume", 0.0)),
            "profit": float(getattr(pos, "profit", 0.0)),
            "price_open": float(getattr(pos, "price_open", 0.0)),
        })
        if len(symbol) >= 6:
            base, quote = symbol[:3], symbol[3:6]
            sign = 1 if action == "BUY" else -1
            currency_exposure[base] = currency_exposure.get(base, 0) + sign
            currency_exposure[quote] = currency_exposure.get(quote, 0) - sign

    open_positions = len(normalized)
    return {
        "open_positions": open_positions,
        "max_positions": max_positions,
        "remaining_slots": max(0, max_positions - open_positions),
        "positions": normalized,
        "currency_exposure": currency_exposure,
        "hard_max_positions": runtime_cfg("hard_max_positions", HARD_MAX_POSITIONS),
        "risk_block_reason": state.get("risk_block_reason"),
    }


def find_top_opportunities(limit: int = 5) -> List[Dict]:
    """Analyse toutes les paires et retourne les meilleures opportunites."""
    print("\nAnalyse de TOUS les couples forex...")
    available_symbols = get_available_symbols()
    print(f"   {len(available_symbols)} couples disponibles")
    analyses = []
    for symbol in available_symbols:
        analysis = analyze_symbol_score(symbol)
        if "error" not in analysis and analysis.get("action") != "NEUTRAL":
            analyses.append(analysis)
    analyses.sort(key=lambda x: x.get("abs_score", 0), reverse=True)
    print("\nTOP 5 OPPORTUNITES:")
    for i, a in enumerate(analyses[:limit]):
        print(f"   {i+1}. {a['symbol']}: {a['action']} (Score: {a['score']}, Conf: {a['confidence']}%)")
    return analyses[:limit]


def ai_allows_trade(opportunity: Dict, portfolio: Dict) -> tuple[bool, str]:
    """Validation IA rapide avant une nouvelle ouverture.

    L'IA ne valide automatiquement que les signaux techniquement forts
    et respecte un cooldown entre deux validations automatiques.
    """
    if not CONFIG.get("use_ai_analysis", True):
        return True, "IA désactivée"

    score_value = float(opportunity.get("score", 0) or 0)

    if abs(score_value) < runtime_cfg("ai_validation_min_score", AI_VALIDATION_MIN_SCORE):
        return False, (
            f"Signal sous le seuil IA "
            f"({abs(score_value):g} < {runtime_cfg("ai_validation_min_score", AI_VALIDATION_MIN_SCORE)})"
        )

    global LAST_AUTO_AI_VALIDATION

    now = time.monotonic()
    elapsed = now - LAST_AUTO_AI_VALIDATION

    if elapsed < runtime_cfg("ai_auto_cooldown", AI_AUTO_COOLDOWN):
        remaining = int(runtime_cfg("ai_auto_cooldown", AI_AUTO_COOLDOWN) - elapsed)
        return False, f"Cooldown IA ({remaining}s restantes)"

    try:
        action = "BUY" if score_value > 0 else "SELL"
        context = {
            "symbol": opportunity.get("symbol"),
            "technical_action": action,
            "technical_score": opportunity.get("score", 0),
            "technical_confidence": opportunity.get("confidence", 0),
            "rsi": opportunity.get("rsi"),
            "trend": opportunity.get("trend"),
            "reasons": opportunity.get("reasons", []),
            "open_positions": portfolio.get("open_positions", 0),
            "max_positions": portfolio.get("max_positions", 1),
            "remaining_slots": portfolio.get("remaining_slots", 0),
            "positions": portfolio.get("positions", []),
            "currency_exposure": portfolio.get("currency_exposure", {}),
        }

        result = analyze_news_with_ollama(
            f"Signal technique {action} sur {opportunity.get('symbol')}",
            context=context,
            quick=True,
        )

        # Le cooldown commence apres une tentative reelle de validation IA.
        LAST_AUTO_AI_VALIDATION = time.monotonic()

        if result.get("error"):
            return False, f"Erreur IA: {result.get('error')}"

        sentiment = str(result.get("sentiment", "neutral")).lower()

        if action == "BUY" and sentiment == "bullish":
            return True, "IA bullish + technique BUY"

        if action == "SELL" and sentiment == "bearish":
            return True, "IA bearish + technique SELL"

        return False, f"Divergence IA ({sentiment}) / technique ({action})"

    except Exception as e:
        LAST_AUTO_AI_VALIDATION = time.monotonic()
        logger.exception("Erreur validation IA automatique")
        return False, f"Erreur IA: {str(e)}"

def ai_enhanced_analysis() -> dict:
    results = {"technical": None, "calendar": [], "ai_analysis": None, "combined_recommendation": None}

    best = None
    if state["connected"]:
        best = find_best_opportunity()
        results["technical"] = best

    events = get_economic_calendar()
    results["calendar"] = events

    if CONFIG.get("use_ai_analysis", True):
        portfolio = get_ai_portfolio()
        context = {
            "pairs_count": len(CONFIG["symbols"]),
            "best_symbol": best["symbol"] if best else "N/A",
            "best_action": best["action"] if best else "N/A",
            "best_score": best["score"] if best else 0,
            "upcoming_events": ", ".join([f"{e['title']} ({e['impact']})" for e in events[:3]]) if events else "Aucun",
            "open_positions": portfolio["open_positions"],
            "max_positions": portfolio["max_positions"],
            "remaining_slots": portfolio["remaining_slots"],
            "positions": portfolio["positions"],
            "currency_exposure": portfolio["currency_exposure"],
        }
        headlines = []
        if events:
            headlines.append(f"Evenements: {', '.join([e['title'] for e in events[:3]])}")
        if best:
            headlines.append(f"Signal {best['action']} sur {best['symbol']} (score {best['score']})")
        headline = " | ".join(headlines) if headlines else "Analyse generale du marche forex"

        ai_result = analyze_news_with_ollama(headline, context=context, quick=False)
        results["ai_analysis"] = ai_result

        if best and not ai_result.get("error"):
            tech_direction = "BUY" if best["score"] > 0 else "SELL"
            ai_sentiment = ai_result.get("sentiment", "neutral")
            if tech_direction == "BUY" and ai_sentiment == "bullish":
                results["combined_recommendation"] = {"action": "STRONG_BUY", "symbol": best["symbol"], "confidence": min(95, best["confidence"] + 15), "reason": "Technique + IA concordent (haussier)"}
            elif tech_direction == "SELL" and ai_sentiment == "bearish":
                results["combined_recommendation"] = {"action": "STRONG_SELL", "symbol": best["symbol"], "confidence": min(95, best["confidence"] + 15), "reason": "Technique + IA concordent (baissier)"}
            elif ai_sentiment == "neutral":
                results["combined_recommendation"] = {"action": best["action"], "symbol": best["symbol"], "confidence": best["confidence"], "reason": "Signal technique seul (IA neutre)"}
            else:
                results["combined_recommendation"] = {"action": "HOLD", "symbol": best["symbol"], "confidence": 30, "reason": "Technique et IA divergent - prudence"}

    return results


@api_router.get("/ai/status")
async def api_ai_status():
    return check_ollama_status()

@api_router.post("/ai/analyze-news")
async def api_ai_analyze_news(body: dict):
    headline = body.get("headline", "")
    content = body.get("content", "")
    if not headline:
        raise HTTPException(status_code=400, detail="headline requis")
    return analyze_news_with_ollama(headline, content)

@api_router.get("/ai/portfolio")
async def api_ai_portfolio():
    return get_ai_portfolio()

@api_router.get("/ai/full-analysis")
async def api_ai_full_analysis():
    return ai_enhanced_analysis()

@api_router.post("/ai/config")
async def api_ai_config(body: dict):
    if "ollama_model" in body:
        CONFIG["ollama_model"] = body["ollama_model"]
    if "use_ai_analysis" in body:
        CONFIG["use_ai_analysis"] = body["use_ai_analysis"]
    if "ollama_url" in body:
        CONFIG["ollama_url"] = body["ollama_url"]
    saved = save_persistent_config()
    return {
        "success": True,
        "model": CONFIG["ollama_model"],
        "enabled": CONFIG["use_ai_analysis"],
        "persistent": saved,
    }

# ============ TELEGRAM ROUTES ============
@api_router.get("/telegram/status")
async def api_telegram_status():
    """Statut de la config Telegram"""
    configured = bool(CONFIG.get("telegram_bot_token")) and bool(CONFIG.get("telegram_chat_id"))
    return {
        "enabled": CONFIG.get("telegram_enabled", False),
        "configured": configured,
        "chat_id": CONFIG.get("telegram_chat_id", "")[:4] + "****" if CONFIG.get("telegram_chat_id") else "",
        "on_open": CONFIG.get("telegram_on_open", True),
        "on_close": CONFIG.get("telegram_on_close", True),
        "on_target": CONFIG.get("telegram_on_target", True),
        "on_stoploss": CONFIG.get("telegram_on_stoploss", True),
        "on_analysis": CONFIG.get("telegram_on_analysis", False),
    }

@api_router.post("/telegram/config")
async def api_telegram_config(body: dict):
    """Met a jour la config Telegram"""
    if "bot_token" in body:
        CONFIG["telegram_bot_token"] = body["bot_token"]
    if "chat_id" in body:
        CONFIG["telegram_chat_id"] = body["chat_id"]
    if "enabled" in body:
        CONFIG["telegram_enabled"] = body["enabled"]
    if "on_open" in body:
        CONFIG["telegram_on_open"] = body["on_open"]
    if "on_close" in body:
        CONFIG["telegram_on_close"] = body["on_close"]
    if "on_target" in body:
        CONFIG["telegram_on_target"] = body["on_target"]
    if "on_stoploss" in body:
        CONFIG["telegram_on_stoploss"] = body["on_stoploss"]
    if "on_analysis" in body:
        CONFIG["telegram_on_analysis"] = body["on_analysis"]
    saved = save_persistent_config()
    logger.info(f"Telegram config: enabled={CONFIG['telegram_enabled']} | persistent={saved}")
    return {
        "success": True,
        "enabled": CONFIG["telegram_enabled"],
        "persistent": saved,
    }

@api_router.post("/telegram/test")
async def api_telegram_test():
    """Envoie un message de test Telegram"""
    if not CONFIG.get("telegram_bot_token") or not CONFIG.get("telegram_chat_id"):
        return {"success": False, "error": "Token ou Chat ID manquant"}
    was_enabled = CONFIG.get("telegram_enabled")
    CONFIG["telegram_enabled"] = True
    now = datetime.now().strftime("%H:%M:%S")
    ok = send_telegram(
        f"<b>TEST - Forex Robot Pro</b>\n"
        f"Les alertes Telegram fonctionnent!\n"
        f"Heure: {now}"
    )
    CONFIG["telegram_enabled"] = was_enabled
    return {"success": ok, "error": "" if ok else "Echec envoi. Verifiez le token et chat_id."}

# ============ SERVIR L'INTERFACE HTML ============
@app.get("/", response_class=HTMLResponse)
async def serve_interface():
    """Sert l'interface web directement depuis le serveur"""
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "interface.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            # Pas de cache navigateur : apres un git pull, l'interface a jour s'affiche.
            return HTMLResponse(content=f.read(), headers={"Cache-Control": "no-store"})
    return HTMLResponse(content="<h1>interface.html non trouve. Placez-le dans le meme dossier que forex_robot.py</h1>")

app.include_router(api_router)

# ============ MAIN ============
if __name__ == "__main__":
    print("="*60)
    print("FOREX ROBOT PRO - UBUNTU EDITION")
    print("="*60)
    print(f"   Lot: {CONFIG['lot_size']} | Objectif: +${CONFIG['target_profit']}")
    print(f"   Stop Loss: ${CONFIG['stop_loss']} | Couples: {len(CONFIG['symbols'])}")
    print(f"   Bridge MT5: {MT5_BRIDGE_HOST}:{MT5_BRIDGE_PORT}")
    print("   Mode demarrage: SAFE (aucun auto-trading automatique)")
    print(f"   Ollama: {CONFIG['ollama_url']} ({CONFIG['ollama_model']})")
    print(f"   Config persistante: {CONFIG_FILE}")
    print("="*60)

    if connect_mt5():
        print(f"\nServeur: http://0.0.0.0:8001")
        print(f"Interface: http://VOTRE_IP:8001")
        print(f"\nRoutes IA:")
        print(f"   GET  /api/ai/status")
        print(f"   POST /api/ai/analyze-news")
        print(f"   GET  /api/ai/portfolio")
        print(f"   GET  /api/ai/full-analysis")
        print()
        uvicorn.run(app, host="0.0.0.0", port=8001)
    else:
        print("\nImpossible de se connecter a MT5 via le bridge.")
        print("Verifiez que :")
        print("  1. MetaTrader 5 est initialise dans le WINEPREFIX .mt5")
        print("  2. Le serveur mt5linux/RPyC est lance")
        print(f"  3. Le bridge ecoute sur {MT5_BRIDGE_HOST}:{MT5_BRIDGE_PORT}")
        print("  4. Le terminal MT5 est connecte au compte souhaite")
        # Code non nul : systemd (Restart=on-failure/always) relance le robot.
        raise SystemExit(1)

