# --- НАЧАЛО ПЕРВОЙ ЧАСТИ (КОНФИГУРАЦИЯ И КОНСТАНТЫ) ---

import sys
import os
print(f"EXECUTABLE: {sys.executable}")
print(f"SYS.PATH: {sys.path}")

import random
import matplotlib.pyplot as plt
import base64
import aiofiles
import asyncio
import logging
import signal
import time
import re
import math
import traceback
import datetime
from datetime import timezone as dt_timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN, InvalidOperation
from functools import partial
import copy
import json
import csv
from typing import Optional, Dict, Any, List, Set, Tuple
import warnings
import html
import httpx
import aiohttp
import io
from collections import deque, defaultdict

from direction_filter import BTCDirectionFilter, EnsembleConfig
from typing import Tuple
from scipy.signal import find_peaks
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from openai import OpenAI
from aiohttp.resolver import AsyncResolver
# ... (другие импорты)
from dataclasses import dataclass, field, asdict

# ... (остальные импорты)

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

import google.generativeai as genai
from google.generativeai import types
# <<< КОНЕЦ ИСПРАВЛЕНИЯ >>>




try:
    from hmmlearn.hmm import GaussianHMM
    import numpy as np
    print("INFO: Библиотека 'hmmlearn' успешно импортирована.")
except ImportError:
    print("!!! КРИТИЧЕСКАЯ ОШИБКА: Библиотека 'hmmlearn' не найдена. Установите ее: pip install hmmlearn")
    sys.exit("Остановка из-за отсутствия библиотеки 'hmmlearn'.")

try:
    from hmmlearn.base import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    print("INFO: Предупреждения о сходимости HMM будут скрыты.")
except ImportError:
    pass

try:
    import aiofiles
except ImportError:
    print("!!! ПРЕДУПРЕЖДЕНИЕ: Библиотека aiofiles не найдена. Логирование ордеров будет синхронным.")
    aiofiles = None

try:
    import mplfinance as mpf
    import matplotlib
    # Эта строка нужна для работы в фоновом режиме на сервере
    matplotlib.use('Agg') 
except ImportError:
    print("!!! ПРЕДУПРЕЖДЕНИЕ: Библиотеки для построения графиков не найдены. Команда /trend_chart не будет работать.")
    print("!!! Установите их: pip install matplotlib mplfinance")
    mpf = None

try:
    import pandas as pd
    import numpy as np
    try:
        import ta
        from ta.volatility import average_true_range, BollingerBands
        from ta.trend import adx, ema_indicator, MACD, psar_up, psar_down
        from ta.momentum import rsi
        print("INFO: Библиотека 'ta' успешно импортирована.")
    except ImportError:
        print("!!! КРИТИЧЕСКАЯ ОШИБКА: Библиотека 'ta' не найдена.")
        sys.exit("Остановка из-за отсутствия библиотеки 'ta'.")

    from binance import (
        AsyncClient, BinanceSocketManager, Client,
        BinanceAPIException, BinanceOrderException
    )

    import telegram
    try:
        from telegram import constants as TG_CONSTANTS
    except ImportError:
        TG_CONSTANTS = type('obj', (object,), {'MessageLimit': type('obj', (object,), {'MAX_TEXT_LENGTH': 4096})})()

    from telegram import Update, BotCommand, __version__ as TG_VER
    try:
        from telegram import __version_info__
    except ImportError:
        __version_info__ = (0, 0, 0, 0, 0)
    if __version_info__ < (20, 0, 0, "alpha", 1):
        raise RuntimeError(f"Этот скрипт требует python-telegram-bot v20.0 или выше (у вас {TG_VER})")

    try:
        from websockets.exceptions import ConnectionClosedError
    except ImportError:
        ConnectionClosedError = ConnectionError

    from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes, Application

    from functools import partial # Убедитесь, что этот импорт есть в начале файла



except ImportError as e:
    print(f"!!! КРИТИЧЕСКАЯ ОШИБКА: Не найдена или не удалось импортировать библиотеку/модуль - {e.name}")
    sys.exit(f"Остановка из-за ошибки импорта: {e.name}")


# Ключи заменены на строки-заглушки
API_KEY = 'EKdpOeAH8Sgs1y6PZ9OndP4qlFPocAyqp4vRTCvaUQNxfnaCvgF3jAx4xCOJKlUU' # ВАШ КЛЮЧ или плейсхолдер
API_SECRET = 'DAvVu4AabC6MYIrBPCl7oHHxlG2Prwf2YeuyoIYPH7uJE42HuWBykbajKfe0IOFA' # ВАШ СЕКРЕТ или плейсхолдер
TELEGRAM_TOKEN = '7845317270:AAH_ehzWYOpGoV9NXyZvZ2CVlz1a2nReI4k' # ВАШ ТОКЕН или плейсхолдер
TELEGRAM_CHAT_ID = '5962765968'

# --- ✅ ПАРАМЕТРЫ DEEPSEEK (упрощенная версия с одним ключом) ---
DEEPSEEK_API_KEY = 'sk-b11ea716c0fa47aaab82edc5ec7efaae'  # ВАШ ОСНОВНОЙ КЛЮЧ

# Базовый URL для OpenAI-совместимого клиента (для Vision API)
DEEPSEEK_API_BASE_URL = "https://api.deepseek.com/v1" 


# Полный URL для прямых запросов через httpx (для текстовых функций)
DEEPSEEK_CHAT_COMPLETIONS_URL = "https://api.deepseek.com/v1/chat/completions"



# Полный URL для прямых запросов через httpx (для старой текстовой функции)
GEMINI_API_KEY = "AIzaSyBKKuUZsaEcU_Jn4gPWqQS_Uf679Csv8EU"  # <--- ВСТАВЬТЕ ВАШ КЛЮЧ ВМЕСТО ЭТОЙ СТРОКИ
# <<< КОНЕЦ ИЗМЕНЕНИЯ >>>



GEMINI_MODEL = "gemini-2.5-pro" 

# Проверка, что ключ был вставлен
if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
    print("!!! ПРЕДУПРЕЖДЕНИЕ: GEMINI_API_KEY не установлен! Визуальный анализ графиков будет отключен.")
    GEMINI_API_KEY = None # Явно устанавливаем в None, чтобы проверка в функции сработала


if not API_KEY or API_KEY == 'YOUR_BINANCE_API_KEY' or not API_SECRET or API_SECRET == 'YOUR_BINANCE_API_SECRET':
    print("!!! КРИТИЧЕСКАЯ ОШИБКА: BINANCE_API_KEY или BINANCE_API_SECRET не установлены!")
    sys.exit("Остановка: Отсутствуют ключи Binance API.")

if not TELEGRAM_TOKEN or TELEGRAM_TOKEN == 'YOUR_TELEGRAM_BOT_TOKEN' or not TELEGRAM_CHAT_ID or TELEGRAM_CHAT_ID == 'YOUR_TELEGRAM_CHAT_ID':
    print("!!! ПРЕДУПРЕЖДЕНИЕ: TELEGRAM_BOT_TOKEN или TELEGRAM_CHAT_ID не установлены!")
    TELEGRAM_TOKEN = None
    TELEGRAM_CHAT_ID = None

ALLOWED_INTERVALS = {
    '1m': Client.KLINE_INTERVAL_1MINUTE, '3m': Client.KLINE_INTERVAL_3MINUTE,
    '5m': Client.KLINE_INTERVAL_5MINUTE, '15m': Client.KLINE_INTERVAL_15MINUTE,
    '30m': Client.KLINE_INTERVAL_30MINUTE, '1h': Client.KLINE_INTERVAL_1HOUR,
    '2h': Client.KLINE_INTERVAL_2HOUR, '4h': Client.KLINE_INTERVAL_4HOUR,
    '1d': Client.KLINE_INTERVAL_1DAY
}

# --- ОСНОВНЫЕ ТОРГОВЫЕ ПАРАМЕТРЫ И ФИЛЬТРЫ ---
TOP_CAP_SYMBOLS = {'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'LTCUSDT'}

SYMBOLS =  ['ENJUSDT', 'CHZUSDT', 'ONEUSDT', 'ANKRUSDT', 'CRVUSDT', 'SUSHIUSDT',
        'ALGOUSDT','SOLUSDT','BNBUSDT','XRPUSDT','ADAUSDT','AVAXUSDT','DOTUSDT','ATOMUSDT','NEARUSDT',
        'LINKUSDT','ICPUSDT','FILUSDT','FETUSDT','IMXUSDT','DOGEUSDT',
        'BTCUSDT','ETHUSDT','TRXUSDT','LTCUSDT','BCHUSDT','XLMUSDT','HBARUSDT','SUIUSDT',
        'OPUSDT','ARBUSDT','INJUSDT','SEIUSDT','TIAUSDT','APTUSDT','APEUSDT','WIFUSDT','ENAUSDT']
LEVERAGE = 13
BASE_RISK_PERCENT = 1.5
MAX_RISK_MODIFIER = 1.5
COLLATERAL_ASSET = 'USDT'
API_RETRY_DELAY_SECONDS = 10
PERIODIC_CHECK_INTERVAL_SECONDS = 60
COMMON_ATR_PERIOD_FOR_SLTP = 14
MIN_RR_RATIO = 1.5
POST_ENTRY_DELAY_SECONDS = 2
SLIP_BP = 2.5 # bps (базисные пункты)
LIMIT_OFFSET_BPS = 1.5 # bps для LIMIT GTX
PARTIAL_TP_RR = 1.5 # Оставим как базовый, если лестница не сработает
TRAILING_TP_CALLBACK_RATE = 0.5 # Оставим, может использоваться как fallback
# --- ШАГ 2: ПАРАМЕТРЫ ЛЕСТНИЦЫ ФИКСАЦИИ ---
# --- ПАРАМЕТРЫ ЛЕСТНИЦЫ ФИКСАЦИИ ---
TP1_RR = 0.7            # R:R для первого частичного закрытия
TP1_CLOSE_FRAC = 0.30   # Доля закрытия на TP1 (30%)
BE_TRIGGER_RR = 1.2     # ✅ R:R для перевода в Б/У (было 1.8, по вашему запросу ставим 1.2)
BE_ATR_BUFFER_FRAC = 0.25 # Буфер для стопа в Б/У (0.25 * ATR)
TP2_RR = 1.4            # R:R для второго частичного закрытия
TP2_CLOSE_FRAC = 0.30   # Доля закрытия на TP2 (еще 30%)
TRAIL_ATR_MULT = 1.8    # Множитель ATR для трейлинга остатка
# --- КОНЕЦ ШАГА 2 ---
# --- ШАГ 4: ПАРАМЕТРЫ ПИРАМИДИНГА ---
PYRAMID_MIN_RR = 1.25          # Минимальный RR для добавления
PYRAMID_MAX_STEPS = 3          # Максимальное количество добавлений
PYRAMID_OBI_GATE = 10          # Порог OBI для подтверждения (для примера)
PYRAMID_STEP_FRACS = (0.30, 0.25, 0.20) # Доли от начального объема для добавлений

# --- НОВЫЕ КОНСТАНТЫ ДЛЯ УЛУЧШЕННОГО SL ---
MIN_SL_ATR_MULT_BASE = 2.0      # Минимальный базовый множитель ATR для SL
ADX_LOW_THRESHOLD_FOR_SL = 18.0 # Порог ADX, ниже которого SL расширяется
SL_ATR_MULT_INCREASE_LOW_ADX = 0.75 # На сколько УВЕЛИЧИВАЕТСЯ множитель при низком ADX
STRUCTURAL_SL_BUFFER_ATR_MULT = 1.0 # Множитель ATR для буфера за структурным уровнем (было 0.3)
BE_TRIGGER_RR = 1.8             # R:R для перевода в безубыток (было 1.0)
CHANDELIER_EXIT_PERIOD = 22     # Период для Chandelier Exit
CHANDELIER_EXIT_MULTIPLIER = 3.0 # Множитель ATR для Chandelier Exit
# --- КОНЕЦ НОВЫХ КОНСТАНТ ---

# Можно также увеличить период ATR по умолчанию, если хотите
# COMMON_ATR_PERIOD_FOR_SLTP = 21 # Вместо 14

TREND_WEAKENING_ADX_THRESHOLD = 20.0 # Порог ADX для выхода при ослаблении тренда
AGGRESSIVE_TRAIL_RR_THRESHOLD = 4.0  # Порог R:R для включения агрессивного трейлинга
AGGRESSIVE_TRAIL_ATR_MULT = 1.2    # Множитель ATR для агрессивного трейлинга
# --- НОВЫЕ ЛИМИТЫ И ПАРАМЕТРЫ БЕЗОПАСНОСТИ РАСЧЕТА ОБЪЕМА ---
MIN_SL_ATR_MULT = 0.25          # Минимальная дистанция до SL в долях ATR (0.25 = 25% от ATR)
MIN_SL_PRICE_STEP_MULT = 5.0    # Минимальная дистанция до SL в количестве шагов цены (tickSize)
MIN_SL_ENTRY_FRAC = 0.001       # Минимальная дистанция до SL в долях от цены входа (0.001 = 0.1%)
CLAMP_KELLY_MIN = 0.05          # Минимальный множитель Келли
CLAMP_KELLY_MAX = 0.25          # Максимальный множитель Келли
CLAMP_FINAL_RISK_MIN_PCT = 0.25 # Минимальный итоговый риск на сделку в %
CLAMP_FINAL_RISK_MAX_PCT = 1.50 # Максимальный итоговый риск на сделку в %
MAX_USD_NOTIONAL_PER_TRADE = 100.0 # Макс. стоимость позиции в USD (можешь изменить)
MAX_QTY_PER_SYMBOL = None       # Макс. кол-во монет в позиции (None = без лимита)
# --- КОНЕЦ НОВЫХ ЛИМИТОВ ---
# --- SL НА ОСНОВЕ AI-АНАЛИЗА ВОЛАТИЛЬНОСТИ ---
MIN_ATR_MULTIPLIER = 1.5
MAX_ATR_MULTIPLIER = 5.0
DEFAULT_ATR_MULTIPLIER = 3.0

# --- НАСТРОЙКИ ДВУХЭТАПНОГО TAKE PROFIT ---
PARTIAL_TP_RR = 1.5
TRAILING_TP_CALLBACK_RATE = 0.5  

# --- ПОГРЕШНОСТЬ ДЛЯ РАСЧЕТА МИНИМАЛЬНОЙ СУММЫ ОРДЕРА ---
MIN_NOTIONAL_TOLERANCE = 1.05

# --- НАСТРОЙКИ ДЛЯ ДИНАМИЧЕСКОГО ПОРОГА ВХОДА (Итоговый балл AI) ---
VOLATILITY_CHECK_TIMEFRAME = '15m'
VOLATILITY_CHECK_PERIOD = 14
VOLATILITY_LOW_THRESHOLD = 0.2 # Сделали бота менее "пугливым"
VOLATILITY_HIGH_THRESHOLD = 2.0
SCORE_THRESHOLD_LOW_VOL = 65   
SCORE_THRESHOLD_NORMAL_VOL = 65
SCORE_THRESHOLD_HIGH_VOL = 65

# --- ✅ НАСТРОЙКИ ВХОДА ПО МОМЕНТУМУ УВЕРЕННОСТИ ---
ENABLE_CONFIDENCE_MOMENTUM = True
MIN_CONFIDENCE_INCREASE = 5
MIN_ABSOLUTE_CONFIDENCE = 60

# --- ДИНАМИЧЕСКИЕ ПОРОГИ УВЕРЕННОСТИ ML ---
AUTOCONFIDENCE_IMPULSE_THRESHOLD = 0.65
AUTOCONFIDENCE_TREND_THRESHOLD = 0.65   
AUTOCONFIDENCE_RANGING_THRESHOLD = 0.65
DEEPSEEK_CONFIDENCE_THRESHOLD = 65

TRADE_HISTORY = deque(maxlen=100) # Хранилище для истории сделок
CONSECUTIVE_WINS = 0
CONSECUTIVE_LOSSES = 0

# --- ОБЩИЕ ФИЛЬТРЫ СТРАТЕГИИ ---
ENFORCE_UNIDIRECTIONAL_TRADES = True
BTC_TREND_FILTER_ENABLED = False             
TREND_VOLUME_MULTIPLIER = 1.5               
AI_CONSOLIDATION_FILTER_ENABLED = True
MULTI_TF_TREND_FILTER_ENABLED = False
CORRELATION_FILTER_ENABLED = False
CORRELATION_THRESHOLD = 0.65
AB_TESTING_ENABLED = False # Отключаем, чтобы логика была одинаковой для всех
GROUP_B_RISK_PERCENT = 2.0
GROUP_B_SCORE_THRESHOLD_MODIFIER = 0.95
MARKET_SENTIMENT_FILTER_ENABLED = True
MAX_OPEN_POSITIONS = 4 # Лимит одновременно открытых позиций
ENABLE_PORTFOLIO_REVERSAL = True
EMERGENCY_ATR_MULTIPLIER = 2.5 
ENABLE_DYNAMIC_SL = True          # Включить/выключить умный стоп-лосс
DYNAMIC_SL_TRIGGER_PROXIMITY = 0.5  # % от ATR. Если цена ближе чем 0.5 * ATR к стопу, запускаем анализ
DYNAMIC_SL_MOVE_DISTANCE = 0.75     # % от ATR. На какое расстояние отодвинуть стоп.
TC_TREND_FILTER_TIMEFRAME = '1h' # Таймфрейм для анализа глобального тренда
BTC_EMA_SLOW_PERIOD = 50         # Период для медленной EMA (основной тренд)
BTC_EMA_FAST_PERIOD = 20         # Период для быстрой EMA (подтверждение)
BTC_ADX_THRESHOLD = 22
FLAT_ADX_THRESHOLD = 22.0 # Порог ADX для определения флэта (взят из 2.py) 
ENABLE_ATR_TRAILING_STOP = True
TRAILING_STOP_ATR_PERIOD = 14
TRAILING_STOP_ATR_MULTIPLIER = 2.0
TRAILING_STOP_ACTIVATION_RR = 1.0
TRAILING_STOP_ACTIVATION_PERCENT = 1.2
ML_CONFIDENCE_THRESHOLD = 0.65  # Минимальная вероятность для сигнала (75%)
ML_CERTAINTY_MARGIN = 0.25      # Сигнал должен быть сильнее второго варианта на 25%
EARLY_BE_TRIGGER_RR = 1.2 
EARLY_BE_MIN_PROFIT_PCT = 0.4 # альтернативный триггер по % от цены входа
ADAPTIVE_ADX_THRESHOLD_LOW = 15.0      # Базовый порог ADX для флэта
ADAPTIVE_ADX_THRESHOLD_HIGH_VOL = 18.0  # Порог ADX в периоды высокой волатильности
MARKET_DEPTH_THRESHOLD_USD = 150000     # Порог ликвидности в стакане ($)
ENSEMBLE_CONFIDENCE_THRESHOLD = 0.50 # Порог уверенности для нового ансамблевого фильтра (от 0.5 до 1.0)
MAX_DIVERGENCE_AGE = 5
DCA_AI_CONFIDENCE_THRESHOLD = 65
MAX_DATAFRAME_ROWS = 1000
DCA_COOLDOWN_MINUTES = 15 # ✅ Время в минутах, на которое DCA ставится на паузу при отказе AI
GLOBAL_MARKET_DIRECTION = 'SIDE' # 'UP', 'DOWN', или 'SIDE'
GLOBAL_DIRECTION_LAST_UPDATE = 0
MIN_DIRECTION_HOLD_SEC = 300  # 5 минут удержания глобального направления

# --- НАСТРОЙКИ СТРАТЕГИЙ ---
# -- Контртрендовая (Ranging) --
RANGING_SIGNAL_TF = '15m'
RANGING_RSI_OVERSOLD = 35      # Классическое значение: 30
RANGING_RSI_OVERBOUGHT = 65    # Классическое значение: 70
RANGING_BB_STD_DEV = 1.9       # Классическое значение: 2.0

# -- Трендовая (Trending) --
TRENDING_SIGNAL_TF = '15m'
TRENDING_FAST_EMA = 20
TRENDING_SLOW_EMA = 50
TRENDING_ADX_THRESHOLD = 20.0
TRENDING_RR_RATIO = 2.0
TRENDING_SL_ATR_MULTIPLIER = 2.5 # Множитель для гибридного стопа
IMPULSE_VOLUME_MULTIPLIER = 2.5 # Объем должен быть в 2.5 раза выше среднего
IMPULSE_BODY_ATR_MULTIPLIER = 2.0 # Тело свечи должно быть в 2 раза больше ATR
# -- Общие --
ADX_REGIME_THRESHOLD = 23.0


DAILY_LOSS_LIMIT_PERCENT = -6.0  # -6% от баланса
WEEKLY_LOSS_LIMIT_PERCENT = -12.0
TRADING_PAUSED = False
_equity_curve = deque(maxlen=10_000) # (timestamp, balance)

LIQ_WINDOW_SEC = 60
LIQ_NOTIONAL_THRESHOLD = 2_000_000  # USD за окно
LIQ_SPIKE_MULTIPLIER = 4.0          # x среднего окна
PYRAMID_MIN_RR = 1.25
PYRAMID_MAX_STEPS = 3
PYRAMID_OBI_GATE = 10
PYRAMID_STEP_FRACS = (0.30, 0.25, 0.20)

_liq_buffer = defaultdict(lambda: deque())  # symbol -> deque[(ts, side, notional)]
_liq_stats = defaultdict(lambda: {"sum": 0.0, "buy": 0.0, "sell": 0.0, "avg": 0.0})



HMM_N_STATES = 3 # Количество режимов (напр., 0=низкая волатильность, 1=средняя, 2=высокая/тренд)
HMM_TRAIN_PERIODS = 1400 # Количество свечей для обучения HMM (рекомендуется > 3*365/свечи_в_день)

REGIME_CONFIG = {
    # Режим 0: Низкая волатильность / Флэт
    0: {'name': 'Low Volatility', 'risk_modifier': 0.75, 'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.0},
    
    # Режим 1: Средняя волатильность / Неопределенность
    1: {'name': 'Medium Volatility', 'risk_modifier': 1.0, 'atr_multiplier_sl': 2.5, 'atr_multiplier_tp': 3.0},
    
    # Режим 2: Высокая волатильность / Тренд
    2: {'name': 'High Volatility/Trend', 'risk_modifier': 1.25, 'atr_multiplier_sl': 3.5, 'atr_multiplier_tp': 5.0}
}

# --- ПАРАМЕТРЫ ML МОДЕЛЕЙ ---
MODELS_DIR = 'models_v7_9_dual_mode' # УКАЗЫВАЕМ НОВУЮ ПАПКУ
ML_MODEL_TIMEFRAMES = ['15m', '1h', '4h']
ML_MODEL_TF_SHORT = ML_MODEL_TIMEFRAMES[0]  # Теперь это '15m'
ML_MODEL_TF_MEDIUM = ML_MODEL_TIMEFRAMES[1] # Теперь это '1h'
ML_MODEL_TF_LONG = ML_MODEL_TIMEFRAMES[2]
FLAT_ADX_THRESHOLD = 22.0 # Порог ADX для определения флэта (взят из 2.py)

SAFETY_FILTER_TIMEFRAME = '15m'

if SAFETY_FILTER_TIMEFRAME not in ALLOWED_INTERVALS:
    sys.exit(f"Остановка: Таймфрейм для фильтра безопасности ({SAFETY_FILTER_TIMEFRAME}) не найден в ALLOWED_INTERVALS.")

KLINE_INTERVAL = ALLOWED_INTERVALS[ML_MODEL_TF_SHORT]

# --- СИСТЕМНЫЕ ПЕРЕМЕННЫЕ ---
MAX_DATAFRAME_ROWS = 1000

hmm_models_manager: Dict[str, Dict[str, Any]] = {} # Хранилище для обученных HMM моделей
market_regimes: Dict[str, int] = {} # Текущий режим для каждого символа

def recalculate_max_rows():
    global MAX_DATAFRAME_ROWS
    # Убедитесь, что здесь есть число, достаточное для дневной EMA
    periods = [COMMON_ATR_PERIOD_FOR_SLTP, HMM_TRAIN_PERIODS, 200, 300] # Добавлено 300 для запаса
    valid_periods = {p for p in periods if isinstance(p, int) and p > 0}
    max_period = max(valid_periods) if valid_periods else 300
    MAX_DATAFRAME_ROWS = max_period + 100
    logging.info(f"Recalculated MAX_DATAFRAME_ROWS = {MAX_DATAFRAME_ROWS} based on Max Period = {max_period}")

client: Optional[AsyncClient] = None
telegram_bot: Optional[telegram.Bot] = None
telegram_app: Optional[Application] = None
exchange_info_cache: dict = {}
market_data_store: dict = {}
current_positions: dict = {}
current_balance: float = 100.0
loop: Optional[asyncio.AbstractEventLoop] = None
websocket_connected: bool = False
last_processed_kline_time: dict = {}
pending_tasks: set = set()
ws_task_handle: Optional[asyncio.Task] = None
periodic_task: Optional[asyncio.Task] = None
order_log_lock = asyncio.Lock()
trade_csv_log_lock = asyncio.Lock()
ai_verdict_log_lock = asyncio.Lock()
trade_decision_locks: Dict[str, asyncio.Lock] = {}
initial_trade_lock = asyncio.Lock() 
state_lock = asyncio.Lock()
market_env_lock = asyncio.Lock()
global_market_environment = 'uncertain'
global_market_env_last_checked = 0
last_ai_confidence_scores = {} # ✅ Переменная для хранения прошлых значений
btc_dir_filter = BTCDirectionFilter(EnsembleConfig())
btc_filter_state = {"direction": 0, "prob_long": 0.5, "last_updated": 0}
GLOBAL_ENTRY_LOCK = asyncio.Lock() #
symbol_cooldown_until = {} # ✅ И ЭТУ СТРОКУ
global_market_environment = 'uncertain'
SERVER_TIME_OFFSET = 0.0

ORDER_LOG_FILE: str = "orders_log.json"
AI_VERDICT_LOG_FILE: str = "ai_verdicts_log.csv"
LATEST_AI_VERDICTS = deque(maxlen=15)
AI_REQUEST_SEMAPHORE = asyncio.Semaphore(5) 
_micro_features_cache = {}
CACHE_TTL_SECONDS = 3  # Время жизни кэша в секундах



message_queue = asyncio.Queue()
NUM_WORKERS = 4 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [%(name)s:%(funcName)s:%(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', handlers=[logging.StreamHandler()], force=True)
for logger_name in ["httpx", "telegram", "binance", "websockets", "ta"]: logging.getLogger(logger_name).setLevel(logging.WARNING)
if 'ml_logger' in globals(): ml_logger.setLevel(logging.INFO)
logging.info("Уровни логирования настроены.")

def is_authorized(chat_id: int) -> bool: return str(chat_id) == str(TELEGRAM_CHAT_ID)

async def send_telegram_message(message: str):
    if not telegram_bot or not TELEGRAM_CHAT_ID: return
    try:
        if len(message) > TG_CONSTANTS.MessageLimit.MAX_TEXT_LENGTH:
            for i in range(0, len(message), TG_CONSTANTS.MessageLimit.MAX_TEXT_LENGTH):
                await telegram_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message[i:i+TG_CONSTANTS.MessageLimit.MAX_TEXT_LENGTH], parse_mode='HTML')
                await asyncio.sleep(0.5)
        else:
            await telegram_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='HTML')
    except telegram.error.BadRequest as e:
        logging.error(f"ОШИБКА HTML-РАЗМЕТКИ В TELEGRAM: {e}")
        clean_message = html.escape(message)
        await telegram_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"⚠️ Ошибка форматирования.\n\n{clean_message}")
    except Exception as e:
        logging.error(f"Другая ошибка при отправке сообщения в Telegram: {e}", exc_info=True)

# --- НАЧАЛО ВТОРОЙ ЧАСТИ (БАЗОВЫЕ КОМАНДЫ TG) ---

def _purge_old_liquidations(symbol, now_ts):
    dq = _liq_buffer[symbol]
    total = buys = sells = 0.0
    while dq and now_ts - dq[0][0] > LIQ_WINDOW_SEC:
        dq.popleft()
    for _, side, notion in dq:
        total += notion
        if side == "BUY": buys += notion
        else: sells += notion
    
    stats = _liq_stats[symbol]
    stats["sum"] = total
    stats["buy"] = buys
    stats["sell"] = sells
    stats["avg"] = 0.98 * stats.get("avg", 0.0) + 0.02 * total # Экспоненциальная средняя

def register_liquidation(symbol: str, side: str, notional_usd: float, ts: float = None):
    now_ts = ts or time.time()
    _liq_buffer[symbol].append((now_ts, "BUY" if side.upper()=="BUY" else "SELL", float(notional_usd)))
    _purge_old_liquidations(symbol, now_ts)

def get_liquidation_signal(symbol: str):
    """Возвращает сигнал о всплеске ликвидаций."""
    stats = _liq_stats[symbol]
    total, avg = stats["sum"], max(1.0, stats["avg"])
    is_spike = total > max(LIQ_NOTIONAL_THRESHOLD, LIQ_SPIKE_MULTIPLIER * avg)
    bias = "BUY" if stats["buy"] > stats["sell"] * 1.5 else "SELL" if stats["sell"] > stats["buy"] * 1.5 else "NEUTRAL"
    score = min(100, int(100 * (total / max(LIQ_NOTIONAL_THRESHOLD, 1.0))))
    return {"spike": is_spike, "bias": bias, "score": score, "window_sec": LIQ_WINDOW_SEC}

async def get_basis_and_funding(symbol: str):
    """Получает премию фьючерса к индексу и текущую ставку финансирования."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as cli:
            r = await cli.get("https://fapi.binance.com/fapi/v1/premiumIndex", params={"symbol": symbol})
            r.raise_for_status()
            j = r.json()
            mark = float(j.get("markPrice", 0))
            indexp = float(j.get("indexPrice", 0))
            funding = float(j.get("lastFundingRate", 0))
            basis_bps = (mark / indexp - 1.0) * 10_000.0 if indexp > 0 else 0.0
            return {"basis_bps": basis_bps, "funding": funding}
    except Exception:
        logging.warning(f"[{symbol}] Не удалось получить данные basis/funding.")
        return {"basis_bps": 0.0, "funding": 0.0}



def estimate_slippage_bps(orderbook: dict, side: str, qty: float):
    """Оценивает проскальзывание в базисных пунктах для заданного объема."""
    levels = orderbook.get("asks") if side.upper()=="BUY" else orderbook.get("bids")
    if not levels: return 9999.0, 0.0
    remain, vwap, filled = qty, 0.0, 0.0
    for p, q in ((float(p), float(q)) for p,q in levels):
        take = min(remain, q); vwap += p * take; filled += take; remain -= take
        if remain <= 1e-12: break
    if filled <= 0: return 9999.0, 0.0
    vwap /= filled
    best = float(orderbook["asks"][0][0] if side.upper()=="BUY" else orderbook["bids"][0][0])
    bps = (vwap / best - 1.0) * 10_000.0 if side.upper()=="BUY" else (1.0 - vwap / best) * 10_000.0
    return max(0.0, bps), vwap

async def stream_liquidations(symbols: list[str]):
    logging.info("🚀 Запуск стрима ликвидаций (v2, исправленная)...")
    url = "wss://fstream.binance.com/ws/!forceOrder@arr"
    symbol_set = set(symbols)
    while True:
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.ws_connect(url, autoping=True) as ws:
                    logging.warning("✅ Стрим ликвидаций успешно подключен.")
                    async for msg in ws:
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            continue
                        payload = json.loads(msg.data)
                        
                        # Обработка как одиночного события, так и массива
                        events = [payload] if "e" in payload and payload["e"] == "forceOrder" else payload
                        
                        for ev in events:
                            o = ev.get("o") or {}
                            sym = o.get("s")
                            if not sym or sym not in symbol_set:
                                continue
                            
                            # Сторона определяется по полю 'S' (Side)
                            # Ликвидация лонга - это SELL ордер, ликвидация шорта - BUY.
                            side = "BUY" if o.get("S") == "BUY" else "SELL"
                            
                            qty = float(o.get("q", 0) or 0)
                            price = float(o.get("ap", 0) or o.get("p", 0) or 0)
                            notional = qty * price
                            if notional > 0:
                                register_liquidation(sym, side, notional)
        except Exception as e:
            logging.error(f"Ошибка в стриме ликвидаций: {e}. Переподключение через 15с...")
            await asyncio.sleep(15)

def is_bullish_microstructure(ms: dict, ofi: dict) -> bool:
    """Проверяет, является ли микроструктура рынка бычьей."""
    return (
        ms and ofi and
        ms.get('bid_ask_spread_pct', 1.0) <= 0.05 and  # Спред не более 0.05%
        ms.get('order_book_imbalance', 0.0) >= 0.05 and     # Давление покупателей в стакане
        ms.get('market_depth_usd', 0.0) >= MARKET_DEPTH_THRESHOLD_USD and
        ofi[1] == 'buy'  # Сигнал от потока ордеров на покупку
    )

def is_bearish_microstructure(ms: dict, ofi: dict) -> bool:
    """Проверяет, является ли микроструктура рынка медвежьей."""
    return (
        ms and ofi and
        ms.get('bid_ask_spread_pct', 1.0) <= 0.05 and
        ms.get('order_book_imbalance', 0.0) <= -0.05 and    # Давление продавцов в стакане
        ms.get('market_depth_usd', 0.0) >= MARKET_DEPTH_THRESHOLD_USD and
        ofi[1] == 'sell' # Сигнал от потока ордеров на продажу
    )

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update.effective_chat.id):
        await update.message.reply_text("Нет доступа.")
        return
    user = update.effective_user
    logging.info(f"User {user.first_name} ({update.effective_chat.id}) /start")
    await update.message.reply_html(f"Привет, {user.mention_html()}! Бот запущен.\n/status - статус\n/help - справка по командам.")

# =================================================================
# ## БЛОК НОВЫХ ФУНКЦИЙ ДЛЯ DEEPSEEK VISION API ##
# =================================================================

def update_equity_curve(balance: float, ts: float = None):
    """Добавляет новую точку баланса в кривую эквити."""
    now_ts = ts or time.time()
    _equity_curve.append((now_ts, balance))

def check_risk_guard() -> Tuple[bool, str]:
    """
    Проверяет, не превышены ли лимиты просадки.
    Возвращает (True, "OK") если торговля разрешена.
    """
    global TRADING_PAUSED, current_balance
    if TRADING_PAUSED:
        return False, "Торговля на паузе из-за Risk Guardian."

    if len(_equity_curve) < 2:
        return True, "OK"

    now_ts = time.time()
    
    # Ищем баланс ~24 часа назад и ~7 дней назад
    balance_24h_ago = current_balance
    balance_7d_ago = current_balance
    
    for ts, balance in reversed(_equity_curve):
        if now_ts - ts <= 86400: # 24 часа
            balance_24h_ago = balance
        if now_ts - ts <= 604800: # 7 дней
            balance_7d_ago = balance
        else:
            break # Дальше смотреть нет смысла

    # Расчет просадки в %
    daily_pnl_percent = (current_balance / balance_24h_ago - 1.0) * 100 if balance_24h_ago > 0 else 0
    weekly_pnl_percent = (current_balance / balance_7d_ago - 1.0) * 100 if balance_7d_ago > 0 else 0

    if daily_pnl_percent <= DAILY_LOSS_LIMIT_PERCENT:
        TRADING_PAUSED = True
        reason = f"STOP: Дневной лимит убытка ({DAILY_LOSS_LIMIT_PERCENT}%) превышен. Текущая просадка: {daily_pnl_percent:.2f}%"
        logging.critical(reason)
        asyncio.create_task(send_telegram_message(f"🚨 **{reason}**"))
        return False, reason

    if weekly_pnl_percent <= WEEKLY_LOSS_LIMIT_PERCENT:
        TRADING_PAUSED = True
        reason = f"STOP: Недельный лимит убытка ({WEEKLY_LOSS_LIMIT_PERCENT}%) превышен. Текущая просадка: {weekly_pnl_percent:.2f}%"
        logging.critical(reason)
        asyncio.create_task(send_telegram_message(f"🚨 **{reason}**"))
        return False, reason
        
    return True, "OK"

async def enhanced_getdeepseekverdict_with_vision(
    symbol: str,
    side: str,
    market_env: str) -> Dict[str, Any]:
    """Гибридный анализ: текстовая аналитика (DeepSeek) + визуальный анализ (Gemini)."""
    try:
        # Запускаем оба запроса параллельно для скорости
        text_analysis_task = get_ai_model_verdict_async(symbol, side, market_env)
        
        df_short = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS.get("5m"))
        if df_short is None or len(df_short) < 20:
            logging.warning(f"[{symbol}] Недостаточно данных для виз. анализа, используется только текст.")
            return await text_analysis_task

        entry_price = float(df_short['close'].iloc[-1])
        
        vision_analysis_task = get_vision_enhanced_trade_decision(
            symbol=symbol, side=side, entry_price=entry_price, market_env=market_env
        )

        text_analysis, vision_analysis = await asyncio.gather(text_analysis_task, vision_analysis_task)

        # Комбинирование результатов
        text_conf = text_analysis.get('confidence', 0)
        vision_conf = vision_analysis.get('confidence', 0)
        
        if text_conf == 0 and vision_conf > 0:
            combined_conf = vision_conf
        elif vision_conf == 0 and text_conf > 0:
            combined_conf = text_conf
        else:
            # Взвешенное среднее (60% текст, 40% графика)
            combined_conf = int((text_conf * 0.6) + (vision_conf * 0.4))

        combined_explanation = (
            f"📊 Текст (DeepSeek): {text_conf}% - {text_analysis.get('justification', 'N/A')}\n"
            f"📈 График (Gemini): {vision_conf}% - {vision_analysis.get('explanation', 'N/A')}\n"
            f"🎯 Паттерны: {', '.join(vision_analysis.get('signals', ['нет']))}"
        )

        logging.info(f"[{symbol}] Гибридный анализ: Text={text_conf}%, Vision={vision_conf}% -> Итог: {combined_conf}%")
        
        return {
            "confidence": combined_conf,
            "justification": combined_explanation,
        }
        
    except Exception as e:
        logging.error(f"[{symbol}] Ошибка в гибридном анализе: {e}")
        return await get_ai_model_verdict_async(symbol, side, market_env)  # Fallback

async def save_chart_to_base64(symbol: str, fig) -> str:
    """
    Сохраняет matplotlib/mplfinance график в base64 для отправки в API.
    """
    try:
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()
        logging.info(f"[{symbol}] График успешно конвертирован в base64")
        return image_base64
    except Exception as e:
        logging.error(f"[{symbol}] Ошибка при конвертации графика в base64: {e}", exc_info=True)
        return ""

async def get_hybrid_analysis(
    symbol: str,
    side: str,
    market_env: str) -> Dict[str, Any]:
    """Гибридный анализ: текстовая аналитика (DeepSeek) + визуальный анализ (Gemini)."""
    try:
        text_analysis_task = get_ai_model_verdict_async(symbol, side, market_env)
        
        df_short = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS.get("5m"))
        if df_short is None or len(df_short) < 20:
            logging.warning(f"[{symbol}] Недостаточно данных для виз. анализа, используется только текст.")
            return await text_analysis_task

        entry_price = float(df_short['close'].iloc[-1])
        
        vision_analysis_task = get_vision_enhanced_trade_decision(
            symbol=symbol, side=side, entry_price=entry_price, market_env=market_env
        )

        # Выполняем оба запроса параллельно
        text_analysis, vision_analysis = await asyncio.gather(text_analysis_task, vision_analysis_task)

        text_conf = text_analysis.get('confidence', 0)
        vision_conf = vision_analysis.get('confidence', 0)
        
        if text_conf == 0 and vision_conf > 0:
            combined_conf = vision_conf
        elif vision_conf == 0 and text_conf > 0:
            combined_conf = text_conf
        else:
            # Взвешенное среднее (60% текст, 40% графика)
            combined_conf = int((text_conf * 0.6) + (vision_conf * 0.4))

        combined_explanation = (
            f"📊 Текст (DeepSeek): {text_conf}% - {text_analysis.get('justification', 'N/A')}\n"
            f"📈 График (Gemini): {vision_conf}% - {vision_analysis.get('explanation', 'N/A')}\n"
            f"🎯 Паттерны: {', '.join(vision_analysis.get('signals', ['нет']))}"
        )

        logging.info(f"[{symbol}] Гибридный анализ: Text={text_conf}%, Vision={vision_conf}% -> Итог: {combined_conf}%")
        
        return {
            "confidence": combined_conf,
            "justification": combined_explanation,
        }
        
    except Exception as e:
        logging.error(f"[{symbol}] Ошибка в гибридном анализе: {e}")
        return await get_ai_model_verdict_async(symbol, side, market_env)




# <<< НОВАЯ ФУНКЦИЯ ДЛЯ АНАЛИЗА ЧЕРЕЗ GEMINI VISION >>>
# <<< НОВАЯ ФУНКЦИЯ ДЛЯ АНАЛИЗА ЧЕРЕЗ GEMINI VISION (Исправленная) >>>
async def analyze_chart_with_gemini_vision(symbol: str,
                                           chart_base64: str,
                                           trade_type: str,
                                           additional_context: str = "") -> dict:
    """
    Анализирует график с помощью Gemini Vision, используя расширенный контекст.
    """
    if not GEMINI_API_KEY:
        logging.error("[Gemini] API-ключ не найден. Визуальный анализ невозможен.")
        return {"confidence": 0, "explanation": "API-ключ Gemini не настроен", "signals": []}

    if not chart_base64:
        return {"confidence": 0, "explanation": "Пустое изображение", "signals": []}

    # --- ✅ ОБНОВЛЕННЫЙ ПРОМПТ ---
    prompt = (
        f"Вы — элитный трейдер-аналитик. Перед вами график {symbol} и детальная рыночная сводка.\n"
        f"**Планируемая сделка:** {trade_type.upper()}\n\n"
        f"**РЫНОЧНАЯ СВОДКА:**\n{additional_context}\n\n" # <--- Здесь все новые данные
        f"**ВАША ЗАДАЧА:**\n"
        f"1. **Проанализируйте график:** Найдите ключевые паттерны, уровни поддержки/сопротивления, тренды.\n"
        f"2. **Сопоставьте с контекстом:** Учитывают ли данные из сводки (OBI, OFI, Funding, OI, Ликвидации, BTC) риски или подтверждают сигнал на графике?\n"
        f"3. **Вынесите вердикт:** На основе комплексного анализа (график + сводка), оцените общую уверенность в успехе сделки {trade_type.upper()} от 0 до 100.\n\n"
        f"**Формат ответа (строго JSON):** {{\"confidence\": 0-100, \"explanation\": \"<Краткое обоснование с учетом графика и сводки>\", \"signals\": [\"<Найденные_сигналы_на_графике>\"]}}"
    )
    # --- КОНЕЦ ОБНОВЛЕННОГО ПРОМПТА ---

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL) # Используем gemini-2.5-pro

        image_part = {"mime_type": "image/png", "data": base64.b64decode(chart_base64)}

        async with AI_REQUEST_SEMAPHORE:
            response = await model.generate_content_async(
                [prompt, image_part],
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.2 # Оставляем низкую температуру
                )
            )

        data = json.loads(response.text)

        if 'confidence' not in data or 'explanation' not in data:
            raise ValueError("Ответ AI не содержит обязательных полей 'confidence' или 'explanation'")

        return {
            "confidence": int(data.get("confidence", 0)),
            "explanation": str(data.get("explanation", "Gemini не предоставил объяснение.")),
            "signals": list(data.get("signals", [])),
        }

    except Exception as e:
        logging.error(f"[{symbol}] Ошибка Gemini Vision (v2): {e}", exc_info=True)
        return {"confidence": 0, "explanation": f"Ошибка Gemini Vision: {e}", "signals": []}

async def create_enhanced_chart(symbol: str, df: pd.DataFrame, side: str, entry_price: float):
    """Создаёт улучшенный график свечей с индикаторами для анализа."""
    try:
        df_plot = df.tail(80).copy()
        
        if not isinstance(df_plot.index, pd.DatetimeIndex):
            df_plot.index = pd.to_datetime(df_plot.index, unit='ms')
        
        df_plot['EMA20'] = ta.trend.ema_indicator(df_plot['close'], window=20)
        df_plot['EMA50'] = ta.trend.ema_indicator(df_plot['close'], window=50)
        df_plot['RSI'] = ta.momentum.rsi(df_plot['close'], window=14)
        
        bb = ta.volatility.BollingerBands(df_plot['close'], window=20, window_dev=2)
        df_plot['BB_upper'] = bb.bollinger_hband()
        df_plot['BB_lower'] = bb.bollinger_lband()
        
        apds = [
            mpf.make_addplot(df_plot['EMA20'], color='blue', width=1.0),
            mpf.make_addplot(df_plot['EMA50'], color='orange', width=1.0),
            mpf.make_addplot(df_plot['BB_upper'], color='grey', linestyle='dotted', width=0.7),
            mpf.make_addplot(df_plot['BB_lower'], color='grey', linestyle='dotted', width=0.7),
        ]

        fig, axes = mpf.plot(
            df_plot, type='candle', style='charles',
            title=f'\n{symbol} | {side.upper()} @ {entry_price:.4f}',
            ylabel='Price', volume=True, addplot=apds,
            returnfig=True, figsize=(12, 8), tight_layout=True
        )

        logging.info(f"[{symbol}] График создан успешно")
        return fig
    except Exception as e:
        logging.error(f"[{symbol}] Ошибка создания графика: {e}", exc_info=True)
        return None



async def get_vision_enhanced_trade_decision(
    symbol: str, side: str, entry_price: float, market_env: str) -> Dict[str, Any]:
    """Комплексная функция: создаёт график + отправляет в Gemini Vision + возвращает решение."""
    global market_data_store, ALLOWED_INTERVALS

    try:
        df_chart = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS.get("15m"))
        if df_chart is None or len(df_chart) < 50:
            return {"confidence": 0, "explanation": "Недостаточно данных для анализа графика"}

        fig = await create_enhanced_chart(symbol, df_chart, side, entry_price)
        if fig is None:
            return {"confidence": 0, "explanation": "Ошибка создания графика"}

        chart_base64 = await save_chart_to_base64(symbol, fig)
        plt.close(fig)

        if not chart_base64:
            return {"confidence": 0, "explanation": "Ошибка конвертации графика"}

        rsi_val = ta.momentum.rsi(df_chart['close'], 14).iloc[-1]
        macd_val = ta.trend.MACD(df_chart['close']).macd_diff().iloc[-1]
        adx_val = ta.trend.adx(df_chart['high'], df_chart['low'], df_chart['close'], 14).iloc[-1]
        additional_context = f"RSI={rsi_val:.1f}, MACD_diff={macd_val:.4f}, ADX={adx_val:.1f}, Market={market_env}"

        # Заменяем вызов DeepSeek на Gemini
        analysis = await analyze_chart_with_gemini_vision(
            symbol=symbol,
            chart_base64=chart_base64,
            trade_type=side,
            additional_context=additional_context
        )
        return analysis
    except Exception as e:
        logging.error(f"[{symbol}] Ошибка в get_vision_enhanced_trade_decision: {e}", exc_info=True)
        return {"confidence": 0, "explanation": f"Ошибка: {str(e)}"}



# =================================================================
# ## КОНЕЦ БЛОКА НОВЫХ ФУНКЦИЙ ##
# =================================================================

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update.effective_chat.id): return
    logging.info(f"User {update.effective_chat.id} /help")
    
    help_text = (
        "<b><u>Основные команды:</u></b>\n"
        "/start - Старт\n"
        "/help - Эта справка\n"
        "/status - Ключевые индикаторы состояния\n"
        "/balance - Текущий баланс\n"
        "/positions - Открытые позиции\n"
        "/get_settings - Показать все текущие настройки\n\n"

        "<b><u>Аналитика и Данные:</u></b>\n"
        "/dailystats - Статистика сделок за сегодня (UTC)\n"
        "/monthlystats - Статистика сделок за 30 дней (UTC)\n"
        "/trend_chart - Прислать график тренда BTC\n\n"
        "/pnl - Подробная PNL-панель по позициям\n"

        "<b><u>Управление Позициями:</u></b>\n"
        "/close_partial <code>&lt;СИМВОЛ&gt; &lt;%&gt;</code> - Частично закрыть позицию (напр., /close_partial BTCUSDT 50)\n\n"

        "<b><u>Управление Риском и Стратегией:</u></b>\n"
        "/set_risk <code>&lt;%&gt;</code> - Базовый риск на сделку (напр., 1.5)\n"
        "/set_leverage <code>&lt;X&gt;</code> - Плечо для всех символов (напр., 10)\n"
        "/set_max_positions <code>&lt;N&gt;</code> - Макс. кол-во открытых позиций (напр., 4)\n\n"

        "<b><u>Фильтры и Другие Настройки:</u></b>\n"
        "/toggle_btc_filter - Вкл/Выкл фильтр тренда BTC\n"
        "/toggle_consolidation_filter - Вкл/Выкл AI-фильтр консолидации\n"
        "/toggle_momentum_entry - Вкл/Выкл вход по моментуму уверенности\n"
        "/emergency_close_all - Немедленно закрыть ВСЕ позиции\n\n"
    )
    await update.message.reply_html(help_text)


async def balance_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update.effective_chat.id): return
    global current_balance
    logging.info(f"User {update.effective_chat.id} /balance")
    await update.message.reply_text(f"Текущий баланс {COLLATERAL_ASSET}: {current_balance:.2f}")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update.effective_chat.id): return
    global websocket_connected, current_positions, btc_filter_state

    # --- НОВАЯ ЛОГИКА ФОРМАТИРОВАНИЯ ---
    dir_map = {1: "🟢 LONG", -1: "🔴 SHORT", 0: "⚪️ FLAT"}
    direction_str = dir_map.get(btc_filter_state.get("direction", 0), "⚪️ N/A")
    
    prob_long = btc_filter_state.get("prob_long", 0.5)
    confidence_str = f"{prob_long:.1%}"
    
    last_update_ts = btc_filter_state.get("last_updated", 0)
    time_since_update = int(time.time() - last_update_ts)
    update_str = f"{time_since_update} сек. назад" if last_update_ts > 0 else "никогда"
    # --- КОНЕЦ НОВОЙ ЛОГИКИ ---

    status_text = (f"<b><u>📊 Статус Бота</u></b>\n"
                   f"<b>Соединение с WebSocket:</b> {'Подключено ✅' if websocket_connected else 'Отключено ❌'}\n"
                   f"<b>Открытые позиции:</b> {len(current_positions)}\n\n"
                   f"<b><u>🚦 Ансамблевый BTC Фильтр</u></b>\n"
                   f"  - <b>Направление:</b> {direction_str}\n"
                   f"  - <b>Уверенность в LONG:</b> {confidence_str}\n"
                   f"  - <b>Обновлено:</b> {update_str}"
                   )
    await update.message.reply_html(status_text)


async def get_settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update.effective_chat.id): return
    logging.info(f"User {update.effective_chat.id} /get_settings")

    settings_text = (
        f"<b>⚙️ Текущие настройки Бота</b>\n\n"
        f"<b><u>Риск-менеджмент</u></b>\n"
        f"  - <b>Плечо:</b> <code>{LEVERAGE}x</code>\n"
        f"  - <b>Базовый риск на сделку:</b> <code>{BASE_RISK_PERCENT}%</code>\n"
        f"  - <b>Макс. модификатор риска:</b> <code>x{MAX_RISK_MODIFIER}</code>\n"
        f"  - <b>Мин. R:R для входа:</b> <code>1:{MIN_RR_RATIO}</code>\n"
        f"  - <b>Частичный Take Profit (R:R):</b> <code>1:{PARTIAL_TP_RR}</code>\n\n"
        
        f"<b><u>Пороги входа</u></b>\n"
        f"  - ML (Тренд/Флэт/Имп): <code>{AUTOCONFIDENCE_TREND_THRESHOLD}/{AUTOCONFIDENCE_RANGING_THRESHOLD}/{AUTOCONFIDENCE_IMPULSE_THRESHOLD}</code>\n"
        f"  - AI (Спокойный/Норм./Волат.): <code>{SCORE_THRESHOLD_LOW_VOL}/{SCORE_THRESHOLD_NORMAL_VOL}/{SCORE_THRESHOLD_HIGH_VOL}</code>\n\n"

        f"<b><u>Вход по моментуму уверенности</u></b>\n"
        f"  - Статус: <code>{'ВКЛ' if ENABLE_CONFIDENCE_MOMENTUM else 'ВЫКЛ'}</code>\n"
        f"  - Мин. рост: <code>{MIN_CONFIDENCE_INCREASE}</code>\n"
        f"  - Мин. уровень: <code>{MIN_ABSOLUTE_CONFIDENCE}</code>\n\n"

        f"<b><u>Фильтры</u></b>\n"
        f"  - Одна сделка за раз: <code>{'ВКЛ' if ENFORCE_UNIDIRECTIONAL_TRADES else 'ВЫКЛ'}</code>\n"
        f"  - Тренд BTC: <code>{'ВКЛ' if BTC_TREND_FILTER_ENABLED else 'ВЫКЛ'}</code>\n"
        f"  - AI-консолидация: <code>{'ВКЛ' if AI_CONSOLIDATION_FILTER_ENABLED else 'ВЫКЛ'}</code>\n"
        f"  - Multi-TF тренд: <code>{'ВКЛ' if MULTI_TF_TREND_FILTER_ENABLED else 'ВЫКЛ'}</code>\n"
        f"  - Корреляция: <code>{'ВКЛ' if CORRELATION_FILTER_ENABLED else 'ВЫКЛ'}</code> (Порог: {CORRELATION_THRESHOLD})\n\n"
        
        f"<b><u>A/B Тестирование</u></b>\n"
        f"  - Статус: <code>{'ВКЛ' if AB_TESTING_ENABLED else 'ВЫКЛ'}</code>\n"
        f"  - Риск группы 'B': <code>{GROUP_B_RISK_PERCENT}%</code>\n"
        f"  - Модификатор порога 'B': <code>x{GROUP_B_SCORE_THRESHOLD_MODIFIER}</code>"
    )
    
    await update.message.reply_html(settings_text)


async def ai_status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update.effective_chat.id): return
    global LATEST_AI_VERDICTS
    
    if not LATEST_AI_VERDICTS:
        await update.message.reply_text("Лента вердиктов AI пока пуста.")
        return
        
    message = "<b>🤖 Последние вердикты AI (новые сверху):</b>\n\n"
    for verdict in reversed(LATEST_AI_VERDICTS):
        message += f"<code>{html.escape(str(verdict))}</code>\n"
        
    await update.message.reply_html(message)

# --- КОНЕЦ ВТОРОЙ ЧАСТИ ---


## --- НАЧАЛО ТРЕТЬЕЙ ЧАСТИ (ДОПОЛНИТЕЛЬНЫЕ И УПРАВЛЯЮЩИЕ КОМАНДЫ TG) ---

async def positions_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update.effective_chat.id): return
    global current_positions, loop

    await update.message.reply_text("Обновляю и синхронизирую состояние позиций...")
    
    try:
        # Шаг 1: Получаем актуальные данные с биржи в виде словарей
        positions_from_api = await loop.run_in_executor(None, get_open_positions_sync)
        
        if positions_from_api is None:
            await update.message.reply_text("Не удалось получить данные с биржи. Отображаю позиции из памяти.")
        else:
            # --- НАЧАЛО ИСПРАВЛЕННОЙ ЛОГИКИ ---
            
            # Определяем, какие позиции были закрыты
            closed_symbols = set(current_positions.keys()) - set(positions_from_api.keys())
            for symbol in closed_symbols:
                logging.warning(f"[/positions] Позиция по {symbol} закрыта на бирже. Удаляю из памяти.")
                if symbol in current_positions:
                    del current_positions[symbol]

            # Обновляем существующие или создаем новые менеджеры
            for symbol, api_pos_data in positions_from_api.items():
                if symbol in current_positions and isinstance(current_positions[symbol], PositionManager):
                    # Если менеджер уже есть - обновляем его данные
                    manager = current_positions[symbol]
                    manager.state.current_quantity = float(api_pos_data.get('amount', manager.state.current_quantity))
                    manager.state.meta['unrealized_pnl'] = float(api_pos_data.get('unrealized_pnl', 0))
                else:
                    # Если менеджера нет (как в случае рассинхрона) - создаем его
                    logging.warning(f"[/positions] Обнаружена позиция по {symbol}, воссоздаю менеджер...")
                    api_pos_data['symbol'] = symbol
                    current_positions[symbol] = PositionManager(api_pos_data)
            
            logging.info(f"Синхронизация через /positions завершена. Актуальных позиций: {len(current_positions)}")
            # --- КОНЕЦ ИСПРАВЛЕННОЙ ЛОГИКИ ---

    except Exception as e:
        logging.error(f"Ошибка в /positions: {e}", exc_info=True)
        await update.message.reply_text("Произошла ошибка при обновлении позиций.")

    # Блок отображения позиций остается без изменений
    if not current_positions:
        await update.message.reply_text("Открытых позиций нет.")
        return

    message = "<b>Открытые позиции (актуализированы):</b>\n"
    total_pnl = Decimal('0.0')

    for symbol, manager in current_positions.items():
        # Теперь мы всегда работаем с объектом PositionManager
        pos_data = manager.get_state_dict()
        
        # Получаем PnL из метаданных, обновленных при синхронизации
        pnl_value = manager.state.meta.get('unrealized_pnl', 0.0)
        
        # ... (остальной код для форматирования сообщения остается таким же) ...
        price_prec = exchange_info_cache.get(symbol, {}).get('pricePrecision', 4)
        qty_prec = exchange_info_cache.get(symbol, {}).get('quantityPrecision', 2)
        side_rus = "ЛОНГ" if pos_data.get('side') == 'LONG' else "ШОРТ"
        
        pnl_dec = Decimal(str(pnl_value))
        total_pnl += pnl_dec
        pnl_str = f"{pnl_dec:+.2f}"

        sl_id, tp1_id, tp2_id = pos_data.get('sl_order_id'), pos_data.get('tp1_order_id'), pos_data.get('tp2_order_id')
        orders_info = []
        if sl_id: orders_info.append(f"SL ID: {sl_id}")
        if tp1_id: orders_info.append(f"TP1 ID: {tp1_id}")
        if tp2_id: orders_info.append(f"TP2 ID: {tp2_id}")
        orders_str = f"\n  <i>({', '.join(orders_info)})</i>" if orders_info else ""

        message += (f"<b>{symbol}</b>: {side_rus} {pos_data.get('amount', 0.0):.{qty_prec}f}\n"
                    f"  Вход: {pos_data.get('entry_price', 0.0):.{price_prec}f}\n"
                    f"  PNL: {pnl_str} {COLLATERAL_ASSET}{orders_str}\n")

    message += f"\n<b>Всего:</b> {len(current_positions)}, <b>Общий PNL:</b> {total_pnl:+.2f} {COLLATERAL_ASSET}"
    await update.message.reply_html(message)


async def debug_positions_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update.effective_chat.id): return
    global current_positions
    if not current_positions:
        await update.message.reply_text("В памяти бота нет открытых позиций.")
        return
    
    try:
        positions_str = json.dumps(current_positions, indent=2, ensure_ascii=False, default=str)
        message = f"<b>🔧 Внутреннее состояние позиций в памяти бота:</b>\n\n<pre>{html.escape(positions_str)}</pre>"
    except Exception as e:
        message = f"Ошибка при форматировании данных: {e}"

    await update.message.reply_html(message)


async def set_leverage_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update.effective_chat.id): return
    global LEVERAGE, loop, client
    if not context.args:
        await update.message.reply_text(f"Текущее плечо: {LEVERAGE}x. Использование: /set_leverage <1-125>")
        return
    try:
        new_leverage = int(context.args[0])
        if not 1 <= new_leverage <= 125: raise ValueError("Плечо должно быть от 1 до 125.")
        
        await update.message.reply_text(f"Устанавливаю плечо {new_leverage}x для всех символов...")
        
        if loop and client:
            LEVERAGE = new_leverage
            asyncio.create_task(apply_global_leverage_async(LEVERAGE))
            await update.message.reply_text(f"✅ Команда отправлена. Плечо будет изменено на {LEVERAGE}x.")
        else:
            await update.message.reply_text("Ошибка: Клиент не инициализирован.")

    except (ValueError, IndexError) as e:
        await update.message.reply_text(f"Ошибка ввода: {e}\nПример: /set_leverage 20")

async def daily_stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update.effective_chat.id): return
    global loop
    await update.message.reply_text("⏳ Рассчитываю детализированную статистику за сегодня (UTC)...")
    if not loop: 
        await update.message.reply_text("❌ Ошибка: Отсутствует цикл событий.")
        return
        
    try:
        now_utc = datetime.datetime.now(dt_timezone.utc)
        start_of_day_utc = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        start_time_ms = int(start_of_day_utc.timestamp() * 1000)
        
        stats = await loop.run_in_executor(None, get_daily_stats_sync, start_time_ms, 1000)
        
        if stats.get('error'):
            await update.message.reply_text(f"❌ Ошибка статистики: {stats['error']}")
            return
            
        if stats['trade_count'] == 0:
            await update.message.reply_text("Сделок за сегодня (UTC) не найдено.")
            return

        result_message = "<b>📊 Статистика за сегодня (UTC)</b>\n\n"
        
        sorted_symbols = sorted(stats['by_symbol'].items(), key=lambda item: item[1]['total_pnl'], reverse=True)
        
        for symbol, data in sorted_symbols:
            pnl_str = f"{data['total_pnl']:+.2f}"
            win_loss_str = f"✅{data['wins']}/🔻{data['losses']}"
            result_message += f"<b>{symbol}:</b> {pnl_str} USDT ({win_loss_str})\n"
            
        result_message += "\n"
        result_message += f"<b>Итого:</b>\n"
        result_message += f"  - Всего сделок: {stats['trade_count']}\n"
        result_message += f"  - Прибыльных: {stats['total_wins']}\n"
        result_message += f"  - Убыточных: {stats['total_losses']}\n"
        result_message += f"  - <b>Общий PNL: {stats['total_pnl']:+.2f} USDT</b>"

        await update.message.reply_html(result_message)
        
    except Exception as e:
        logging.error(f"Крит. ошибка в /dailystats: {e}", exc_info=True)
        await update.message.reply_text("❌ Внутренняя ошибка при расчете статистики.")

async def set_risk_percent_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update.effective_chat.id): return
    global BASE_RISK_PERCENT
    if not context.args:
        await update.message.reply_text(f"Текущий базовый риск: {BASE_RISK_PERCENT}%\nИспользование: /set_risk <процент>")
        return
    try:
        new_risk = float(context.args[0].replace(',', '.'))
        if not 0.1 <= new_risk <= 5.0:
            raise ValueError("Риск должен быть в диапазоне от 0.1 до 5.0")
        BASE_RISK_PERCENT = new_risk
        await update.message.reply_html(f"✅ Базовый риск на сделку установлен: <b>{BASE_RISK_PERCENT}%</b> от баланса.")
    except (ValueError, IndexError) as e:
        await update.message.reply_text(f"Ошибка ввода: {e}\nПример: /set_risk 1.5")

async def set_min_rr_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update.effective_chat.id): return
    global MIN_RR_RATIO
    if not context.args:
        await update.message.reply_text(f"Текущее минимальное R:R: 1:{MIN_RR_RATIO}\nИспользование: /set_min_rr <число>")
        return
    try:
        new_rr = float(context.args[0].replace(',', '.'))
        if not 0.5 <= new_rr <= 10.0:
            raise ValueError("Соотношение должно быть в диапазоне от 0.5 до 10.0")
        MIN_RR_RATIO = new_rr
        await update.message.reply_html(f"✅ Минимальное R:R для входа в сделку: <b>1:{MIN_RR_RATIO}</b>.")
    except (ValueError, IndexError) as e:
        await update.message.reply_text(f"Ошибка ввода: {e}\nПример: /set_min_rr 2.0")

async def set_normal_score_threshold_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update.effective_chat.id): return
    global SCORE_THRESHOLD_NORMAL_VOL
    if not context.args:
        await update.message.reply_text(f"Текущий порог AI (нормальный рынок): {SCORE_THRESHOLD_NORMAL_VOL}\nИспользование: /set_normal_score <50-95>")
        return
    try:
        new_threshold = int(context.args[0])
        if not 50 <= new_threshold <= 95:
            raise ValueError("Порог должен быть в диапазоне от 50 до 95")
        SCORE_THRESHOLD_NORMAL_VOL = new_threshold
        await update.message.reply_html(f"✅ Порог AI для нормального рынка установлен: <b>{SCORE_THRESHOLD_NORMAL_VOL}</b>.")
    except (ValueError, IndexError) as e:
        await update.message.reply_text(f"Ошибка ввода: {e}\nПример: /set_normal_score 75")

async def set_low_vol_score_threshold_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update.effective_chat.id): return
    global SCORE_THRESHOLD_LOW_VOL
    if not context.args:
        await update.message.reply_text(f"Текущий порог AI (спокойный рынок): {SCORE_THRESHOLD_LOW_VOL}\nИспользование: /set_low_vol_score <50-99>")
        return
    try:
        new_threshold = int(context.args[0])
        if not 50 <= new_threshold <= 99:
            raise ValueError("Порог должен быть в диапазоне от 50 до 99")
        SCORE_THRESHOLD_LOW_VOL = new_threshold
        await update.message.reply_html(f"✅ Порог AI для спокойного рынка установлен: <b>{SCORE_THRESHOLD_LOW_VOL}</b>.")
    except (ValueError, IndexError) as e:
        await update.message.reply_text(f"Ошибка ввода: {e}\nПример: /set_low_vol_score 85")
        
async def toggle_btc_filter_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update.effective_chat.id): return
    global BTC_TREND_FILTER_ENABLED
    BTC_TREND_FILTER_ENABLED = not BTC_TREND_FILTER_ENABLED
    await update.message.reply_html(f"🚦 Фильтр тренда BTC теперь: <b>{'ВКЛЮЧЕН' if BTC_TREND_FILTER_ENABLED else 'ВЫКЛЮЧЕН'}</b>.")

async def set_volume_mult_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update.effective_chat.id): return
    global TREND_VOLUME_MULTIPLIER
    if not context.args:
        status = f"x{TREND_VOLUME_MULTIPLIER:.2f}" if TREND_VOLUME_MULTIPLIER > 0 else "ВЫКЛЮЧЕН"
        await update.message.reply_text(f"Текущий множитель объема: {status}\nИспользование: /set_volume_mult <число>")
        return
    try:
        new_mult = float(context.args[0].replace(',', '.'))
        if new_mult < 0: raise ValueError("Множитель не может быть отрицательным.")
        TREND_VOLUME_MULTIPLIER = new_mult
        status_text = f"x{TREND_VOLUME_MULTIPLIER:.2f}" if TREND_VOLUME_MULTIPLIER > 0 else "<b>ВЫКЛЮЧЕН</b> (множитель 0)"
        await update.message.reply_html(f"📊 Множитель для фильтра объема обновлен: {status_text}.")
    except (ValueError, IndexError) as e: await update.message.reply_text(f"Ошибка ввода: {e}\nПример: /set_volume_mult 1.2")

async def toggle_consolidation_filter_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update.effective_chat.id): return
    global AI_CONSOLIDATION_FILTER_ENABLED
    AI_CONSOLIDATION_FILTER_ENABLED = not AI_CONSOLIDATION_FILTER_ENABLED
    await update.message.reply_html(f"🤖 AI-фильтр консолидации теперь: <b>{'ВКЛЮЧЕН' if AI_CONSOLIDATION_FILTER_ENABLED else 'ВЫКЛЮЧЕН'}</b>.")

async def toggle_correlation_filter_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update.effective_chat.id): return
    global CORRELATION_FILTER_ENABLED
    CORRELATION_FILTER_ENABLED = not CORRELATION_FILTER_ENABLED
    await update.message.reply_html(f"🔗 Фильтр корреляции теперь: <b>{'ВКЛЮЧЕН' if CORRELATION_FILTER_ENABLED else 'ВЫКЛЮЧЕН'}</b>.")

async def toggle_ab_testing_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update.effective_chat.id): return
    global AB_TESTING_ENABLED
    AB_TESTING_ENABLED = not AB_TESTING_ENABLED
    await update.message.reply_html(f"🅰️/🅱️ A/B тестирование теперь: <b>{'ВКЛЮЧЕНО' if AB_TESTING_ENABLED else 'ВЫКЛЮЧЕНО'}</b>.")

async def toggle_unidirectional_trades_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update.effective_chat.id): return
    global ENFORCE_UNIDIRECTIONAL_TRADES
    ENFORCE_UNIDIRECTIONAL_TRADES = not ENFORCE_UNIDIRECTIONAL_TRADES
    status = 'ВКЛЮЧЕН (только одна сделка)' if ENFORCE_UNIDIRECTIONAL_TRADES else 'ВЫКЛЮЧЕН (несколько сделок)'
    await update.message.reply_html(f"🛡️ Режим одной сделки теперь: <b>{status}</b>.")

async def toggle_multi_tf_filter_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Переключает фильтр подтверждения тренда по нескольким таймфреймам."""
    if not is_authorized(update.effective_chat.id): return
    global MULTI_TF_TREND_FILTER_ENABLED
    MULTI_TF_TREND_FILTER_ENABLED = not MULTI_TF_TREND_FILTER_ENABLED
    status = 'ВКЛЮЧЕН' if MULTI_TF_TREND_FILTER_ENABLED else 'ВЫКЛЮЧЕН'
    await update.message.reply_html(f"📈 Фильтр Multi-TF тренда теперь: <b>{status}</b>.")

async def toggle_momentum_entry_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Включает/выключает логику входа по моментуму уверенности."""
    if not is_authorized(update.effective_chat.id): return
    global ENABLE_CONFIDENCE_MOMENTUM
    ENABLE_CONFIDENCE_MOMENTUM = not ENABLE_CONFIDENCE_MOMENTUM
    status = 'ВКЛЮЧЕН' if ENABLE_CONFIDENCE_MOMENTUM else 'ВЫКЛЮЧЕН'
    await update.message.reply_html(f"📈 Вход по моментуму уверенности теперь: <b>{status}</b>.")

async def set_momentum_increase_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Устанавливает минимальный рост уверенности для входа по моментуму."""
    if not is_authorized(update.effective_chat.id): return
    global MIN_CONFIDENCE_INCREASE
    if not context.args:
        await update.message.reply_text(f"Текущий мин. рост: {MIN_CONFIDENCE_INCREASE}\nИспользование: /set_momentum_increase <число>")
        return
    try:
        new_value = int(context.args[0])
        if not 1 <= new_value <= 20:
            raise ValueError("Значение должно быть от 1 до 20")
        MIN_CONFIDENCE_INCREASE = new_value
        await update.message.reply_html(f"✅ Минимальный рост уверенности для входа установлен: <b>{MIN_CONFIDENCE_INCREASE}</b>.")
    except (ValueError, IndexError) as e:
        await update.message.reply_text(f"Ошибка ввода: {e}\nПример: /set_momentum_increase 5")

async def set_momentum_level_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Устанавливает минимальный абсолютный уровень уверенности для входа по моментуму."""
    if not is_authorized(update.effective_chat.id): return
    global MIN_ABSOLUTE_CONFIDENCE
    if not context.args:
        await update.message.reply_text(f"Текущий мин. уровень: {MIN_ABSOLUTE_CONFIDENCE}\nИспользование: /set_momentum_level <число>")
        return
    try:
        new_value = int(context.args[0])
        if not 50 <= new_value <= 80:
            raise ValueError("Значение должно быть от 50 до 80")
        MIN_ABSOLUTE_CONFIDENCE = new_value
        await update.message.reply_html(f"✅ Минимальный уровень уверенности для входа по моментуму: <b>{MIN_ABSOLUTE_CONFIDENCE}</b>.")
    except (ValueError, IndexError) as e:
        await update.message.reply_text(f"Ошибка ввода: {e}\nПример: /set_momentum_level 60")


async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update.effective_chat.id): return
    logging.warning(f"Unknown command from {update.effective_chat.id}: {update.message.text}")

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    error = context.error
    if isinstance(error, telegram.error.Conflict):
        logging.warning(f"TG Conflict Error: Another instance of the bot is running. {error}")
    else:
        logging.error(f"TG Error Handler Caught Exception:", exc_info=context.error)

# --- КОНЕЦ ТРЕТЬЕЙ ЧАСТИ ---

# --- НАЧАЛО ЧЕТВЕРТОЙ ЧАСТИ (ФИНАЛЬНАЯ ВЕРСИЯ) ---
# (Синхронные Хелперы, Инициализация ML-моделей и Фильтры)

# main.py - Блок 1

# main.py - Диагностическая версия функции



async def close_partial_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Частично закрывает открытую позицию."""
    if not is_authorized(update.effective_chat.id): return
    global current_positions, client

    try:
        if len(context.args) != 2:
            raise ValueError("Неверное количество аргументов.")
        
        symbol_to_close = context.args[0].upper()
        percentage_to_close = float(context.args[1])

        if not (0 < percentage_to_close <= 100):
            raise ValueError("Процент должен быть от 1 до 100.")

        if symbol_to_close not in current_positions:
            await update.message.reply_text(f"Открытая позиция по {symbol_to_close} не найдена.")
            return

        pos_manager = current_positions[symbol_to_close]
        pos_data = pos_manager.get_state_dict() if isinstance(pos_manager, PositionManager) else pos_manager
        
        amount_to_close = pos_data['amount'] * (percentage_to_close / 100.0)
        formatted_qty = format_quantity(amount_to_close, symbol_to_close)

        if not formatted_qty or float(formatted_qty) <= 0:
            await update.message.reply_text("Ошибка: Рассчитанный объем для закрытия слишком мал.")
            return

        await update.message.reply_text(f"Закрываю {percentage_to_close}% позиции по {symbol_to_close} ({formatted_qty} ед.)...")

        close_side = 'SELL' if pos_data['side'] == 'LONG' else 'BUY'
        
        # Используем вашу существующую функцию для отправки ордера
        closing_order = await place_order_async(
            symbol=symbol_to_close, 
            side=close_side, 
            order_type="MARKET", 
            quantity=float(formatted_qty), 
            reduce_only=True
        )

        if closing_order and closing_order.get('orderId'):
            await update.message.reply_html(f"✅ Часть позиции по <b>{symbol_to_close}</b> успешно отправлена на закрытие.")
            # Важно: бот автоматически обновит состояние позиции при следующей периодической проверке
        else:
            await update.message.reply_text("❌ Не удалось отправить ордер на закрытие. Проверьте логи.")

    except (ValueError, IndexError) as e:
        await update.message.reply_text(f"Ошибка ввода: {e}\nПример: /close_partial XRPUSDT 50")

async def trend_chart_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Генерирует и отправляет график тренда BTC."""
    if not is_authorized(update.effective_chat.id): return
    if not mpf:
        await update.message.reply_text("Ошибка: библиотека mplfinance не установлена на сервере.")
        return

    await update.message.reply_text("⏳ Генерирую актуальный график тренда BTC...")

    try:
        # Используем часовой таймфрейм для анализа
        tf_key = '1h'
        df_btc = market_data_store.get('BTCUSDT', {}).get(ALLOWED_INTERVALS[tf_key])

        if df_btc is None or len(df_btc) < 100:
            await update.message.reply_text("Недостаточно данных для построения графика.")
            return
        
        # Берем последние 100 свечей для наглядности
        df_plot = df_btc.tail(100).copy()

        # Расчет индикаторов
        ema_fast = ta.trend.ema_indicator(df_plot['close'], window=BTC_EMA_FAST_PERIOD)
        ema_slow = ta.trend.ema_indicator(df_plot['close'], window=BTC_EMA_SLOW_PERIOD)
        adx_series = ta.trend.adx(df_plot['high'], df_plot['low'], df_plot['close'], window=14)
        
        # Подготовка данных для mplfinance
        plots = [
            mpf.make_addplot(ema_fast, color='blue', width=0.7),
            mpf.make_addplot(ema_slow, color='orange', width=0.7),
            mpf.make_addplot(adx_series, panel=1, color='purple', ylabel='ADX')
        ]
        
        # Создание графика в памяти
        buf = io.BytesIO()
        style = mpf.make_marketcolors(up='green', down='red', inherit=True)
        mc_style = mpf.make_mpf_style(marketcolors=style, gridstyle=':')
        
        fig, axes = mpf.plot(
            df_plot,
            type='candle',
            style=mc_style,
            title=f'BTC/USDT Trend Analysis ({tf_key}) | ADX: {adx_series.iloc[-1]:.2f}',
            ylabel='Price ($)',
            addplot=plots,
            panel_ratios=(3, 1),
            returnfig=True,
            figsize=(12, 7)
        )
        # Добавляем горизонтальную линию на ADX
        axes[2].axhline(BTC_ADX_THRESHOLD, color='gray', linestyle='--', linewidth=1)
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        
        # Отправка графика в Telegram
        await update.message.reply_photo(photo=buf, caption=f"Актуальный график тренда BTC на {tf_key}.")

    except Exception as e:
        logging.error(f"Ошибка при создании графика: {e}", exc_info=True)
        await update.message.reply_text("Произошла ошибка при генерации графика.")


async def detect_market_environment(symbol: str) -> str:
    """
    Определяет среду рынка ('impulse', 'trending', 'ranging', 'uncertain')
    с настраиваемыми параметрами.
    """
    global market_data_store, SAFETY_FILTER_TIMEFRAME, ALLOWED_INTERVALS

    tf_15m_val = ALLOWED_INTERVALS[SAFETY_FILTER_TIMEFRAME]
    df_15m = market_data_store.get(symbol, {}).get(tf_15m_val)

    if df_15m is None or len(df_15m) < 21:
        logging.debug(f"[{symbol}] Недостаточно данных для определения рыночной среды.")
        return 'uncertain'
    
    # --- Определение ИМПУЛЬСА ---
    try:
        atr = average_true_range(df_15m['high'], df_15m['low'], df_15m['close'], window=14).iloc[-1]
        avg_volume = df_15m['volume'].rolling(window=20).mean().iloc[-2]
        last_candle = df_15m.iloc[-1]
        last_candle_body = abs(last_candle['close'] - last_candle['open'])
        
        # Используем переменные из настроек
        is_high_volume = last_candle['volume'] > (avg_volume * IMPULSE_VOLUME_MULTIPLIER) if avg_volume > 0 else False
        is_large_body = last_candle_body > (atr * IMPULSE_BODY_ATR_MULTIPLIER) if atr > 0 else False

        if is_high_volume and is_large_body:
            logging.info(f"[{symbol}] ОБНАРУЖЕНА ФАЗА 'ИМПУЛЬС'!")
            return 'impulse'
    except Exception:
        pass # Если здесь ошибка, просто переходим к анализу тренда/флэта

    # --- Определение ТРЕНДА/ФЛЭТА ---
    tf_1h_val = ALLOWED_INTERVALS.get('1h')
    df_1h = market_data_store.get(symbol, {}).get(tf_1h_val)

    if df_1h is None or len(df_1h) < 21:
        logging.debug(f"[{symbol}] Недостаточно данных на 1h для определения тренда/флэта.")
        return 'uncertain'
        
    try:
        adx_15m = adx(df_15m['high'], df_15m['low'], df_15m['close'], window=14).iloc[-1]
        bb_1h = BollingerBands(close=df_1h['close'], window=20, window_dev=2)
        bb_width_1h = bb_1h.bollinger_wband().iloc[-1]

        logging.info(f"[{symbol}] Детектор рынка: ADX(15m)={adx_15m:.2f}, BB_Width(1h)={bb_width_1h:.2f}")
        
        if adx_15m >= ADX_REGIME_THRESHOLD:
            return 'trending'
        elif adx_15m < ADX_REGIME_THRESHOLD: 
            return 'ranging'
        else:
            return 'uncertain'
            
    except Exception as e:
        logging.error(f"[{symbol}] Ошибка в детекторе тренда/флэта: {e}", exc_info=True)
        return 'uncertain'




def initialize_exchange_info_sync():
    global exchange_info_cache, SYMBOLS, API_KEY, API_SECRET
    logging.info("Запуск синхронной инициализации Exchange Info...")
    try:
        sync_client = Client(API_KEY, API_SECRET, requests_params={"timeout": 30})
        info = sync_client.futures_exchange_info()
        temp_cache = {}
        symbols_data = {item['symbol']: item for item in info.get('symbols', [])}
        for symbol_key in SYMBOLS:
            if symbol_key in symbols_data:
                data = symbols_data[symbol_key]
                if data.get('contractType') == 'PERPETUAL' and data.get('status') == 'TRADING':
                    try:
                        filters = {f['filterType']: f for f in data['filters']}
                        temp_cache[symbol_key] = {
                            'tickSize': filters['PRICE_FILTER']['tickSize'],
                            'stepSize': filters['LOT_SIZE']['stepSize'],
                            'minNotional': filters['MIN_NOTIONAL']['notional'],
                            'quantityPrecision': data['quantityPrecision'],
                            'pricePrecision': data['pricePrecision'],
                            'status': 'TRADING'
                        }
                    except KeyError as e:
                        logging.warning(f"Не удалось обработать фильтры для {symbol_key}: отсутствует ключ {e}. Символ пропущен.")
        exchange_info_cache.update(temp_cache)
        logging.warning(f"Exchange Info: {len(exchange_info_cache)}/{len(SYMBOLS)} символов из списка SYMBOLS кэшировано.")
        return True
    except Exception as e:
        logging.critical(f"Крит. API Ошибка Exchange Info: {e}")
        return False


def set_leverage_sync(symbols_list: list, target_leverage: int) -> bool:
    logging.info(f"Запуск синхр. уст. плеча {target_leverage}x для {len(symbols_list)} симв...")
    try:
        sync_client = Client(API_KEY, API_SECRET)
        ok_count, fail_count = 0, 0
        for symbol in symbols_list:
            if symbol in exchange_info_cache:
                try:
                    sync_client.futures_change_leverage(symbol=symbol, leverage=target_leverage)
                    ok_count += 1
                except BinanceAPIException as e:
                    if e.code == -4046: ok_count += 1 # "No need to change leverage" is not an error
                    else: fail_count += 1; logging.error(f"API Ошибка уст. плеча {symbol}: {e.message}")
        logging.info(f"Синхр. уст. плеча завершена. OK:{ok_count}, Err:{fail_count}.")
        return fail_count == 0
    except Exception as e:
        logging.error(f"Общая ошибка в set_leverage_sync: {e}")
        return False


def get_ohlcv_sync(symbol: str, interval: str, limit: int = 500):
    """
    Синхронно загружает исторические данные OHLCV с Binance (версия с улучшенной отладкой).
    """
    try:
        sync_client = Client(API_KEY, API_SECRET, requests_params={"timeout": 20})
        logging.info(f"[Data Loader] Запрос {limit} свечей для {symbol} на таймфрейме {interval}...")
        bars = sync_client.futures_klines(symbol=symbol, interval=interval, limit=limit)
        
        if not bars:
            logging.error(f"!!! [Data Loader] Запрос для {symbol}/{interval} вернул ПУСТОЙ список.")
            return None
            
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)
        
        cols_to_numeric = [
            'open', 'high', 'low', 'close', 'volume', 
            'quote_asset_volume', 'number_of_trades', 
            'taker_buy_base', 'taker_buy_quote'
        ]
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        df.set_index('timestamp', inplace=True)
        return df.dropna()
        
    except BinanceAPIException as e:
        logging.critical(f"!!! КРИТИЧЕСКАЯ API ОШИБКА при загрузке {symbol}/{interval}: Код={e.code}, Сообщение='{e.message}'")
        return None
    except Exception as e:
        logging.critical(f"!!! НЕИЗВЕСТНАЯ КРИТИЧЕСКАЯ ОШИБКА при загрузке {symbol}/{interval}: {e}", exc_info=True)
        return None



def get_initial_balance_sync() -> Optional[float]:
    try:
        sync_client = Client(API_KEY, API_SECRET)
        balances = sync_client.futures_account_balance()
        asset_info = next((item for item in balances if item.get('asset') == COLLATERAL_ASSET), None)
        return float(asset_info['balance']) if asset_info else None
    except Exception as e:
        logging.error(f"Ошибка получения баланса: {e}")
        return None

def calculate_kelly_position_size(win_rate: float, avg_win: float, avg_loss: float, balance: float, max_kelly: float = 0.25) -> Optional[float]:
    """Критерий Келли для оптимального размера позиции."""
    if avg_loss <= 0:
        return None # Невозможно рассчитать, если нет убытков
        
    win_loss_ratio = abs(avg_win / avg_loss)
    if win_loss_ratio == 0:
        return None

    # Формула Келли: K% = W - [(1 - W) / R]
    kelly_fraction = (win_rate - ((1 - win_rate) / win_loss_ratio))
    
    # Ограничиваем Kelly для снижения риска и применяем дробную часть
    effective_kelly = max(0, kelly_fraction) * max_kelly
    
    if effective_kelly <= 0:
        return None
        
    position_value = balance * effective_kelly
    logging.info(f"Kelly Calc: Win Rate={win_rate:.2%}, W/L Ratio={win_loss_ratio:.2f} -> Kelly Frac={effective_kelly:.3f} -> Size=${position_value:.2f}")
    return position_value

async def multi_timeframe_trend_confirmation(symbol: str, expected_direction: str) -> tuple[bool, float]:
    """
    Упрощённая согласованность: достаточно совпадения 2 из 3 (5m, 15m, 1h),
    либо 1h совпадает и даёт суммарный вес >= 0.5.
    """
    tfs = [("5m", 1.0), ("15m", 1.3), ("1h", 1.7)]  # усиливаем вес 1h
    score = 0.0
    weight_sum = sum(w for _, w in tfs)
    agree_count = 0
    
    for tf, w in tfs:
        df = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS.get(tf))
        if df is None or len(df) < 55:
            weight_sum -= w # Уменьшаем общий вес, если данных нет
            continue
        try:
            ema_fast = ta.trend.ema_indicator(df["close"], window=TRENDING_FAST_EMA).iloc[-1]
            ema_slow = ta.trend.ema_indicator(df["close"], window=TRENDING_SLOW_EMA).iloc[-1]
            dir_up = ema_fast >= ema_slow
            want_up = expected_direction.lower() == "long"
            agree = (dir_up and want_up) or ((not dir_up) and (not want_up))
            if agree:
                score += w
                agree_count += 1
        except Exception as e:
            logging.error(f"[{symbol}] Ошибка в MTF анализе для {tf}: {e}")
            weight_sum -= w # Уменьшаем общий вес при ошибке
            continue
            
    norm_score = score / max(1e-9, weight_sum)
    
    # Правила прохождения:
    pass_by_count = agree_count >= 2
    pass_by_weight = norm_score >= 0.5
    
    is_confirmed = pass_by_count or pass_by_weight
    
    logging.info(f"[{symbol}] Multi-TF Check for {expected_direction.upper()}: {'PASSED' if is_confirmed else 'FAILED'}. "
                 f"Agree Count: {agree_count}/3, Weighted Score: {norm_score:.2f}")
                 
    return is_confirmed, float(round(norm_score, 3))

def calculate_adaptive_risk(base_risk_percent: float) -> float:
    """Адаптирует риск на основе серий побед/поражений."""
    global CONSECUTIVE_WINS, CONSECUTIVE_LOSSES

    if CONSECUTIVE_LOSSES >= 3:
        new_risk = base_risk_percent * 0.5 # Снижаем риск вдвое
        logging.warning(f"⚠️ {CONSECUTIVE_LOSSES} поражений подряд. Риск временно снижен до {new_risk}%.")
        return new_risk

    # Антимартингейл: прогрессивное увеличение после побед
    multiplier = min(1.0 + (CONSECUTIVE_WINS * 0.25), 2.0) # 1.0 -> 1.25 -> 1.5 ... -> 2.0
    adjusted_risk = base_risk_percent * multiplier
    if multiplier > 1.0:
        logging.warning(f"🔥 {CONSECUTIVE_WINS} побед подряд. Риск временно увеличен до {adjusted_risk}%.")
        
    return adjusted_risk

async def track_trade_result(pnl: float):
    """Обновляет счетчики серий побед/поражений."""
    global CONSECUTIVE_WINS, CONSECUTIVE_LOSSES, TRADE_HISTORY
    
    if pnl > 0:
        CONSECUTIVE_WINS += 1
        CONSECUTIVE_LOSSES = 0
    elif pnl < 0:
        CONSECUTIVE_LOSSES += 1
        CONSECUTIVE_WINS = 0

def get_open_positions_sync() -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Синхронизирует позиции с биржей.
    ИСПРАВЛЕНО: Добавлен recvWindow для предотвращения ошибки -1021.
    """
    global current_positions
    try:
        sync_client = Client(API_KEY, API_SECRET)
        positions_new = {}
        # ✅ ИСПРАВЛЕНИЕ: Добавлен параметр recvWindow=10000 (10 секунд)
        position_info = sync_client.futures_position_information(recvWindow=10000)
        
        for pos in position_info:
            if Decimal(pos.get('positionAmt', '0')) != Decimal('0'):
                symbol = pos['symbol']
                old_pos_data = {}
                if symbol in current_positions:
                    manager_or_dict = current_positions[symbol]
                    if isinstance(manager_or_dict, PositionManager):
                        old_pos_data = manager_or_dict.get_state_dict()
                    elif isinstance(manager_or_dict, dict):
                        old_pos_data = manager_or_dict
                
                positions_new[symbol] = {
                    'side': 'LONG' if Decimal(pos['positionAmt']) > 0 else 'SHORT',
                    'amount': float(abs(Decimal(pos['positionAmt']))),
                    'entry_price': float(pos['entryPrice']),
                    'liquidation_price': float(pos['liquidationPrice']),
                    'unrealized_pnl': float(pos['unRealizedProfit']),
                    'sl_order_id': old_pos_data.get('sl_order_id'),
                    'tp1_order_id': old_pos_data.get('tp1_order_id'),
                    'tp2_order_id': old_pos_data.get('tp2_order_id'),
                    'state': old_pos_data.get('state', 'initial'),
                    'entry_context': old_pos_data.get('entry_context'),
                    'sl_price': old_pos_data.get('sl_price'),
                    'initial_sl_price': old_pos_data.get('initial_sl_price'),
                    'tp1_price': old_pos_data.get('tp1_price'),
                    'tp2_price': old_pos_data.get('tp2_price'),
                    'initial_amount': old_pos_data.get('initial_amount'),
                    'ai_verdict': old_pos_data.get('ai_verdict'),
                    'strategy_group': old_pos_data.get('strategy_group', 'A')
                }
        return positions_new
    except BinanceAPIException as e:
        # ✅ ИСПРАВЛЕНИЕ: Логируем ошибку, но возвращаем None, чтобы не остановить бота.
        logging.error(f"Ошибка получения открытых позиций: {e}", exc_info=True)
        if e.code == -1021:
            logging.warning("Обнаружена ошибка времени (-1021). Рекомендуется проверить синхронизацию времени на сервере.")
        return None
    except Exception as e:
        logging.error(f"Критическая ошибка в get_open_positions_sync: {e}", exc_info=True)
        return None

def get_historical_stats_sync(start_time_ms: int, limit: int = 1000) -> Dict[str, Any]:
    """Универсальная функция для получения и агрегации статистики PnL с биржи."""
    stats_by_symbol = {}
    total_pnl, total_wins, total_losses = Decimal('0.0'), 0, 0
    
    try:
        sync_client = Client(API_KEY, API_SECRET)
        # Получаем все сделки, где был реализован PnL
        trades = sync_client.futures_account_trades(startTime=start_time_ms, limit=limit)
        
        realized_pnls = {}
        for trade in trades:
            pnl = Decimal(trade.get('realizedPnl', '0.0'))
            if pnl != 0:
                # Группируем сделки по ID ордера, чтобы одна позиция считалась как одна сделка
                order_id = trade['orderId']
                if order_id not in realized_pnls:
                    realized_pnls[order_id] = {'pnl': Decimal('0.0'), 'symbol': trade['symbol']}
                realized_pnls[order_id]['pnl'] += pnl
        
        for order_info in realized_pnls.values():
            symbol, pnl = order_info['symbol'], order_info['pnl']
            if symbol not in stats_by_symbol:
                stats_by_symbol[symbol] = {'wins': 0, 'losses': 0, 'total_pnl': Decimal('0.0')}
            
            stats_by_symbol[symbol]['total_pnl'] += pnl
            total_pnl += pnl
            
            if pnl > 0:
                stats_by_symbol[symbol]['wins'] += 1
                total_wins += 1
            elif pnl < 0:
                stats_by_symbol[symbol]['losses'] += 1
                total_losses += 1
                
        return {
            'by_symbol': stats_by_symbol, 'total_pnl': total_pnl,
            'total_wins': total_wins, 'total_losses': total_losses,
            'trade_count': len(realized_pnls), 'error': None
        }
    except Exception as e:
        logging.error(f"Ошибка при получении истории сделок: {e}", exc_info=True)
        return {'error': str(e)}

def compute_hmm_trend_prob(series_returns: np.ndarray, symbol='BTCUSDT') -> pd.Series:
    """
    Рассчитывает вероятность нахождения в "трендовом" состоянии на основе обученной HMM-модели.
    """
    global hmm_models_manager
    
    # Возвращаем нейтральную вероятность, если HMM-модель недоступна
    if symbol not in hmm_models_manager or 'model' not in hmm_models_manager[symbol]:
        return pd.Series([0.5] * len(series_returns))

    model_data = hmm_models_manager[symbol]
    mdl = model_data['model']
    state_order_map = model_data.get('order', {}) # Карта состояний, отсортированных по волатильности
    
    try:
        # Предсказываем вероятности для каждого состояния
        probs = mdl.predict_proba(series_returns.reshape(-1, 1))
        
        # Предполагаем, что состояние 2 (самое волатильное) является "трендовым"
        # Это допущение из вашей HMM-конфигурации (0=Low, 1=Medium, 2=High/Trend)
        trend_state_index = 2 
        
        # Если у нас есть карта состояний, находим, какой "сырой" индекс соответствует трендовому
        raw_trend_state_idx = next((raw_idx for raw_idx, sorted_idx in state_order_map.items() if sorted_idx == trend_state_index), None)
        
        if raw_trend_state_idx is not None:
            trend_prob = probs[:, raw_trend_state_idx]
        else:
            # Если карты нет, используем состояние с максимальной средней вероятностью как "трендовое"
            trend_prob = probs[:, np.argmax(probs.mean(axis=0))]
            
        return pd.Series(trend_prob)
        
    except Exception as e:
        logging.error(f"[HMM Prob Compute] Ошибка при расчете вероятности тренда: {e}")
        return pd.Series([0.5] * len(series_returns))

async def get_basis_and_funding(symbol: str) -> Dict[str, float]:
    """Получает данные о премии фьючерса к индексу и ставке финансирования."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as cli:
            r = await cli.get("https://fapi.binance.com/fapi/v1/premiumIndex", params={"symbol": symbol})
            r.raise_for_status()
            j = r.json()
            mark = float(j.get("markPrice", 0))
            indexp = float(j.get("indexPrice", 0))
            funding = float(j.get("lastFundingRate", 0)) * 100 # в %
            basis_bps = 0.0
            if indexp > 0:
                basis_bps = (mark / indexp - 1.0) * 10_000.0
            return {"basis_bps": basis_bps, "funding_pct": funding, "index_price": indexp, "mark_price": mark}
    except Exception as e:
        logging.warning(f"[{symbol}] Не удалось получить данные basis/funding: {e}")
        return {"basis_bps": 0.0, "funding_pct": 0.0, "index_price": 0.0, "mark_price": 0.0}

def basis_funding_filter(side: str, basis_bps: float, funding: float) -> Tuple[bool, str]:
    """
    Фильтрует сделки при "перегретом" рынке, принимая числовые значения напрямую.
    """
    # Теперь функция не пытается извлечь данные, а работает с ними напрямую
    if side.upper() == "LONG" and (basis_bps > 25 and funding > 0.0001): # Сделаем порог фандинга чуть строже
        return False, f"Блок ЛОНГ: рынок перегрет (basis {basis_bps:.0f} bps, funding {funding:.4f})"
    
    if side.upper() == "SHORT" and (basis_bps < -25 and funding < -0.0001):
        return False, f"Блок ШОРТ: рынок переохлажден (basis {basis_bps:.0f} bps, funding {funding:.4f})"
        
    return True, "OK"

def get_daily_stats_sync(start_time_ms: int, limit: int = 1000) -> Dict[str, Any]:
    stats_by_symbol = {}
    total_pnl, total_wins, total_losses = Decimal('0.0'), 0, 0
    try:
        sync_client = Client(API_KEY, API_SECRET)
        trades = sync_client.futures_account_trades(startTime=start_time_ms, limit=limit)
        realized_pnls = {}
        for trade in trades:
            pnl = Decimal(trade.get('realizedPnl', '0.0'))
            if pnl != 0:
                order_id = trade['orderId']
                if order_id not in realized_pnls:
                    realized_pnls[order_id] = {'pnl': Decimal('0.0'), 'symbol': trade['symbol']}
                realized_pnls[order_id]['pnl'] += pnl
        
        for order_info in realized_pnls.values():
            symbol, pnl = order_info['symbol'], order_info['pnl']
            if symbol not in stats_by_symbol:
                stats_by_symbol[symbol] = {'wins': 0, 'losses': 0, 'total_pnl': Decimal('0.0')}
            stats_by_symbol[symbol]['total_pnl'] += pnl
            total_pnl += pnl
            if pnl > 0:
                stats_by_symbol[symbol]['wins'] += 1; total_wins += 1
            elif pnl < 0:
                stats_by_symbol[symbol]['losses'] += 1; total_losses += 1
        
        return {
            'by_symbol': stats_by_symbol, 'total_pnl': total_pnl, 
            'total_wins': total_wins, 'total_losses': total_losses, 
            'trade_count': len(realized_pnls), 'error': None
        }
    except Exception as e:
        return {'error': str(e)}

# --- КОНЕЦ ЧЕТВЕРТОЙ ЧАСТИ ---
# --- НАЧАЛО ПЯТОЙ ЧАСТИ ---
logging.info("INFO: Часть 5 подтверждает: логика сигналов теперь в TradingModel и process_message.")
# --- КОНЕЦ ПЯТОЙ ЧАСТИ ---


# --- НАЧАЛО ШЕСТОЙ ЧАСТИ ---
logging.info("INFO: Часть 6 подтверждает: логика SL/TP встроена в процесс входа в позицию.")
# --- КОНЕЦ ШЕСТОЙ ЧАСТИ ---


# --- НАЧАЛО СЕДЬМОЙ ЧАСТИ (ПОЛНАЯ ФИНАЛЬНАЯ ВЕРСИЯ) ---

async def save_state_async():
    """Асинхронно сохраняет текущее состояние позиций в файл, преобразуя объекты в словари."""
    async with state_lock:
        try:
            # ✅✅✅ ИЗМЕНЕНИЕ ЗДЕСЬ ✅✅✅
            positions_to_save = {}
            for symbol, manager_or_dict in current_positions.items():
                if isinstance(manager_or_dict, PositionManager):
                    # Если это наш новый объект, получаем из него словарь для сохранения
                    positions_to_save[symbol] = manager_or_dict.get_state_dict()
                elif isinstance(manager_or_dict, dict):
                    # Обратная совместимость на всякий случай
                    positions_to_save[symbol] = manager_or_dict
            
            if not aiofiles:
                # Синхронный fallback, если aiofiles не установлен
                with open("bot_state.json", "w", encoding='utf-8') as f:
                    json.dump(positions_to_save, f, indent=2, default=str)
                logging.info("Состояние бота (синхронно) сохранено в bot_state.json.")
                return

            async with aiofiles.open("bot_state.json", "w", encoding='utf-8') as f:
                await f.write(json.dumps(positions_to_save, indent=2, default=str))
            logging.info("Состояние бота успешно сохранено в bot_state.json.")
        except Exception as e:
            logging.error(f"Ошибка сохранения состояния: {e}", exc_info=True)

async def monthly_stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Показывает статистику PnL за последние 30 дней."""
    if not is_authorized(update.effective_chat.id): return
    global loop
    
    await update.message.reply_text("⏳ Рассчитываю статистику за последние 30 дней (UTC)...")
    if not loop:
        await update.message.reply_text("❌ Ошибка: Отсутствует цикл событий.")
        return
        
    try:
        # Рассчитываем временную метку 30 дней назад
        now_utc = datetime.datetime.now(dt_timezone.utc)
        start_of_period_utc = now_utc - timedelta(days=30)
        start_time_ms = int(start_of_period_utc.timestamp() * 1000)
        
        # Вызываем нашу новую универсальную функцию
        stats = await loop.run_in_executor(None, get_historical_stats_sync, start_time_ms, 1000)
        
        if stats.get('error'):
            await update.message.reply_text(f"❌ Ошибка при получении статистики: {stats['error']}")
            return
            
        if stats['trade_count'] == 0:
            await update.message.reply_text("Сделок за последние 30 дней не найдено.")
            return

        result_message = "<b>📊 Статистика за последние 30 дней (UTC)</b>\n\n"
        
        # Сортируем монеты по прибыльности
        sorted_symbols = sorted(stats['by_symbol'].items(), key=lambda item: item[1]['total_pnl'], reverse=True)
        
        for symbol, data in sorted_symbols:
            pnl_str = f"{data['total_pnl']:+.2f}"
            win_loss_str = f"✅{data['wins']}/🔻{data['losses']}"
            result_message += f"<b>{symbol}:</b> {pnl_str} USDT ({win_loss_str})\n"
            
        result_message += "\n"
        result_message += f"<b><u>Итого за 30 дней:</u></b>\n"
        result_message += f"  - Всего сделок: {stats['trade_count']}\n"
        result_message += f"  - Прибыльных: {stats['total_wins']}\n"
        result_message += f"  - Убыточных: {stats['total_losses']}\n"
        win_rate = (stats['total_wins'] / stats['trade_count'] * 100) if stats['trade_count'] > 0 else 0
        result_message += f"  - Винрейт: {win_rate:.1f}%\n"
        result_message += f"  - <b>Общий PNL: {stats['total_pnl']:+.2f} USDT</b>"

        await update.message.reply_html(result_message)
        
    except Exception as e:
        logging.error(f"Критическая ошибка в /monthlystats: {e}", exc_info=True)
        await update.message.reply_text("❌ Внутренняя ошибка при расчете статистики.")

def sync_deepseek_request(prompt: str, api_url: str, use_json_format: bool = False) -> Dict[str, Any]:
    """Универсальная функция для запросов к DeepSeek API с одним ключом."""
    global DEEPSEEK_API_KEY
    if not DEEPSEEK_API_KEY:
        logging.error("DEEPSEEK_API_KEY не установлен.")
        return {}

    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}], "max_tokens": 1024, "temperature": 0.6}
    
    if use_json_format:
        payload["response_format"] = {"type": "json_object"}
        
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(api_url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logging.error(f"Ошибка в sync_deepseek_request: {e}")
        return {}

# --- Вспомогательные функции для адаптивных промптов ---

def format_rsi_context(rsi_val: float) -> str:
    """Преобразует значение RSI в текстовое описание."""
    if rsi_val > 75: return f"{rsi_val:.1f} (сильная перекупленность)"
    if rsi_val > 68: return f"{rsi_val:.1f} (близко к перекупленности)"
    if rsi_val < 25: return f"{rsi_val:.1f} (сильная перепроданность)"
    if rsi_val < 32: return f"{rsi_val:.1f} (близко к перепроданности)"
    return f"{rsi_val:.1f} (нейтральный)"

def format_adx_context(adx_val: float) -> str:
    """Преобразует значение ADX в текстовое описание."""
    if adx_val > 35: return f"{adx_val:.1f} (очень сильный тренд)"
    if adx_val > 25: return f"{adx_val:.1f} (сильный тренд)"
    if adx_val < 20: return f"{adx_val:.1f} (слабый тренд/флэт)"
    return f"{adx_val:.1f} (умеренный тренд)"

def format_macd_context(macd_val: float) -> str:
    """Преобразует значение MACD в текстовое описание."""
    if macd_val > 0.0015: return f"{macd_val:.4f} (бычий)"
    if macd_val < -0.0015: return f"{macd_val:.4f} (медвежий)"
    return f"{macd_val:.4f} (нейтральный)"

def format_stoch_context(k: float, d: float) -> str:
    """Преобразует значения стохастика в текстовое описание."""
    if k > 80 and d > 75: return f"%K={k:.1f}, %D={d:.1f} (сильная перекупленность)"
    if k < 20 and d < 25: return f"%K={k:.1f}, %D={d:.1f} (сильная перепроданность)"
    if k > d: return f"%K={k:.1f}, %D={d:.1f} (бычий моментум)"
    if d > k: return f"%K={k:.1f}, %D={d:.1f} (медвежий моментум)"
    return f"%K={k:.1f}, %D={d:.1f} (нейтральный)"

async def get_market_environment_from_ai(symbol: str) -> str:
    """Использует AI для определения рыночной среды."""
    global DEEPSEEK_API_URL, market_data_store, ALLOWED_INTERVALS

    df_15m = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS['15m'])
    df_1h = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS['1h'])

    if df_15m is None or len(df_15m) < 21 or df_1h is None or len(df_1h) < 21:
        return 'uncertain'
    
    try:
        # Расчет основных индикаторов
        atr_15m = average_true_range(df_15m['high'], df_15m['low'], df_15m['close'], window=14).iloc[-1]
        adx_15m = adx(df_15m['high'], df_15m['low'], df_15m['close'], window=14).iloc[-1]
        rsi_15m = rsi(df_15m['close'], window=14).iloc[-1]
        
        bb_1h = BollingerBands(close=df_1h['close'], window=20, window_dev=2)
        bb_width_1h = bb_1h.bollinger_wband().iloc[-1]
        
        # Проверка на импульс
        avg_volume = df_15m['volume'].rolling(window=20).mean().iloc[-2]
        last_candle = df_15m.iloc[-1]
        last_candle_body = abs(last_candle['close'] - last_candle['open'])
        
        is_high_volume = last_candle['volume'] > (avg_volume * IMPULSE_VOLUME_MULTIPLIER) if avg_volume > 0 else False
        is_large_body = last_candle_body > (atr_15m * IMPULSE_BODY_ATR_MULTIPLIER) if atr_15m > 0 else False
        
        if is_high_volume and is_large_body:
            logging.info(f"[{symbol}] AI определил: ИМПУЛЬСНОЕ движение")
            return 'impulse'
        
        # Определение тренда/флэта
        if adx_15m >= ADX_REGIME_THRESHOLD:
            logging.info(f"[{symbol}] AI определил: ТРЕНДОВЫЙ режим (ADX={adx_15m:.1f})")
            return 'trending'
        elif adx_15m < FLAT_ADX_THRESHOLD:
            logging.info(f"[{symbol}] AI определил: ФЛЭТ (ADX={adx_15m:.1f})")
            return 'ranging'
        else:
            return 'uncertain'
            
    except Exception as e:
        logging.error(f"[{symbol}] Ошибка в get_market_environment_from_ai: {e}", exc_info=True)
        return 'uncertain'

def find_rsi_divergence(
    df: pd.DataFrame,
    rsi_period: int = 14,
    peak_lookback: int = 50, # За сколько свечей назад ищем пики/впадины
    peak_prominence: float = 1.5 # Насколько "выдающимся" должен быть пик (в % от RSI)
) -> tuple[int, int]:
    """
    Находит классическую бычью или медвежью дивергенцию RSI.

    Args:
        df: DataFrame с колонками 'high', 'low', 'close'.
        rsi_period: Период для расчета RSI.
        peak_lookback: Окно в свечах для поиска последних 2-х экстремумов.
        peak_prominence: Минимальная "заметность" пика RSI для его учета.

    Returns:
        Кортеж (signal, age):
        - signal: 1 (бычья дивергенция), -1 (медвежья дивергенция), 0 (нет дивергенции).
        - age: Сколько свечей назад была обнаружена дивергенция.
    """
    if len(df) < peak_lookback + rsi_period:
        return 0, 0 # Недостаточно данных для анализа

    # 1. Рассчитываем RSI
    rsi = ta.momentum.rsi(df['close'], window=rsi_period)
    
    # Ограничиваем данные последними `peak_lookback` свечами для поиска
    price_data_high = df['high'].tail(peak_lookback)
    price_data_low = df['low'].tail(peak_lookback)
    rsi_data = rsi.tail(peak_lookback)

    # 2. Находим пики (для медвежьей дивергенции) и впадины (для бычьей)
    # prominence - это мера того, насколько пик выделяется над окружающим ландшафтом
    rsi_peaks, _ = find_peaks(rsi_data, prominence=peak_prominence)
    rsi_troughs, _ = find_peaks(-rsi_data, prominence=peak_prominence)

    # --- Проверка на медвежью дивергенцию (Higher Highs на цене, Lower Highs на RSI) ---
    if len(rsi_peaks) >= 2:
        # Берем два последних пика RSI
        p1_idx, p2_idx = rsi_peaks[-2], rsi_peaks[-1]
        
        # Получаем их значения и соответствующие им значения цены
        rsi_p1, rsi_p2 = rsi_data.iloc[p1_idx], rsi_data.iloc[p2_idx]
        price_p1, price_p2 = price_data_high.iloc[p1_idx], price_data_high.iloc[p2_idx]

        # Условие дивергенции
        if price_p2 > price_p1 and rsi_p2 < rsi_p1:
            age = len(rsi_data) - 1 - p2_idx # Сколько свечей прошло с момента фиксации
            return -1, age # Медвежий сигнал

    # --- Проверка на бычью дивергенцию (Lower Lows на цене, Higher Lows на RSI) ---
    if len(rsi_troughs) >= 2:
        # Берем две последние впадины RSI
        t1_idx, t2_idx = rsi_troughs[-2], rsi_troughs[-1]

        # Получаем их значения и соответствующие им значения цены
        rsi_t1, rsi_t2 = rsi_data.iloc[t1_idx], rsi_data.iloc[t2_idx]
        price_t1, price_t2 = price_data_low.iloc[t1_idx], price_data_low.iloc[t2_idx]
        
        # Условие дивергенции
        if price_t2 < price_t1 and rsi_t2 > rsi_t1:
            age = len(rsi_data) - 1 - t2_idx # Сколько свечей прошло с момента фиксации
            return 1, age # Бычий сигнал

    return 0, 0 # Дивергенция не найдена

async def train_and_predict_hmm_regime(symbol: str) -> Optional[int]:
    """
    Обучает Скрытую Марковскую Модель (HMM) и предсказывает текущий рыночный режим.
    Версия с улучшенной обработкой пустых состояний.
    """
    global hmm_models_manager, market_regimes

    df_source = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS[ML_MODEL_TF_MEDIUM])

    if df_source is None or len(df_source) < HMM_TRAIN_PERIODS:
        logging.debug(f"[{symbol}] Недостаточно данных для обучения HMM ({len(df_source) if df_source is not None else 0}/{HMM_TRAIN_PERIODS}).")
        return None

    df = df_source.copy()
    df['returns'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['returns'].rolling(window=21).std() * np.sqrt(21)
    df.dropna(inplace=True)
    
    if df.empty:
        logging.warning(f"[{symbol}] DataFrame пуст после подготовки данных для HMM.")
        return None
        
    features = df[['returns', 'volatility']].values

    try:
        if symbol not in hmm_models_manager:
            logging.info(f"[{symbol}] Обучение новой HMM модели...")
            # Стало:
            model = GaussianHMM(n_components=HMM_N_STATES, covariance_type="diag", n_iter=1500, tol=0.001, random_state=42)
            model.fit(features)
            
            # ✅ ИСПРАВЛЕНИЕ: Более надёжный способ сортировки режимов
            # Предсказываем состояния для всех данных один раз
            predicted_states = model.predict(features)
            
            # Рассчитываем волатильность для каждого состояния
            regime_volatilities = []
            for i in range(HMM_N_STATES):
                # Выбираем волатильность для текущего состояния
                state_volatility_features = features[predicted_states == i, 1]
                if state_volatility_features.size > 0:
                    # Если состояние не пустое, считаем среднюю волатильность
                    regime_volatilities.append(np.mean(state_volatility_features))
                else:
                    # Если состояние пустое, присваиваем "бесконечную" волатильность,
                    # чтобы оно гарантированно оказалось последним при сортировке
                    regime_volatilities.append(np.inf)

            state_order = np.argsort(regime_volatilities)
            
            hmm_models_manager[symbol] = {'model': model, 'order': {old: new for new, old in enumerate(state_order)}}
            logging.info(f"[{symbol}] HMM модель обучена и отсортирована по волатильности.")

        model_data = hmm_models_manager[symbol]
        model = model_data['model']
        state_order_map = model_data['order']

        hidden_states = model.predict(features)
        last_raw_regime = hidden_states[-1]
        
        sorted_regime = state_order_map[last_raw_regime]
        
        if market_regimes.get(symbol) != sorted_regime:
            regime_name = REGIME_CONFIG.get(sorted_regime, {}).get('name', 'Unknown')
            logging.warning(f"[{symbol}] СМЕНА РЕЖИМА: Новый режим -> {sorted_regime} ({regime_name})")

        market_regimes[symbol] = sorted_regime
        return sorted_regime

    except Exception as e:
        logging.error(f"[{symbol}] КРИТИЧЕСКАЯ ОШИБКА в HMM: {e}", exc_info=True)
        if symbol in hmm_models_manager:
            del hmm_models_manager[symbol]
        return None

# =================================================================
# ## УНИВЕРСАЛЬНЫЙ МЕНЕДЖЕР ПОЗИЦИЙ (ИСПРАВЛЕННАЯ ВЕРСЯ) ##
# =================================================================



def _rr(side: str, entry: float, sl: float, price: float) -> float:
    """Рассчитывает текущее соотношение Риск/Прибыль (R:R)."""
    # Рассчитываем размер риска как абсолютную разницу между ценой входа и стоп-лоссом.
    risk = abs(entry - sl)
    
    # Если риск пренебрежимо мал, возвращаем бесконечность, чтобы избежать деления на ноль.
    if risk < 1e-9: 
        return float('inf')
        
    # Рассчитываем текущую прибыль (или убыток).
    # Для лонга это разница между текущей ценой и ценой входа.
    # Для шорта — наоборот.
    pnl = (price - entry) if side == 'long' else (entry - price)
    
    # Возвращаем отношение прибыли к риску.
    return pnl / risk

def _atr(df: pd.DataFrame, period: int = 14) -> float:
    """Простой и быстрый расчет ATR."""
    if df is None or len(df) < period + 1: return 0.0
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    return float(np.nanmean(tr[-period:])) if len(tr) >= period else 0.0

def _last_swing(df: pd.DataFrame, lookback: int = 10) -> Tuple[Optional[float], Optional[float]]:
    """Находит последние локальные экстремумы (свинги)."""
    if df is None or len(df) < lookback: return None, None
    recent_data = df.iloc[-lookback:]
    return float(recent_data['high'].max()), float(recent_data['low'].min())

# =================================================================
# ## УНИВЕРСАЛЬНЫЙ МЕНЕДЖЕР ПОЗИЦИЙ (ИСПРАВЛЕННАЯ ВЕРСЯ) ##
# =================================================================

@dataclass
class PositionState:
    """Хранит все данные о состоянии текущей позиции."""
    side: str
    current_quantity: float
    entry_price: float
    sl_price: Optional[float] = None
    tp1_price: Optional[float] = None # <-- ✅ ДОБАВЛЕНА ЭТА СТРОКА
    initial_sl_price: Optional[float] = None
    initial_quantity: Optional[float] = None
    state: str = 'initial'
    entry_timestamp: float = field(default_factory=time.time)
    meta: Dict[str, Any] = field(default_factory=dict)

class PositionManager:
    """
    Продвинутый менеджер для управления состоянием и жизненным циклом одной позиции.
    Реализует многоступенчатую защиту прибыли, частичную фиксацию и прогрессивный трейлинг.
    """
    
    def __init__(self, position_data: Dict[str, Any]):
        """Инициализирует менеджер, восстанавливая состояние из словаря (v3, с защитой от сбоев).
        ИСПРАВЛЕНО (v3.1): SL инициализируется как None, а не 0.0.
        """
        self.symbol = position_data.get('symbol', '')
        
        # --- БЕЗОПАСНОЕ ПОЛУЧЕНИЕ КЛЮЧЕВЫХ ДАННЫХ (ИСПРАВЛЕНО) ---
        entry_p = position_data.get('entry_price')
        if not entry_p or float(entry_p) <= 0:
            logging.critical(f"[{self.symbol}] КРИТИЧЕСКАЯ ОШИБКА ВОССТАНОВЛЕНИЯ: ЦЕНА ВХОДА НЕ НАЙДЕНА ИЛИ РАВНА НУЛЮ!")
            entry_p = 0.0 # Оставляем 0.0 здесь, т.к. без цены входа менеджер бесполезен

        # ✅ ИСПРАВЛЕНИЕ: Заменяем `or 0.0` на `or None`
        initial_sl_raw = position_data.get('initial_sl_price') or position_data.get('sl_price') or None
        current_sl_raw = position_data.get('sl_price') or initial_sl_raw or None

        # Преобразуем в float только если значение не None
        initial_sl = float(initial_sl_raw) if initial_sl_raw is not None and float(initial_sl_raw) > 0 else None
        current_sl = float(current_sl_raw) if current_sl_raw is not None and float(current_sl_raw) > 0 else None
        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

        initial_quantity = float(position_data.get('initial_amount') or position_data.get('amount') or 0.0)

        self.state = PositionState(
            side=position_data.get('side', 'LONG'),
            current_quantity=float(position_data.get('amount') or 0.0),
            entry_price=float(entry_p),
            sl_price=current_sl, # Теперь здесь None или float
            tp1_price=position_data.get('tp1_price'),
            initial_quantity=initial_quantity,
            initial_sl_price=initial_sl, # Теперь здесь None или float
            state=position_data.get('state', 'initial'),
            meta=position_data.get('meta', {})
        )
        
        if not isinstance(self.state.meta, dict):
            self.state.meta = {}

    def get_state_dict(self) -> Dict[str, Any]:
        """Возвращает полное состояние позиции в виде словаря для сохранения.
        ИСПРАВЛЕНО (v3.1): Использует asdict вместо .dict.copy().
        """
        # ✅ ИСПРАВЛЕНИЕ: Используем asdict для dataclass
        state_dict = asdict(self.state).copy()
        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---
        
        state_dict['symbol'] = self.symbol
        # Добавляем поля для обратной совместимости
        state_dict['amount'] = self.state.current_quantity
        state_dict['initial_amount'] = self.state.initial_quantity
        # Добавляем ордера из meta для полноты картины
        state_dict['sl_order_id'] = self.state.meta.get('sl_order_id')
        state_dict['tp1_order_id'] = self.state.meta.get('tp1_order_id')
        state_dict['tp2_order_id'] = self.state.meta.get('tp2_order_id')
        return state_dict
    
    async def _update_sl_callback(self, new_sl_price: float, reason: str):
        """
        Асинхронно и надежно обновляет стоп-лосс на бирже.
        ИСПРАВЛЕНО (v3.2):
        1. Корректно форматирует цену (format_price) перед отправкой.
        2. Сохраняет в state (sl_price) ФАКТИЧЕСКУЮ цену ордера (с буфером и форматированием).
        3. Немедленно вызывает save_state_async() после обновления.
        """
        log_prefix = f"[{self.symbol}] [Update SL]"
        
        # 0. Отменяем старый ордер
        old_sl_order_id = self.state.meta.get('sl_order_id')
        if old_sl_order_id:
            await cancel_single_order_async(self.symbol, old_sl_order_id)
            self.state.meta.pop('sl_order_id', None) # Сразу очищаем, на случай если новый не установится

        close_side = 'SELL' if self.state.side.upper() == 'LONG' else 'BUY'
        
        # --- 1. Расчет буфера (как и было) ---
        final_sl_price_target = new_sl_price
        price_step = get_symbol_price_step(self.symbol)
        if price_step:
            buffer = price_step * 5
            # ✅ ИСПРАВЛЕНИЕ (Ошибка №5): Явная логика буфера
            if close_side == 'SELL': # Стоп-лосс для LONG (цена НИЖЕ)
                final_sl_price_target -= buffer
            else: # Стоп-лосс для SHORT (цена ВЫШЕ)
                final_sl_price_target += buffer
            logging.info(f"{log_prefix} Применен буфер безопасности к SL. Расчетный: {new_sl_price}, Целевой (с буфером): {final_sl_price_target}")

        # --- ✅ 2. Форматирование цены (Ошибка №6) ---
        formatted_sl_price_str = format_price(final_sl_price_target, self.symbol, side=close_side)
        if not formatted_sl_price_str:
             logging.critical(f"{log_prefix} КРИТИЧЕСКАЯ ОШИБКА: Не удалось отформатировать цену SL: {final_sl_price_target}. Позиция не защищена.")
             await send_telegram_message(f"🚨 <b>{self.symbol}</b>: КРИТИЧЕСКАЯ ОШИБКА ФОРМАТИРОВАНИЯ SL!")
             return # Не продолжаем

        final_sl_price_formatted = float(formatted_sl_price_str)
        
        # --- 3. Размещение ордера ---
        new_order = await place_order_async(
            symbol=self.symbol, side=close_side, order_type="STOP_MARKET",
            quantity=self.state.current_quantity, 
            stopPrice=final_sl_price_formatted, # Используем отформатированную цену
            reduce_only=True
        )
        
        if new_order and new_order.get('orderId'):
            # --- ✅ 4. Исправление Ошибок №1 и №2 ---
            self.state.sl_price = final_sl_price_formatted # BUG 1: Сохраняем ФАКТИЧЕСКУЮ цену ордера
            self.state.meta['sl_order_id'] = new_order.get('orderId')
            await save_state_async() # BUG 2: Немедленно сохраняем состояние
            # --- КОНЕЦ ИСПРАВЛЕНИЙ ---

            await send_telegram_message(f"📈 <b>{self.symbol} ({self.state.side.upper()})</b>: SL обновлен -> <b>{formatted_sl_price_str}</b> ({reason})")
        else:
            logging.critical(f"[{self.symbol}] КРИТИЧЕСКАЯ ОШИБКА ОБНОВЛЕНИЯ SL! Позиция может быть не защищена.")
            await send_telegram_message(f"🚨 <b>{self.symbol}</b>: КРИТИЧЕСКАЯ ОШИБКА ОБНОВЛЕНИЯ SL!")

    async def _execute_partial_tp(self, fraction: float, reason: str):
        """
        Надежно закрывает часть позиции, отменяет все TP ордера и
        ПЕРЕУСТАНАВЛИВАЕТ SL (В БЕЗУБЫТОК) и АКТИВИРУЕТ ТРЕЙЛИНГ.
        ИСПРАВЛЕНО (v3.2): Не закрывает всю позицию при ошибке minNotional,
                         корректно переустанавливает SL на ОСТАТОК.
        """
        pos = self.state
        log_prefix = f"[{self.symbol}] [Partial Close]"
        
        # --- 1. Отменяем ВСЕ существующие связанные TP ордера ---
        order_ids_to_cancel = [
            pos.meta.get(key) for key in ['tp1_order_id', 'tp2_order_id', 'near_tp_partial_oid', 'near_tp_guard_oid'] 
            if pos.meta.get(key)
        ]
        if order_ids_to_cancel:
            logging.info(f"{log_prefix} Отмена {len(order_ids_to_cancel)} TP ордеров...")
            await asyncio.gather(*[cancel_single_order_async(self.symbol, oid) for oid in order_ids_to_cancel])
            for key in ['tp1_order_id', 'tp2_order_id', 'near_tp_partial_oid', 'near_tp_guard_oid']:
                pos.meta.pop(key, None)

        # --- 2. Рассчитываем и закрываем часть позиции ---
        qty_to_close_raw = (pos.initial_quantity or pos.current_quantity) * fraction
        qty_to_close = float(format_quantity(qty_to_close_raw, self.symbol) or 0.0)
        qty_to_close = min(qty_to_close, pos.current_quantity) 

        df_5m = market_data_store.get(self.symbol, {}).get(ALLOWED_INTERVALS["5m"])
        current_price = df_5m['close'].iloc[-1] if df_5m is not None and not df_5m.empty else pos.entry_price
        min_notional = float(exchange_info_cache.get(self.symbol, {}).get('minNotional', 5.1))
        notional_value = qty_to_close * current_price
        
        # ✅ ИСПРАВЛЕНИЕ (Ошибка №3): Не закрываем всю позицию
        if notional_value < min_notional and qty_to_close < pos.current_quantity:
            logging.warning(
                f"{log_prefix} {reason}: Частичное закрытие невозможно (стоимость ${notional_value:.2f} < ${min_notional:.2f}). "
                f"ПРОПУСКАЮ частичное закрытие."
            )
            # НЕ закрываем всю позицию, а просто пропускаем этот шаг
            qty_to_close = 0.0 
        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---
        
        if qty_to_close > 0:
            close_side = 'SELL' if pos.side.upper() == 'LONG' else 'BUY'
            closing_order = await place_order_async(symbol=self.symbol, side=close_side, order_type="MARKET", quantity=qty_to_close, reduce_only=True)
            if closing_order and closing_order.get('orderId'):
                pos.current_quantity -= qty_to_close
                await send_telegram_message(f"💰 **{self.symbol}**: {reason} - зафиксировано **{qty_to_close}** ед.")
            else:
                logging.error(f"{log_prefix} Не удалось отправить ордер на частичное закрытие! ({reason})")
                return 
        
        # --- 3. УСТАНОВКА BE И АКТИВАЦИЯ ТРЕЙЛИНГА (ИСПРАВЛЕНО Ошибка №4, №6, №7) ---
        pos.meta["tp1_done"] = True # Отмечаем, что TP1 выполнен
        
        if pos.current_quantity > 0:
            # Рассчитываем BE + буфер
            df_work_tf = market_data_store.get(self.symbol, {}).get(ALLOWED_INTERVALS['15m'])
            atr_value = _atr(df_work_tf, 21) if (df_work_tf is not None and not df_work_tf.empty) else (pos.entry_price * 0.005)
            if atr_value <= 0: atr_value = pos.entry_price * 0.005
            
            be_buffer = BE_ATR_BUFFER_FRAC * atr_value
            be_sl_price_target = (pos.entry_price + be_buffer) if pos.side.upper() == "LONG" else (pos.entry_price - be_buffer)
            
            close_side_for_be = "SELL" if pos.side.upper() == "LONG" else "BUY"
            formatted_be_sl_price_str = format_price(be_sl_price_target, self.symbol, side=close_side_for_be)
            
            if not formatted_be_sl_price_str:
                 logging.error(f"{log_prefix} НЕ УДАЛОСЬ отформатировать цену BE SL: {be_sl_price_target}. Обновление SL пропущено.")
                 await save_state_async()
                 return

            final_be_sl_price = float(formatted_be_sl_price_str)
            
            # Используем replace_order_safe для атомарной замены SL
            if await replace_order_safe(self, "sl_order_id", 
                                      lambda: self._place_updated_sl(final_be_sl_price), # _place_updated_sl САМ возьмет pos.current_quantity
                                      f"BE + Buffer after {reason}"):
                
                pos.sl_price = final_be_sl_price # Сохраняем ФАКТИЧЕСКУЮ цену ордера
                pos.meta["be_done"] = True
                pos.meta["atr_trail_active"] = True 
                logging.info(f"{log_prefix} BE установлен ({pos.sl_price:.5f}). АВТО-ТРЕЙЛИНГ АКТИВИРОВАН.")
        
        else:
            # Если позиция полностью закрыта, отменяем SL
            logging.warning(f"{log_prefix} Позиция полностью закрыта после фиксации прибыли.")
            await replace_order_safe(self, "sl_order_id", lambda: asyncio.sleep(0), "Position fully closed on partial")
            if self.symbol in current_positions:
                del current_positions[self.symbol]
        
        await save_state_async()

    async def _cancel_order_by_key(self, key: str, reason: str):
        """Вспомогательная функция: отменяет ордер по ключу в meta и очищает ключ."""
        pos = self.state
        old_oid = pos.meta.get(key)
        if old_oid:
            logging.info(f"[{self.symbol}] Отмена ордера ({key}: {old_oid}). Причина: {reason}.")
            await cancel_single_order_async(self.symbol, old_oid)
            pos.meta.pop(key, None)
            # Не нужно save_state_async здесь, т.к. 'manage' сделает это в конце

    async def manage(self, df_work_tf: pd.DataFrame):
        """
        ФИНАЛЬНАЯ ВЕРСИЯ v2.8.7:
        - ✅ Исправлена логика отмены TP1/TP2 (используется _cancel_order_by_key).
        - BE на PnL >= 40%.
        - Активация ATR/CE Трейлинга СРАЗУ ПОСЛЕ BE.
        """
        pos = self.state
        log_prefix = f"[{self.symbol}]"

        MIN_ROWS_FOR_MANAGE = 60
        if df_work_tf is None or len(df_work_tf) < MIN_ROWS_FOR_MANAGE:
            tf_name = df_work_tf.index.name if (df_work_tf is not None and hasattr(df_work_tf, 'index') and df_work_tf.index.name) else 'N/A'
            logging.debug(f"{log_prefix} Управление пропущено (TF {tf_name}): "
                          f"недостаточно данных ({len(df_work_tf) if df_work_tf is not None else 0}/{MIN_ROWS_FOR_MANAGE})")
            return

        strategy_type = pos.meta.get('strategy_type', 'trend')

        if pos.current_quantity <= 0 or pos.initial_sl_price is None or pos.sl_price is None:
            logging.debug(f"{log_prefix} Управление пропущено: позиция закрыта или SL не установлен (is None).")
            return

        try:
            current_price = float(df_work_tf['close'].iloc[-1])
            atr_value = _atr(df_work_tf, period=21)
            if atr_value <= 0:
                atr_value = abs(pos.entry_price * 0.005) if pos.entry_price else 0.001
                logging.warning(f"{log_prefix} ATR(21) <= 0 для manage, используется fallback: {atr_value:.5f}")
            if atr_value <= 0: return

            current_rr = _rr(pos.side, pos.entry_price, pos.initial_sl_price, current_price)
            try:
                adx_val = ta.trend.adx(df_work_tf['high'], df_work_tf['low'], df_work_tf['close'], window=14).iloc[-1]
            except Exception as e_adx:
                 logging.error(f"{log_prefix} Ошибка расчета ADX для manage: {e_adx}. Использую fallback 25.")
                 adx_val = 25.0

            # Расчет PnL % (Return on Margin)
            pnl_usd = (current_price - pos.entry_price) * pos.current_quantity if pos.side.upper() == "LONG" else (pos.entry_price - current_price) * pos.current_quantity
            initial_qty = pos.initial_quantity if pos.initial_quantity and pos.initial_quantity > 0 else pos.current_quantity
            initial_margin = (pos.entry_price * initial_qty / LEVERAGE) if pos.entry_price > 0 and initial_qty > 0 and LEVERAGE > 0 else 0
            pnl_percent = (pnl_usd / initial_margin) * 100 if initial_margin > 0 else 0
            
            BE_TRIGGER_PNL_PCT = 40.0 # Ваш порог 40%

            logging.info(f"{log_prefix} Управление ({strategy_type}): Цена={current_price:.4f}, SL={pos.sl_price:.4f}, RR={current_rr:.2f}, ADX={adx_val:.1f}, PnL%={pnl_percent:.2f}%")

            # --- 1. ПЕРЕВОД В БЕЗУБЫТОК (BE + Buffer) ---
            if not pos.meta.get("be_done", False) and pnl_percent >= BE_TRIGGER_PNL_PCT:
                logging.warning(f"{log_prefix} 🎯 BE Trigger (PnL% >= {BE_TRIGGER_PNL_PCT:.1f}%). Текущий PnL: {pnl_percent:.2f}%. Установка BE+Buffer...")
                
                # --- ✅ ИСПРАВЛЕНИЕ: Используем _cancel_order_by_key ---
                await self._cancel_order_by_key("tp1_order_id", "On BE (TP1 passed)")
                await self._cancel_order_by_key("near_tp_partial_oid", "On BE")
                await self._cancel_order_by_key("near_tp_guard_oid", "On BE")
                # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

                be_buffer = BE_ATR_BUFFER_FRAC * atr_value
                be_sl_price_target = (pos.entry_price + be_buffer) if pos.side.upper() == "LONG" else (pos.entry_price - be_buffer)

                close_side_for_be = "SELL" if pos.side.upper() == "LONG" else "BUY"
                formatted_be_sl_price_str = format_price(be_sl_price_target, self.symbol, side=close_side_for_be)
                
                if not formatted_be_sl_price_str:
                    logging.error(f"{log_prefix} НЕ УДАЛОСЬ отформатировать цену BE SL: {be_sl_price_target}. Обновление SL пропущено.")
                else:
                    final_be_sl_price = float(formatted_be_sl_price_str)
                    
                    if await replace_order_safe(self, "sl_order_id", lambda: self._place_updated_sl(final_be_sl_price), f"BE + Buffer @ {pnl_percent:.1f}% PnL"):
                       pos.sl_price = final_be_sl_price 
                       pos.meta["be_done"] = True
                       pos.meta["atr_trail_active"] = True 
                       await save_state_async()
                       logging.info(f"{log_prefix} BE установлен ({pos.sl_price:.5f}). АВТО-ТРЕЙЛИНГ АКТИВИРОВАН.")

            # --- 2. ВЫБОР ЛОГИКИ ТРЕЙЛИНГА (ПОСЛЕ BE или TP1) ---
            is_trailing_active = pos.meta.get("be_done", False) or pos.meta.get("tp1_done", False)

            if is_trailing_active: 
                
                close_side_for_trail = "SELL" if pos.side.upper() == "LONG" else "BUY"
                
                # --- ЛОГИКА ТРЕНДА (CE + ATR) ---
                if strategy_type == 'trend':
                    logging.debug(f"{log_prefix} Применяется логика Trend (Трейлинг активен).")

                    if adx_val < TREND_WEAKENING_ADX_THRESHOLD:
                         try:
                             fast_ema = ta.trend.ema_indicator(df_work_tf["close"], window=TRENDING_FAST_EMA).iloc[-1]
                             slow_ema = ta.trend.ema_indicator(df_work_tf["close"], window=TRENDING_SLOW_EMA).iloc[-1]
                             ema_crossed_against = (pos.side.upper() == "LONG" and fast_ema < slow_ema) or (pos.side.upper() == "SHORT" and fast_ema > slow_ema)
                             if ema_crossed_against:
                                  logging.warning(f"{log_prefix} TREND ВЫХОД: Тренд ослаб (ADX={adx_val:.1f}) + EMA Cross. Закрываю.")
                                  await close_position_emergency(self.symbol, self.get_state_dict(), f"Trend Weakening Exit (ADX={adx_val:.1f})")
                                  return
                         except Exception as e_ema: logging.error(f"{log_prefix} Ошибка EMA для ADX-выхода: {e_ema}")

                    new_sl_ce = calculate_chandelier_exit_stop(self.symbol, pos.side, market_data_store, tf_key='1h', period=CHANDELIER_EXIT_PERIOD, multiplier=CHANDELIER_EXIT_MULTIPLIER)
                    atr_trail_mult = AGGRESSIVE_TRAIL_ATR_MULT if current_rr >= AGGRESSIVE_TRAIL_RR_THRESHOLD else TRAIL_ATR_MULT
                    atr_distance = atr_value * atr_trail_mult
                    new_sl_atr = (current_price - atr_distance) if pos.side.upper() == "LONG" else (current_price + atr_distance)
                    potential_sls = [pos.sl_price]
                    if new_sl_ce: potential_sls.append(new_sl_ce)
                    if new_sl_atr: potential_sls.append(new_sl_atr)
                    new_best_sl_target = 0.0
                    if pos.side.upper() == "LONG":
                        valid_sls = [sl for sl in potential_sls if sl < current_price]
                        new_best_sl_target = max(valid_sls) if valid_sls else pos.sl_price
                    else: # SHORT
                        valid_sls = [sl for sl in potential_sls if sl > current_price]
                        new_best_sl_target = min(valid_sls) if valid_sls else pos.sl_price

                    formatted_new_best_sl_str = format_price(new_best_sl_target, self.symbol, side=close_side_for_trail)
                    if not formatted_new_best_sl_str:
                         logging.error(f"{log_prefix} НЕ УДАЛОСЬ отформатировать цену Trend Trail SL: {new_best_sl_target}.")
                         return
                    
                    final_new_best_sl = float(formatted_new_best_sl_str)
                    
                    if (pos.side.upper() == "LONG" and final_new_best_sl > pos.sl_price) or \
                       (pos.side.upper() == "SHORT" and final_new_best_sl < pos.sl_price):
                        
                        reason = "CE Trail" if new_sl_ce and abs(final_new_best_sl - float(format_price(new_sl_ce, self.symbol, side=close_side_for_trail) or 0.0)) < 1e-9 else f"ATR Trail x{atr_trail_mult:.1f}"
                        
                        # --- ✅ ИСПРАВЛЕНИЕ: Используем _cancel_order_by_key ---
                        if pos.meta.get("tp2_order_id"):
                            logging.warning(f"{log_prefix} Трейлинг ({reason}) активирован. Отмена TP2.")
                            await self._cancel_order_by_key("tp2_order_id", "Trend Trail Activated")
                            pos.meta.pop("tp2_price", None)
                        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---
                        
                        if await replace_order_safe(self, "sl_order_id", lambda: self._place_updated_sl(final_new_best_sl), f"{reason} @ {current_rr:.1f}R"):
                            pos.sl_price = final_new_best_sl 
                            await save_state_async()
                            logging.info(f"{log_prefix} Trend Трейлинг SL ({reason}) обновлен: {pos.sl_price:.4f}")

                # --- ЛОГИКА ФЛЭТА (ATR) ---
                elif strategy_type == 'ranging':
                    logging.debug(f"{log_prefix} Применяется логика Ranging (Трейлинг активен).")

                    distance = atr_value * TRAIL_ATR_MULT_RANGE
                    new_sl_atr_target = (current_price - distance) if pos.side.upper() == "LONG" else (current_price + distance)
                    
                    formatted_new_sl_atr_str = format_price(new_sl_atr_target, self.symbol, side=close_side_for_trail)
                    if not formatted_new_sl_atr_str:
                         logging.error(f"{log_prefix} НЕ УДАЛОСЬ отформатировать цену Ranging Trail SL: {new_sl_atr_target}.")
                         return
                         
                    final_new_sl_atr = float(formatted_new_sl_atr_str)
                    
                    if (pos.side.upper() == "LONG" and final_new_sl_atr > pos.sl_price and final_new_sl_atr < current_price) or \
                       (pos.side.upper() == "SHORT" and final_new_sl_atr < pos.sl_price and final_new_sl_atr > current_price):
                        
                        # --- ✅ ИСПРАВЛЕНИЕ: Используем _cancel_order_by_key ---
                        if pos.meta.get("tp2_order_id"):
                            logging.warning(f"{log_prefix} Ranging ATR Trail активирован. Отмена TP2.")
                            await self._cancel_order_by_key("tp2_order_id", "Ranging Trail Activated")
                            pos.meta.pop("tp2_price", None)
                        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

                        if await replace_order_safe(self, "sl_order_id", lambda: self._place_updated_sl(final_new_sl_atr), f"Ranging ATR Trail x{TRAIL_ATR_MULT_RANGE:.1f}"):
                             pos.sl_price = final_new_sl_atr
                             await save_state_async()
                             logging.info(f"{log_prefix} Ranging Трейлинг SL (ATR) обновлен: {pos.sl_price:.4f}")

            # --- 3. УПРАВЛЕНИЕ ФЛЭТОМ (ДО АКТИВАЦИИ ТРЕЙЛИНГА) ---
            if strategy_type == 'ranging' and not is_trailing_active:
                # (Логика без изменений)
                logging.debug(f"{log_prefix} Применяется логика Ranging (Ожидание TP1 или BE).")
                ADX_EXIT_THRESHOLD_RANGING = 28.0
                if adx_val >= ADX_EXIT_THRESHOLD_RANGING:
                    await close_position_emergency(self.symbol, self.get_state_dict(), f"Ranging ADX Exit (ADX={adx_val:.1f})")
                    return
                MAX_BARS_BEFORE_TP1_RANGING = 15
                bars_since_entry = pos.meta.get("bars_since_entry", 0) + 1
                pos.meta["bars_since_entry"] = bars_since_entry
                if not pos.meta.get("tp1_done", False) and bars_since_entry > MAX_BARS_BEFORE_TP1_RANGING:
                    await close_position_emergency(self.symbol, self.get_state_dict(), f"Ranging Time Stop ({bars_since_entry} bars)")
                    return
                tp1_price = pos.meta.get("tp1_price")
                tp2_price = pos.meta.get("tp2_price")
                if not tp1_price:
                     logging.error(f"{log_prefix} Ошибка Ranging: Отсутствует цена TP1.")
                     return
                tp1_reached = (pos.side.upper() == "LONG" and current_price >= tp1_price) or \
                              (pos.side.upper() == "SHORT" and current_price <= tp1_price)
                if not pos.meta.get("tp1_done", False) and tp1_reached:
                    logging.warning(f"{log_prefix} 🎯 RANGING TP1 (Цена @ {tp1_price:.4f}). Фиксация {TP1_CLOSE_FRAC*100}%...")
                    await self._execute_partial_tp(fraction=TP1_CLOSE_FRAC, reason=f"TP1 @ {current_rr:.1f}R")
                    await save_state_async()
                    logging.info(f"{log_prefix} Ranging TP1 обработан.")
                tp2_reached = False
                if tp2_price:
                     tp2_reached = (pos.side.upper() == "LONG" and current_price >= tp2_price) or \
                                   (pos.side.upper() == "SHORT" and current_price <= tp2_price)
                if pos.meta.get("tp1_done", False) and not pos.meta.get("tp2_done", False) and tp2_reached:
                    logging.warning(f"{log_prefix} 🎯 RANGING TP2 (Цена @ {tp2_price:.4f}). Фиксация {TP2_CLOSE_FRAC*100}%...")
                    await self._execute_partial_tp(fraction=TP2_CLOSE_FRAC, reason=f"TP2 @ {current_rr:.1f}R")
                    pos.meta["tp2_done"] = True
                    await save_state_async()

        except Exception as e:
            logging.error(f"{log_prefix} Критическая ошибка в PositionManager.manage (V2.8.7): {e}", exc_info=True)


    # Вспомогательный метод для создания SL ордера (остается без изменений)
    # Вспомогательный метод для создания SL ордера (остается без изменений)
    async def _place_updated_sl(self, new_sl_price: float):
        """
        Вспомогательная функция для создания нового SL ордера.
        ИСПРАВЛЕНО (v3.1): Принимает УЖЕ отформатированную и/или забуференную цену.
        """
        close_side = 'SELL' if self.state.side.upper() == 'LONG' else 'BUY'
        # Используем текущее количество
        formatted_qty_str = format_quantity(self.state.current_quantity, self.symbol)
        
        # ✅ ИСПРАВЛЕНИЕ (Ошибка №6): Убеждаемся, что format_price вызывается с 'side'
        # (Эта цена УЖЕ должна быть отформатирована вызывающей функцией, 
        # но мы форматируем ее здесь еще раз на всякий случай, если new_sl_price не отформатирован)
        formatted_stop_price_str = format_price(new_sl_price, self.symbol, side=close_side)

        if not formatted_qty_str or not formatted_stop_price_str or float(formatted_qty_str) <= 0:
            logging.error(f"[{self.symbol}] Ошибка форматирования данных для SL: Qty='{formatted_qty_str}', Price='{formatted_stop_price_str}'")
            return None

        # Используем place_order_async без smart_entry для защитных ордеров
        return await place_order_async(
            symbol=self.symbol,
            side=close_side,
            order_type="STOP_MARKET",
            quantity=float(formatted_qty_str),
            stopPrice=float(formatted_stop_price_str), # Используем отформатированную цену
            reduce_only=True
        )
    
    # --- Остальные методы класса (остаются без изменений) ---
    
    async def get_detailed_status(self) -> Dict[str, Any]:
        """Возвращает детальную информацию о текущем состоянии позиции."""
        pos = self.state
        df_5m = market_data_store.get(self.symbol, {}).get(ALLOWED_INTERVALS['5m'])
        if df_5m is None or df_5m.empty: return {"error": "Нет рыночных данных"}
        
        current_price = df_5m['close'].iloc[-1]
        initial_qty = pos.initial_quantity if pos.initial_quantity and pos.initial_quantity > 0 else pos.current_quantity
        pnl = (current_price - pos.entry_price) * pos.current_quantity if pos.side.upper() == 'LONG' else (pos.entry_price - current_price) * pos.current_quantity
        pnl_percent = (pnl / (pos.entry_price * initial_qty / LEVERAGE)) * 100 if pos.entry_price > 0 and initial_qty > 0 else 0
        
        is_breakeven = (pos.side.upper() == 'LONG' and pos.sl_price >= pos.entry_price) or \
                       (pos.side.upper() == 'SHORT' and pos.sl_price <= pos.entry_price) if pos.sl_price is not None else False
        
        status_str = "В прибыли" if pnl > 0 else "В просадке"
        if pos.meta.get('tp1_done'): status_str += ", трейлинг активен"
        if is_breakeven: status_str += ", в безубытке"

        tp_status_parts = []
        if pos.meta.get('tp1_done') is not None: tp_status_parts.append(f"TP1 {'✅' if pos.meta.get('tp1_done') else '⏳'}")
        if pos.meta.get('tp2_done') is not None: tp_status_parts.append(f"TP2 {'✅' if pos.meta.get('tp2_done') else '⏳'}")
        
        current_rr = _rr(pos.side, pos.entry_price, pos.initial_sl_price, current_price) if pos.initial_sl_price is not None else 0
        age_seconds = time.time() - pos.entry_timestamp
        age_str = str(datetime.timedelta(seconds=int(age_seconds)))

        return {
            "pnl": pnl, "pnl_percent": pnl_percent, "status_str": status_str,
            "current_sl": pos.sl_price, "is_breakeven": is_breakeven,
            "tp_status": ", ".join(tp_status_parts), "current_rr": current_rr, "age_str": age_str
        }

    @staticmethod
    @staticmethod
    def create_scaling_plan_v2(symbol: str, side: str, entry_price: float, df_5m: pd.DataFrame) -> dict:
        """Создает план для будущих докупок (DCA V2) на основе режима рынка/волатильности."""
        if df_5m is None or len(df_5m) < 60:
            return None
        atr_val = _atr(df_5m, 14)
        if atr_val <= 0:
            return None
        
        mults_side = [0.8, 1.4, 2.0, 2.8, 3.6]
        qty_fracs  = [0.35, 0.25, 0.20, 0.12, 0.08]
        
        if side.upper() == "LONG":
            levels = [entry_price - m * atr_val for m in mults_side]
        else:
            levels = [entry_price + m * atr_val for m in mults_side]
            
        rounding_side = 'BUY' if side.upper() == 'SHORT' else 'SELL'
        levels_fmt = [float(format_price(p, symbol, side=rounding_side)) for p in levels if format_price(p, symbol, side=rounding_side)]
        
        plan = {
            "max_adds": len(levels_fmt),           # ✅ Исправлено: max_adds
            "adds_done": 0,                         # ✅ Исправлено: adds_done
            "qty_fractions": qty_fracs[:len(levels_fmt)],
            "levels": levels_fmt,
            "next_add_price": levels_fmt[0] if levels_fmt else None, # ✅ Исправлено: next_add_price
        }
        logging.info(f"[{symbol}] План DCA V2 создан. Уровни: {levels_fmt}")
        return plan

# (Вставьте этот блок в ваш файл main.py)


async def _oi_features(symbol: str, tf_key: str = '1h', lookback: int = 6) -> Dict[str, Optional[float]]:
    """
    Рассчитывает моментум Открытого Интереса (ОИ) и дивергенцию Цена/ОИ.
    Использует 1-часовые данные для ОИ и цены.
    """
    log_prefix = f"[{symbol}] [OI Features]"
    try:
        # 1. Получаем данные о цене
        df_price = _safe_df(symbol, tf_key, need=lookback + 2)
        if df_price is None:
            logging.warning(f"{log_prefix} Недостаточно данных о цене на {tf_key}.")
            return {"oi_mom_pct": None, "price_oi_divergence": None, "oi_latest_usd": None}

        # 2. Получаем историю Открытого Интереса
        # (Binance API отдает историю ОИ в USD для фьючерсов)
        oi_history = await client.futures_open_interest_hist(symbol=symbol, period='1h', limit=lookback + 2)
        if not oi_history or len(oi_history) < lookback + 1:
            logging.warning(f"{log_prefix} Недостаточно истории ОИ ({len(oi_history) if oi_history else 0}).")
            return {"oi_mom_pct": None, "price_oi_divergence": None, "oi_latest_usd": None}

        # 3. Расчет изменений (предполагаем, что данные синхронизированы по времени)
        
        # Изменение цены в %
        price_latest = float(df_price['close'].iloc[-1])
        price_past = float(df_price['close'].iloc[-1 - lookback])
        price_change_pct = ((price_latest - price_past) / price_past) * 100.0 if price_past > 0 else 0.0

        # Изменение ОИ в %
        oi_latest_usd = float(oi_history[-1]['sumOpenInterestValue'])
        oi_past_usd = float(oi_history[-1 - lookback]['sumOpenInterestValue'])
        oi_change_pct = ((oi_latest_usd - oi_past_usd) / oi_past_usd) * 100.0 if oi_past_usd > 0 else 0.0

        # 4. Расчет Дивергенции/Конвергенции
        # Используем np.sign() для получения направления (-1, 0, или 1)
        price_dir = np.sign(price_change_pct)
        oi_dir = np.sign(oi_change_pct)
        
        # Конвергенция (здоровый тренд): price_dir * oi_dir = 1 (1*1=1 или -1*-1=1)
        # Дивергенция (нездоровый тренд): price_dir * oi_dir = -1 (1*-1=-1)
        divergence_score = price_dir * oi_dir 

        logging.info(f"{log_prefix} ({lookback}h): Изм. Цены={price_change_pct:.2f}%, Изм. ОИ={oi_change_pct:.2f}%. Оценка Конвергенции={divergence_score}")

        return {
            "oi_mom_pct": oi_change_pct, 
            "price_oi_divergence": divergence_score, # 1 = Конвергенция (хорошо), -1 = Дивергенция (плохо)
            "oi_latest_usd": oi_latest_usd
        }

    except BinanceAPIException as e:
        logging.warning(f"{log_prefix} API Ошибка при получении истории ОИ: {e}")
        return {"oi_mom_pct": None, "price_oi_divergence": None, "oi_latest_usd": None}
    except Exception as e:
        logging.error(f"{log_prefix} Ошибка расчета признаков ОИ: {e}", exc_info=False)
        return {"oi_mom_pct": None, "price_oi_divergence": None, "oi_latest_usd": None}

async def get_vision_enhanced_trade_decision(
    symbol: str, side: str, entry_price: float, market_env: str) -> Dict[str, Any]:
    """
    Комплексная функция v2: собирает расширенный контекст, создаёт график,
    отправляет в Gemini Vision и возвращает решение.
    """
    global market_data_store, ALLOWED_INTERVALS, client # Убедимся, что client доступен

    default_response = {"confidence": 0, "explanation": "Ошибка получения данных для Gemini"}

    try:
        # --- 1. Сбор данных для графика и базового контекста ---
        df_chart = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS.get("15m"))
        if df_chart is None or len(df_chart) < 50:
            return {"confidence": 0, "explanation": "Недостаточно данных для анализа графика (15m)"}

        # --- 2. Параллельный сбор расширенного контекста ---
        context_tasks = {
            "microstructure": get_market_microstructure_features(symbol, client),
            "derivatives": get_derivatives_sentiment(symbol, client),
            "ofi": calculate_order_flow_imbalance(symbol),
            "liquidations": asyncio.to_thread(get_liquidation_signal, symbol), # Синхронная функция
            "basis_funding": get_basis_and_funding(symbol),
            "btc_trend": get_btc_trend_strength(),
            "btc_corr": get_correlation_snapshot(symbol, 'BTCUSDT')
        }
        # Запускаем все задачи параллельно
        results = await asyncio.gather(*context_tasks.values(), return_exceptions=True)
        # Собираем результаты в словарь
        gathered_data = {}
        for i, key in enumerate(context_tasks.keys()):
             if isinstance(results[i], Exception):
                  logging.error(f"[{symbol}] Ошибка при сборе контекста '{key}': {results[i]}")
                  gathered_data[key] = {} # Используем пустой словарь при ошибке
             else:
                  gathered_data[key] = results[i]

        # --- 3. Формирование строки additional_context ---
        # Базовые индикаторы
        rsi_val = ta.momentum.rsi(df_chart['close'], 14).iloc[-1]
        macd_val = ta.trend.MACD(df_chart['close']).macd_diff().iloc[-1]
        adx_val = ta.trend.adx(df_chart['high'], df_chart['low'], df_chart['close'], 14).iloc[-1]

        # Собираем строку контекста, безопасно извлекая данные
        ctx_parts = [
            f"Base: RSI={rsi_val:.1f}, MACD_diff={macd_val:.4f}, ADX={adx_val:.1f}, MarketEnv={market_env}.",
            f"Micro: OBI={gathered_data.get('microstructure', {}).get('order_book_imbalance', 'N/A'):.3f}, "
            f"Depth=${gathered_data.get('microstructure', {}).get('market_depth_usd', 'N/A'):,.0f}.",
            f"Flow: OFI={gathered_data.get('ofi', (0.0,))[0]:.1f}.",
            f"Derivs: OI=${gathered_data.get('derivatives', {}).get('open_interest_usd', 'N/A'):,.0f}, "
            f"LS_Ratio={gathered_data.get('derivatives', {}).get('long_short_ratio', 'N/A'):.3f}.",
            f"Liqs: Spike={'Yes (' + gathered_data.get('liquidations', {}).get('bias', 'N/A') + ')' if gathered_data.get('liquidations', {}).get('spike') else 'No'}.",
            f"Rates: Funding={gathered_data.get('basis_funding', {}).get('funding_pct', 'N/A'):.4f}%, "
            f"Basis={gathered_data.get('basis_funding', {}).get('basis_bps', 'N/A'):.1f}bps.",
            f"BTC: Trend={gathered_data.get('btc_trend', 'N/A')}, Corr={gathered_data.get('btc_corr', 'N/A'):.2f}."
        ]
        # Убираем 'N/A' и None перед объединением
        cleaned_parts = [part for part in ctx_parts if 'N/A' not in part and 'None' not in part]
        additional_context = " ".join(cleaned_parts)

        logging.info(f"[{symbol}] Расширенный контекст для Gemini: {additional_context}")

        # --- 4. Создание графика ---
        fig = await create_enhanced_chart(symbol, df_chart, side, entry_price)
        if fig is None:
            return {"confidence": 0, "explanation": "Ошибка создания графика"}

        chart_base64 = await save_chart_to_base64(symbol, fig)
        plt.close(fig) # Закрываем фигуру matplotlib

        if not chart_base64:
            return {"confidence": 0, "explanation": "Ошибка конвертации графика"}

        # --- 5. Вызов Gemini Vision с расширенным контекстом ---
        analysis = await analyze_chart_with_gemini_vision(
            symbol=symbol,
            chart_base64=chart_base64,
            trade_type=side,
            additional_context=additional_context # Передаем собранный контекст
        )
        return analysis

    except Exception as e:
        logging.error(f"[{symbol}] Критическая ошибка в get_vision_enhanced_trade_decision_v2: {e}", exc_info=True)
        return default_response

# --- ШАГ 5: Функция reconcile_brackets (ДЛЯ ДОБАВЛЕНИЯ В КОД) ---
async def reconcile_brackets(symbol: str):
    """
    Сверяет состояние локального менеджера с биржей и корректирует
    защитные ордера (SL/TP) при необходимости.
    """
    log_prefix = f"[{symbol}] [Reconcile]"
    logging.info(f"{log_prefix} Запуск сверки скобок...")

    manager = current_positions.get(symbol)
    # Убеждаемся, что это наш менеджер позиций
    if not isinstance(manager, PositionManager):
        logging.info(f"{log_prefix} Менеджер не найден или имеет неверный тип, сверка отменена.")
        return

    pos = manager.state
    # Используем сохраненные ID ордеров из метаданных
    sl_order_id = pos.meta.get("sl_order_id")
    tp1_order_id = pos.meta.get("tp1_order_id") # TP1 из adaptive_execute_entry (если есть)
    # Добавляем ID ордеров от near-TP логики, если они используются
    near_tp_partial_oid = pos.meta.get("near_tp_partial_oid")
    near_tp_guard_oid = pos.meta.get("near_tp_guard_oid")
    # Добавьте сюда другие ключи ID ордеров, если они у вас есть

    try:
        # 1. Получаем актуальную позицию и открытые ордера с биржи
        # Убедитесь, что `client` - это ваш инициализированный AsyncClient
        if not client:
             logging.error(f"{log_prefix} Клиент Binance не инициализирован.")
             return

        position_info_task = client.futures_position_information(symbol=symbol)
        open_orders_api_task = client.futures_get_open_orders(symbol=symbol)
        position_info, open_orders_api = await asyncio.gather(position_info_task, open_orders_api_task, return_exceptions=True)

        # Обработка ошибок при получении данных
        if isinstance(position_info, Exception):
            logging.error(f"{log_prefix} Ошибка получения информации о позиции: {position_info}")
            return
        if isinstance(open_orders_api, Exception):
            logging.error(f"{log_prefix} Ошибка получения открытых ордеров: {open_orders_api}")
            open_orders_api = [] # Продолжаем без информации об ордерах

        # Находим данные именно по нашей позиции
        current_pos_on_api = next((p for p in position_info if p.get('symbol') == symbol), None)
        api_pos_qty = Decimal(current_pos_on_api.get('positionAmt', '0')) if current_pos_on_api else Decimal('0')
        api_pos_qty_abs = abs(api_pos_qty)

        # 2. Обработка случая, когда позиция закрыта на бирже
        if api_pos_qty_abs == 0:
            logging.warning(f"{log_prefix} Позиция на бирже равна нулю. Отменяю все связанные ордера...")
            # Собираем все ID, которые нужно отменить
            orders_to_cancel_ids = [sl_order_id, tp1_order_id, near_tp_partial_oid, near_tp_guard_oid]
            # Добавьте сюда другие ID из pos.meta, если нужно
            cancelled_count = 0
            # Отменяем только те, у которых есть ID
            cancellation_tasks = [cancel_single_order_async(symbol, oid) for oid in filter(None, orders_to_cancel_ids)]
            results = await asyncio.gather(*cancellation_tasks, return_exceptions=True)
            for result in results:
                if result is True: # cancel_single_order_async возвращает True при успехе или если ордера уже нет
                    cancelled_count += 1

            # Очищаем все ID ордеров из метаданных
            keys_to_pop = [k for k, v in pos.meta.items() if k.endswith("_order_id") or k.endswith("_oid")]
            for key in keys_to_pop:
                pos.meta.pop(key, None)

            # Удаляем менеджера локально
            if symbol in current_positions:
                del current_positions[symbol]
            await save_state_async()
            logging.info(f"{log_prefix} Сверка завершена. Отменено ордеров: {cancelled_count}. Позиция удалена локально.")
            return

        # 3. Обработка случая, когда позиция частично закрыта (или объем рассинхронизирован)
        local_qty_dec = Decimal(str(pos.current_quantity))
        # Используем stepSize для точного сравнения
        step_size = Decimal(exchange_info_cache.get(symbol, {}).get('stepSize', '0.00000001')) # Безопасный fallback
        qty_diff = abs(local_qty_dec - api_pos_qty_abs)
        # Считаем расхождением, если разница больше половины шага
        is_partially_closed_or_desynced = qty_diff > (step_size * Decimal('0.5'))

        if is_partially_closed_or_desynced:
            logging.warning(f"{log_prefix} Обнаружено расхождение объемов! Локально: {local_qty_dec}, Биржа: {api_pos_qty_abs}. Корректирую ордера...")
            # Обновляем локальный объем
            pos.current_quantity = float(api_pos_qty_abs)

            # Пересоздаем SL ордер с НОВЫМ объемом, используя replace_order_safe
            sl_price = pos.sl_price # Берем текущую цену SL из состояния
            if sl_price:
                 # Передаем функцию для создания нового SL ордера с актуальным объемом
                 await replace_order_safe(manager, "sl_order_id",
                                          lambda: manager._place_updated_sl(sl_price),
                                          "Reconcile after partial close/desync")
            else: # Если цены SL нет (ошибка?), просто отменяем старый ордер
                 await replace_order_safe(manager, "sl_order_id", lambda: asyncio.sleep(0), "Cancel SL on reconcile (no price)")

            # Отменяем старые TP ордера (т.к. они были на старый объем)
            await replace_order_safe(manager, "tp1_order_id", lambda: asyncio.sleep(0), "Cancel TP1 on reconcile")
            await replace_order_safe(manager, "near_tp_partial_oid", lambda: asyncio.sleep(0), "Cancel near_tp_partial on reconcile")
            await replace_order_safe(manager, "near_tp_guard_oid", lambda: asyncio.sleep(0), "Cancel near_tp_guard on reconcile")
            # Примечание: Новые TP ордера (если они реальные, а не виртуальные) нужно будет пересоздать здесь,
            # если ваша логика этого требует. В текущей реализации TP в основном виртуальные или управляются `manage_near_tp_band`.

            await save_state_async()
            logging.info(f"{log_prefix} Сверка завершена. Объем скорректирован до {pos.current_quantity}. SL ордер переустановлен (если был).")

        # 4. Проверка существования ордеров (если объемы совпали)
        else:
            logging.info(f"{log_prefix} Объемы совпадают ({local_qty_dec}). Проверка существования ордеров...")
            # Получаем ID активных ордеров с биржи
            active_api_oids = {str(o['orderId']) for o in open_orders_api if 'orderId' in o}
            missing_orders_keys = []

            # Проверяем все ID ордеров, хранящиеся в meta
            keys_to_check = [k for k, v in pos.meta.items() if (k.endswith("_order_id") or k.endswith("_oid")) and v is not None]
            for key in keys_to_check:
                oid = pos.meta.get(key)
                if oid and str(oid) not in active_api_oids:
                    missing_orders_keys.append(key)
                    logging.warning(f"{log_prefix} Ордер {key} (ID: {oid}) отсутствует на бирже!")
                    # Удаляем ID отсутствующего ордера из meta, чтобы не пытаться его отменить в будущем
                    pos.meta.pop(key, None)

            # Если пропал SL ордер - это критично, пытаемся восстановить
            if "sl_order_id" in missing_orders_keys:
                if pos.sl_price:
                    logging.warning(f"{log_prefix} КРИТИЧНО: SL ордер отсутствует! Попытка восстановления на {pos.sl_price:.5f}...")
                    # Используем replace_order_safe (он сначала проверит, нет ли уже ID в meta и отменит, если нужно)
                    await replace_order_safe(manager, "sl_order_id",
                                             lambda: manager._place_updated_sl(pos.sl_price),
                                             "Reconcile restore missing SL")
                else:
                    logging.error(f"{log_prefix} КРИТИЧНО: SL ордер отсутствует и нет цены для восстановления!")
                    # Возможно, стоит закрыть позицию аварийно?
                    # await close_position_emergency(symbol, pos.__dict__, "Missing SL order and price")

            # Если пропали другие ордера (TP, near-TP), просто удаляем их ID из meta (они пересоздадутся при необходимости)
            keys_to_pop_on_missing = ["tp1_order_id", "tp2_order_id", "near_tp_partial_oid", "near_tp_guard_oid"]
            for key in keys_to_pop_on_missing:
                if key in missing_orders_keys:
                    logging.warning(f"{log_prefix} Ордер {key} отсутствует, удаляю ID из состояния.")
                    # pos.meta уже был обновлен выше при проверке

            if missing_orders_keys:
                await save_state_async() # Сохраняем изменения в meta
            else:
                logging.info(f"{log_prefix} Все активные ордера найдены на бирже. Сверка завершена.")

    except BinanceAPIException as e:
        # Игнорируем ошибку "Нет открытых ордеров", если она возникает при запросе get_open_orders
        if e.code == -2011 and "futures_get_open_orders" in str(e):
             logging.info(f"{log_prefix} Нет открытых ордеров на бирже для сверки.")
        else:
             logging.error(f"{log_prefix} API Ошибка во время сверки: {e}")
    except Exception as e:
        logging.error(f"{log_prefix} Неизвестная ошибка во время сверки: {e}", exc_info=True)
# --- КОНЕЦ ШАГА 5 ---

async def manage_pnl_percentage_tp(pm, target_pnl_pct: float = 15.0, close_fraction: float = 0.5):
    """
    Проверяет PnL позиции в процентах и при достижении цели:
    1. Частично фиксирует прибыль (например, 50%).
    2. Переводит стоп-лосс в безубыток для оставшейся части.
    """
    pos = pm.state
    
    if pos.meta.get("pnl_tp_done", False):
        return False

    try:
        details = await pm.get_detailed_status()
        current_pnl_pct = details.get("pnl_percent", 0.0)

        if current_pnl_pct >= target_pnl_pct:
            logging.warning(
                f"💰 [{pm.symbol}] PnL достиг {current_pnl_pct:.2f}% (цель: {target_pnl_pct}%). "
                f"Фиксирую {close_fraction * 100:.0f}% позиции."
            )
            
            await pm._execute_partial_tp(fraction=close_fraction, reason=f"PnL TP @ {current_pnl_pct:.1f}%")
            
            pos.meta["pnl_tp_done"] = True
            
            # ✅ ИСПРАВЛЕНО: Гарантированная установка BE с обновлением флага
            if not pos.meta.get("be_done", False):
                df = market_data_store.get(pm.symbol, {}).get(ALLOWED_INTERVALS["5m"])
                if df is not None and not df.empty:
                    atr_value = _atr(df, period=14)
                    if atr_value > 0:
                        be_buffer = atr_value * 0.25 # Буфер для покрытия комиссий
                        be_sl_price = pos.entry_price + be_buffer if pos.side.upper() == "LONG" else pos.entry_price - be_buffer
                        await pm._update_sl_callback(be_sl_price, "BE after PnL TP")
                        pos.meta["be_done"] = True

            await save_state_async()
            return True # Возвращаем True, т.к. позиция была изменена

    except Exception as e:
        logging.error(f"[{pm.symbol}] Ошибка в manage_pnl_percentage_tp: {e}")
    
    return False

import numpy as np
import pandas as pd

# --- ADD THIS FUNCTION ---
async def _funding_rate_momentum(symbol: str, lookback: int = 6) -> Dict[str, Optional[float]]:
    """
    Calculates the recent change (momentum) in the funding rate.
    Uses hourly funding rate data. lookback=6 means comparing the latest rate to 6 hours ago.
    """
    try:
        # Fetch historical funding rates (adjust limit based on lookback)
        funding_history = await client.futures_funding_rate(symbol=symbol, limit=lookback + 2) # Fetch a bit more for safety
        if not funding_history or len(funding_history) < lookback + 1:
            logging.warning(f"[{symbol}] Insufficient funding rate history ({len(funding_history) if funding_history else 0}).")
            return {"funding_rate_mom": None, "funding_rate_latest": None}

        # Extract rates into a pandas Series for easier calculation
        rates = pd.Series([float(entry['fundingRate']) for entry in funding_history],
                          index=pd.to_datetime([entry['fundingTime'] for entry in funding_history], unit='ms', utc=True))
        rates = rates.sort_index() # Ensure chronological order

        if len(rates) < 2:
             return {"funding_rate_mom": None, "funding_rate_latest": float(rates.iloc[-1]) if len(rates)>0 else None}

        # Calculate momentum: (latest rate - rate N periods ago)
        # We multiply by 10000 to express it in basis points per lookback period for better scaling
        latest_rate = float(rates.iloc[-1])
        past_rate = float(rates.iloc[-lookback-1]) # Get the rate from 'lookback' periods ago
        momentum_bps = (latest_rate - past_rate) * 10000

        logging.info(f"[{symbol}] Funding Rate Momentum ({lookback}h): {momentum_bps:.2f} bps. Latest Rate: {latest_rate*100:.4f}%")
        return {"funding_rate_mom": momentum_bps, "funding_rate_latest": latest_rate}

    except BinanceAPIException as e:
        # Funding rate history might not be available for all symbols or might have rate limits
        logging.warning(f"[{symbol}] API Error fetching funding rate history: {e}")
        return {"funding_rate_mom": None, "funding_rate_latest": None}
    except Exception as e:
        logging.error(f"[{symbol}] Error calculating funding rate momentum: {e}", exc_info=False)
        return {"funding_rate_mom": None, "funding_rate_latest": None}

# --- ADD THIS FUNCTION ---
async def _realized_volatility(symbol: str, tf_key: str = '1h', short_window: int = 24, long_window: int = 168) -> Dict[str, Optional[float]]:
    """
    Calculates short-term vs long-term realized volatility ratio.
    Uses log returns. short_window=24 (1 day), long_window=168 (1 week) on hourly data.
    """
    df = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS.get(tf_key))
    # Need enough data for the long window + 1 for diff()
    if df is None or len(df) < long_window + 1:
        logging.warning(f"[{symbol}] Insufficient data ({len(df) if df is not None else 0}) for Realized Volatility on {tf_key}.")
        return {"vol_ratio_s_l": None, "vol_short_ann": None}

    try:
        # Calculate log returns
        log_returns = np.log(df['close'] / df['close'].shift(1)).dropna()

        if len(log_returns) < long_window:
             return {"vol_ratio_s_l": None, "vol_short_ann": None}

        # Calculate rolling standard deviation (volatility) of log returns
        rolling_std = log_returns.rolling(window=short_window).std()

        # Short-term volatility (last window)
        vol_short = rolling_std.iloc[-1]

        # Long-term volatility (average over the long window, excluding the most recent short window)
        # Ensure we don't index out of bounds
        long_term_start_index = max(0, len(rolling_std) - long_window)
        long_term_end_index = max(0, len(rolling_std) - short_window)
        
        # Check if the range is valid
        if long_term_start_index >= long_term_end_index:
             vol_long = vol_short # Fallback if range is too small
        else:
             vol_long = rolling_std.iloc[long_term_start_index:long_term_end_index].mean()


        # Avoid division by zero
        if vol_long is None or pd.isna(vol_long) or vol_long == 0:
            vol_ratio = 1.0 # Neutral if long-term vol is zero or NaN
        else:
            vol_ratio = vol_short / vol_long

        # Annualized short-term volatility (assuming hourly data -> sqrt(24*365))
        annualized_factor = np.sqrt(24 * 365)
        vol_short_ann = vol_short * annualized_factor * 100 # In percent

        logging.info(f"[{symbol}] Realized Volatility ({tf_key}): Ratio (S/L)={vol_ratio:.2f}, Short Ann={vol_short_ann:.1f}%")
        return {"vol_ratio_s_l": vol_ratio, "vol_short_ann": vol_short_ann}

    except Exception as e:
        logging.error(f"[{symbol}] Error calculating realized volatility: {e}", exc_info=False)
        return {"vol_ratio_s_l": None, "vol_short_ann": None}

# --- ШАГ 4: Обновленная функция check_and_execute_scaling_v2 с проверкой RR и Total Risk ---
async def check_and_execute_scaling_v2(symbol: str):
    """
    MODIFIED (v2.1): Incorporates RR check, Max Steps check,
    and ensures total initial risk is preserved (using _recalc_avg_and_sl_preserve_total_risk logic).
    Uses smart_entry for adding.
    """
    log_prefix = f"⚖️ [{symbol}]"
    if symbol not in current_positions: return False
    pm = current_positions.get(symbol)
    if not isinstance(pm, PositionManager): return False

    df5m = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS.get('5m'))
    if df5m is None or df5m.empty: return False

    pos = pm.state
    plan = pos.meta.get("scaling_plan", {})
    if not isinstance(plan, dict): return False

    next_price_str = plan.get("next_add_price")
    qty_fracs = plan.get("qty_fractions") or []
    adds_done = plan.get("adds_done", 0)
    max_adds = plan.get("max_adds", PYRAMID_MAX_STEPS) # Используем константу

    # --- УСИЛЕННЫЕ ПРОВЕРКИ ПЕРЕД ДОБАВЛЕНИЕМ ---
    current_price = float(df5m["close"].iloc[-1])
    # 1. Проверка RR от НАЧАЛЬНОГО стопа
    if pos.initial_sl_price is None: return False # Не можем добавить без начального SL
    current_rr = rr(pos.side, pos.entry_price, pos.initial_sl_price, current_price)
    if current_rr < PYRAMID_MIN_RR:
        logging.info(f"{log_prefix} Добавление пропущено: RR={current_rr:.2f} < {PYRAMID_MIN_RR}")
        return False
    # 2. Проверка максимального количества шагов
    if adds_done >= max_adds:
        # logging.info(f"{log_prefix} Достигнут лимит добавлений ({adds_done}/{max_adds}).") # Можно раскомментировать
        return False
    # 3. Проверка триггерной цены
    if not next_price_str: return False
    try: next_price = float(next_price_str)
    except (ValueError, TypeError): return False
    is_long = pos.side.upper() == "LONG"
    triggered = (is_long and current_price <= next_price) or (not is_long and current_price >= next_price)
    if not triggered: return False
    # 4. Проверка микроструктуры (пример)
    try:
        ofi_score, _ = await calculate_order_flow_imbalance(symbol)
        if is_long and ofi_score < PYRAMID_OBI_GATE:
            logging.info(f"{log_prefix} Добавление пропущено: OBI {ofi_score:.1f} < {PYRAMID_OBI_GATE}")
            return False
        if not is_long and ofi_score > -PYRAMID_OBI_GATE:
            logging.info(f"{log_prefix} Добавление пропущено: OBI {ofi_score:.1f} > {-PYRAMID_OBI_GATE}")
            return False
    except Exception: pass # Пропускаем, если OBI недоступен
    # --- КОНЕЦ УСИЛЕННЫХ ПРОВЕРОК ---

    logging.warning(f"{log_prefix} DCA/PYRAMID TRIGGERED! Price={current_price:.5f} reached Target={next_price:.5f}")

    if adds_done < len(qty_fracs):
        try: add_frac = float(qty_fracs[adds_done])
        except (ValueError, TypeError, IndexError): add_frac = 0.0
    else: add_frac = 0.0

    if pos.initial_quantity is None or pos.initial_quantity <= 0: return False
    add_qty_raw = pos.initial_quantity * add_frac
    add_qty_str = format_quantity(add_qty_raw, symbol)
    add_qty = float(add_qty_str) if add_qty_str else 0.0

    if add_qty <= 0:
        # Продвигаем план, даже если объем 0, чтобы не застрять
        plan["adds_done"] = adds_done + 1
        levels = plan.get("levels") or []
        plan["next_add_price"] = levels[plan["adds_done"]] if plan["adds_done"] < len(levels) else None
        pos.meta["scaling_plan"] = plan
        await save_state_async()
        return False

    # --- ПРОВЕРКА СОХРАНЕНИЯ РИСКА ---
    # Псевдо-функция для расчета нового SL, сохраняющего суммарный риск
    def _recalc_avg_and_sl_preserve_total_risk(side, old_qty, old_avg, old_initial_sl, add_qty, add_price):
        if old_qty <= 0 or add_qty <= 0: return None, None
        new_total_qty = old_qty + add_qty
        new_avg_price = (old_avg * old_qty + add_price * add_qty) / new_total_qty
        initial_risk_per_unit_at_entry = abs(old_avg - old_initial_sl)
        total_initial_risk_usd = old_qty * initial_risk_per_unit_at_entry
        new_risk_per_unit_allowed = total_initial_risk_usd / new_total_qty
        new_sl_price = new_avg_price - new_risk_per_unit_allowed if side.upper() == "LONG" else new_avg_price + new_risk_per_unit_allowed
        return new_avg_price, new_sl_price

    new_avg_price_calc, new_sl_price_calc = _recalc_avg_and_sl_preserve_total_risk(
        side=pos.side, old_qty=pos.current_quantity, old_avg=pos.entry_price,
        old_initial_sl=pos.initial_sl_price, add_qty=add_qty, add_price=current_price
    )
    if new_avg_price_calc is None or new_sl_price_calc is None:
        logging.error(f"{log_prefix} Ошибка расчета нового SL для сохранения риска.")
        return False
    # --- КОНЕЦ ПРОВЕРКИ РИСКА ---

    logging.info(f"{log_prefix} Попытка добавить {add_qty:.6f} (Frac: {add_frac:.2f})")
    add_side = "BUY" if is_long else "SELL"

    # Используем place_order_async со smart_entry
    add_order = await place_order_async(
        symbol, add_side, "MARKET", add_qty,
        use_smart_entry=True, slip_threshold_bps=SLIP_BP, limit_off_bps=LIMIT_OFFSET_BPS
    )

    if not add_order or not add_order.get("orderId"):
        logging.error(f"{log_prefix} Ордер на добавление не прошел.")
        return False

    logging.info(f"{log_prefix} Ордер на добавление размещен: ID {add_order.get('orderId')}")

    # Ожидаем цену исполнения
    fill_price = await await_filled_avg_price(symbol, add_order.get("orderId"), current_price)
    if fill_price <= 0:
         logging.error(f"{log_prefix} Не удалось получить цену исполнения. Состояние не обновлено.")
         return False

    async with state_lock: # Атомарное обновление состояния
        current_pm = current_positions.get(symbol)
        if not isinstance(current_pm, PositionManager) or current_pm.state != pos:
             logging.error(f"{log_prefix} Состояние позиции изменилось во время добавления. Отмена.")
             return False

        old_avg, old_qty = pos.entry_price, pos.current_quantity
        new_total_qty = old_qty + add_qty
        if new_total_qty <= 1e-12: return False

        # Используем ФАКТИЧЕСКУЮ цену исполнения для пересчета средней
        actual_new_avg_price = (old_avg * old_qty + fill_price * add_qty) / new_total_qty
        # Пересчитываем SL еще раз с ФАКТИЧЕСКОЙ средней ценой
        _, actual_new_sl_price = _recalc_avg_and_sl_preserve_total_risk(
            side=pos.side, old_qty=old_qty, old_avg=old_avg,
            old_initial_sl=pos.initial_sl_price, add_qty=add_qty, add_price=fill_price # Используем fill_price
        )
        if actual_new_sl_price is None:
             logging.error(f"{log_prefix} Критическая ошибка: Не удалось пересчитать SL после получения цены исполнения.")
             # Возможно, стоит аварийно закрыть позицию здесь?
             return False

        logging.info(f"{log_prefix} Обновление состояния: Qty {old_qty:.6f}->{new_total_qty:.6f}, Avg {old_avg:.5f}->{actual_new_avg_price:.5f}")
        pos.current_quantity = new_total_qty
        pos.entry_price = actual_new_avg_price # Сохраняем фактическую среднюю

        # Обновляем SL через replace_order_safe
        logging.info(f"{log_prefix} Перестановка SL на {actual_new_sl_price:.5f} для сохранения риска.")
        await replace_order_safe(pm, "sl_order_id",
                                 lambda: pm._place_updated_sl(actual_new_sl_price),
                                 f"DCA/Pyr #{adds_done + 1}")
        pos.sl_price = actual_new_sl_price # Обновляем цену SL в состоянии

        # Продвигаем план
        plan["adds_done"] = adds_done + 1
        levels = plan.get("levels") or []
        next_add_idx = plan["adds_done"]
        plan["next_add_price"] = levels[next_add_idx] if next_add_idx < len(levels) else None
        pos.meta["scaling_plan"] = plan

        await save_state_async()

    logging.warning(f"✅ {log_prefix} Шаг {adds_done + 1}/{max_adds} исполнен.")
    await send_telegram_message(
        f"⚖️ **{symbol} DCA/Pyr #{adds_done + 1}**\n"
        f"   Добавлено: {add_qty:.4f} @ ${fill_price:.4f}\n"
        f"   Новая ср: ${pos.entry_price:.4f}\n"
        f"   Новый SL: ${pos.sl_price:.4f}"
    )
    return True
# --- КОНЕЦ ШАГА 4 ---

# --- Вспомогательные функции для сбора признаков ---


def _safe_df(symbol: str, tf_key: str, need: int = 120) -> Optional[pd.DataFrame]:
    df = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS.get(tf_key))
    return df if df is not None and len(df) >= need else None

def _slope(series: pd.Series, k: int = 5) -> float:
    if len(series) < k + 1: return 0.0
    y = series.iloc[-(k + 1):].values.astype(float)
    x = np.arange(len(y))
    A = np.vstack([x, np.ones_like(x)]).T
    m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(m)

def _ema_features(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> Dict[str, float]:
    ef = ta.trend.ema_indicator(df["close"], fast)
    es = ta.trend.ema_indicator(df["close"], slow)
    efv, esv = float(ef.iloc[-1]), float(es.iloc[-1])
    dist = (efv - esv) / max(esv, 1e-9) * 100.0
    return {"ema_dist_pct": dist, "ema_slope": _slope(ef, 5)}

def _mtf_core_features(symbol: str, tfs: List[str] = ["15m", "1h", "4h"]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for tf in tfs:
        df = _safe_df(symbol, tf, need=max(60, 3 * 50))
        if df is None: continue
        out[tf] = {
            **_ema_features(df),
            "adx": float(ta.trend.adx(df["high"], df["low"], df["close"], 14).iloc[-1]),
            "natr": float(ta.volatility.average_true_range(df["high"],df["low"],df["close"],14).iloc[-1] / max(float(df["close"].iloc[-1]), 1e-9) * 100.0),
        }
    return out

async def replace_order_safe(pm: PositionManager, old_key: str, create_fn, reason: str):
    """
    Безопасно заменяет ордер: сначала отменяет старый (если есть), затем ставит новый.
    Args:
        pm: Экземпляр PositionManager.
        old_key: Ключ в pos.meta для хранения orderId (напр., 'sl_order_id').
        create_fn: Асинхронная функция (lambda или partial), которая создает и возвращает НОВЫЙ ордер.
        reason: Причина замены для логов.
    """
    symbol = pm.symbol
    pos = pm.state
    old_oid = pos.meta.get(old_key)

    if old_oid:
        logging.info(f"[{symbol}] Отмена старого ордера ({old_key}: {old_oid}) перед заменой ({reason})...")
        await cancel_single_order_async(symbol, old_oid)
        pos.meta.pop(old_key, None) # Удаляем старый ID из метаданных

    logging.info(f"[{symbol}] Попытка установки нового ордера ({reason})...")
    new_order = await create_fn() # Выполняем функцию создания нового ордера

    if new_order and new_order.get("orderId"):
        pos.meta[old_key] = new_order["orderId"] # Сохраняем ID нового ордера
        logging.info(f"[{symbol}] Новый ордер ({old_key}) успешно установлен: {new_order['orderId']}")
        await save_state_async() # Сохраняем обновленное состояние
        return True
    else:
        logging.error(f"[{symbol}] НЕ УДАЛОСЬ установить новый ордер ({old_key}) после отмены старого! ({reason})")
        return False

async def _micro_features(symbol: str) -> Dict[str, Optional[float]]:
    ofi, spread_bps, depth_usd = None, None, None
    try: ofi, _ = await calculate_order_flow_imbalance(symbol)
    except Exception: pass
    try:
        m = await get_market_microstructure_features(symbol, client)
        if m:
            if m.get("bid_ask_spread_pct") is not None: spread_bps = float(m["bid_ask_spread_pct"]) * 100.0
            if m.get("market_depth_usd") is not None: depth_usd = float(m["market_depth_usd"])
    except Exception: pass
    return {"ofi": ofi, "bidask_bps": spread_bps, "depth_usd": depth_usd}

async def _derivatives_features(symbol: str) -> Dict[str, Optional[float]]:
    try:
        data = await get_basis_and_funding(symbol)
        basis = float(data["basis_bps"]) if data and data.get("basis_bps") is not None else None
        funding = float(data["funding_pct"]) if data and data.get("funding_pct") is not None else None
        return {"basis_bps": basis, "funding_pct": funding}
    except Exception:
        return {"basis_bps": None, "funding_pct": None}

def _obv_features(df: pd.DataFrame) -> Dict[str, float]:
    try:
        obv = ta.volume.on_balance_volume(df["close"], df["volume"])
        body = (df["close"] - df["open"]).astype(float)
        bodyv = (body * df["volume"]).rolling(20).mean()
        return {"obv_slope": _slope(obv, 10), "bodyv_slope": _slope(bodyv, 10)}
    except Exception:
        return {"obv_slope": 0.0, "bodyv_slope": 0.0}

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

# --- Основной класс агрегатора с исправлениями ---


@dataclass
class EnsembleWeights:
    w_ema_dist_15m: float = 0.8; w_ema_slope_15m: float = 0.4; w_adx_15m: float = 0.05
    w_ema_dist_1h: float = 1.2; w_ema_slope_1h: float = 0.6; w_adx_1h: float = 0.08
    w_ema_dist_4h: float = 1.5; w_ema_slope_4h: float = 0.7; w_adx_4h: float = 0.10
    w_natr_15m: float = -0.05; w_natr_1h: float = -0.04; w_natr_4h: float = -0.03
    w_ofi: float = 0.02; w_spread_bps: float = -0.02; w_depth_usd: float = 0.01
    w_basis_bps: float = 0.01
    # -- Adjusted funding weight ---
    w_funding_pct_latest: float = 0.20 # Reduced weight for static funding
    # --- New Weights ---
    w_funding_mom: float = 0.35      # Weight for funding momentum (significant)
    w_price_oi_divergence: float = 0.50  # Это сильный сигнал, даем ему хороший вес
    w_vol_ratio: float = -0.10       # Negative weight: high ratio (expansion) slightly bearish, low ratio (squeeze) slightly bullish
    # --- End New Weights ---
    w_obv_slope: float = 0.02; w_bodyv_slope: float = 0.02
    bias: float = 0.0

@dataclass
class HysteresisConfig:
    p_high: float = 0.62; p_low: float = 0.38
    hold_base_sec: int = 240; hold_max_sec: int = 900

@dataclass
class BTCProbDirection:
    symbol: str = "BTCUSDT"
    weights: EnsembleWeights = field(default_factory=EnsembleWeights)
    hyst: HysteresisConfig = field(default_factory=HysteresisConfig)
    last_state: str = "SIDEWAYS"
    last_switch_ts: float = 0.0
    last_p_up: float = 0.5  # Для EMA-сглаживания

    def _adapt_weights_by_regime(self, mtf: Dict[str, Any]) -> EnsembleWeights:
        # Create a fresh copy of the base weights
        w = EnsembleWeights(**vars(self.weights)) # Use vars() for dataclass copy

        # Get key indicators (handle potential missing data)
        adx1h = mtf.get("1h", {}).get("adx", 20.0)
        adx4h = mtf.get("4h", {}).get("adx", 20.0)
        # Assuming vol_data is calculated and available - we need it here
        # For simplicity now, let's use NATR as a proxy for volatility adaptation
        natr1h = mtf.get("1h", {}).get("natr", 1.0) # Using 1h NATR

        # --- Trend Strength Adaptation ---
        if adx1h >= 28.0 and adx4h >= 25.0: # Strong, confirmed trend
             logging.info("[BTC Filter] Adapting weights for STRONG TREND")
             w.w_ema_dist_4h *= 1.25 # Emphasize long-term EMA
             w.w_ema_slope_4h *= 1.15
             w.w_ema_dist_1h *= 1.15
             w.w_adx_1h *= 1.10
             w.w_funding_mom *= 1.10 # Funding momentum more important in trends
             w.w_vol_ratio *= 0.8 # Volatility ratio less important
        elif adx1h <= 18.0 and adx4h <= 20.0: # Clear sideways / weak trend
             logging.info("[BTC Filter] Adapting weights for SIDEWAYS")
             w.w_ema_dist_4h *= 0.8
             w.w_ema_slope_4h *= 0.8
             w.w_ema_dist_1h *= 0.9
             w.w_adx_1h *= 0.7
             w.w_ofi *= 1.15         # Microstructure more important
             w.w_spread_bps *= 1.15  # Avoid wide spreads
             w.w_vol_ratio *= 1.2    # Volatility squeeze/expansion becomes more relevant

        # --- Volatility Adaptation (using NATR proxy) ---
        if natr1h >= 2.0: # High volatility
             logging.info("[BTC Filter] Adapting weights for HIGH VOLATILITY")
             w.w_natr_1h *= 1.2          # Penalize high NATR more
             w.w_spread_bps *= 1.2       # Penalize wide spreads more
             w.w_funding_mom *= 0.9      # Slightly reduce impact of funding mom in chaos
        elif natr1h < 0.8: # Low volatility
             logging.info("[BTC Filter] Adapting weights for LOW VOLATILITY")
             w.w_vol_ratio *= 1.25       # Squeeze/expansion signal is stronger
             w.w_ofi *= 1.10             # OFI might be clearer

        return w # Return the modified weights object

    def _dynamic_thresholds(self, mtf: Dict[str, Any]) -> Tuple[float, float, int]:
        adx1h = mtf.get("1h", {}).get("adx", 20.0)
        natr1h = mtf.get("1h", {}).get("natr", 1.0)
        ph, pl = self.hyst.p_high, self.hyst.p_low
        if adx1h >= 28.0: ph -= 0.03; pl += 0.03
        elif adx1h <= 18.0: ph += 0.03; pl -= 0.03
        # ✅ ИСПРАВЛЕНИЕ: Жесткий клиппинг порогов в допустимых границах
        ph = min(0.75, max(0.55, ph))
        pl = max(0.25, min(0.45, pl))
        hold = int(min(self.hyst.hold_max_sec, self.hyst.hold_base_sec * (1.0 + natr1h / 2.5)))
        return ph, pl, hold

    def _linear_score(self, mtf, micro, deriv, obv, funding_mom_data, vol_data, oi_data, weights: EnsembleWeights) -> float:
        # ✅ Добавлены funding_mom_data, vol_data, oi_data
        w, s = weights, weights.bias
        def g(tf: str, k: str): return mtf.get(tf, {}).get(k, 0.0)
        
        # --- Существующие расчеты ---
        s += w.w_ema_dist_15m*g("15m","ema_dist_pct") + w.w_ema_slope_15m*g("15m","ema_slope") + w.w_adx_15m*g("15m","adx")
        s += w.w_ema_dist_1h*g("1h","ema_dist_pct") + w.w_ema_slope_1h*g("1h","ema_slope") + w.w_adx_1h*g("1h","adx")
        s += w.w_ema_dist_4h*g("4h","ema_dist_pct") + w.w_ema_slope_4h*g("4h","ema_slope") + w.w_adx_4h*g("4h","adx")
        s += w.w_natr_15m*g("15m","natr") + w.w_natr_1h*g("1h","natr") + w.w_natr_4h*g("4h","natr")
        
        if (ofi := micro.get("ofi")): s += w.w_ofi * ofi
        if (spr := micro.get("bidask_bps")): s += w.w_spread_bps * spr
        if (depth := micro.get("depth_usd")): s += w.w_depth_usd * min(depth / 1_000_000.0, 2.0)
        if (basis := deriv.get("basis_bps")): s += w.w_basis_bps * basis
        
        # Используем последнюю ставку фандинга
        if (funding_latest := funding_mom_data.get("funding_rate_latest")):
             s += w.w_funding_pct_latest * (funding_latest * 100) # Используем %
             
        # --- Интеграция Новых Признаков ---
        
        # Моментум Фандинга
        if (funding_mom := funding_mom_data.get("funding_rate_mom")):
             s += w.w_funding_mom * funding_mom
             
        # Соотношение Волатильности
        if (vol_ratio := vol_data.get("vol_ratio_s_l")):
             s += w.w_vol_ratio * np.clip(vol_ratio - 1.0, -2.0, 2.0)
             
        # --- ✅ НОВАЯ ИНТЕГРАЦИЯ ОИ ---
        if (divergence_score := oi_data.get("price_oi_divergence")) is not None:
             # divergence_score = 1 (Цена и ОИ движутся вместе - хорошо, балл растет)
             # divergence_score = -1 (Цена и ОИ движутся врозь - плохо, балл падает)
             s += w.w_price_oi_divergence * divergence_score
        # --- КОНЕЦ НОВОЙ ИНТЕГРАЦИИ ОИ ---
        
        # --- Расчет OBV ---
        if (obv_s := obv.get("obv_slope")): s += w.w_obv_slope * obv_s
        if (bodyv_s := obv.get("bodyv_slope")): s += w.w_bodyv_slope * bodyv_s
        return float(s)

    async def compute(self) -> Dict[str, Any]:
        # --- Profiling code remains here if you added it ---
        # pr = cProfile.Profile(); pr.enable()

        mtf = _mtf_core_features(self.symbol)
        if not mtf:
            # --- Profiling stop code remains here ---
            # pr.disable(); ... logging.info(...)
            return {"p_up": self.last_p_up, "state": self.last_state, "details": {}}

        w = self._adapt_weights_by_regime(mtf) # Keep adaptive weights

        df1h = _safe_df(self.symbol, "1h", 120)
        obv_feats = _obv_features(df1h) if df1h is not None else {}

        # --- ✅ ИСПРАВЛЕНИЕ: Добавлен oi_task ---
        # --- Call all feature functions concurrently ---
        micro_task = _micro_features(self.symbol)
        deriv_task = _derivatives_features(self.symbol)
        funding_mom_task = _funding_rate_momentum(self.symbol) # Call new function
        vol_task = _realized_volatility(self.symbol, tf_key='1h') # Call new function
        oi_task = _oi_features(self.symbol, tf_key='1h') # ✅ Вызываем новую функцию ОИ

        micro, deriv, funding_mom_data, vol_data, oi_data = await asyncio.gather(
            micro_task, deriv_task, funding_mom_task, vol_task, oi_task
        )
        # --- End concurrent calls ---

        # --- ✅ ИСПРАВЛЕНИЕ: Добавлен oi_data в вызов ---
        # Pass new data to _linear_score
        z = self._linear_score(mtf, micro, deriv, obv_feats, funding_mom_data, vol_data, oi_data, w)
        p_raw = _sigmoid(z / 15.0) # Adjust divisor if needed based on new score range

        # --- Rest of the compute method (EMA smoothing, hysteresis) remains the same ---
        adx1h = mtf.get("1h", {}).get("adx", 20.0)
        alpha = 0.45 if adx1h < 20.0 else (0.60 if adx1h < 28.0 else 0.75)
        p_up = alpha * p_raw + (1.0 - alpha) * self.last_p_up
        self.last_p_up = p_up

        p_h, p_l, hold = self._dynamic_thresholds(mtf)
        now = time.time()
        can_switch = (now - self.last_switch_ts) >= hold
        state = self.last_state

        if state in ("SIDEWAYS", "DOWN") and p_up >= p_h and can_switch:
            state = "UP"; self.last_switch_ts = now
        elif state in ("SIDEWAYS", "UP") and p_up <= p_l and can_switch:
            state = "DOWN"; self.last_switch_ts = now

        self.last_state = state
        
        # --- ✅ ИСПРАВЛЕНИЕ: Добавлен 'oi_divergence' в details ---
        result = {"p_up": p_up, "state": state, "details": {"z": z, "p_raw": p_raw, "p_high": p_h, "p_low": p_l,
                   # Add new details for logging/debugging
                   "funding_mom": funding_mom_data.get("funding_rate_mom"),
                   "vol_ratio": vol_data.get("vol_ratio_s_l"),
                   "oi_divergence": oi_data.get("price_oi_divergence") # ✅ Добавляем новый ключ
                   }}

        # --- Profiling stop code remains here ---
        # pr.disable(); ... logging.info(...)
        return result

# --- Глобальный экземпляр и функция-гейт ---
BTC_DIRECTION = BTCProbDirection(symbol="BTCUSDT")

async def global_direction_verdict(trade_type: str) -> Tuple[bool, str, Dict[str, Any]]:
    snap = await BTC_DIRECTION.compute()
    state, p_up = snap["state"], snap["p_up"]
    
    if trade_type.lower() == "long":
        ok = (state == "UP")
        reason = f"BTC state={state}, p_up={p_up:.2f}"
        return ok, reason, snap
    else: # short
        ok = (state == "DOWN")
        reason = f"BTC state={state}, p_up={p_up:.2f}"
        return ok, reason, snap

async def estimate_market_liquidity(symbol: str) -> Tuple[float, float]:
    """
    Продвинутая оценка ликвидности (0-10) И Спреда (%).
    Возвращает (final_score, spread_percent)
    """
    try:
        # Используем один вызов для получения последней цены, которая нужна для расчета глубины в USDT
        ticker, depth = await asyncio.gather(
            client.futures_ticker(symbol=symbol),
            client.futures_order_book(symbol=symbol, limit=20)
        )
        
        last_price = float(ticker['lastPrice'])
        volume_24h_usdt = float(ticker['quoteVolume'])

        bid_depth_usdt = sum(float(b[1]) * last_price for b in depth['bids'][:10])
        ask_depth_usdt = sum(float(a[1]) * last_price for a in depth['asks'][:10])
        total_depth_usdt = bid_depth_usdt + ask_depth_usdt

        best_bid = float(depth['bids'][0][0])
        best_ask = float(depth['asks'][0][0])
        spread_percent = ((best_ask - best_bid) / best_bid) * 100 if best_bid > 0 else float('inf')

        # Комбинированная оценка (0-10)
        volume_score = min(10, (volume_24h_usdt / 10_000_000) * 5) # 10M USDT = 5 баллов
        depth_score = min(10, (total_depth_usdt / 250_000) * 5)    # 250K USDT в стакане = 5 баллов
        spread_score = 10 if spread_percent < 0.04 else (5 if spread_percent < 0.08 else 0)
        
        final_score = (volume_score * 0.4) + (depth_score * 0.4) + (spread_score * 0.2)
        logging.info(f"[{symbol}] Оценка ликвидности: {final_score:.1f}/10 (Объем: {volume_24h_usdt:,.0f}, Глубина: {total_depth_usdt:,.0f}, Спред: {spread_percent:.4f}%)")
        
        # --- ИЗМЕНЕНИЕ: Возвращаем кортеж (score, spread) ---
        return final_score, spread_percent
        
    except Exception as e:
        logging.error(f"[{symbol}] Ошибка оценки ликвидности: {e}")
        
        # --- ИЗМЕНЕНИЕ: Возвращаем кортеж с плохими значениями ---
        return 0.0, 999.0 # В случае ошибки считаем ликвидность 0 и спред огромным

async def get_correlation_snapshot(symbol1: str, symbol2: str, period: int = 50) -> float:
    """Рассчитывает корреляцию Пирсона между двумя активами."""
    try:
        df1 = market_data_store.get(symbol1, {}).get(ALLOWED_INTERVALS['15m'])
        df2 = market_data_store.get(symbol2, {}).get(ALLOWED_INTERVALS['15m'])
        if df1 is None or df2 is None or len(df1) < period or len(df2) < period:
            return 0.0

        # Используем процентное изменение для расчета корреляции
        returns1 = df1['close'].tail(period).pct_change().dropna()
        returns2 = df2['close'].tail(period).pct_change().dropna()
        
        correlation = returns1.corr(returns2)
        return correlation if pd.notna(correlation) else 0.0
    except Exception as e:
        logging.error(f"Ошибка расчета корреляции между {symbol1} и {symbol2}: {e}")
        return 0.0

async def calculate_adaptive_risk_by_volatility(base_risk: float, symbol: str) -> float:
    """Корректирует риск на основе волатильности (0.6x - 1.3x)."""
    try:
        df = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS['1h'])
        if df is None or len(df) < 120: return base_risk
        
        returns = df['close'].pct_change()
        current_vol = returns.iloc[-24:].std()     # Волатильность за последние сутки
        historical_vol = returns.iloc[-120:-24].std() # Волатильность за предыдущие 4 дня

        if historical_vol == 0 or current_vol == 0: return base_risk
            
        vol_ratio = current_vol / historical_vol
        
        # Адаптивный множитель
        if vol_ratio > 1.8:    # Очень высокая волатильность
            multiplier = 0.6
            logging.warning(f"[{symbol}] Волатильность ВЫСОКАЯ (x{vol_ratio:.2f}). Риск снижен x{multiplier}")
        elif vol_ratio < 0.6:  # Очень низкая волатильность
            multiplier = 1.3
            logging.info(f"[{symbol}] Волатильность НИЗКАЯ (x{vol_ratio:.2f}). Риск повышен x{multiplier}")
        else:
            multiplier = 1.0
            
        return base_risk * multiplier
    except Exception as e:
        logging.error(f"[{symbol}] Ошибка расчета адаптивного риска: {e}")
        return base_risk

async def check_and_execute_scaling(symbol: str):
    """
    Проверяет и выполняет докупку (DCA) с полной диагностикой, логикой временной паузы и форматированием объема.
    """
    log_prefix = f"⚖️ [{symbol}]"

    # --- 1. Базовые проверки с детальным логированием ---
    if symbol not in current_positions: return
    pos_manager = current_positions[symbol]
    if not isinstance(pos_manager, PositionManager): return
    pos_state = pos_manager.state
    
    scaling_plan = pos_state.meta.get('scaling_plan')
    if not scaling_plan:
        logging.info(f"{log_prefix} DCA пропущен: план докупок не найден.")
        return

    paused_until = scaling_plan.get('paused_until')
    if paused_until and time.time() < paused_until:
        logging.info(f"{log_prefix} DCA на паузе до {datetime.datetime.fromtimestamp(paused_until).strftime('%H:%M:%S')}.")
        return
        
    adds_done = scaling_plan.get('adds_done', 0)
    max_adds = scaling_plan.get('max_adds', 0)
    if adds_done >= max_adds:
        return

    # --- 2. Проверка триггера по цене с логированием ---
    next_add_price = scaling_plan.get('next_add_price')
    df_short = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS['5m'])
    if not next_add_price or df_short is None or df_short.empty:
        logging.info(f"{log_prefix} DCA пропущен: нет цены для следующей докупки или отсутствуют данные 5m.")
        return

    current_price = df_short['close'].iloc[-1]
    side = pos_state.side
    triggered = (side == 'long' and current_price <= next_add_price) or \
                (side == 'short' and current_price >= next_add_price)
    if not triggered:
        logging.info(f"{log_prefix} Ожидание уровня DCA: Цена={current_price:.5f}, Цель={next_add_price:.5f}.")
        return

    logging.warning(f"{log_prefix} ЦЕНА ДОСТИГЛА УРОВНЯ ДОКУПКИ ({next_add_price:.5f}). Запуск проверок...")

    try:
        # --- 3. Фильтры (BTC и ликвидность) ---
        btc_ok, btc_reason = await btc_filter_for_ranging_market(symbol, side)
        if not btc_ok:
            raise ValueError(f"Отклонено фильтром BTC: {btc_reason}")

        liquid_ok, liq_reason = await enforce_liquidity_ban(symbol, 'SCALING')
        if not liquid_ok:
            raise ValueError(f"Отклонено фильтром ликвидности: {liq_reason}")
            
        # --- 4. Одобрение от AI с логикой паузы ---
        market_env = await detect_market_environment(symbol)

        # <<< ИЗМЕНЕНИЕ: Вызов новой гибридной функции анализа >>>
        ai_verdict = await enhanced_getdeepseekverdict_with_vision(symbol, side, market_env)
        # <<< КОНЕЦ ИЗМЕНЕНИЯ >>>

        ai_confidence = ai_verdict.get("confidence", 0)
        
        if ai_confidence < DCA_AI_CONFIDENCE_THRESHOLD:
            pause_timestamp = (datetime.datetime.now(dt_timezone.utc) + datetime.timedelta(minutes=DCA_COOLDOWN_MINUTES)).timestamp()
            pos_state.meta['scaling_plan']['paused_until'] = pause_timestamp
            await save_state_async()
            raise ValueError(f"Низкая уверенность AI ({ai_confidence}%), DCA на паузе на {DCA_COOLDOWN_MINUTES} мин.")
        
        if 'paused_until' in pos_state.meta['scaling_plan']:
            del pos_state.meta['scaling_plan']['paused_until']

        logging.warning(f"{log_prefix} ✅ ДОКУПКА ОДОБРЕНА AI ({ai_confidence}%). Исполнение...")

        # --- 5. Расчет и исполнение ордера с форматированием ---
        add_qty_fraction = scaling_plan.get('qty_fractions', [])[adds_done]
        add_qty_raw = pos_state.initial_quantity * add_qty_fraction
        
        add_qty_formatted = format_quantity(add_qty_raw, symbol)
        if not add_qty_formatted or float(add_qty_formatted) <= 0:
            raise ValueError("Расчетный объем для докупки равен нулю после форматирования.")
        
        side_str = 'BUY' if side == 'long' else 'SELL'
        add_order_info = await place_order_async(symbol=symbol, side=side_str, order_type="MARKET", quantity=float(add_qty_formatted))
        
        if not add_order_info or not add_order_info.get('orderId'):
            raise ValueError("Ошибка ордера на докупку.")

        await update_position_after_scaling(pos_manager, add_order_info, float(add_qty_formatted))

    except ValueError as e:
        logging.warning(f"{log_prefix} Докупка отложена. Причина: {e}")
        return

import uuid

@dataclass
class TradePassport:
    """Хранит полную историю жизни и принятия решения по одному сигналу."""
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.datetime.now(dt_timezone.utc).isoformat())
    symbol: str = "N/A"
    strategy: str = "N/A"
    trigger: str = "N/A"
    side: str = "N/A"
    
    # Результаты проверок
    btc_filter_passed: Optional[bool] = None
    btc_filter_reason: Optional[str] = None
    ai_confidence: Optional[int] = None
    required_threshold: Optional[float] = None
    
    # Итоговый результат
    status: str = "PENDING" # PENDING -> REJECTED | EXECUTED
    reject_reason: Optional[str] = None
    
    # Данные по сделке (если исполнена)
    entry_price: Optional[float] = None
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None

async def generate_entry_chart_async(passport: TradePassport, df: pd.DataFrame) -> Optional[io.BytesIO]:
    """Генерирует и возвращает график в виде байтового буфера."""
    if not mpf or df is None or len(df) < 100:
        return None
        
    try:
        df_plot = df.tail(100).copy()
        
        # Добавляем EMA для контекста
        ema_fast = ta.trend.ema_indicator(df_plot['close'], window=TRENDING_FAST_EMA)
        ema_slow = ta.trend.ema_indicator(df_plot['close'], window=TRENDING_SLOW_EMA)
        
        plots = [
            mpf.make_addplot(ema_fast, color='cyan', width=0.7),
            mpf.make_addplot(ema_slow, color='orange', width=0.7),
        ]
        
        # Горизонтальные линии для сделки
        hlines = dict(hlines=[], colors=[], linestyle='--')
        if passport.entry_price: hlines['hlines'].append(passport.entry_price); hlines['colors'].append('white')
        if passport.sl_price: hlines['hlines'].append(passport.sl_price); hlines['colors'].append('red')
        if passport.tp_price: hlines['hlines'].append(passport.tp_price); hlines['colors'].append('green')

        buf = io.BytesIO()
        style = mpf.make_marketcolors(up='#26a69a', down='#ef5350', inherit=True)
        mc_style = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=style, gridstyle=':')
        
        mpf.plot(
            df_plot, type='candle', style=mc_style,
            title=f"{passport.symbol} | {passport.strategy} ({passport.side.upper()})",
            ylabel='Price', addplot=plots, hlines=hlines,
            savefig=dict(fname=buf, dpi=120),
            figsize=(12, 7)
        )
        buf.seek(0)
        return buf
    except Exception as e:
        logging.error(f"[{passport.symbol}] Ошибка при создании графика: {e}")
        return None

import csv

import os

async def log_and_notify_async(passport: TradePassport, df_chart: pd.DataFrame):
    """
    Центральная функция для логирования в CSV и отправки уведомлений с графиком в Telegram.
    """
    file_path = "signals_log.csv"
    log_entry = passport.__dict__
    headers = list(log_entry.keys())

    # --- Блок логирования в CSV (без изменений) ---
    try:
        file_exists = os.path.exists(file_path)
        async with aiofiles.open(file_path, mode='a', encoding='utf-8', newline='') as f:
            if not file_exists:
                await f.write(','.join(headers) + '\n')
            
            def escape_csv(val):
                s_val = str(val).replace('"', '""')
                return f'"{s_val}"'

            row_values = [escape_csv(log_entry.get(h, '')) for h in headers]
            await f.write(','.join(row_values) + '\n')
            
        logging.info(f"Сигнал {passport.signal_id} ({passport.status}) успешно записан в {file_path}")
    except Exception as e:
        logging.error(f"Не удалось записать лог сигнала в {file_path}: {e}")

    # --- ✅ ИСПРАВЛЕНИЕ: БЛОК, МЕШАЮЩИЙ ОТПРАВКЕ УВЕДОМЛЕНИЙ, УДАЛЕН ---
    # Старый блок `if passport.status == "EXECUTED": return` был полностью удален.
    # Теперь эта функция снова отвечает за отправку всех уведомлений.
    
    if passport.status == "EXECUTED":
        # Проверяем, что все данные для уведомления на месте
        if not all([passport.entry_price, passport.sl_price, passport.tp_price]):
            logging.error(f"[{passport.symbol}] Невозможно отправить паспорт: отсутствуют данные о ценах.")
            return

        msg = (
            f"✅ <b>Вход {passport.side.upper()} по {passport.symbol}</b>\n\n"
            f"<b>Стратегия:</b> {passport.strategy}\n"
            f"<b>Триггер:</b> {passport.trigger}\n"
            f"<b>AI Уверенность:</b> {passport.ai_confidence}% (Порог: {passport.required_threshold:.0f}%)\n\n"
            f"<b>Цена входа:</b> {float(passport.entry_price):.4f}\n"
            f"<b>Stop Loss:</b> {float(passport.sl_price):.4f}\n"
            f"<b>Take Profit:</b> {float(passport.tp_price):.4f}"
        )
        
        chart_buffer = await generate_entry_chart_async(passport, df_chart)
        
        if chart_buffer and telegram_bot:
            try:
                await telegram_bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=chart_buffer, caption=msg, parse_mode='HTML')
            except Exception as e:
                logging.error(f"Не удалось отправить фото паспорта сделки: {e}")
                await send_telegram_message(msg)
        else:
            await send_telegram_message(msg)

# --- НОВАЯ ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ---
async def wait_order_new(symbol: str, order_id: Optional[int], timeout_s: float = 4.0) -> bool:
    """
    Ожидает, пока ордер не получит статус NEW на бирже.
    Возвращает True, если статус подтвержден в течение таймаута, иначе False.
    """
    if not order_id or not client:
        return False

    log_prefix = f"[{symbol}] [Wait Status NEW]"
    deadline = asyncio.get_event_loop().time() + timeout_s
    logging.info(f"{log_prefix} Ожидание статуса NEW для ордера {order_id} (таймаут {timeout_s}с)...")

    while asyncio.get_event_loop().time() < deadline:
        try:
            # Используем get_order для проверки статуса
            od = await client.futures_get_order(symbol=symbol, orderId=int(order_id))
            status = od.get("status") if od else None

            # NEW - идеальный статус для STOP/TAKE_PROFIT ордеров, которые еще не сработали
            # ACCEPTED/TRIGGERED - могут быть условными ордерами, которые биржа приняла
            if status in ("NEW", "ACCEPTED", "TRIGGERED"):
                logging.info(f"{log_prefix} Статус '{status}' для ордера {order_id} подтвержден.")
                return True
            # Если ордер уже исполнился или отменен - выходим
            elif status in ("FILLED", "CANCELED", "EXPIRED", "REJECTED"):
                 logging.warning(f"{log_prefix} Ордер {order_id} уже в финальном статусе '{status}'. Ожидание прекращено.")
                 return False # Считаем неудачей, т.к. ожидали NEW

        except BinanceAPIException as e:
            # Игнорируем ошибку "Order does not exist" в первые секунды
            if e.code == -2013 and (asyncio.get_event_loop().time() < deadline - timeout_s + 1.5):
                 logging.debug(f"{log_prefix} Ордер {order_id} еще не найден, ждем...")
            else:
                 logging.error(f"{log_prefix} API Ошибка при проверке статуса ордера {order_id}: {e}")
                 return False # Выходим при других ошибках API
        except Exception as e_generic:
            logging.error(f"{log_prefix} Неизвестная ошибка при проверке статуса {order_id}: {e_generic}")
            return False # Выходим

        await asyncio.sleep(0.1) # Короткая пауза перед следующей проверкой

    logging.error(f"{log_prefix} Таймаут ожидания статуса NEW для ордера {order_id}.")
    return False
# --- КОНЕЦ НОВОЙ ФУНКЦИИ ---

# --- ОБНОВЛЕННАЯ ФУНКЦИЯ ПОСЛЕ DCA/ПИРАМИДИНГА V-АТОМАРНАЯ ---
async def update_position_after_scaling(pos_manager: "PositionManager", add_order_info: Dict, add_qty: float):
    """
    АТОМАРНАЯ ВЕРСИЯ: Сначала ставит новый SL, ждет подтверждения,
    затем отменяет старые ордера. Убрана пауза sleep(2.5).
    TP остается виртуальным.
    """
    symbol = pos_manager.symbol # Используем атрибут менеджера
    side = pos_manager.state.side
    pos_state = pos_manager.state
    log_prefix = f"⚖️ [{symbol}] [Post-Scale V-Atomic]"

    await send_telegram_message_safe(f"⚙️ **{symbol}**: Обработка докупки {add_qty:.4f} (Атомарный SL)...")

    # Сохраняем ID старых ордеров ДО их возможной отмены
    old_sl_id = pos_state.meta.get('sl_order_id')
    old_tp_orders_ids = [pos_state.meta.get(key) for key in ['tp1_order_id', 'tp2_order_id', 'near_tp_partial_oid', 'near_tp_guard_oid'] if pos_state.meta.get(key)]

    try:
        # --- 1. Получение актуальных данных о позиции ---
        # Даем небольшую паузу для обновления данных на бирже после докупки
        await asyncio.sleep(0.5)
        positions_from_api = await loop.run_in_executor(None, get_open_positions_sync)
        if not positions_from_api or symbol not in positions_from_api:
            # Попробуем еще раз через секунду
            await asyncio.sleep(1.0)
            positions_from_api = await loop.run_in_executor(None, get_open_positions_sync)
            if not positions_from_api or symbol not in positions_from_api:
                raise ValueError("Позиция не найдена на бирже после докупки!")

        updated_pos_data = positions_from_api[symbol]
        new_avg_price = float(updated_pos_data['entry_price'])
        new_total_qty = float(updated_pos_data['amount'])

        # --- 2. Обновление локального состояния (средняя цена, объем) ---
        pos_state.entry_price = new_avg_price
        pos_state.current_quantity = new_total_qty

        # --- 3. Расчет нового Stop Loss ---
        # Используем функцию, которая сохраняет общий риск (была в check_and_execute_scaling_v2)
        def _recalc_sl_preserve_total_risk(side, old_qty, old_avg, old_initial_sl, new_qty, new_avg):
             if old_qty <= 0 or new_qty <= 0 or old_initial_sl is None: return None
             initial_risk_per_unit = abs(old_avg - old_initial_sl)
             total_initial_risk = old_qty * initial_risk_per_unit
             new_risk_per_unit = total_initial_risk / new_qty
             new_sl = new_avg - new_risk_per_unit if side.upper() == "LONG" else new_avg + new_risk_per_unit
             return new_sl

        # Нужны данные ДО добавления для расчета
        # Предполагаем, что они доступны (их нужно было бы передавать или хранить)
        # В check_and_execute_scaling_v2 они были, здесь используем приближение
        # ВАЖНО: Для точного сохранения риска нужно передать old_qty, old_avg из check_and_execute_scaling_v2!
        # Пока используем fallback на гибридный расчет SL
        regime_id = market_regimes.get(symbol, 1)
        new_sl_price = await calculate_hybrid_stop_loss(symbol, side, new_avg_price, regime_id)

        if not new_sl_price:
            raise ValueError("НЕ УДАЛОСЬ РАССЧИТАТЬ НОВЫЙ SL!")

        # --- 4. УСТАНОВКА НОВОГО Stop Loss (с верификацией) ---
        close_side = "SELL" if side.upper() == "LONG" else "BUY"
        formatted_sl_price = format_price(new_sl_price, symbol, side=close_side)
        if not formatted_sl_price: raise ValueError(f"Не удалось отформатировать цену SL: {new_sl_price}")

        new_sl_order_id = None
        new_sl_order = await place_order_async(
            symbol=symbol, side=close_side, order_type="STOP_MARKET",
            quantity=new_total_qty, # На новый полный объем
            stopPrice=formatted_sl_price,
            reduce_only=True
            # НЕ ИСПОЛЬЗУЕМ closePosition=True здесь, так как управляем объемом
        )

        if new_sl_order and new_sl_order.get('orderId'):
            new_sl_order_id = new_sl_order.get('orderId')
            # Ждем подтверждения статуса NEW
            if not await wait_order_new(symbol, new_sl_order_id, timeout_s=4.0):
                # Если статус не подтвержден - КРИТИЧЕСКАЯ ОШИБКА
                await cancel_single_order_async(symbol, new_sl_order_id) # Пытаемся отменить некорректный ордер
                raise ValueError(f"Новый SL ордер {new_sl_order_id} не получил статус NEW вовремя!")
            logging.warning(f"{log_prefix} ✅ Новый SL ордер {new_sl_order_id} на {new_total_qty} ед. УСПЕШНО УСТАНОВЛЕН и ВЕРИФИЦИРОВАН на {formatted_sl_price}.")
        else:
            # Если даже разместить не удалось - КРИТИЧЕСКАЯ ОШИБКА
            raise ValueError("КРИТИЧЕСКАЯ ОШИБКА УСТАНОВКИ НОВОГО SL!")

        # --- 5. ОТМЕНА СТАРЫХ ОРДЕРОВ (только после успешной установки нового SL) ---
        orders_to_cancel_finally = [old_sl_id] + old_tp_orders_ids
        orders_to_cancel_finally = [oid for oid in orders_to_cancel_finally if oid]
        if orders_to_cancel_finally:
            logging.warning(f"{log_prefix} Отменяем {len(orders_to_cancel_finally)} старых ордеров...")
            await asyncio.gather(*[cancel_single_order_async(symbol, oid) for oid in orders_to_cancel_finally])

        # --- 6. Обновление состояния и плана DCA ---
        pos_state.sl_price = new_sl_price # Сохраняем рассчитанный (неформатированный) SL
        pos_state.meta['sl_order_id'] = new_sl_order_id # ID нового SL
        # Очищаем ID старых TP ордеров
        for key in ['tp1_order_id', 'tp2_order_id', 'near_tp_partial_oid', 'near_tp_guard_oid']:
             pos_state.meta.pop(key, None)

        # Обновляем план DCA (логика без изменений)
        scaling_plan = pos_state.meta.get("scaling_plan")
        if scaling_plan and isinstance(scaling_plan, dict):
            current_adds_done = scaling_plan.get('adds_done', 0)
            new_adds_done = current_adds_done + 1
            scaling_plan['adds_done'] = new_adds_done
            if new_adds_done < scaling_plan.get('max_adds', 0):
                levels = scaling_plan.get('levels', [])
                if new_adds_done < len(levels): scaling_plan['next_add_price'] = levels[new_adds_done]
                else: scaling_plan['next_add_price'] = None
            else: scaling_plan['next_add_price'] = None
        else: logging.warning(f"{log_prefix} scaling_plan не найден/невалиден.")

        await save_state_async()
        logging.warning(f"{log_prefix} ✅ Обработка докупки (Атомарный SL) завершена.")
        await send_telegram_message_safe(
            f"✅ **{symbol} Докупка + Перестановка SL!**\n"
            f"📊 Новый объем: {new_total_qty:.4f}\n"
            f"💰 Новая ср. цена: {new_avg_price:.5f}\n"
            f"🛡️ **Новый SL установлен:** {formatted_sl_price}"
        )

    # --- 7. Обработка КРИТИЧЕСКИХ ошибок ---
    except Exception as e:
        logging.critical(f"{log_prefix} КРИТИЧЕСКАЯ ОШИБКА при обновлении позиции после докупки: {e}. Запуск аварийного закрытия...", exc_info=True)
        # В случае любой ошибки (расчет SL, установка SL, верификация) - закрываем позицию
        await close_position_emergency(symbol, pos_state.__dict__, f"Крит. ошибка после DCA: {e}")

# --- КОНЕЦ ОБНОВЛЕННОЙ ФУНКЦИИ ---

def build_tp_ladder(entry: float, sl: float, side: str, rr: float) -> Dict[str, float]:
    """
    Создает двухуровневую лестницу Take Profit.
    TP1 для быстрой фиксации и перевода в Б/У, TP2 - основная цель.
    """
    risk = abs(entry - sl)
    # TP1 устанавливается на 60% от полной дистанции R:R для быстрой фиксации
    tp1 = entry + (0.6 * rr * risk) if side.upper() == "LONG" else entry - (0.6 * rr * risk)
    # TP2 - это полная, рассчитанная адаптивная цель
    tp2 = entry + (1.0 * rr * risk) if side.upper() == "LONG" else entry - (1.0 * rr * risk)
    return {
        "tp1": tp1,
        "tp2": tp2
    }

def get_market_structure(df: pd.DataFrame, lookback: int = 20) -> str:
    """
    Определяет структуру рынка (Higher Highs/Higher Lows или Lower Lows/Lower Highs).
    Возвращает: 'Uptrend', 'Downtrend', или 'Sideways'.
    """
    if df is None or len(df) < lookback * 2:
        return 'Sideways'

    recent_data = df.iloc[-(lookback * 2):]
    
    # Находим последние 2 максимума и 2 минимума
    high1 = recent_data['high'].tail(lookback).max()
    high2 = recent_data['high'].head(lookback).max()
    
    low1 = recent_data['low'].tail(lookback).min()
    low2 = recent_data['low'].head(lookback).min()

    if high1 > high2 and low1 > low2:
        return 'Uptrend'
    elif high1 < high2 and low1 < low2:
        return 'Downtrend'
    else:
        return 'Sideways'

async def calculate_quantitative_sltp(symbol: str, side: str, entry_price: float, regime: int) -> Dict[str, Optional[float]]:
    """
    Рассчитывает гибридные SL/TP с ГАРАНТИРОВАННЫМ R:R.
    --- ВЕРСИЯ 2: РАСШИРЕННЫЙ SL при низком ADX ---
    """
    global market_data_store, ALLOWED_INTERVALS, ML_MODEL_TF_SHORT, COMMON_ATR_PERIOD_FOR_SLTP, REGIME_CONFIG, ta, find_structural_level

    # Используем 15м для ATR, как и раньше
    df = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS[ML_MODEL_TF_SHORT])
    if df is None or len(df) < 30:
        logging.warning(f"[{symbol}] Недостаточно данных для гибридного расчета SL/TP.")
        return {"stop_loss": None, "take_profit": None}

    try: # Добавим try-except для большей надежности
        atr_value = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=COMMON_ATR_PERIOD_FOR_SLTP).iloc[-1]
        if pd.isna(atr_value) or atr_value <= 0:
            # Попробуем рассчитать ATR с большим периодом как fallback
            atr_value = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=COMMON_ATR_PERIOD_FOR_SLTP * 2).iloc[-1]
            if pd.isna(atr_value) or atr_value <= 0:
                 logging.error(f"[{symbol}] Не удалось рассчитать валидный ATR.")
                 return {"stop_loss": None, "take_profit": None}
            logging.warning(f"[{symbol}] Использован ATR с удвоенным периодом.")


        adx_val = None
        try:
            # ADX берем с 1h для оценки общего тренда
            df_long = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS.get('1h'))
            if df_long is not None and len(df_long) >= 20:
                adx_val = ta.trend.adx(df_long['high'], df_long['low'], df_long['close'], window=14).iloc[-1]
        except Exception:
            adx_val = None # Не критично, если ADX не рассчитался

        config = REGIME_CONFIG.get(regime, REGIME_CONFIG.get(1, {}))
        # Базовый множитель из конфига, но не ниже MIN_SL_ATR_MULT_BASE
        m_sl_base = max(config.get('atr_multiplier_sl', 3.0), MIN_SL_ATR_MULT_BASE)

        m_sl = m_sl_base
        # --- ✅ ИЗМЕНЕННАЯ ЛОГИКА ADX ---
        if adx_val is not None and adx_val < ADX_LOW_THRESHOLD_FOR_SL:
            # УВЕЛИЧИВАЕМ множитель при низком ADX
            m_sl += SL_ATR_MULT_INCREASE_LOW_ADX
            logging.info(f"[{symbol}] Низкий ADX ({adx_val:.1f}). Множитель SL УВЕЛИЧЕН до {m_sl:.2f}")
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

        risk_distance_atr = atr_value * m_sl
        sl_atr_based = entry_price - risk_distance_atr if side.upper() == 'LONG' else entry_price + risk_distance_atr

        # Структурный SL здесь не рассчитываем, он будет в hybrid

        final_stop_loss = sl_atr_based # Пока используем только ATR
        actual_risk_distance = abs(entry_price - final_stop_loss)

        if actual_risk_distance <= 1e-9: # Защита от нулевого риска
            logging.error(f"[{symbol}] Рассчитанная дистанция риска равна нулю. SL/TP не установлены.")
            return {"stop_loss": None, "take_profit": None}

        # Расчет TP (логика R:R остается)
        m_tp = config.get('atr_multiplier_tp', 4.5)
        if adx_val is not None:
            if adx_val < 18: m_tp *= 0.9
            elif adx_val > 28: m_tp *= 1.1

        # Гарантируем минимальный R:R для TP
        min_tp_rr = MIN_RR_RATIO # Используем глобальную константу
        reward_distance = actual_risk_distance * max(m_tp, min_tp_rr) # Берем больший из расчетного и минимального
        final_take_profit = entry_price + reward_distance if side.upper() == 'LONG' else entry_price - reward_distance

        logging.info(f"[{symbol}] Количественный SL/TP (V2): SL={final_stop_loss:.5f} (ATR*{m_sl:.2f}), TP={final_take_profit:.5f} (R:R ≈ {max(m_tp, min_tp_rr):.2f})")

        return {"stop_loss": final_stop_loss, "take_profit": final_take_profit}

    except Exception as e:
        logging.error(f"[{symbol}] Крит. ошибка в calculate_quantitative_sltp_v2: {e}", exc_info=True)
        return {"stop_loss": None, "take_profit": None}

async def unified_liquidity_filter(symbol: str) -> tuple[bool, str]:
    """
    Реализует двухуровневый гейт ликвидности (V2):
    1. TOP_CAP_SYMBOLS пропускаются всегда.
    2. Для остальных: скор >= 5.0 И спред <= 0.15%.
    """
    # Уровень 1: ...
    if symbol in TOP_CAP_SYMBOLS:
        return True, "Top-cap symbol, liquidity check skipped."

    # Уровень 2: Проверка для остальных монет по скоринг-системе
    LIQUIDITY_SCORE_THRESHOLD = 4.0 
    MAX_SPREAD_PERCENT = 0.15 # <-- НОВЫЙ ПОРОГ
    
    liquidity_score, spread_percent = await estimate_market_liquidity(symbol)
    
    if liquidity_score < LIQUIDITY_SCORE_THRESHOLD:
        reason = f"Низкая ликвидность (скор: {liquidity_score:.1f} < {LIQUIDITY_SCORE_THRESHOLD})"
        return False, reason
        
    if spread_percent > MAX_SPREAD_PERCENT:
        reason = f"Слишком большой спред ({spread_percent:.3f}% > {MAX_SPREAD_PERCENT}%)"
        return False, reason
        
    return True, f"Liquidity score OK ({liquidity_score:.1f}), Spread OK ({spread_percent:.3f}%)"

def get_symbol_price_step(symbol: str) -> Optional[float]:
    """Извлекает шаг цены (tickSize) из кэша биржевой информации."""
    global exchange_info_cache
    try:
        symbol_info = exchange_info_cache.get(symbol)
        if symbol_info and 'tickSize' in symbol_info:
            return float(symbol_info['tickSize'])
        return None
    except (ValueError, KeyError, TypeError):
        logging.warning(f"[{symbol}] Не удалось получить tickSize из exchange_info_cache.")
        return None

async def find_donchian_breakout_trade(symbol: str):
    """
    Пробой Канала Дончиана с быстрыми пре-фильтрами и финальным вердиктом от Gemini.
    --- ВЕРСИЯ С ГЛОБАЛЬНЫМ ФИЛЬТРОМ, ОБЪЕМОМ, ВОЛАТИЛЬНОСТЬЮ И СЕТКОЙ ГРАФИКОВ 4-В-1 ---
    """
    timeframe = '1h'
    df = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS[timeframe])
    donchian_period = 40
    if df is None or len(df) < max(donchian_period, 50):
        return

    passport = None
    try:
        # --- Этап 1: Проверка на 'сжатие' волатильности ДО пробоя ---
        if not is_in_volatility_squeeze(df):
            return

        # --- Этап 2: Поиск первичного триггера ---
        upper_ch, lower_ch, _ = calculate_donchian_channels(df, period=donchian_period)
        fast_ema = ta.trend.ema_indicator(df['close'], window=20)
        slow_ema = ta.trend.ema_indicator(df['close'], window=50)
        last_close = df['close'].iloc[-1]
        
        trade_type, trigger_reason = None, ""
        if df['close'].iloc[-2] < upper_ch.iloc[-2] and last_close > upper_ch.iloc[-2] and fast_ema.iloc[-1] > slow_ema.iloc[-1]:
            trade_type, trigger_reason = "long", f"Donchian Breakout H1 (P{donchian_period})"
        elif df['close'].iloc[-2] > lower_ch.iloc[-2] and last_close < lower_ch.iloc[-2] and fast_ema.iloc[-1] < slow_ema.iloc[-1]:
            trade_type, trigger_reason = "short", f"Donchian Breakdown H1 (P{donchian_period})"

        if not trade_type:
            return

        # --- Этап 3: Быстрые локальные пре-фильтры ---
        logging.warning(f"📈 [{symbol}] НАЙДЕН ПОТЕНЦИАЛЬНЫЙ СИГНАЛ BREAKOUT. Проверка фильтров...")
        
        if not is_breakout_volume_confirmed(df, multiplier=1.8):
            return

        atr = float(ta.volatility.average_true_range(df['high'], df['low'], df['close'], 14).iloc[-1])
        dist_from_ema_in_atr = abs(last_close - fast_ema.iloc[-1]) / max(1e-9, atr)
        if dist_from_ema_in_atr > 2.5:
            logging.info(f"~ [{symbol}] Breakout сигнал пропущен: слишком далеко от EMA.")
            return
            
        if is_quiet_session():
            logging.info(f"~ [{symbol}] Breakout сигнал пропущен: тихая торговая сессия.")
            return

        # --- Этап 4: Создание паспорта и глубокий анализ AI ---
        passport = TradePassport(symbol=symbol, strategy="Breakout_AI_Filtered", trigger=trigger_reason, side=trade_type)
        logging.warning(f"✅ [{symbol}] СИГНАЛ BREAKOUT ПРОШЕЛ ВСЕ ФИЛЬТРЫ. Отправляю на анализ в Gemini...")

        async with GLOBAL_ENTRY_LOCK:
            # --- ✅ ИЗМЕНЕНИЕ: ФИЛЬТР ГЛОБАЛЬНОГО НАПРАВЛЕНИЯ ---
            ok, reason = await global_direction_filter(trade_type, symbol)
            if not ok:
                raise ValueError(reason)
            
            dynamic_limit = get_dynamic_max_positions()
            if len(current_positions) >= dynamic_limit:
                logging.info(f"[{symbol}] Сигнал 'Breakout' пропущен: достигнут лимит позиций ({len(current_positions)}/{dynamic_limit})")
                return

            basis_data = await get_basis_and_funding(symbol)
            liq_signal = get_liquidation_signal(symbol)
            context_text = (
                f"Funding Rate: {basis_data.get('funding_pct', 0):.4f}%. "
                f"Basis: {basis_data.get('basis_bps', 0):.1f} bps. "
                f"Liquidation Spike: {'Yes, bias ' + liq_signal['bias'] if liq_signal['spike'] else 'No'}. "
                f"Volume Confirmed: Yes."
            )
            logging.info(f"[{symbol}] Контекст для AI: {context_text}")

            # --- ✅ ИЗМЕНЕНИЕ: СОЗДАЕМ СЕТКУ 4-В-1 ---
            df_sym_15m = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS['15m'])
            df_btc_15m = market_data_store.get('BTCUSDT', {}).get(ALLOWED_INTERVALS['15m'])
            df_btc_1h = market_data_store.get('BTCUSDT', {}).get(ALLOWED_INTERVALS['1h'])
            
            combined_chart_base64 = await create_multi_timeframe_chart_grid(
                symbol, df_sym_15m, df, df_btc_15m, df_btc_1h # df - это уже df_sym_1h
            )
            if not combined_chart_base64: raise ValueError("Не удалось создать сетку графиков")
            
            # --- ✅ ИЗМЕНЕНИЕ: ОБНОВЛЯЕМ ПРОМПТ ДЛЯ AI ---
            prompt = (
                f"Вы — аналитик, оцениваете сигнал '{trade_type.upper()}' (пробой канала Дончиана) по {symbol}.\n"
                f"Перед вами 4 графика: {symbol} (15м/1ч) и BTC (15м/1ч).\n"
                f"Пробой на 1ч графике подтвержден объемом и произошел после сжатия волатильности.\n\n"
                f"**ЗАДАЧА:** Проведите мульти-таймфрейм анализ. Является ли этот пробой на 1ч истинным началом нового тренда, или это ложный 'вынос'? "
                f"Поддерживает ли краткосрочный контекст (15м) и общая ситуация на BTC открытие этой сделки?\n\n"
                f"Ответ строго в формате JSON: {{\"confidence\": 0-100, \"explanation\": \"...\"}}"
            )
            
            ai_verdict = await analyze_chart_with_gemini_vision(symbol, combined_chart_base64, trade_type, prompt)
            ai_confidence = ai_verdict.get("confidence", 0)
            
            entry_threshold = (await get_adaptive_entry_threshold(symbol))
            if ai_confidence < entry_threshold:
                raise ValueError(f"Gemini отклонил сделку (уверенность {ai_confidence}% < порога {entry_threshold:.1f}%)")

            entry_price = last_close
            sltp = await get_sltp_from_ai_async(symbol, trade_type, entry_price, context={})
            if not (sltp and sltp.get("stop_loss") and sltp.get("take_profit")):
                raise ValueError("AI не предоставил SL/TP")
            
            final_sl_price = float(sltp["stop_loss"])
            final_tp_price = float(sltp["take_profit"])
            
            qty = await calculate_smart_quantity_v2(symbol, entry_price, final_sl_price, ai_confidence, BASE_RISK_PERCENT, entry_threshold)
            if qty <= 0: raise ValueError("Расчетный объем равен нулю")

            execution_result = await adaptive_execute_entry(
                symbol=symbol, 
                side_str="BUY" if trade_type == "long" else "SELL", 
                quantity=qty, 
                sl_price=final_sl_price, 
                tp_price=final_tp_price, 
                entry_context=f"AI Conf: {ai_confidence}%, Trigger: {trigger_reason}, ID: {passport.signal_id}",
                signal_price=entry_price
            )

            if not execution_result: raise ValueError("Исполнение сделки не удалось")

            passport.status = "EXECUTED"
            passport.entry_price = execution_result['entry_price']
            passport.sl_price = final_sl_price
            passport.tp_price = final_tp_price
            passport.ai_confidence = ai_confidence
            passport.required_threshold = entry_threshold
            
            current_positions[symbol] = PositionManager(execution_result)
            await save_state_async()

    except ValueError as e:
        if passport:
            passport.status = "REJECTED"; passport.reject_reason = str(e)
            logging.warning(f"🚫 [{symbol}] Сигнал ОТКЛОНЕН. Причина: {passport.reject_reason}")
    except Exception as e:
        logging.error(f"[{symbol}] Критическая ошибка в find_donchian_breakout_trade: {e}", exc_info=True)
    finally:
        if passport and passport.status != "PENDING":
            await log_and_notify_async(passport, df)

async def find_and_execute_pullback_trade(symbol: str):
    """
    Полноценная торговая стратегия на основе "Pullback Engine" со всеми фильтрами.
    --- ВЕРСИЯ С ГЛОБАЛЬНЫМ ФИЛЬТРОМ, СЕТКОЙ ГРАФИКОВ 4-В-1 и сохранением типа стратегии ---
    """
    # 1. Получаем сигнал от "движка"
    signal = await compute_pullback_signal(symbol)
    if not signal.get("ok"):
        # logging.debug(f"[{symbol}] Pullback: Нет сигнала от compute_pullback_signal.") # Можно раскомментировать для отладки
        return

    passport = None
    # Используем 15м DF и для паспорта, и для сетки графиков
    df = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS.get('15m'))
    if df is None or len(df) < 60: # Убедимся, что данных достаточно
        logging.warning(f"[{symbol}] Pullback: Недостаточно данных 15m для анализа.")
        return

    try:
        # Извлекаем данные из сигнала
        trade_type = signal['side'].lower()
        trigger_reason = signal['reason']
        entry_price = signal['entry'] # Используем цену из сигнала как сигнальную
        final_sl_price = signal['sl']
        final_tp_price = signal['tp1'] # Используем TP1 из сигнала как базовый TP

        passport = TradePassport(symbol=symbol, strategy="Pullback Engine", trigger=trigger_reason, side=trade_type)
        logging.warning(f"↪️ [{symbol}] СИГНАЛ PULLBACK ID: {passport.signal_id} ({trigger_reason}). Верификация...")

        async with GLOBAL_ENTRY_LOCK:
            # --- ФИЛЬТР ГЛОБАЛЬНОГО НАПРАВЛЕНИЯ ---
            ok, reason = await global_direction_filter(trade_type, symbol)
            if not ok:
                raise ValueError(f"Глобальный фильтр: {reason}") # Добавим префикс для ясности

            # --- ПРОВЕРКА ЛИМИТА ПОЗИЦИЙ И ОТКРЫТОЙ ПОЗИЦИИ ---
            dynamic_limit = get_dynamic_max_positions()
            if len(current_positions) >= dynamic_limit:
                raise ValueError(f"Достигнут лимит позиций ({len(current_positions)}/{dynamic_limit})")
            if symbol in current_positions:
                raise ValueError("Позиция уже открыта")

            # --- Остальные фильтры (Basis/Funding, OFI, Reversal, Liquidity) ---
            basis_data = await get_basis_and_funding(symbol)
            # Передаем basis_bps и funding_pct напрямую
            ok_basis, reason_basis = basis_funding_filter(trade_type, basis_data.get("basis_bps", 0.0), basis_data.get("funding_pct", 0.0) / 100.0)
            if not ok_basis:
                raise ValueError(f"Basis/funding block: {reason_basis}")

            ofi_score, _ = await calculate_order_flow_imbalance(symbol)
            # Оставляем порог OFI для дополнительной проверки
            if trade_type == "long" and ofi_score <= -25:
                raise ValueError(f"Strong OFI against long: {ofi_score:.1f}")
            if trade_type == "short" and ofi_score >= 25:
                raise ValueError(f"Strong OFI against short: {ofi_score:.1f}")

            was_reversed = await handle_position_reversal(symbol, trade_type)
            if was_reversed: raise ValueError("Позиция была реверсирована")

            # AI BTC фильтр (можно оставить или убрать, если доверяете global_direction_filter)
            btc_ok, btc_reason = await btc_trend_filter_ai(trade_type, symbol)
            if not btc_ok: raise ValueError(f"AI BTC filter: {btc_reason}")

            liquidity_ok, liquidity_reason = await unified_liquidity_filter(symbol)
            if not liquidity_ok: raise ValueError(liquidity_reason)

            # Проверка R:R
            risk = abs(entry_price - final_sl_price)
            reward = abs(final_tp_price - entry_price)
            current_rr = (reward / risk) if risk > 0 else 0
            if current_rr < MIN_RR_RATIO: # Используем глобальную константу
                raise ValueError(f"Низкий R:R ({current_rr:.2f} < {MIN_RR_RATIO}) для отката.")

            # --- AI Вердикт на основе сетки 4-в-1 ---
            logging.info(f"[{symbol}] Создание сетки графиков 4-в-1 для Gemini...")
            df_sym_1h = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS['1h'])
            df_btc_15m = market_data_store.get('BTCUSDT', {}).get(ALLOWED_INTERVALS['15m'])
            df_btc_1h = market_data_store.get('BTCUSDT', {}).get(ALLOWED_INTERVALS['1h'])

            # Передаем df (15м) как df_sym_15m
            combined_chart_base64 = await create_multi_timeframe_chart_grid(
                symbol, df, df_sym_1h, df_btc_15m, df_btc_1h
            )
            if not combined_chart_base64: raise ValueError("Не удалось создать сетку графиков")

            # Промпт для Gemini
            prompt = (
                f"Вы — аналитик, оцениваете сигнал '{trade_type.upper()}' (откат к EMA) по {symbol}.\n"
                f"Перед вами 4 графика: {symbol} (15м/1ч) и BTC (15м/1ч).\n\n"
                f"**ЗАДАЧА:** Проведите мульти-таймфрейм анализ. Является ли этот откат на 15м графике хорошей точкой входа в рамках старшего тренда на 1ч? "
                f"Не противоречит ли этому движению общая ситуация на BTC? Оцените риски и вынесите вердикт.\n\n"
                f"Ответ строго в формате JSON: {{\"confidence\": 0-100, \"explanation\": \"...\"}}"
            )

            ai_verdict = await analyze_chart_with_gemini_vision(symbol, combined_chart_base64, trade_type, prompt)
            ai_confidence = ai_verdict.get("confidence", 0)

            # Адаптивный порог входа
            base_threshold = await get_adaptive_entry_threshold(symbol)
            garch_vol = await forecast_garch_volatility(symbol, market_data_store, "15m")
            threshold = base_threshold
            # Повышаем порог при низкой волатильности
            if garch_vol is not None and garch_vol < 0.8:
                threshold = min(base_threshold + 5, 75) # Ограничим сверху

            if ai_confidence < threshold:
                raise ValueError(f"AI confidence {ai_confidence}% < {threshold:.1f}%")

            logging.warning(f"✅ [{symbol}] ВХОД Pullback ОДОБРЕН AI ({ai_confidence}%). Финальные расчеты...")

            # Расчет объема
            # Передаем threshold для корректного расчета модификатора
            qty = await calculate_smart_quantity_v2(symbol, entry_price, final_sl_price, ai_confidence, BASE_RISK_PERCENT, threshold)
            if qty <= 0: raise ValueError("Расчетный объем равен нулю")

            # Исполнение входа
            passport.status = "EXECUTED"
            passport.entry_price = entry_price # Сигнальная цена
            passport.sl_price = final_sl_price
            passport.tp_price = final_tp_price # TP1
            passport.ai_confidence = ai_confidence
            passport.required_threshold = threshold

            execution_result = await adaptive_execute_entry(
                symbol=symbol, side_str="BUY" if trade_type == "long" else "SELL", quantity=qty,
                sl_price=final_sl_price, tp_price=final_tp_price, # Передаем TP1
                entry_context=f"AI Conf: {ai_confidence}%, Trigger: {trigger_reason}, ID: {passport.signal_id}",
                signal_price=entry_price # Передаем сигнальную цену
            )

            if execution_result:
                # --- ✅ ДОБАВЛЕНА СТРОКА ---
                execution_result['meta']['strategy_type'] = 'trend' # Указываем тип стратегии
                # --- КОНЕЦ ИЗМЕНЕНИЯ ---

                current_positions[symbol] = PositionManager(execution_result)
                await save_state_async()
            else:
                raise ValueError("Исполнение не удалось (adaptive_execute_entry провалился)")

    # Обработка отказов
    except ValueError as e:
        if passport:
            passport.status, passport.reject_reason = "REJECTED", str(e)
            logging.info(f"[{symbol}] Pullback Signal {passport.signal_id} rejected: {e}")
    # Обработка критических ошибок
    except Exception as e:
        if passport:
            passport.status, passport.reject_reason = "REJECTED", f"Критическая ошибка: {e}"
        logging.error(f"[{symbol}] Pullback trade critical error: {e}", exc_info=True)
    # Логирование и уведомление
    finally:
        if passport and passport.status != "PENDING": # Логируем только если был сигнал
            # Передаем df (15м) для генерации графика в уведомлении
            await log_and_notify_async(passport, df)


# Пирамидинг V2: риск-инвариантный, с микроструктурой и RR-гейтом
async def pyramiding_entry_v2(symbol: str, manager, min_rr: float = 1.25, max_steps: int = 3, step_fracs=(0.30, 0.25, 0.20), obi_gate: int = 10):
    """
    Исправленная версия с корректным пересчетом SL для сохранения константного риска
    и проверкой RR от текущего, а не начального, стопа.
    """
    pos = manager.state
    df = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS.get("5m"))
    
    # ПРОВЕРКА ОТ ТЕКУЩЕГО SL
    if df is None or pos.sl_price is None or pos.current_quantity is None or not pos.initial_quantity:
        return False

    price = float(df["close"].iloc[-1])
    # ПРАВИЛЬНАЯ ПРОВЕРКА RR
    rr_now = rr(pos.side, pos.entry_price, pos.sl_price, price)
    
    if rr_now < min_rr:
        return False
        
    done = int(pos.meta.get("pyramids_count", 0))
    if done >= max_steps:
        return False

    try:
        ofi_score, _ = await calculate_order_flow_imbalance(symbol)
        if pos.side.upper() == "LONG" and ofi_score < obi_gate: return False
        if pos.side.upper() == "SHORT" and ofi_score > -obi_gate: return False
    except Exception:
        pass

    frac = step_fracs[min(done, len(step_fracs) - 1)]
    add_raw = float(pos.initial_quantity) * float(frac)
    add_qty = float(format_quantity(add_raw, symbol) or 0.0)
    if add_qty <= 0.0:
        return False

    # Исполнение ордера на добавление
    side_str = "BUY" if pos.side.upper() == "LONG" else "SELL"
    ord_info = await place_order_async(symbol=symbol, side=side_str, order_type="MARKET", quantity=add_qty)
    if not ord_info or not ord_info.get("orderId"):
        return False

    # --- ВСТРОЕННАЯ ЛОГИКА ПЕРЕСЧЁТА ---
    E0, S0, Q0 = pos.entry_price, float(pos.sl_price), float(pos.current_quantity)
    Q1 = max(1e-9, Q0 + add_qty)
    E1 = (E0 * Q0 + price * add_qty) / Q1
    
    # Правильная формула для сохранения риска
    if pos.side.upper() == "LONG":
        S1 = E1 - ((E0 - S0) * Q0) / Q1
    else: # SHORT
        S1 = E1 + ((S0 - E0) * Q0) / Q1
    # --- КОНЕЦ ВСТРОЕННОЙ ЛОГИКИ ---
    
    # Обновление состояния позиции
    pos.entry_price = E1
    pos.current_quantity = Q1
    pos.meta["pyramids_count"] = done + 1

    await manager._update_sl_callback(float(S1), f"Pyramiding step {done+1}")
    await save_state_async()
    await send_telegram_message(f"🔺 **{symbol} Pyramiding #{done+1}**: +{add_qty:.4f} @ ${price:.4f}, New Avg: ${E1:.4f}, New SL: ${S1:.4f}")
    
    return True

# План DCA V2: многократно-адаптивный


async def btc_trend_filter_ai(trade_type: str, symbol_for_logging: str) -> Tuple[bool, str]:
    """
    Продвинутый AI-фильтр тренда BTC. Анализирует график BTC через Gemini Vision.
    """
    global market_data_store, ALLOWED_INTERVALS
    log_prefix = f"[{symbol_for_logging}]"
    
    try:
        logging.info(f"{log_prefix} Запуск AI-фильтра по BTC для сделки {trade_type.upper()}...")
        
        # 1. Получаем данные и создаем график для BTC
        df_btc = market_data_store.get('BTCUSDT', {}).get(ALLOWED_INTERVALS.get("15m"))
        if df_btc is None or len(df_btc) < 80:
            logging.warning(f"{log_prefix} AI-фильтр BTC пропущен: недостаточно данных.")
            return True, "Недостаточно данных BTC для AI-анализа"

        # Создаем график BTC
        fig = await create_enhanced_chart('BTCUSDT', df_btc, trade_type, df_btc['close'].iloc[-1])
        if fig is None:
            return True, "Ошибка создания графика BTC"

        chart_base64 = await save_chart_to_base64('BTCUSDT', fig)
        plt.close(fig)
        if not chart_base64:
            return True, "Ошибка конвертации графика BTC"

        # 2. Формируем специальный промпт для Gemini
        prompt = (
            f"Вы — главный риск-менеджер. Перед вами 15-минутный график BTC. "
            f"Другой бот хочет открыть сделку {trade_type.upper()} по одному из альткоинов. "
            f"Ваша задача — проанализировать ТОЛЬКО этот график BTC и решить, является ли текущая рыночная фаза на BTC благоприятной для такого действия.\n\n"
            f"**Проанализируйте:**\n"
            f"- Есть ли сильный импульс ПРОТИВ предлагаемой сделки?\n"
            f"- Не находится ли рынок в опасной 'пиле' или зоне высокой неопределенности?\n"
            f"- Поддерживает ли общая структура рынка открытие {trade_type.upper()} позиций прямо сейчас?\n\n"
            f"**Ваш вердикт:** Дайте оценку уверенности от 0 до 100, что сейчас безопасно открывать {trade_type.upper()} по альтам. "
            f"Высокая уверенность (>65) означает 'ДА, РАЗРЕШАЮ'. Низкая - 'НЕТ, БЛОКИРУЮ'.\n\n"
            f'Ответ строго в формате JSON: {{"confidence": 0-100, "explanation": "..."}}'
        )
        
        # 3. Вызываем Gemini Vision
        ai_verdict = await analyze_chart_with_gemini_vision('BTCUSDT', chart_base64, trade_type, prompt)
        ai_confidence = ai_verdict.get("confidence", 0)
        explanation = ai_verdict.get("explanation", "Нет объяснения.")
        
        # 4. Принимаем решение
        # Порог можно сделать настраиваемым, но 65 - хорошее начало.
        if ai_confidence >= 65:
            logging.info(f"{log_prefix} ✅ AI-фильтр BTC ПРОЙДЕН для {trade_type.upper()} (Уверенность: {ai_confidence}%). Причина: {explanation}")
            return True, f"AI Одобрено: {explanation} (conf: {ai_confidence}%)"
        else:
            logging.warning(f"{log_prefix} 🚫 AI-фильтр BTC ЗАБЛОКИРОВАЛ сделку {trade_type.upper()} (Уверенность: {ai_confidence}%). Причина: {explanation}")
            return False, f"AI Заблокировано: {explanation} (conf: {ai_confidence}%)"

    except Exception as e:
        logging.error(f"{log_prefix} Критическая ошибка в AI-фильтре BTC: {e}", exc_info=True)
        return True, "Ошибка AI-фильтра -> разрешено по умолчанию" # Безопасный фолбэк

async def compute_impulse_signal(symbol: str) -> dict:
    """
    Детектор "Impulse Engine": ищет зарождающееся импульсное движение.
    """
    df15 = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS.get('15m'))
    if df15 is None or len(df15) < 60:
        return {"ok": False, "reason": "Insufficient 15m data"}

    try:
        # --- Расчет индикаторов ---
        atr15 = float(ta.volatility.average_true_range(df15['high'], df15['low'], df15['close'], 14).iloc[-1])
        adx15 = float(ta.trend.adx(df15['high'], df15['low'], df15['close'], 14).iloc[-1])
        bb = ta.volatility.BollingerBands(df15['close'], window=20, window_dev=2)
        bbw = float(bb.bollinger_wband().iloc[-1])
        bbw_prev = float(bb.bollinger_wband().iloc[-5])
        bbw_expansion = (bbw - bbw_prev) / max(1e-9, bbw_prev)
        
        vol_avg = float(df15['volume'].rolling(20).mean().iloc[-2])
        last = df15.iloc[-1]
        body = abs(float(last['close']) - float(last['open']))
        is_big_body = body > 1.0 * atr15
        is_big_vol = float(last['volume']) > 1.5 * vol_avg if vol_avg > 0 else False
        
        # --- Микроструктура и внешние факторы ---
        micro = await get_market_microstructure_features(symbol, client)
        obi = float(micro.get("order_book_imbalance") or 0.0)
        ofi_score, _ = await calculate_order_flow_imbalance(symbol)
        liq = get_liquidation_signal(symbol)
        basis = await get_basis_and_funding(symbol)
        
        # --- Определение направления и проверка правил ---
        ema20 = float(ta.trend.ema_indicator(df15['close'], 20).iloc[-1])
        ema50 = float(ta.trend.ema_indicator(df15['close'], 50).iloc[-1])
        is_long_bias = ema20 > ema50
        
        ok_basis, basis_reason = basis_funding_filter("LONG" if is_long_bias else "SHORT", basis.get("basis_bps",0), basis.get("funding_pct", 0) / 100)
        if not ok_basis:
            return {"ok": False, "reason": f"Basis/Funding block: {basis_reason}"}

        if liq.get("spike") and ((is_long_bias and liq.get("bias")=="SELL") or (not is_long_bias and liq.get("bias")=="BUY")):
            return {"ok": False, "reason": f"Adverse liquidation spike {liq.get('bias')}"}

        vol_expansion = bbw_expansion > 0.3 and adx15 > 18.0
        micro_support_long = (obi > 0.2 or ofi_score >= 10)
        micro_support_short = (obi < -0.2 or ofi_score <= -10)
        
        if is_big_body and is_big_vol and vol_expansion:
            side = "LONG" if is_long_bias and micro_support_long else "SHORT" if (not is_long_bias) and micro_support_short else None
            if side is None:
                return {"ok": False, "reason": "Microstructure mismatch"}
            
            # Расчет триггера и зоны ретеста
            high = float(df15['high'].iloc[-1]); low = float(df15['low'].iloc[-1])
            delta = 0.20 * atr15
            trigger_price = high + delta if side == "LONG" else low - delta
            retest_price = high if side == "LONG" else low
            
            return {"ok": True, "kind": "impulse", "side": side, "atr": atr15, "trigger": trigger_price, "retest": retest_price,
                    "reason": f"BBW↑{bbw_expansion:.2f}, ADX={adx15:.1f}, OFI={ofi_score:.1f}, OBI={obi:.2f}"}

        return {"ok": False, "reason": "No impulse conditions"}
    except Exception as e:
        logging.error(f"[{symbol}] Ошибка в compute_impulse_signal: {e}", exc_info=True)
        return {"ok": False, "reason": str(e)}

async def compute_pullback_signal(symbol: str) -> dict:
    """
    Детектор "Pullback Engine" с УЛУЧШЕННЫМ R:R Ratio (v2.0).
    """
    df15 = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS.get('15m'))
    if df15 is None or len(df15) < 80:
        return {"ok": False, "reason": "Insufficient 15m data"}

    try:
        # --- Расчет индикаторов ---
        ema20 = float(ta.trend.ema_indicator(df15['close'], 20).iloc[-1])
        ema50 = float(ta.trend.ema_indicator(df15['close'], 50).iloc[-1])
        atr14 = float(ta.volatility.average_true_range(df15['high'], df15['low'], df15['close'], 14).iloc[-1])
        rsi14 = float(ta.momentum.rsi(df15['close'], 14).iloc[-1])
        last_close = float(df15['close'].iloc[-1])

        # --- Микроструктура и внешние факторы ---
        micro = await get_market_microstructure_features(symbol, client)
        obi = float(micro.get("order_book_imbalance") or 0.0)
        ofi_score, _ = await calculate_order_flow_imbalance(symbol)
        basis = await get_basis_and_funding(symbol)
        
        # --- Проверка правил для LONG ---
        is_uptrend = ema20 > ema50
        ok_basis_long, _ = basis_funding_filter("LONG", basis.get("basis_bps",0), basis.get("funding_pct", 0) / 100)
        is_retesting_ema = abs(last_close - ema20) <= 0.3 * atr14
        is_rsi_reset = rsi14 < 60
        has_micro_support = (ofi_score >= 10 or obi > 0.1)

        if is_uptrend and ok_basis_long and is_retesting_ema and is_rsi_reset and has_micro_support:
            entry_price = ema20
            sl_price = entry_price - 2.0 * atr14
            tp_price = entry_price + 3.0 * atr14
            expected_rr = (tp_price - entry_price) / (entry_price - sl_price)
            if expected_rr < 1.2:
                return {"ok": False, "reason": f"Low RR {expected_rr:.2f} < 1.2"}
            return {"ok": True, "kind": "pullback", "side": "LONG", "entry": entry_price, "sl": sl_price, "tp1": tp_price,
                    "reason": f"EMA20 retest, RSI={rsi14:.1f}, OFI={ofi_score:.1f}, OBI={obi:.2f}"}

        # --- Проверка правил для SHORT ---
        is_downtrend = ema20 < ema50
        ok_basis_short, _ = basis_funding_filter("SHORT", basis.get("basis_bps",0), basis.get("funding_pct", 0) / 100)
        is_rsi_reset_short = rsi14 > 40
        has_micro_support_short = (ofi_score <= -10 or obi < -0.1)

        if is_downtrend and ok_basis_short and is_retesting_ema and is_rsi_reset_short and has_micro_support_short:
            entry_price = ema20
            sl_price = entry_price + 2.0 * atr14
            tp_price = entry_price - 3.0 * atr14
            expected_rr = (entry_price - tp_price) / (sl_price - entry_price)
            if expected_rr < 1.2:
                return {"ok": False, "reason": f"Low RR {expected_rr:.2f} < 1.2"}
            return {"ok": True, "kind": "pullback", "side": "SHORT", "entry": entry_price, "sl": sl_price, "tp1": tp_price,
                    "reason": f"EMA20 retest, RSI={rsi14:.1f}, OFI={ofi_score:.1f}, OBI={obi:.2f}"}

        return {"ok": False, "reason": "No pullback conditions"}
    except Exception as e:
        logging.error(f"[{symbol}] Ошибка в compute_pullback_signal: {e}", exc_info=True)
        return {"ok": False, "reason": str(e)}

async def create_multi_timeframe_chart_grid(symbol: str, df_sym_15m, df_sym_1h, df_btc_15m, df_btc_1h) -> Optional[str]:
    """
    Создает единое изображение (2x2) с 4 графиками: 
    Символ (15м, 1ч) и BTC (15м, 1ч) для комплексного анализа в Gemini Vision.
    Возвращает изображение в формате base64.
    """
    try:
        # 1. Создаем фигуру с сеткой 2x2 для графиков
        fig, axes = plt.subplots(2, 2, figsize=(24, 16))
        fig.suptitle(f'Комплексный анализ для {symbol}', fontsize=24)

        # Настраиваем стиль для mplfinance
        mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350', inherit=True)
        style = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc, gridstyle=':')

        # --- 2. Рисуем каждый из 4 графиков ---
        
        # График 1: Символ, 15 минут
        if df_sym_15m is not None and len(df_sym_15m) > 50:
            df_plot = df_sym_15m.tail(80).copy()
            df_plot['EMA20'] = ta.trend.ema_indicator(df_plot['close'], 20)
            apds = [mpf.make_addplot(df_plot['EMA20'], ax=axes[0, 0], color='cyan')]
            mpf.plot(df_plot, ax=axes[0, 0], type='candle', style=style, addplot=apds, axtitle=f'{symbol} / 15m')

        # График 2: BTC, 15 минут
        if df_btc_15m is not None and len(df_btc_15m) > 50:
            df_plot = df_btc_15m.tail(80).copy()
            df_plot['EMA20'] = ta.trend.ema_indicator(df_plot['close'], 20)
            apds = [mpf.make_addplot(df_plot['EMA20'], ax=axes[0, 1], color='cyan')]
            mpf.plot(df_plot, ax=axes[0, 1], type='candle', style=style, addplot=apds, axtitle='BTC / 15m')

        # График 3: Символ, 1 час
        if df_sym_1h is not None and len(df_sym_1h) > 50:
            df_plot = df_sym_1h.tail(80).copy()
            df_plot['EMA20'] = ta.trend.ema_indicator(df_plot['close'], 20)
            apds = [mpf.make_addplot(df_plot['EMA20'], ax=axes[1, 0], color='orange')]
            mpf.plot(df_plot, ax=axes[1, 0], type='candle', style=style, addplot=apds, axtitle=f'{symbol} / 1h')

        # График 4: BTC, 1 час
        if df_btc_1h is not None and len(df_btc_1h) > 50:
            df_plot = df_btc_1h.tail(80).copy()
            df_plot['EMA20'] = ta.trend.ema_indicator(df_plot['close'], 20)
            apds = [mpf.make_addplot(df_plot['EMA20'], ax=axes[1, 1], color='orange')]
            mpf.plot(df_plot, ax=axes[1, 1], type='candle', style=style, addplot=apds, axtitle='BTC / 1h')

        # 3. Сохраняем результат в память
        buf = io.BytesIO()
        fig.tight_layout(rect=[0, 0.03, 1, 0.97]) # Оптимизируем расположение с учетом заголовка
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        
        # 4. Кодируем в base64 и закрываем фигуру, чтобы освободить память
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        logging.info(f"[{symbol}] Сетка графиков 2x2 для Gemini успешно создана.")
        return image_base64

    except Exception as e:
        logging.error(f"[{symbol}] Ошибка при создании сетки графиков: {e}", exc_info=True)
        if 'fig' in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)
        return None

async def find_and_execute_impulse_trade(symbol: str):
    """
    Импульсная стратегия с быстрыми пре-фильтрами и финальным вердиктом от Gemini.
    --- ВЕРСИЯ 1.1 (Патч для ключей TP) ---
    """
    df = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS.get('15m'))
    if df is None or len(df) < 60:
        return

    passport = None
    try:
        # --- Этап 1: Поиск первичного триггера ---
        signal = await compute_impulse_signal(symbol)
        if not signal.get("ok"):
            return

        trade_type = signal['side'].lower()
        trigger_reason = signal['reason']

        # --- Этап 2: Быстрые локальные пре-фильтры ---
        if is_quiet_session():
            logging.info(f"~ [{symbol}] Impulse сигнал пропущен: тихая торговая сессия.")
            return

        # --- Этап 3: Создание паспорта и глубокий анализ AI ---
        passport = TradePassport(symbol=symbol, strategy="Impulse_AI_Filtered", trigger=trigger_reason, side=trade_type)
        logging.warning(f"💥 [{symbol}] НАЙДЕН СИГНАЛ IMPULSE: {passport.side.upper()}. Прошел пре-фильтры. Отправляю на анализ в Gemini...")

        async with GLOBAL_ENTRY_LOCK:
            # --- ФИЛЬТР ГЛОБАЛЬНОГО НАПРАВЛЕНИЯ ---
            ok, reason = await global_direction_filter(trade_type, symbol)
            if not ok:
                raise ValueError(reason)

            # --- ПРОВЕРКА ЛИМИТА ПОЗИЦИЙ ---
            dynamic_limit = get_dynamic_max_positions()
            if len(current_positions) >= dynamic_limit:
                logging.info(f"[{symbol}] Сигнал 'Impulse' пропущен: достигнут лимит позиций ({len(current_positions)}/{dynamic_limit})")
                return
            if symbol in current_positions:
                 raise ValueError("Позиция уже открыта")

            # --- Сбор контекста для AI ---
            btc_state, _ = await get_btc_trend_state('1h')
            liq_signal = get_liquidation_signal(symbol)
            context_text = (
                f"BTC Trend State: {btc_state}. "
                f"Liquidation Spike: {'Yes, bias ' + liq_signal['bias'] if liq_signal['spike'] else 'No'}. "
                f"Signal Details: {trigger_reason}"
            )
            logging.info(f"[{symbol}] Контекст для AI: {context_text}")
            
            df_sym_1h = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS['1h'])
            df_btc_15m = market_data_store.get('BTCUSDT', {}).get(ALLOWED_INTERVALS['15m'])
            df_btc_1h = market_data_store.get('BTCUSDT', {}).get(ALLOWED_INTERVALS['1h'])
            
            combined_chart_base64 = await create_multi_timeframe_chart_grid(
                symbol, df, df_sym_1h, df_btc_15m, df_btc_1h
            )
            if not combined_chart_base64: raise ValueError("Не удалось создать сетку графиков")

            prompt = (
                f"Вы трейдер, ищущий импульсы. Оцените сигнал '{trade_type.upper()}' по {symbol}.\n"
                f"Перед вами 4 графика: {symbol} (15м/1ч) и BTC (15м/1ч).\n\n"
                f"**Дополнительный контекст:**\n{context_text}\n\n"
                f"**ЗАДАЧА:** Проанализируйте, действительно ли это начало сильного импульса на 15м, который поддерживается старшим трендом на 1ч. "
                f"Не противоречит ли этому импульсу ситуация на BTC? Оцените, не является ли это движение ложным 'выбросом' перед разворотом.\n\n"
                f"Ответ строго в формате JSON: {{\"confidence\": 0-100, \"explanation\": \"...\"}}"
            )
            
            ai_verdict = await analyze_chart_with_gemini_vision(symbol, combined_chart_base64, trade_type, prompt)
            ai_confidence = ai_verdict.get("confidence", 0)
            
            entry_threshold = (await get_adaptive_entry_threshold(symbol)) + 5 
            if ai_confidence < entry_threshold:
                raise ValueError(f"Gemini отклонил сделку (уверенность {ai_confidence}% < порога {entry_threshold:.1f}%)")

            # --- Этап 4: Исполнение сделки ---
            entry_price = await get_last_price(symbol)
            
            sltp = await get_sltp_from_ai_async(symbol, trade_type, entry_price, context={})
            
            # --- ✅ ИСПРАВЛЕНИЕ: Проверяем "take_profit_1" ---
            if not (sltp and sltp.get("stop_loss") and sltp.get("take_profit_1")):
                raise ValueError("AI не предоставил SL/TP (ошибка ключа 'take_profit_1')")

            final_sl_price = float(sltp["stop_loss"])
            final_tp_price = float(sltp["take_profit_1"]) # Используем TP1
            # --- КОНЕЦ ИСПРАВЛЕНИЯ ---
            
            qty = await calculate_smart_quantity_v2(symbol, entry_price, final_sl_price, ai_confidence, BASE_RISK_PERCENT, entry_threshold)
            if qty <= 0: raise ValueError("Расчетный объем равен нулю")

            execution_result = await adaptive_execute_entry(
                symbol=symbol, 
                side_str="BUY" if trade_type == "long" else "SELL", 
                quantity=qty, 
                sl_price=final_sl_price, 
                tp_price=final_tp_price, # Передаем TP1
                entry_context=f"AI Conf={ai_confidence}%, Trigger={trigger_reason}, ID:{passport.signal_id}",
                signal_price=entry_price
            )

            if not execution_result: raise ValueError("Исполнение сделки не удалось")

            passport.status = "EXECUTED"
            passport.entry_price = execution_result['entry_price']
            passport.sl_price = final_sl_price
            passport.tp_price = final_tp_price
            passport.ai_confidence = ai_confidence
            passport.required_threshold = entry_threshold
            
            # --- ✅ ДОБАВЛЕНО: Сохранение типа стратегии "trend" ---
            execution_result['meta']['strategy_type'] = 'trend'
            # --- КОНЕЦ ИСПРАВЛЕНИЯ ---
            current_positions[symbol] = PositionManager(execution_result)
            await save_state_async()
            
    except ValueError as e:
        if passport:
            passport.status = "REJECTED"; passport.reject_reason = str(e)
            logging.warning(f"🚫 [{symbol}] Сигнал Impulse ОТКЛОНЕН. Причина: {passport.reject_reason}")
    except Exception as e:
        logging.error(f"[{symbol}] Критическая ошибка в find_and_execute_impulse_trade: {e}", exc_info=True)
    finally:
        if passport and passport.status != "PENDING":
            await log_and_notify_async(passport, df)

def calculate_daily_vwap(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Рассчитывает дневной VWAP (сбрасывается каждый день).
    """
    if df is None or df.empty:
        return None
    try:
        # Убедимся, что индекс - это DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            logging.error("[VWAP] DataFrame index is not DatetimeIndex.")
            return None
            
        # Группируем по дате
        df_grouped = df.groupby(df.index.date)
        
        # Расчет VWAP
        cumulative_price_volume = (df['close'] * df['volume']).groupby(df.index.date).cumsum()
        cumulative_volume = df['volume'].groupby(df.index.date).cumsum()
        
        vwap = cumulative_price_volume / cumulative_volume
        
        # Заполняем NaN в начале дня (если объем был 0)
        # Новый, правильный вариант
        vwap = vwap.ffill()
        
        return vwap
        
    except Exception as e:
        logging.error(f"[VWAP] Ошибка расчета VWAP: {e}")
        return None

async def find_and_execute_vwap_cross_trade(symbol: str, expected_side: str):
    """
    Новая трендовая стратегия: Пересечение VWAP.
    Использует дневной VWAP на 15м графике.
    """
    log_prefix = f"[{symbol}] [VWAP Cross]"
    df = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS.get('15m'))
    
    if df is None or len(df) < 50:
        logging.debug(f"{log_prefix} Недостаточно данных.")
        return

    passport = None
    try:
        # --- Этап 1: Расчет VWAP и поиск сигнала ---
        vwap_series = calculate_daily_vwap(df)
        if vwap_series is None or len(vwap_series) < 3:
            logging.debug(f"{log_prefix} Не удалось рассчитать VWAP.")
            return
            
        last_close = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2]
        
        last_vwap = vwap_series.iloc[-1]
        prev_vwap = vwap_series.iloc[-2]

        trade_type, trigger_reason = None, None

        # Условие Long: Цена была НИЖЕ VWAP, а стала ВЫШЕ
        if expected_side == 'long' and (prev_close < prev_vwap) and (last_close > last_vwap):
            trade_type = "long"
            trigger_reason = f"VWAP Cross Up ({last_vwap:.4f})"
            
        # Условие Short: Цена была ВЫШЕ VWAP, а стала НИЖЕ
        elif expected_side == 'short' and (prev_close > prev_vwap) and (last_close < last_vwap):
            trade_type = "short"
            trigger_reason = f"VWAP Cross Down ({last_vwap:.4f})"

        if not trade_type:
            return

        # --- Этап 2: Создание паспорта и Глубокий Анализ AI ---
        passport = TradePassport(symbol=symbol, strategy="VWAP_Cross_AI_Filtered", trigger=trigger_reason, side=trade_type)
        logging.warning(f"🎯 [{symbol}] НАЙДЕН СИГНАЛ VWAP CROSS: {passport.side.upper()}. Отправляю на анализ...")

        async with GLOBAL_ENTRY_LOCK:
            # --- ФИЛЬТР ГЛОБАЛЬНОГО НАПРАВЛЕНИЯ ---
            ok, reason = await global_direction_filter(trade_type, symbol)
            if not ok:
                raise ValueError(reason)

            # --- ПРОВЕРКА ЛИМИТА ПОЗИЦИЙ ---
            dynamic_limit = get_dynamic_max_positions()
            if len(current_positions) >= dynamic_limit:
                raise ValueError(f"Достигнут лимит позиций ({len(current_positions)}/{dynamic_limit})")
            if symbol in current_positions:
                 raise ValueError("Позиция уже открыта")

            was_reversed = await handle_position_reversal(symbol, trade_type)
            if was_reversed:
                raise ValueError("Позиция была реверсирована, пропуск нового входа")

            # --- AI Вердикт (Контекст 'trending') ---
            ai_verdict = await get_ai_model_verdict_async(symbol, trade_type, "trending")
            ai_confidence = ai_verdict.get("confidence", 0)
            entry_threshold = await get_adaptive_entry_threshold(symbol)
            if ai_confidence < entry_threshold:
                raise ValueError(f"Gemini/DeepSeek отклонил сделку (уверенность {ai_confidence}% < порога {entry_threshold:.1f}%)")

            logging.warning(f"✅ [{symbol}] ВХОД VWAP Cross ОДОБРЕН AI ({ai_confidence}%). Финальные расчеты...")

            # --- Этап 3: Исполнение сделки ---
            entry_price = float(df["close"].iloc[-1])
            
            # Используем нашу новую функцию SL/TP
            sltp_levels = await get_sltp_from_ai_async(symbol, trade_type, entry_price, context={})
            
            if not (sltp_levels and sltp_levels.get("stop_loss") and sltp_levels.get("take_profit_1")):
                raise ValueError("AI не предоставил SL/TP (ошибка ключа take_profit_1)")
            
            final_sl_price = float(sltp_levels["stop_loss"])
            final_tp_price = float(sltp_levels["take_profit_1"]) # Используем TP1

            # Валидация геометрии
            final_sl_price, final_tp_price, norm_notes = normalize_sltp_for_side(
                 side="BUY" if trade_type == "long" else "SELL", entry=entry_price, sl=final_sl_price, tp=final_tp_price,
                 min_rr=MIN_RR_RATIO, price_step=get_symbol_price_step(symbol)
            )
            if norm_notes: logging.warning(f"[{symbol}] Уровни SL/TP нормализованы: {', '.join(norm_notes)}")

            # Расчет объема
            quantity = await calculate_smart_quantity_v2(symbol, entry_price, final_sl_price, ai_confidence, BASE_RISK_PERCENT, entry_threshold)
            if quantity <= 0: raise ValueError("Расчетный объем равен нулю")

            # Исполнение входа
            passport.status = "EXECUTED"
            passport.entry_price = entry_price
            passport.sl_price = final_sl_price
            passport.tp_price = final_tp_price
            passport.ai_confidence = ai_confidence
            passport.required_threshold = entry_threshold

            execution_result = await adaptive_execute_entry(
                symbol=symbol,
                side_str="BUY" if trade_type == "long" else "SELL",
                quantity=quantity,
                sl_price=final_sl_price,
                tp_price=final_tp_price, # Передаем TP1
                entry_context=f"AI Conf: {ai_confidence}%, Trigger: {trigger_reason}, ID: {passport.signal_id}",
                signal_price=entry_price
            )
            if not execution_result: raise ValueError("Исполнение сделки не удалось")

            # Сохраняем как 'trend'
            execution_result['meta']['strategy_type'] = 'trend' 
            current_positions[symbol] = PositionManager(execution_result)
            await save_state_async()

    except ValueError as e:
        if passport:
            passport.status = "REJECTED"; passport.reject_reason = str(e)
            logging.warning(f"🚫 [{symbol}] Сигнал VWAP Cross ОТКЛОНЕН. Причина: {passport.reject_reason}")
    except Exception as e:
        if passport: passport.status = "REJECTED"; passport.reject_reason = f"КРИТИЧЕСКАЯ ОШИБКА: {e}"
        logging.error(f"[{symbol}] Критическая ошибка в VWAP Cross: {e}", exc_info=True)
    finally:
        if passport and passport.status != "PENDING":
            await log_and_notify_async(passport, df)

async def get_last_price(symbol: str) -> float:
    """Быстро получает последнюю цену закрытия с самого быстрого таймфрейма."""
    # Используем 1м или 5м для максимальной скорости реакции
    for tf in ['1m', '5m']:
        df = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS.get(tf))
        if df is not None and not df.empty:
            return float(df['close'].iloc[-1])
    raise ValueError(f"Нет актуальных данных о цене для {symbol}")

def get_dynamic_max_positions() -> int:
    """
    Рассчитывает динамический лимит открытых позиций.
    Если есть прибыльные позиции в безубытке, лимит увеличивается.
    """
    global current_positions, MAX_OPEN_POSITIONS
    
    base_limit = MAX_OPEN_POSITIONS
    # Установим максимальный предел, чтобы избежать слишком большого риска
    MAX_OVERALL_POSITIONS = 8 

    risk_free_count = 0
    for symbol, manager in current_positions.items():
        if not isinstance(manager, PositionManager):
            continue

        pos_state = manager.state
        
        # Проверяем, что позиция в безубытке (флаг be_done установлен)
        is_breakeven = pos_state.meta.get("be_done", False)
        
        # Проверяем, что позиция в прибыли
        is_profitable = False
        try:
            # Используем ваш же метод для получения актуального PnL
            details = asyncio.run_coroutine_threadsafe(manager.get_detailed_status(), loop).result(timeout=5)
            if 'pnl' in details and details['pnl'] > 0:
                is_profitable = True
        except Exception:
            # Если не удалось получить PnL, считаем позицию не прибыльной
            pass

        if is_breakeven and is_profitable:
            risk_free_count += 1

    # --- ПРАВИЛО: Добавляем 1 слот за каждые 2 "безопасные" позиции ---
    # Вы можете изменить `2` на `1` (агрессивно) или `3` (консервативно)
    bonus_slots = risk_free_count // 2
    
    new_limit = base_limit + bonus_slots
    
    # Ограничиваем итоговый лимит максимальным значением
    final_limit = min(new_limit, MAX_OVERALL_POSITIONS)
    
    if final_limit != base_limit:
        logging.warning(
            f"📈 Динамический лимит позиций увеличен! "
            f"База: {base_limit}, Безопасных поз.: {risk_free_count}, Бонус: {bonus_slots} -> Итог: {final_limit}"
        )
        
    return final_limit

# СНАЧАЛА ИДЕТ ЭТА ФУНКЦИЯ
# --- ОБНОВЛЕННАЯ ФУНКЦИЯ process_message (v7.5 - Диспетчер Режимов) ---
async def process_message(msg: Dict[str, Any]):
    """
    ФИНАЛЬНАЯ ВЕРСИЯ v7.5: Диспетчер Режимов на основе ADX (Патч 1.3).
    Добавлена стратегия VWAP Cross.
    """
    try:
        if msg.get('e') != 'kline' or not msg.get('k', {}).get('x'):
            return  # Обрабатываем только закрытые свечи

        kline_data = msg['k']
        symbol = kline_data['s']
        interval_ws_key = kline_data['i']  # Таймфрейм из WebSocket
        interval_api_value = ALLOWED_INTERVALS.get(interval_ws_key)
        if not interval_api_value: return  # Неизвестный таймфрейм

        # --- Обновление данных ---
        kline_start_time_ms = kline_data['t']
        current_kline_ts = pd.to_datetime(kline_start_time_ms, unit='ms', utc=True)
        data_key = f"{symbol}_{interval_ws_key}"
        if last_processed_kline_time.get(data_key) and current_kline_ts <= last_processed_kline_time.get(data_key, pd.Timestamp.min.tz_localize('UTC')):
            return
        last_processed_kline_time[data_key] = current_kline_ts
        new_data = {'open': float(kline_data['o']), 'high': float(kline_data['h']), 'low': float(kline_data['l']), 'close': float(kline_data['c']), 'volume': float(kline_data['v']), 'close_time': pd.to_datetime(kline_data['T'], unit='ms', utc=True)}
        if symbol not in market_data_store: market_data_store[symbol] = {}
        if interval_api_value not in market_data_store[symbol]: market_data_store[symbol][interval_api_value] = pd.DataFrame(columns=new_data.keys()).set_index(pd.to_datetime([]))
        
        df_to_update = market_data_store[symbol][interval_api_value]
        new_row = pd.DataFrame(new_data, index=[current_kline_ts])

        # --- ✅ ИСПРАВЛЕННАЯ ЛОГИКА ОБНОВЛЕНИЯ DATAFRAME ---
        dfs_to_concat = []
        if not df_to_update.empty:
            dfs_to_concat.append(df_to_update[~df_to_update.index.isin(new_row.index)])
        if not new_row.empty:
            dfs_to_concat.append(new_row)
        
        if dfs_to_concat:
            df_to_update = pd.concat(dfs_to_concat)
        else:
            df_to_update = pd.DataFrame() # На случай, если оба пусты
        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

        if len(df_to_update) > MAX_DATAFRAME_ROWS: df_to_update = df_to_update.iloc[-MAX_DATAFRAME_ROWS:]
        market_data_store[symbol][interval_api_value] = df_to_update
        # --- Конец обновления данных ---

        # === 1. УПРАВЛЕНИЕ ОТКРЫТЫМИ ПОЗИЦИЯМИ (на ЛЮБОМ TF) ===
        if symbol in current_positions:
            manager = current_positions.get(symbol)
            if isinstance(manager, PositionManager):
                await manager.manage(df_work_tf=df_to_update)
                
                if interval_ws_key == '15m':
                    logging.info(f"~ [{symbol}] Новая свеча (15m). Проверка пирамидинга...")
                    await check_and_execute_scaling_v2(symbol)

                if interval_ws_key == '1h':
                    await manage_donchian_trailing_stop(manager)

        # === 2. ПОИСК НОВЫХ СИГНАЛОВ (ТОЛЬКО на 15м) ===
        if interval_ws_key == '15m' and symbol not in current_positions:
            logging.info(f"~ [{symbol}] Новая свеча (15m). Анализ рыночного режима...")

            # --- ДИСПЕТЧЕР РЕЖИМОВ (ПАТЧ 1.2) ---
            df15 = df_to_update # Это 15м DF
            adx_value = 0.0
            market_regime = "UNCERTAIN"
            try:
                if len(df15) >= 20:
                    adx_series = ta.trend.adx(df15['high'], df15['low'], df15['close'], window=14)
                    if not adx_series.empty:
                        adx_value = adx_series.iloc[-1]
                        if pd.notna(adx_value):
                            if adx_value < 20.0:
                                market_regime = "FLAT"
                            elif adx_value > 25.0:
                                market_regime = "TREND"
                            else:
                                market_regime = "TRANSITION"
                else:
                    logging.warning(f"[{symbol}] Недостаточно данных для ADX-диспетчера ({len(df15)}).")
            except Exception as e_adx:
                logging.error(f"[{symbol}] Ошибка ADX-диспетчера: {e_adx}")

            # --- ВЫЗОВ СООТВЕТСТВУЮЩИХ СТРАТЕГИЙ ---
            if market_regime == "FLAT":
                logging.warning(f"[{symbol}] РЕЖИМ: ФЛЭТ (ADX={adx_value:.1f}). Запуск Ranging (V5.0)...")
                # await find_and_execute_ranging_trade_v4_3(symbol)

            elif market_regime == "TREND":
                logging.warning(f"[{symbol}] РЕЖИМ: ТРЕНД (ADX={adx_value:.1f}). Запуск Трендовых (V2.0)...")
                if symbol not in current_positions: await find_and_execute_pullback_trade(symbol)
                
                if symbol not in current_positions:
                    df_trend = df15
                    if len(df_trend) > TRENDING_SLOW_EMA:
                        try:
                            fast_ema = ta.trend.ema_indicator(df_trend["close"], window=TRENDING_FAST_EMA).iloc[-1]
                            slow_ema = ta.trend.ema_indicator(df_trend["close"], window=TRENDING_SLOW_EMA).iloc[-1]
                            expected_side = 'long' if fast_ema > slow_ema else 'short'
                            
                            if symbol not in current_positions:
                                await find_and_execute_trend_trade(symbol, expected_side=expected_side)
                            
                            if symbol not in current_positions:
                                await find_and_execute_vwap_cross_trade(symbol, expected_side=expected_side)
                                
                        except Exception as e:
                            logging.error(f"[{symbol}] Ошибка расчета EMA для определения стороны тренда: {e}")
                
                if symbol not in current_positions: await find_and_execute_impulse_trade(symbol)

            elif market_regime == "TRANSITION":
                logging.info(f"[{symbol}] РЕЖИМ: ПЕРЕХОДНЫЙ (ADX={adx_value:.1f}, 20-25). Входы пропускаются.")
            
            else: # UNCERTAIN
                logging.warning(f"[{symbol}] РЕЖИМ: НЕОПРЕДЕЛЕННЫЙ. Входы пропускаются.")

        # === 3. ПОИСК BREAKOUT (ТОЛЬКО на 1h) ===
        elif interval_ws_key == '1h' and symbol not in current_positions:
            df1h = df_to_update
            try:
                if len(df1h) >= 20:
                    adx1h_series = adx(df1h['high'], df1h['low'], df1h['close'], window=14)
                    if not adx1h_series.empty:
                        adx1h = adx1h_series.iloc[-1]
                        if pd.notna(adx1h) and adx1h > 25.0:
                            logging.info(f"[{symbol}] Новая свеча (1h). Поиск Breakout сигнала...")
                            await find_donchian_breakout_trade(symbol)
                        else:
                            logging.info(f"[{symbol}] 1h Breakout пропущен: ADX={adx1h:.1f if pd.notna(adx1h) else 'NaN'} < 25.0")
            except Exception as e_adx1h:
                logging.error(f"[{symbol}] Ошибка ADX 1h для Breakout: {e_adx1h}")

    except Exception as e:
        symbol_in_error = msg.get('k', {}).get('s', 'UNKNOWN')
        logging.critical(f"[{symbol_in_error}] КРИТИЧЕСКАЯ ОШИБКА в process_message: {e}", exc_info=True)


async def enforce_liquidity_ban(symbol: str, side: str) -> Tuple[bool, str]:
    """
    Фильтр, который запрещает торговлю, если ликвидность слишком низкая.
    """
    MIN_LIQUIDITY_USD = 100000.0  # Порог ликвидности. Сделки на рынках с меньшей ликвидностью будут блокированы.

    liquidity = await estimate_market_liquidity(symbol)
    if liquidity is None:
        return False, f"Запрет: Не удалось получить данные о ликвидности."

    if liquidity < MIN_LIQUIDITY_USD:
        reason = f"Запрет {side.upper()}: Низкая ликвидность (${int(liquidity)} < ${int(MIN_LIQUIDITY_USD)})."
        logging.warning(f"[{symbol}] {reason}")
        return False, reason

    return True, f"Ликвидность в норме (${int(liquidity)})."


def calculate_regime_based_quantity(symbol: str, entry_price: float, sl_price: float, regime: int, base_risk_percent: float) -> float:
    """
    Рассчитывает размер позиции на основе режима рынка с проверкой на минимальную стоимость ордера $20.
    """
    global current_balance, MAX_RISK_MODIFIER

    BINANCE_MIN_NOTIONAL_USD = 20.0
    
    if entry_price <= 0 or sl_price <= 0 or entry_price == sl_price:
        return 0.0

    config = REGIME_CONFIG.get(regime, {})
    risk_modifier = config.get('risk_modifier', 1.0)
    adjusted_risk_percent = min(base_risk_percent * risk_modifier, base_risk_percent * MAX_RISK_MODIFIER)
    risk_capital_usd = current_balance * (adjusted_risk_percent / 100.0)
    
    risk_per_coin_usd = abs(entry_price - sl_price)
    if risk_per_coin_usd == 0:
        return 0.0
    
    calculated_quantity = risk_capital_usd / risk_per_coin_usd
    
    # --- БЛОК ПРОВЕРКИ И УВЕЛИЧЕНИЯ ОБЪЕМА ---
    final_quantity = calculated_quantity
    min_notional_with_buffer = BINANCE_MIN_NOTIONAL_USD * 1.01
    calculated_notional = calculated_quantity * entry_price
    
    if calculated_notional < min_notional_with_buffer:
        logging.warning(
            f"[{symbol}] Расчетный объем ($ {calculated_notional:.2f}) ниже минимума биржи (${BINANCE_MIN_NOTIONAL_USD:.2f})."
        )
        if current_balance * LEVERAGE >= min_notional_with_buffer: # Проверяем, хватает ли баланса
            logging.warning(f"[{symbol}] ⚠️ ПРИНУДИТЕЛЬНОЕ УВЕЛИЧЕНИЕ ОБЪЕМА до минимального.")
            final_quantity = min_notional_with_buffer / entry_price
        else:
            logging.error(f"[{symbol}] Вход отменен: баланса недостаточно для минимального ордера.")
            return 0.0
            
    # --- Форматирование ---
    formatted_qty = format_quantity(final_quantity, symbol)
    logging.info(f"[{symbol}] Расчет Qty (Regime): Режим={regime}, Модификатор={risk_modifier:.2f}, Итоговый объем={formatted_qty}")
    return float(formatted_qty) if formatted_qty else 0.0

async def pnl_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update.effective_chat.id):
        return
    
    global current_positions, loop
    
    if not current_positions:
        await update.message.reply_text("📊 Нет открытых позиций.")
        return
    
    # Синхронизация с API для получения актуальных PnL
    try:
        positions_from_api = await loop.run_in_executor(None, get_open_positions_sync)
        if positions_from_api:
            for symbol, api_pos_data in positions_from_api.items():
                if symbol in current_positions and isinstance(current_positions[symbol], PositionManager):
                    manager = current_positions[symbol]
                    # Обновляем PnL в метаданных менеджера
                    manager.state.meta['unrealized_pnl'] = float(api_pos_data.get('unrealized_pnl', 0))
    except Exception as e:
        logging.error(f"⚠️ Ошибка синхронизации позиций для /pnl: {e}")
    
    message = "<b>📊 PnL по открытым позициям:</b>\n"
    total_pnl = 0.0
    
    # Сортируем позиции по символу для удобства
    sorted_positions = sorted(current_positions.items())

    for symbol, manager in sorted_positions:
        if not isinstance(manager, PositionManager):
            continue
        
        base_info = manager.state
        
        # Получаем детальную информацию о позиции
        try:
            details = await manager.get_detailed_status()
            if 'error' in details:
                message += f"\n--- \n<b>{symbol}</b> - ⚠️ {details['error']}\n"
                continue
        except Exception as e:
            logging.error(f"Ошибка получения статуса для {symbol}: {e}")
            message += f"\n--- \n<b>{symbol}</b> - ⚠️ Ошибка получения статуса\n"
            continue
        
        pnl = details.get('pnl', 0.0)
        total_pnl += pnl
        
        side_icon = "🟢" if base_info.side.upper() == 'LONG' else "🔴"
        pnl_sign = "+" if pnl >= 0 else ""
        
        message += (
            f"\n--- \n"
            f"<b>{symbol}</b> | {base_info.side.upper()} {side_icon}\n"
            f"  - <b>PNL: {pnl_sign}{pnl:.2f} USDT</b> ({details.get('pnl_percent', 0):+.2f}%)\n"
            f"  - <b>Вход:</b> {base_info.entry_price:.4f}, <b>Объём:</b> {base_info.current_quantity}\n"
            f"  - <b>Статус:</b> <i>{details.get('status_str', 'N/A')}</i>\n"
        )
        if details.get('current_sl'):
            be_text = " (в б/у)" if details.get('is_breakeven') else ""
            message += f"  - <b>SL:</b> {details['current_sl']:.4f}{be_text}\n"
        if details.get('tp_status'):
            message += f"  - <b>Цели:</b> {details['tp_status']}\n"
        
        message += f"  - <b>R:R (тек.):</b> {details.get('current_rr', 0):.1f}:1 | <b>В сделке:</b> {details.get('age_str', 'N/A')}\n"

    pnl_sign_total = "+" if total_pnl >= 0 else ""
    message += f"\n---\n<b>Общий PNL ({len(current_positions)} поз.): {pnl_sign_total}{total_pnl:.2f} USDT</b>"
    
    await update.message.reply_html(message)

async def update_global_market_direction(force_update: bool = False):
    """
    Определяет ГЛАВНОЕ направление рынка по BTC, кэширует и удерживает его для стабильности.
    """
    global GLOBAL_MARKET_DIRECTION, GLOBAL_DIRECTION_LAST_UPDATE
    
    now = time.time()
    # Удерживаем текущее направление минимум MIN_DIRECTION_HOLD_SEC секунд
    if not force_update and (now - GLOBAL_DIRECTION_LAST_UPDATE) < MIN_DIRECTION_HOLD_SEC:
        return GLOBAL_MARKET_DIRECTION

    logging.warning(">>> ОБНОВЛЕНИЕ ГЛОБАЛЬНОГО НАПРАВЛЕНИЯ РЫНКА (по BTC)...")
    
    df_btc_15m = market_data_store.get('BTCUSDT', {}).get(ALLOWED_INTERVALS['15m'])
    
    if df_btc_15m is None or len(df_btc_15m) < 60:
        logging.error("Недостаточно данных по BTC для определения глобального направления!")
        return GLOBAL_MARKET_DIRECTION # Возвращаем старое значение в случае ошибки

    new_direction = await get_authoritative_market_state('BTCUSDT', df_btc_15m)
    
    if new_direction != GLOBAL_MARKET_DIRECTION:
        logging.critical(f"🔥🔥🔥 СМЕНА ГЛОБАЛЬНОГО НАПРАВЛЕНИЯ РЫНКА: {GLOBAL_MARKET_DIRECTION} -> {new_direction} 🔥🔥🔥")
    
    GLOBAL_MARKET_DIRECTION = new_direction
    GLOBAL_DIRECTION_LAST_UPDATE = now
    
    return GLOBAL_MARKET_DIRECTION

async def get_deepseek_verdict_async(symbol: str, trade_type: str, market_env: str) -> Dict[str, Any]:
    """Адаптивный ИИ-вердикт с новыми инструкциями."""
    global DEEPSEEK_API_URL, market_data_store, ALLOWED_INTERVALS, AI_REQUEST_SEMAPHORE

    default_response = {"confidence": 0, "explanation": "Проверка ИИ не удалась"}

    df_short = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS[ML_MODEL_TF_SHORT])
    df_medium = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS[ML_MODEL_TF_MEDIUM])
    df_long = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS[ML_MODEL_TF_LONG])
    if any(df is None or len(df) < 50 for df in [df_short, df_medium, df_long]):
        return default_response

    rsi_short_val = rsi(df_short['close'], 14).iloc[-1]
    macd_medium_val = MACD(df_medium['close']).macd_diff().iloc[-1]
    adx_long_val = adx(df_long['high'], df_long['low'], df_long['close'], 14).iloc[-1]
    volume_analysis = await get_volume_analysis(df_short)
    btc_context = await get_btc_context()
    
    try:
        # 1. Импортируем правильный класс
        from ta.momentum import StochasticOscillator
        
        # 2. Создаем экземпляр осциллятора
        stoch_indicator = StochasticOscillator(
            high=df_short['high'],
            low=df_short['low'],
            close=df_short['close'],
            window=14,
            smooth_window=3 # Этот параметр отвечает за сглаживание
        )
        
        # 3. Получаем %K и %D (сигнальная линия) отдельно
        stoch_k = stoch_indicator.stoch().iloc[-1]
        stoch_d = stoch_indicator.stoch_signal().iloc[-1]
        
        stoch_context = format_stoch_context(stoch_k, stoch_d)
    except Exception as e:
        logging.error(f"Ошибка расчета стохастика для {symbol}: {e}")
        stoch_context = "ошибка расчета"

    prompt = (
        f"Вы — трейдинг-эксперт. Оцените сигнал {trade_type.upper()} для {symbol} в среде {market_env.upper()}.\n\n"
        f"**АНАЛИТИЧЕСКАЯ СВОДКА:**\n"
        f"- Сила Тренда (ADX 1h): {format_adx_context(adx_long_val)}.\n"
        f"- Моментум (RSI 5m): {format_rsi_context(rsi_short_val)}.\n"
        f"- Моментум (MACD 15m): {format_macd_context(macd_medium_val)}.\n"
        f"- Моментум (Stochastic 5m): {stoch_context}.\n"
        f"- Поддержка Объемом: {'Есть всплеск' if volume_analysis['spike'] else 'Отсутствует'}, тренд OBV: {volume_analysis['obv_trend']}.\n"
        f"- Общий Рыночный Фон (BTC): Тренд {btc_context['trend']}.\n\n"
        f"**ЗАДАЧА:**\n"
        f"На основе этой сводки дай **общую оценку уверенности** в успехе сделки от 0 до 100. "
        f"Кратко поясни свое решение.\n\n"
        f"**ПРАВИЛА ОЦЕНКИ:**\n"
        f"1.  **Приоритет тренда:** Если ADX > 25, это сильный бычий сигнал для оценки. Небольшая перекупленность/перепроданность по RSI в таком случае — это нормально и не должно сильно снижать уверенность.\n"
        f"2.  **СИГНАЛ ПРОТИВ ТРЕНДА:** Если сигнал идет против сильного тренда (напр., LONG при падающем рынке), уверенность должна быть очень низкой (меньше 40).\n"
        f"3.  **СЛАБЫЙ РЫНОК:** Если ADX < 20 (флэт), любой сигнал является рискованным. Уверенность не должна быть высокой (макс. 60-65), если нет других сильных подтверждений (например, паттернов).\n\n"
        f"**Формат вывода (JSON):** {{\"confidence\": int, \"explanation\": string}}"
    )

    loop = asyncio.get_running_loop()
    async with AI_REQUEST_SEMAPHORE:
        logging.info(f"[{symbol}] ---> ОТПРАВКА АДАПТИВНОГО ЗАПРОСА НА AI-ВЕРДИКТ")
        response_data = await loop.run_in_executor(None, partial(sync_deepseek_request, prompt, DEEPSEEK_API_URL, True))

    if not response_data or 'choices' not in response_data:
        return default_response

    try:
        content = response_data['choices'][0]['message']['content']
        ai_data = json.loads(content)
        if 'confidence' not in ai_data or 'explanation' not in ai_data:
            raise ValueError("Неверный формат ответа ИИ")
        logging.info(f"[{symbol}] Вердикт DeepSeek ИИ (адаптивный): {ai_data}")
        return ai_data
    except Exception as e:
        logging.error(f"[{symbol}] Ошибка парсинга вердикта ИИ: {e}. Ответ: {response_data}")
        return default_response

async def await_filled_avg_price(symbol: str, order_id: Optional[int], fallback_price: float, max_wait_sec: int = 10) -> float:
    """
    Надежно ожидает исполнения ордера и возвращает среднюю цену исполнения.
    Опрашивает get_order, а затем account_trades для точности.
    """
    if not order_id:
        logging.error(f"[{symbol}] await_filled_avg_price: Передан невалидный order_id ({order_id})")
        return fallback_price # Возвращаем fallback, если ID ордера нет

    start_time = time.time()
    log_prefix = f"[{symbol}] [Order {order_id}]"

    while time.time() - start_time < max_wait_sec:
        try:
            # Сначала проверяем статус ордера через get_order
            od = await client.futures_get_order(symbol=symbol, orderId=int(order_id))

            if od and od.get("status") == "FILLED":
                avg_price_order = float(od.get("avgPrice", 0))
                filled_qty_order = float(od.get("executedQty", 0))
                logging.info(f"{log_prefix} Статус FILLED получен через get_order. AvgPrice={avg_price_order}, Qty={filled_qty_order}")

                # Если avgPrice > 0, используем его
                if avg_price_order > 0:
                    return avg_price_order

                # Если avgPrice == 0 (редкий случай), пытаемся получить цену через трейды
                logging.warning(f"{log_prefix} avgPrice из get_order равен 0. Запрашиваю трейды...")
                trades = await client.futures_account_trades(symbol=symbol, orderId=int(order_id), limit=10) # Увеличим лимит на всякий случай
                if trades:
                    total_qty = sum(float(t.get("qty", 0)) for t in trades)
                    total_quote_qty = sum(float(t.get("quoteQty", 0)) for t in trades)
                    if total_qty > 0:
                        avg_price_trades = total_quote_qty / total_qty
                        logging.info(f"{log_prefix} Цена исполнения получена из трейдов: {avg_price_trades:.5f}")
                        return avg_price_trades
                    else:
                         logging.error(f"{log_prefix} Трейды найдены, но суммарный объем равен 0.")
                         return fallback_price # Не удалось рассчитать
                else:
                     logging.error(f"{log_prefix} Ордер FILLED, avgPrice=0, но трейды не найдены!")
                     return fallback_price # Не удалось найти цену

            elif od and od.get("status") in ("CANCELED", "EXPIRED", "REJECTED"):
                logging.error(f"{log_prefix} Ордер получил финальный статус {od.get('status')}, исполнение не ожидается.")
                return 0.0 # Возвращаем 0, чтобы показать, что исполнения не было

        except BinanceAPIException as e:
            # Игнорируем ошибку "Order does not exist", если она возникает сразу после отправки
            if e.code == -2013 and (time.time() - start_time < 2):
                 logging.info(f"{log_prefix} Ордер еще не появился в API, ожидание...")
            else:
                 logging.error(f"{log_prefix} API Ошибка при проверке статуса: {e}")
                 # При других API ошибках прекращаем ожидание
                 return fallback_price
        except Exception as e:
            logging.error(f"{log_prefix} Неизвестная ошибка при проверке статуса: {e}", exc_info=True)
            # При неизвестных ошибках прекращаем ожидание
            return fallback_price

        # Небольшая пауза перед следующей проверкой
        await asyncio.sleep(0.3)

    logging.error(f"{log_prefix} Превышено время ожидания ({max_wait_sec} сек) для получения статуса FILLED. Используется fallback цена.")
    return fallback_price

async def is_consolidating_deepseek_async(symbol: str) -> bool:
    return False

async def get_sl_tp_from_ai_async(symbol: str, side: str, entry_price: float) -> Dict[str, Optional[float]]:
    """
    ФИНАЛЬНАЯ ВЕРСИЯ: Запрашивает у AI структурные уровни для SL/TP с учетом волатильности (ATR).
    """
    df_short = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS[ML_MODEL_TF_SHORT])
    if df_short is None or len(df_short) < 20:
        logging.warning(f"[{symbol}] Недостаточно данных для get_sl_tp_from_ai_async.")
        return {"stop_loss": None, "take_profit_1": None, "take_profit_2": None}

    # --- Расчет ATR для контекста AI и для фолбэка ---
    try:
        atr_val = _atr(df_short, 14)
        if atr_val <= 0: raise ValueError("Невалидный ATR")
    except Exception as e:
        logging.error(f"[{symbol}] Ошибка расчета ATR: {e}. Расчет SL/TP невозможен.")
        return {"stop_loss": None, "take_profit_1": None, "take_profit_2": None}
    
    # Расчет фолбэка на случай полной ошибки AI
    sl_atr = entry_price - (atr_val * DEFAULT_ATR_MULTIPLIER) if side == "LONG" else entry_price + (atr_val * DEFAULT_ATR_MULTIPLIER)
    tp1_atr = entry_price + (atr_val * DEFAULT_ATR_MULTIPLIER * PARTIAL_TP_RR) if side == "LONG" else entry_price - (atr_val * DEFAULT_ATR_MULTIPLIER * PARTIAL_TP_RR)
    fallback_sltp = {"stop_loss": sl_atr, "take_profit_1": tp1_atr, "take_profit_2": None}

    # --- Улучшенный промпт для AI ---
    recent_data_str = df_short.tail(20)[['high', 'low', 'close']].to_string()
    prompt = (
        f"Вы — профессиональный трейдер. Ваша задача — определить самый надежный Stop Loss и Take Profit для сделки.\n\n"
        f"**ДАННЫЕ СДЕЛКИ:**\n"
        f"- Инструмент: {symbol}\n"
        f"- Направление: {side.upper()}\n"
        f"- Цена входа: {entry_price:.5f}\n\n"
        f"**КОНТЕКСТ РЫНКА:**\n"
        f"- Текущая волатильность (1x ATR): {atr_val:.5f}\n"
        f"- Данные последних 20 свечей:\n{recent_data_str}\n\n"
        f"**ИНСТРУКЦИИ ДЛЯ РАСЧЕТА:**\n"
        f"1.  **Stop Loss:** Найди самый очевидный и надежный структурный уровень (локальный минимум для LONG, максимум для SHORT) за последние 20 свечей. Твой SL должен быть ЗА этим уровнем. **Критически важно:** дистанция от цены входа до твоего SL должна быть НЕ МЕНЬШЕ, чем {MIN_ATR_MULTIPLIER} * ATR ({MIN_ATR_MULTIPLIER * atr_val:.5f}). Выбери уровень, который удовлетворяет обоим условиям.\n"
        f"2.  **Take Profit:** Рассчитай две цели. TP1 с соотношением Риск/Прибыль {PARTIAL_TP_RR}:1. TP2 с соотношением 3:1.\n"
        f"3.  **Формат ответа (строго JSON):** {{\"stop_loss\": float, \"take_profit_1\": float, \"take_profit_2\": float, \"justification\": \"Краткое объяснение, почему выбран именно этот SL.\"}}"
    )

    loop = asyncio.get_running_loop()
    logging.info(f"[{symbol}] ---> ОТПРАВКА УЛУЧШЕННОГО ЗАПРОСА В AI ДЛЯ РАСЧЕТА SL/TP...")
    response_data = await loop.run_in_executor(None, partial(sync_deepseek_request, prompt, DEEPSEEK_API_URL, True))

    if not response_data or 'choices' not in response_data:
        logging.warning(f"[{symbol}] AI не вернул SL/TP. Используется ATR-фолбэк.")
        return fallback_sltp

    try:
        content = response_data['choices'][0]['message']['content']
        sltp_data = json.loads(content)
        sl = float(sltp_data["stop_loss"])
        tp1 = float(sltp_data["take_profit_1"])
        tp2 = float(sltp_data["take_profit_2"])

        # Валидация ответа от AI
        if (side == "LONG" and sl >= entry_price) or (side == "SHORT" and sl <= entry_price):
            raise ValueError("Нелогичный SL от AI (находится не с той стороны от цены входа)")
        
        risk = abs(entry_price - sl)
        if risk < (MIN_ATR_MULTIPLIER * atr_val * 0.9): # Доп. проверка с 10% погрешностью
             raise ValueError(f"Слишком близкий SL от AI ({risk:.5f} < {MIN_ATR_MULTIPLIER} * ATR)")

        logging.warning(f"[{symbol}] AI предложил SL/TP: SL={sl:.5f}, TP1={tp1:.5f}, TP2={tp2:.5f}. Обоснование: {sltp_data.get('justification', 'N/A')}")
        return {"stop_loss": sl, "take_profit_1": tp1, "take_profit_2": tp2}
    except Exception as e:
        logging.error(f"[{symbol}] Ошибка парсинга SL/TP от AI: {e}. Используется ATR-фолбэк.")
        return fallback_sltp

async def btc_filter_for_ranging_market(symbol: str, trade_type: str) -> tuple[bool, str]:
    """
    Флэт-фильтр BTC: разрешаем при SIDEWAYS/WEAK состояниях.
    Блокируем только явный 2-свечный импульс против входа на BTC 1h (>=~0.6*ATR).
    """
    try:
        btc_df = market_data_store.get("BTCUSDT", {}).get(ALLOWED_INTERVALS.get("1h"))
        if btc_df is None or len(btc_df) < 60:
            return True, "BTC 1h insufficient -> allow"
        
        state, _ = await get_btc_trend_state('1h') # Используем вашу функцию
        
        if state in ("SIDEWAYS", "WEAK_UP", "WEAK_DOWN"):
            return True, f"BTC {state} -> allow"

        # Проверка на 2-свечный импульс
        c1_o, c1_c = float(btc_df["open"].iloc[-2]), float(btc_df["close"].iloc[-2])
        c2_o, c2_c = float(btc_df["open"].iloc[-1]), float(btc_df["close"].iloc[-1])
        
        atr = float(ta.volatility.average_true_range(
            btc_df["high"], btc_df["low"], btc_df["close"], window=14
        ).iloc[-1])
        
        if atr <= 0:
            return True, "ATR<=0 -> allow"

        def big(c_o, c_c) -> bool:
            return abs(c_c - c_o) >= 0.6 * atr

        t = trade_type.lower()
        if t == "long":
            if (c1_c < c1_o and big(c1_o, c1_c)) and (c2_c < c2_o and big(c2_o, c2_c)):
                return False, "Blocked LONG: BTC strong 2-candle down impulse"
        else: # short
            if (c1_c > c1_o and big(c1_o, c1_c)) and (c2_c > c2_o and big(c2_o, c2_c)):
                return False, "Blocked SHORT: BTC strong 2-candle up impulse"
                
        return True, f"BTC {state} -> allow"
        
    except Exception as e:
        logging.error(f"btc_filter_for_ranging_market error: {e}", exc_info=True)
        return True, "BTC ranging filter error -> allow"

async def toggle_portfolio_reversal_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Включает/выключает режим однонаправленной торговли (реверс портфеля)."""
    if not is_authorized(update.effective_chat.id): return
    global ENABLE_PORTFOLIO_REVERSAL
    ENABLE_PORTFOLIO_REVERSAL = not ENABLE_PORTFOLIO_REVERSAL
    status = 'ВКЛЮЧЕН' if ENABLE_PORTFOLIO_REVERSAL else 'ВЫКЛЮЧЕН'
    await update.message.reply_html(f"🚨 Режим реверсирования всего портфеля теперь: <b>{status}</b>.")

# coding: utf-8
# Предполагается, что у вас уже есть все необходимые импорты,
# включая `asyncio`, `logging`, `ta`, `html` и вашу функцию `send_telegram_message`.
# Также предполагается, что функции `calculate_chandelier_exit_stop`, `close_position_emergency`,
# `cancel_single_order_async`, `place_order_async`, `format_price` и `save_state_async`
# уже определены в вашем коде.


async def handle_position_reversal(symbol: str, new_trade_type: str) -> bool:
    """
    Проверяет и закрывает противоположную позицию, корректно работая с объектами PositionManager.
    """
    global current_positions
    
    if symbol in current_positions:
        manager = current_positions.get(symbol)
        if not isinstance(manager, PositionManager):
            return False

        # --- ИСПРАВЛЕНИЕ 1: Доступ к атрибуту напрямую, а не через .get() ---
        # Было: existing_side = manager.get('side', '').lower()
        existing_side = manager.state.side.lower()
        # --- КОНЕЦ ИСПРАВЛЕНИЯ 1 ---
        
        is_opposite = (new_trade_type.lower() == 'long' and existing_side == 'short') or \
                      (new_trade_type.lower() == 'short' and existing_side == 'long')
                      
        if is_opposite:
            logging.warning(
                f"🚨 [{symbol}] ОБНАРУЖЕН СИГНАЛ РЕВЕРСА! "
                f"Новый сигнал: {new_trade_type.upper()}, открыта позиция: {existing_side.upper()}. "
                f"Закрываю старую позицию..."
            )
            
            # --- ИСПРАВЛЕНИЕ 2: Преобразуем объект в словарь перед передачей в функцию закрытия ---
            # Функция close_position_emergency ожидает словарь, а не объект.
            # Было: await close_position_emergency(symbol, manager, ...)
            pos_data_for_closing = manager.get_state_dict()
            await close_position_emergency(
                symbol, 
                pos_data_for_closing, 
                reason=f"Реверс позиции из-за нового сигнала на {new_trade_type.upper()}"
            )
            # --- КОНЕЦ ИСПРАВЛЕНИЯ 2 ---
            return True # Сообщаем, что позиция была закрыта
            
    return False


async def is_trade_setup_valid(symbol: str, trade_type: str) -> Tuple[bool, str]:
    """
    Проверяет качество сетапа по нескольким факторам (объем, волатильность).
    Возвращает (True/False, "причина").
    """
    # 1. Проверка поддержки объемом
    df_short = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS[ML_MODEL_TF_SHORT])
    if df_short is None or len(df_short) < 21:
        return False, "Недостаточно данных для анализа объема."
        
    avg_volume = df_short['volume'].rolling(window=20).mean().iloc[-2]
    last_volume = df_short['volume'].iloc[-1]
    
    if last_volume < avg_volume * 0.8: # Объем ниже среднего на 20%
        return False, f"Низкий объем ({last_volume:.0f} < {avg_volume:.0f}). Сигнал не подтвержден."

    # 2. Проверка на аномальную волатильность (по ATR)
    atr_val = average_true_range(df_short['high'], df_short['low'], df_short['close'], 14).iloc[-1]
    last_close = df_short['close'].iloc[-1]
    natr_percent = (atr_val / last_close) * 100 if last_close > 0 else 0
    
    # Слишком высокая волатильность (свеча > 5% от цены) - рискованно
    if natr_percent > 5.0:
        return False, f"Аномальная волатильность (NATR={natr_percent:.1f}%). Вход слишком рискованный."

    # 3. (Опционально) Проверка на дивергенцию - более сложная, но мощная техника
    
    return True, "Сетап прошел проверку качества."

def get_symbol_price_step(symbol: str) -> Optional[float]:
    """Извлекает шаг цены (tickSize) из кэша биржевой информации."""
    global exchange_info_cache
    try:
        symbol_info = exchange_info_cache.get(symbol)
        if symbol_info and 'tickSize' in symbol_info:
            return float(symbol_info['tickSize'])
        # Если в основной информации нет, ищем в фильтрах (старая структура)
        filters = symbol_info.get('filters', [])
        price_filter = next((f for f in filters if f.get('filterType') == 'PRICE_FILTER'), None)
        if price_filter and 'tickSize' in price_filter:
            return float(price_filter['tickSize'])
        return None
    except (ValueError, KeyError, TypeError):
        logging.warning(f"[{symbol}] Не удалось получить tickSize из exchange_info_cache.")
        return None

def normalize_sltp_for_side(
    side: str,
    entry: float,
    sl: float,
    tp: float,
    *,
    min_rr: float = 1.2,
    price_step: Optional[float] = None
) -> tuple[float, float, list[str]]:
    """
    Гарантирует корректную геометрию брекетов:
    - LONG: SL < entry < TP и TP >= entry + min_rr * (entry - SL)
    - SHORT: TP < entry < SL и TP <= entry - min_rr * (SL - entry)
    Возвращает (sl_fixed, tp_fixed, warnings)
    """
    notes = []
    # Определяем сторону для ордера ('BUY' или 'SELL')
    order_side = side.upper() if side.upper() in ["BUY", "SELL"] else ("BUY" if side.upper() == "LONG" else "SELL")

    risk = abs(entry - sl) if sl is not None else None

    def rnd(x: float) -> float:
        if price_step and price_step > 0:
            k = round(x / price_step)
            return float(f"{k * price_step:.8f}")
        return float(f"{x:.8f}")

    if order_side == "BUY": # Для LONG сделок
        if sl is None or sl >= entry:
            # Исправляем SL, чтобы он был ниже входа
            sl = entry - (risk if risk else max(entry * 0.002, 1e-6))
            notes.append("Fixed SL for LONG below entry")
        
        risk_dist = entry - sl
        min_tp = entry + max(min_rr * risk_dist, risk_dist) # TP должен быть как минимум 1R, но не меньше min_rr
        
        if tp is None or tp <= entry or tp < min_tp:
            tp = min_tp
            notes.append(f"Fixed TP for LONG above entry with min RR >={min_rr}")

    else:  # SELL, для SHORT сделок
        if sl is None or sl <= entry:
            # Исправляем SL, чтобы он был выше входа
            sl = entry + (risk if risk else max(entry * 0.002, 1e-6))
            notes.append("Fixed SL for SHORT above entry")

        risk_dist = sl - entry
        max_tp = entry - max(min_rr * risk_dist, risk_dist)

        if tp is None or tp >= entry or tp > max_tp:
            tp = max_tp
            notes.append(f"Fixed TP for SHORT below entry with min RR >={min_rr}")
            
    return rnd(sl), rnd(tp), notes

async def set_max_positions_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Устанавливает максимальное количество одновременных позиций."""
    if not is_authorized(update.effective_chat.id): return
    global MAX_OPEN_POSITIONS
    
    if not context.args:
        await update.message.reply_text(f"Текущий лимит позиций: {MAX_OPEN_POSITIONS}\nИспользование: /set_max_positions <число>")
        return
        
    try:
        new_limit = int(context.args[0])
        if not 1 <= new_limit <= 10:
            raise ValueError("Лимит должен быть в диапазоне от 1 до 10.")
            
        MAX_OPEN_POSITIONS = new_limit
        await update.message.reply_html(f"✅ Новый лимит одновременных позиций: <b>{MAX_OPEN_POSITIONS}</b>.")
        
    except (ValueError, IndexError) as e:
        await update.message.reply_text(f"Ошибка ввода: {e}\nПример: /set_max_positions 5")

def btc_alt_trend_momentum_gate(symbol: str, trade_type: str) -> Tuple[bool, str]:
    """
    ГИБКАЯ ВЕРСЯ v2. Считает WEAK_UP/WEAK_DOWN как ту же сторону тренда.
    Добавлен подробный лог, чтобы видеть диагноз по таймфреймам.
    """
    def trend_state(df, fast=21, slow=50, adx_thr=25):
        try:
            ef = ta.trend.ema_indicator(df['close'], window=fast).iloc[-1]
            es = ta.trend.ema_indicator(df['close'], window=slow).iloc[-1]
            a  = ta.trend.adx(df['high'], df['low'], df['close'], window=14).iloc[-1]
            if a >= adx_thr:
                if ef > es: return 'UP'
                if ef < es: return 'DOWN'
            if ef > es: return 'WEAK_UP'
            if ef < es: return 'WEAK_DOWN'
            return 'SIDE'
        except Exception:
            return 'UNKNOWN'

    def is_up(state: str) -> bool:
        return state in ('UP', 'WEAK_UP')

    def is_down(state: str) -> bool:
        return state in ('DOWN', 'WEAK_DOWN')

    # 1) BTC контекст (1h обязательно, 4h — усиление)
    btc_1h = market_data_store.get('BTCUSDT', {}).get(ALLOWED_INTERVALS.get('1h'))
    btc_4h = market_data_store.get('BTCUSDT', {}).get(ALLOWED_INTERVALS.get('4h'))

    if btc_1h is None or len(btc_1h) < 60:
        return False, "Недостаточно данных BTC (1h) для фильтра."

    st_btc_1h = trend_state(btc_1h)
    
    if trade_type == 'long' and is_down(st_btc_1h):
        return False, f"ЛОНГ запрещён: BTC тренд на 1ч указывает вниз ({st_btc_1h})."
    if trade_type == 'short' and is_up(st_btc_1h):
        return False, f"ШОРТ запрещён: BTC тренд на 1ч указывает вверх ({st_btc_1h})."

    if btc_4h is not None and len(btc_4h) >= 60:
        st_btc_4h = trend_state(btc_4h)
        if trade_type == 'long' and st_btc_4h == 'DOWN':
            return False, "ЛОНГ запрещён: Глобальный тренд BTC на 4ч направлен строго вниз."
        if trade_type == 'short' and st_btc_4h == 'UP':
            return False, "ШОРТ запрещён: Глобальный тренд BTC на 4ч направлен строго вверх."

    # 2) ALT тренд (15m и 1h должны совпадать по направлению)
    alt_15m = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS.get('15m'))
    alt_1h  = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS.get('1h'))
    
    if any(df is None or len(df) < 60 for df in [alt_15m, alt_1h]):
        return False, f"Недостаточно данных {symbol} (15m/1h) для тренда."
        
    st_alt_15m = trend_state(alt_15m)
    st_alt_1h  = trend_state(alt_1h)

    if trade_type == 'long' and not (is_up(st_alt_15m) and is_up(st_alt_1h)):
        return False, f"{symbol} тренд не подтверждён для LONG на 15m/1h (15m={st_alt_15m}, 1h={st_alt_1h})."
    if trade_type == 'short' and not (is_down(st_alt_15m) and is_down(st_alt_1h)):
        return False, f"{symbol} тренд не подтверждён для SHORT на 15m/1h (15m={st_alt_15m}, 1h={st_alt_1h})."

    return True, "BTC/ALT тренд согласован."

async def enforce_hard_bans(symbol: str, trade_type: str) -> Tuple[bool, str]:
    """
    Жёсткие отказы с адаптивным порогом ADX и настраиваемой ликвидностью.
    """
    # 1. Fear & Greed (без изменений)
    ok_sentiment, reason_sentiment = await get_market_sentiment_filter(symbol, trade_type)
    if not ok_sentiment:
        return False, reason_sentiment

    # 2. ADX экстремы по BTC на 1h (теперь с адаптивным порогом)
    btc_1h = market_data_store.get('BTCUSDT', {}).get(ALLOWED_INTERVALS.get('1h'))
    if btc_1h is None or len(btc_1h) < 50:
        return False, "Нет достаточных данных BTC (1h) для оценки ADX."
    try:
        # --- ✅ АДАПТИВНАЯ ЛОГИКА ---
        adx_series = ta.trend.adx(btc_1h['high'], btc_1h['low'], btc_1h['close'], window=14)
        current_adx = adx_series.iloc[-1]
        
        # Получаем HMM-режим и состояние BTC для более точного порога
        btc_trend_state, _ = await get_btc_trend_state('1h')
        hmm_regime_id = market_regimes.get(symbol, 1) # 1 - нейтральный режим по умолчанию
        
        # Используем вашу продвинутую функцию для расчета порога
        adx_dynamic_threshold = compute_dynamic_adx_threshold(
            adx_series=adx_series,
            btc_state=btc_trend_state,
            regime_id=hmm_regime_id
        )
        # --- КОНЕЦ АДАПТИВНОЙ ЛОГИКИ ---

        if current_adx < adx_dynamic_threshold:
            return False, f"Запрет: ADX ({current_adx:.1f}) ниже динамического порога ({adx_dynamic_threshold:.1f})."
        
        if current_adx > 45: # Эту проверку можно оставить, она защищает от аномалий
            price = btc_1h['close'].iloc[-1]
            direction_ok = (trade_type == 'long' and price > ta.trend.ema_indicator(btc_1h['close'], window=21).iloc[-1]) or \
                           (trade_type == 'short' and price < ta.trend.ema_indicator(btc_1h['close'], window=21).iloc[-1])
            if not direction_ok:
                # ✅ Превращаем в более мягкое предупреждение, а не жёсткий бан, но пока оставим как есть для безопасности
                return False, f"Запрет: ADX экстремальный {current_adx:.1f}, вход только по тренду BTC."
    except Exception as e:
        logging.error(f"Не удалось оценить ADX для жёсткого бана: {e}")
        return False, "Не удалось оценить ADX для жёсткого бана."

    # 3. Ликвидность (можно снизить порог или сделать его адаптивным позже)
    try:
        liq_score_raw = await estimate_market_liquidity(symbol)
        if liq_score_raw is None:
             return False, "Ошибка оценки ликвидности"
        
        # ✅ Снижаем порог liq_score с 2.0 до 1.5, чтобы пропускать больше монет
        if (liq_score_raw / 100000.0) < 1.5: return False, f"Запрет: Недостаточная ликвидность (score={(liq_score_raw/100000.0):.1f})."
        
        ob = await client.futures_order_book(symbol=symbol, limit=20)
        best_bid = float(ob['bids'][0][0])
        best_ask = float(ob['asks'][0][0])
        mid = (best_bid + best_ask) / 2.0
        depth_usd = sum(float(p) * float(q) for p, q in ob['bids'] if float(p) > mid * 0.99) + \
                    sum(float(p) * float(q) for p, q in ob['asks'] if float(p) < mid * 1.01)
        
        if depth_usd < MARKET_DEPTH_THRESHOLD_USD:
            return False, f"Запрет: Низкая локальная глубина рынка (${depth_usd:,.0f})."
    except Exception:
        return False, "Ошибка ликвидностного контроля."
        
    return True, "Жёсткие запреты пройдены."

def get_dynamic_trailing_multiplier(current_rr: float) -> Optional[float]:
    """
    Возвращает множитель для ATR трейлинга в зависимости от текущей прибыли.
    Чем выше R:R, тем меньше (агрессивнее) множитель.
    """
    if current_rr < 1.0:
        # Трейлинг не активен, пока прибыль меньше 1R
        return None 
    elif 1.0 <= current_rr < 2.5:
        # Стандартный, широкий трейлинг в начале движения
        return 2.5 
    elif 2.5 <= current_rr < 4.0:
        # Умеренно-агрессивный, когда прибыль уже ощутима
        return 2.0 
    else: # current_rr >= 4.0
        # Агрессивный, для защиты большой прибыли
        return 1.5

async def calculate_contextual_sl_tp(symbol: str, side: str, entry_price: float) -> Dict[str, Optional[float]]:
    """
    Контекстные уровни с динамическим R:R для тренда.
    """
    regime, det = await get_market_regime(symbol, timeframe_key='1h')
    df_15m = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS.get('15m'))
    if df_15m is None or len(df_15m) < 50:
        return {"stop_loss": None, "take_profit": None}

    def structural_sl(df, side, lookback=25, offset_pct=0.15):
        data = df.iloc[-lookback:-1]
        base = data['low'].min() if side.upper() == 'LONG' else data['high'].max()
        offset_multiplier = (Decimal('1') - Decimal(str(offset_pct/100.0))) if side.upper() == 'LONG' else (Decimal('1') + Decimal(str(offset_pct/100.0)))
        return float(Decimal(str(base)) * offset_multiplier)

    try:
        if regime == 'Ranging':
            sl = structural_sl(df_15m, side, lookback=25, offset_pct=0.15)
            risk = abs(entry_price - sl)
            if risk <= 1e-9: return {"stop_loss": None, "take_profit": None}
            tp = entry_price + 1.5 * risk if side.lower() == 'long' else entry_price - 1.5 * risk
            return {"stop_loss": sl, "take_profit": tp}
        
        # --- ДИНАМИЧЕСКИЙ R:R ДЛЯ ТРЕНДА ---
        adx_val = ta.trend.adx(df_15m['high'], df_15m['low'], df_15m['close'], window=14).iloc[-1]
        if adx_val > 35: # Очень сильный тренд
            rr_trending = 4.0
        elif adx_val > 25: # Уверенный тренд
            rr_trending = 3.0
        else: # Слабый тренд
            rr_trending = 2.0
        logging.info(f"[{symbol}] Динамический R:R: ADX={adx_val:.1f}, выбран R:R={rr_trending}:1")
        # --- КОНЕЦ ДИНАМИЧЕСКОЙ ЛОГИКИ ---

        ema_fast = ta.trend.ema_indicator(df_15m['close'], window=21).iloc[-1]
        ema_slow = ta.trend.ema_indicator(df_15m['close'], window=50).iloc[-1]
        sl_struct = structural_sl(df_15m, side, lookback=30, offset_pct=0.10)
        
        if side.upper() == 'LONG':
            sl_ema = min(ema_fast, ema_slow) * 0.998
            sl = min(sl_struct, sl_ema)
        else:
            sl_ema = max(ema_fast, ema_slow) * 1.002
            sl = max(sl_struct, sl_ema)
            
        risk = abs(entry_price - sl)
        if risk <= 1e-9: return {"stop_loss": None, "take_profit": None}
        tp = entry_price + rr_trending * risk if side.lower() == 'long' else entry_price - rr_trending * risk
        return {"stop_loss": sl, "take_profit": tp}

    except Exception:
        return {"stop_loss": None, "take_profit": None}

def get_market_structure(df: pd.DataFrame, lookback: int = 20) -> str:
    """
    Определяет структуру рынка (Higher Highs/Higher Lows или Lower Lows/Lower Highs).
    
    Args:
        df (pd.DataFrame): DataFrame с данными OHLCV.
        lookback (int): Период для поиска максимумов и минимумов.

    Returns:
        str: 'Uptrend', 'Downtrend', или 'Sideways'.
    """
    if df is None or len(df) < lookback * 2:
        return 'Sideways'

    recent_data = df.iloc[-(lookback * 2):]
    
    # Находим последние 2 максимума и 2 минимума
    high1 = recent_data['high'].tail(lookback).max()
    high2 = recent_data['high'].head(lookback).max()
    
    low1 = recent_data['low'].tail(lookback).min()
    low2 = recent_data['low'].head(lookback).min()

    if high1 > high2 and low1 > low2:
        return 'Uptrend'
    elif high1 < high2 and low1 < low2:
        return 'Downtrend'
    else:
        return 'Sideways'

# --- ЗАМЕНИТЕ ВАШУ СТАРУЮ ФУНКЦИЮ ЭТОЙ ---

async def get_btc_trend_score() -> Tuple[float, Dict[str, Any]]:
    """
    Рассчитывает комплексный скоринг-балл для тренда BTC (v3.2)
    с многоуровневым отказоустойчивым анализом (D1 -> 4H -> 1H).
    """
    details = {}

    # --- Уровень 1: Макро-анализ на Дневном графике (D1) ---
    df_daily = market_data_store.get('BTCUSDT', {}).get(ALLOWED_INTERVALS.get('1d'))
    if df_daily is not None and len(df_daily) >= 200:
        ema_200_daily = ta.trend.ema_indicator(df_daily['close'], window=200).iloc[-1]
        daily_price = df_daily['close'].iloc[-1]
        macro_regime = 'Bullish' if daily_price > ema_200_daily else 'Bearish'
        macro_score = 50.0 if macro_regime == 'Bullish' else -50.0
        details['macro_analysis'] = f"D1 ({macro_regime})"
    else:
        macro_score = 0.0 # Нейтральный счет, если данных нет
        details['macro_analysis'] = "D1 (пропущено)"

    # --- Уровень 2: Тактический анализ на 4-часовом графике (4H) ---
    df_4h = market_data_store.get('BTCUSDT', {}).get(ALLOWED_INTERVALS.get('4h'))
    if df_4h is not None and len(df_4h) >= 50:
        structure_4h = get_market_structure(df_4h)
        ema_slow_4h = ta.trend.ema_indicator(df_4h['close'], window=50).iloc[-1]
        price_4h = df_4h['close'].iloc[-1]
        tactical_score = 0.0
        if price_4h > ema_slow_4h: tactical_score += 25 # Бычий импульс
        if price_4h < ema_slow_4h: tactical_score -= 25 # Медвежий импульс
        if structure_4h == 'Uptrend': tactical_score += 25
        if structure_4h == 'Downtrend': tactical_score -= 25
        details['tactical_analysis'] = f"4H (Score: {tactical_score})"
        
        final_score = macro_score + tactical_score if macro_score != 0.0 else tactical_score * 2
        details['final_score'] = round(final_score, 2)
        logging.info(f"[BTC Trend Score] Расчет выполнен на основе D1+4H. Итог: {final_score:.2f}")
        return final_score, details

    # --- Уровень 3: ЗАПАСНОЙ ВАРИАНТ на 1-часовом графике (1H) ---
    logging.warning("[BTC Trend Score] Недостаточно данных на D1 и 4H. Используется запасной анализ на 1H.")
    df_1h = market_data_store.get('BTCUSDT', {}).get(ALLOWED_INTERVALS.get('1h'))
    if df_1h is None or len(df_1h) < 50:
        logging.critical("[BTC Trend Score] Нет данных даже на 1H. Дальнейший анализ невозможен.")
        return 0.0, {'error': 'Нет данных на 1H'}

    # Упрощенный анализ для 1H
    ema_50_1h = ta.trend.ema_indicator(df_1h['close'], window=50).iloc[-1]
    price_1h = df_1h['close'].iloc[-1]
    adx_1h = ta.trend.adx(df_1h['high'], df_1h['low'], df_1h['close'], window=14).iloc[-1]
    
    fallback_score = 0.0
    if price_1h > ema_50_1h: fallback_score += 50 # Основной балл за направление
    if price_1h < ema_50_1h: fallback_score -= 50
    
    if adx_1h > 25: # Если есть сильный тренд, удваиваем балл
        fallback_score *= 2
        
    final_score = max(-100, min(100, fallback_score)) # Ограничиваем от -100 до 100
    details['final_score'] = round(final_score, 2)
    details['fallback_analysis'] = f"1H (Score: {final_score})"
    logging.info(f"[BTC Trend Score] Расчет выполнен на основе 1H. Итог: {final_score:.2f}")
    return final_score, details

async def get_sltp_from_support_resistance(symbol: str, side: str, entry_price: float) -> Dict[str, Optional[float]]:
    """
    Рассчитывает Stop Loss и Take Profit на основе локальных уровней поддержки/сопротивления
    с заданным процентным отступом.
    """
    # --- НАСТРОЙКИ ---
    LOOKBACK_PERIOD = 50  # Количество свечей для поиска минимумов и максимумов.
    OFFSET_PERCENTAGE = 0.2 # Процент отступа от найденного уровня. 0.2% = безопасный буфер.
                            # Если вы уверены, можете изменить это значение, но 10% использовать не рекомендуется.
    MIN_RR_RATIO = 1.2    # Минимальное соотношение риска к прибыли (1 к 1.2).

    # Используем 15-минутный таймфрейм для поиска более значимых уровней
    tf_key = ML_MODEL_TF_MEDIUM # '15m'
    df = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS.get(tf_key))

    if df is None or len(df) < LOOKBACK_PERIOD:
        logging.warning(f"[{symbol}] Недостаточно данных ({len(df) if df is not None else 0}/{LOOKBACK_PERIOD}) для расчета SL/TP по S/R.")
        return {"final_sl_price": None, "final_tp_price": None}

    try:
        # Берем срез данных, исключая последнюю (текущую) свечу, чтобы уровни были историческими
        recent_data = df.iloc[-LOOKBACK_PERIOD:-1]
        
        highest_high = recent_data['high'].max()
        lowest_low = recent_data['low'].min()

        logging.info(f"[{symbol}] Поиск S/R на {LOOKBACK_PERIOD} свечах: High={highest_high}, Low={lowest_low}")

        stop_loss = None
        take_profit = None
        offset_multiplier = Decimal(str(OFFSET_PERCENTAGE / 100.0))

        if side.upper() == "LONG":
            # SL ставится ниже уровня поддержки (lowest low)
            support_level = Decimal(str(lowest_low))
            stop_loss = float(support_level * (Decimal('1') - offset_multiplier))
            
            # TP ставится чуть ниже уровня сопротивления (highest high)
            resistance_level = Decimal(str(highest_high))
            take_profit = float(resistance_level * (Decimal('1') - offset_multiplier))

            # Проверка на логичность: TP должен быть выше цены входа
            if take_profit <= entry_price:
                 logging.warning(f"[{symbol}] TP ({take_profit}) для LONG ниже цены входа ({entry_price}). Уровень не подходит.")
                 risk_distance = entry_price - stop_loss
                 take_profit = entry_price + (risk_distance * MIN_RR_RATIO)
                 logging.info(f"[{symbol}] Установлен TP по R:R {MIN_RR_RATIO}:1 -> {take_profit}")

        elif side.upper() == "SHORT":
            # SL ставится выше уровня сопротивления (highest high)
            resistance_level = Decimal(str(highest_high))
            stop_loss = float(resistance_level * (Decimal('1') + offset_multiplier))

            # TP ставится чуть выше уровня поддержки (lowest low)
            support_level = Decimal(str(lowest_low))
            take_profit = float(support_level * (Decimal('1') + offset_multiplier))

            # Проверка на логичность: TP должен быть ниже цены входа
            if take_profit >= entry_price:
                 logging.warning(f"[{symbol}] TP ({take_profit}) для SHORT выше цены входа ({entry_price}). Уровень не подходит.")
                 risk_distance = stop_loss - entry_price
                 take_profit = entry_price - (risk_distance * MIN_RR_RATIO)
                 logging.info(f"[{symbol}] Установлен TP по R:R {MIN_RR_RATIO}:1 -> {take_profit}")

        # Финальная проверка соотношения риска к прибыли
        risk_dist = abs(entry_price - stop_loss)
        reward_dist = abs(take_profit - entry_price)
        if risk_dist == 0 or (reward_dist / risk_dist) < MIN_RR_RATIO:
            logging.warning(f"[{symbol}] Итоговое R:R ({reward_dist/risk_dist if risk_dist > 0 else 0 :.2f}) ниже минимума ({MIN_RR_RATIO}). Сделка отменена.")
            return {"final_sl_price": None, "final_tp_price": None}

        logging.warning(f"[{symbol}] Расчет SL/TP по S/R ({side}): SL={stop_loss:.5f}, TP={take_profit:.5f}")
        return {"final_sl_price": stop_loss, "final_tp_price": take_profit}

    except Exception as e:
        logging.error(f"[{symbol}] Ошибка в расчете SL/TP по S/R: {e}", exc_info=True)
        return {"final_sl_price": None, "final_tp_price": None}

async def get_dynamic_tp_levels(symbol: str, entry_price: float, sl_price: float, market_env: str, side: str) -> Dict[str, Optional[float]]:
    """
    Рассчитывает динамические уровни Take Profit на основе силы рынка.
    """
    logging.info(f"[{symbol}] Расчет динамических TP для среды '{market_env}'...")
    risk_distance = abs(entry_price - sl_price)
    if risk_distance == 0:
        return {"take_profit_1": None, "take_profit_2": None}

    reward_multiplier_1 = 1.5  # Базовый R:R для TP1
    reward_multiplier_2 = 3.0  # Базовый R:R для TP2

    df_long = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS[ML_MODEL_TF_LONG])
    
    if df_long is not None and not df_long.empty:
        try:
            adx_val = adx(df_long['high'], df_long['low'], df_long['close'], 14).iloc[-1]
            
            # Если тренд сильный, ставим цели дальше
            if market_env == 'trending' and adx_val > 28:
                reward_multiplier_1 = 2.5
                reward_multiplier_2 = 5.0
                logging.info(f"[{symbol}] Сильный тренд (ADX={adx_val:.1f}). Устанавливаем агрессивные цели.")
            
            # Если рынок во флэте, ставим цели ближе
            elif market_env == 'ranging' and adx_val < 20:
                reward_multiplier_1 = 1.5
                reward_multiplier_2 = 2.0
                logging.info(f"[{symbol}] Флэт (ADX={adx_val:.1f}). Устанавливаем консервативные цели.")
        except Exception as e:
            logging.warning(f"[{symbol}] Не удалось рассчитать ADX для динамического TP: {e}")

    # ✅ ИСПРАВЛЕНИЕ: Вместо 'pos_data' используется параметр 'side'
    side_modifier = 1 if side.lower() == 'long' else -1
            
    tp1 = entry_price + (risk_distance * reward_multiplier_1 * side_modifier)
    tp2 = entry_price + (risk_distance * reward_multiplier_2 * side_modifier)
    
    return {"take_profit_1": tp1, "take_profit_2": tp2}

async def place_regime_based_trade_async(symbol: str, side_str: str, qty_to_trade: float, initial_entry_price: float, initial_sl_price: float, initial_tp_price: float, regime: int):
    """
    Размещает сделку с пересчетом SL/TP ПОСЛЕ получения фактической цены входа.
    Это самая надежная версия для предотвращения ошибок 'Order would immediately trigger'.
    """
    side = 'LONG' if side_str == 'BUY' else 'SHORT'
    logging.warning(f"[{symbol}] [HMM-режим] ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ. Отправка рыночного ордера на {side_str} объемом {qty_to_trade}...")
    
    # Шаг 1: Вход в рынок
    entry_order_info = await place_order_async(symbol=symbol, side=side_str, order_type="MARKET", quantity=qty_to_trade)
    if not entry_order_info or not entry_order_info.get('orderId'):
        logging.error(f"[{symbol}] [HMM-режим] Не удалось разместить рыночный ордер на вход.")
        return

    await log_order_to_json(entry_order_info, "entry_market_hmm")
    
    # Шаг 2: Получение фактической цены входа
    await asyncio.sleep(POST_ENTRY_DELAY_SECONDS)
    actual_entry_price = 0.0
    try:
        trades = await client.futures_account_trades(symbol=symbol, orderId=entry_order_info.get('orderId'), limit=1)
        if trades:
            actual_entry_price = float(trades[0]['price'])
        else: # Fallback, если история трейдов пуста
            filled_order = await client.futures_get_order(symbol=symbol, orderId=entry_order_info.get('orderId'))
            if filled_order and float(filled_order.get('avgPrice', 0)) > 0:
                actual_entry_price = float(filled_order['avgPrice'])

    except Exception as e:
        logging.error(f"[{symbol}] [HMM-режим] Не удалось получить цену исполнения для ордера {entry_order_info.get('orderId')}: {e}")

    if actual_entry_price <= 0:
        await close_position_emergency(symbol, {'side': side, 'amount': qty_to_trade}, "Критическая ошибка: не удалось подтвердить цену входа.")
        return

    logging.info(f"[{symbol}] [HMM-режим] Позиция успешно открыта по средней цене {actual_entry_price:.4f}. Пересчет и установка SL/TP.")
    
    # Шаг 3: Пересчет SL/TP от ФАКТИЧЕСКОЙ цены входа
    # Сохраняем изначально запланированные дистанции до стопа и тейка
    planned_risk_distance = abs(initial_entry_price - initial_sl_price)
    planned_reward_distance = abs(initial_tp_price - initial_entry_price)

    # Применяем эти же дистанции к новой, фактической цене входа
    if side == 'LONG':
        final_sl_price = actual_entry_price - planned_risk_distance
        final_tp_price = actual_entry_price + planned_reward_distance
    else: # SHORT
        final_sl_price = actual_entry_price + planned_risk_distance
        final_tp_price = actual_entry_price - planned_reward_distance

    logging.info(f"[{symbol}] Итоговый план: SL={final_sl_price:.4f}, TP={final_tp_price:.4f} (на основе фактической цены входа {actual_entry_price:.4f})")
    
    # Шаг 4: Сохранение состояния и установка SL/TP
    pos_to_save = {
        'side': side, 'amount': qty_to_trade, 'initial_amount': qty_to_trade,
        'entry_price': actual_entry_price, 'entry_context': f"HMM Regime {regime}",
        'state': 'placing_brackets', 'sl_price': final_sl_price, 'initial_sl_price': final_sl_price,
        'tp1_price': final_tp_price, 'tp2_price': None,
        'entry_regime': regime,
        'strategy_group': 'HMM'
    }
    current_positions[symbol] = pos_to_save

    opposite_side_str = "SELL" if side_str == "BUY" else "BUY"
    
    sl_task = place_order_async(symbol=symbol, side=opposite_side_str, order_type="STOP_MARKET", quantity=qty_to_trade, stopPrice=final_sl_price, reduce_only=True)
    tp_task = place_order_async(symbol=symbol, side=opposite_side_str, order_type="TAKE_PROFIT_MARKET", quantity=qty_to_trade, stopPrice=final_tp_price, reduce_only=True)

    sl_res, tp_res = await asyncio.gather(sl_task, tp_task, return_exceptions=True)
    
    msg_tg = f"✅ Вход по HMM {'Лонг' if side_str == 'BUY' else 'Шорт'} <b>{symbol}</b>\nРежим: {regime}\nЦена: {actual_entry_price:.4f}, Qty: {qty_to_trade}"

    if isinstance(sl_res, dict) and sl_res.get('orderId'):
        current_positions[symbol]['sl_order_id'] = sl_res.get('orderId')
        msg_tg += f"\nSL: {format_price(final_sl_price, symbol)}"
    else:
        error_details = str(sl_res) if sl_res is not None else "Ответ от place_order_async был None"
        logging.error(f"[{symbol}] [HMM-режим] Критическая ошибка установки SL ордера. Закрытие позиции. Ошибка: {error_details}")
        await close_position_emergency(symbol, current_positions[symbol], "Критическая ошибка установки SL ордера.")
        return

    if isinstance(tp_res, dict) and tp_res.get('orderId'):
        current_positions[symbol]['tp1_order_id'] = tp_res.get('orderId')
        msg_tg += f"\nTP: {format_price(final_tp_price, symbol)}"
    else:
        logging.warning(f"[{symbol}] [HMM-режим] Ошибка установки TP ордера: {tp_res}")
        msg_tg += "\n⚠️ ОШИБКА TP!"

    current_positions[symbol]['state'] = 'initial'
    asyncio.create_task(advanced_position_manager(symbol, current_positions[symbol]))
    logging.info(f"[{symbol}] [HMM-режим] Успешный вход. Запущено управление позицией.")
    await save_state_async()
    await send_telegram_message(msg_tg)


# ##################################################################
# ## ЗАДАЧА 1: ПРОДВИНУТОЕ ОПРЕДЕЛЕНИЕ РЫНОЧНЫХ РЕЖИМОВ          ##
# ##################################################################

async def calculate_order_flow_imbalance(symbol: str) -> Tuple[float, str]:
    """
    Order Flow Imbalance (OFI) - показывает давление покупателей/продавцов
    через анализ глубины ордербука.

    Возвращает: (OFI_score от -100 до +100, signal)
    """
    try:
        # Получаем глубину ордербука
        order_book = await client.futures_order_book(symbol=symbol, limit=20)
        
        bids = order_book.get('bids')
        asks = order_book.get('asks')

        if not bids or not asks:
            logging.warning(f"[{symbol}] OFI: Стакан ордеров пуст.")
            return 0.0, 'neutral'

        # 1. Объемный дисбаланс на топ-5 уровнях
        bid_volume_top5 = sum(float(bid[1]) for bid in bids[:5])
        ask_volume_top5 = sum(float(ask[1]) for ask in asks[:5])
        
        total_volume = bid_volume_top5 + ask_volume_top5
        if total_volume == 0:
            return 0.0, 'neutral'
            
        volume_imbalance = ((bid_volume_top5 - ask_volume_top5) / total_volume) * 100

        # 2. Ценовой дисбаланс (анализ спреда)
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid_price = (best_bid + best_ask) / 2
        
        spread_pct = ((best_ask - best_bid) / mid_price) * 100 if mid_price > 0 else 0
        
        # Штраф за широкий спред (низкая ликвидность)
        if spread_pct > 0.08:
            volume_imbalance *= 0.7

        # 3. Глубинный анализ - проверяем "стены"
        avg_bid_size = bid_volume_top5 / 5 if bid_volume_top5 > 0 else 1
        avg_ask_size = ask_volume_top5 / 5 if ask_volume_top5 > 0 else 1
        
        wall_score = 0
        # Ищем крупный ордер (>3x средний размер) в топ-3 уровнях
        for i, bid in enumerate(bids[:3]):
            if float(bid[1]) > avg_bid_size * 3:
                wall_score += 20  # Близкая стена покупателей = бычий сигнал
                break
                
        for i, ask in enumerate(asks[:3]):
            if float(ask[1]) > avg_ask_size * 3:
                wall_score -= 20  # Близкая стена продавцов = медвежий сигнал
                break

        # 4. Итоговый OFI скор
        ofi_score = volume_imbalance + wall_score
        ofi_score = max(-100, min(100, ofi_score))

        # Интерпретация
        if ofi_score > 40:
            signal = 'strong_buy'
        elif ofi_score > 15:
            signal = 'buy'
        elif ofi_score < -40:
            signal = 'strong_sell'
        elif ofi_score < -15:
            signal = 'sell'
        else:
            signal = 'neutral'
            
        logging.info(f"[{symbol}] OFI = {ofi_score:.1f} (Сигнал: {signal})")
        return ofi_score, signal
        
    except Exception as e:
        logging.error(f"[{symbol}] Ошибка при расчете OFI: {e}")
        return 0.0, 'neutral'

async def get_ai_model_verdict_async(symbol: str, side: str, market_env: str) -> Dict:
    """
    Собирает данные, включая СТРУКТУРУ РЫНКА, и получает вердикт от AI.
    --- ВЕРСИЯ V3: Переведена на OpenAI клиент для унификации ---
    """
    global market_data_store, client, DEEPSEEK_API_KEY, DEEPSEEK_API_BASE_URL, AI_REQUEST_SEMAPHORE

    logging.info(f"[{symbol}] Сбор данных для Экспертной AI-модели (LLM)...")
    
    default_response = {"confidence": 0, "justification": "Ошибка: отсутствуют исторические данные."}

    # --- 1. Сбор данных и расчет индикаторов ---
    df_5m = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS['5m'])
    df_15m = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS['15m'])
    df_1h = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS['1h'])
    
    if any(df is None or df.empty for df in [df_5m, df_15m, df_1h]):
        logging.error(f"[{symbol}] Отсутствуют необходимые данные для анализа.")
        return default_response

    try:
        market_structure_1h = get_market_structure(df_1h, lookback=30)
        context_tasks = {
            "derivatives": get_derivatives_sentiment(symbol, client),
            "microstructure": get_market_microstructure_features(symbol, client),
            "garch": forecast_garch_volatility(symbol, market_data_store),
            "btc_trend_strength": get_btc_trend_strength(),
            "btc_correlation": get_correlation_snapshot(symbol, 'BTCUSDT')
        }
        results = await asyncio.gather(*context_tasks.values(), return_exceptions=True)
        gathered_data = dict(zip(context_tasks.keys(), results))

        json_payload = {
            "symbol": symbol, "signal_direction": side.upper(), "market_environment": market_env,
            "trend_indicators": {
                "adx_1h": ta.trend.adx(df_1h['high'], df_1h['low'], df_1h['close'], 14).iloc[-1],
                "market_structure_1h": market_structure_1h,
                "btc_trend_strength": gathered_data.get("btc_trend_strength", "Sideways"),
                "btc_correlation": gathered_data.get("btc_correlation", 0.0)
            },
            "momentum_indicators": {
                "rsi_15m": ta.momentum.rsi(df_15m['close'], 14).iloc[-1],
                "macd_hist_15m": ta.trend.MACD(df_15m['close']).macd_diff().iloc[-1]
            },
            "volatility_indicators": {
                "bollinger_width_1h": ta.volatility.BollingerBands(df_1h['close']).bollinger_wband().iloc[-1],
                "garch_volatility_forecast": gathered_data.get("garch")
            },
            "market_microstructure": {
                "order_book_imbalance": gathered_data.get("microstructure", {}).get("order_book_imbalance"),
                "long_short_ratio": gathered_data.get("derivatives", {}).get("long_short_ratio")
            }
        }
    except Exception as e:
        logging.error(f"[{symbol}] Ошибка при расчете индикаторов для AI: {e}", exc_info=True)
        return {"confidence": 0, "justification": f"Ошибка расчета индикаторов: {e}"}

    # --- 2. Формирование промпта ---
    prompt_template = """
Вы — элитный количественный аналитик (Quant Analyst). Ваша задача — провести всесторонний анализ рыночных данных и вынести вердикт о потенциальной сделке, вернув строго форматированный JSON-объект.
**АНАЛИТИЧЕСКАЯ СВОДКА (JSON):**
{json_data}
**ВАША ЗАДАЧА:**
Проведите пошаговый анализ данных и верните итоговую уверенность в успехе сделки по направлению `signal_direction`.
**ПРАВИЛА ОЦЕНКИ (СИСТЕМА БОНУСОВ И ШТРАФОВ):**
1. **Базовая оценка:** Начни с базовой оценки в 50 баллов.
2. **Анализ Тренда (Главный фактор):**
   * Если тренд сильный (ADX >= 25) И он совпадает с `signal_direction` И совпадает с трендом BTC -> **Добавь +30 баллов.**
   * Если `market_structure_1h` совпадает с `signal_direction` (напр. 'Uptrend' и LONG) -> **Добавь +15 баллов.**
   * Если тренд слабый (ADX < 20), рынок в боковике -> **Вычти -10 баллов.**
   * **Критический штраф:** Если сигнал идет против сильного тренда BTC (например, LONG против 'Strong Down') -> **Вычти -40 баллов.**
3. **Анализ Моментума (Подтверждение):**
   * Если гистограмма MACD сильно растет в сторону сигнала -> **Добавь +10 баллов.**
   * **Штраф за перекупленность/перепроданность:** Если это LONG-сигнал при RSI > 75 ИЛИ SHORT-сигнал при RSI < 25 -> **Вычти -15 баллов.**
4. **Анализ Волатильности (Оценка риска):**
   * Если волатильность высокая (GARCH > 4.0 или BB Width сильно расширяется) И тренд НЕ сильный (ADX < 25) -> **Вычти -10 баллов** (высокий риск).
5. **Анализ Микроструктуры (Дополнительное подтверждение):**
   * Если OBI и L/S Ratio оба подтверждают направление сигнала -> **Добавь +10 баллов.**
**ФОРМАТ ВЫВОДА (СТРОГО JSON):**
На основе твоего анализа верни JSON-объект со следующими ключами:
- `confidence`: Итоговая уверенность в сделке (целое число от 0 до 100).
- `justification`: Краткое (1-2 предложения) обоснование твоего решения, основанное на анализе.
"""
    
    def clean_data(d):
        if isinstance(d, dict): return {k: clean_data(v) for k, v in d.items()}
        elif isinstance(d, list): return [clean_data(i) for i in d]
        elif isinstance(d, (float, np.floating)) and (np.isnan(d) or np.isinf(d)): return None
        elif isinstance(d, (float, np.floating)): return round(d, 5)
        return d

    cleaned_payload = clean_data(json_payload)
    final_prompt = prompt_template.format(json_data=json.dumps(cleaned_payload, indent=2, ensure_ascii=False))
    
    # --- 3. Отправка запроса через OpenAI клиент ---
    try:
        async with AI_REQUEST_SEMAPHORE:
            logging.info(f"[{symbol}] ---> Отправка текстового запроса в AI (через OpenAI клиент)...")
            
            # Создаем клиент так же, как и в Vision-функции
            api_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_BASE_URL)
            
            # Запускаем в отдельном потоке, чтобы не блокировать основной цикл
            response = await asyncio.to_thread(
                api_client.chat.completions.create,
                model="deepseek-chat", # Используем текстовую модель
                messages=[{"role": "user", "content": final_prompt}],
                response_format={"type": "json_object"},
                temperature=0.6,
                max_tokens=1024
            )
        
        content = response.choices[0].message.content
        ai_data = json.loads(content)
        
        if 'confidence' not in ai_data or 'justification' not in ai_data:
            raise ValueError("Ответ AI не содержит ключи 'confidence' или 'justification'")
            
        confidence_score = int(ai_data['confidence'])
        justification_text = str(ai_data['justification'])

        logging.warning(f"🎯 [{symbol}] ВЕРДИКТ ОТ ЭКСПЕРТА AI (Текст): Уверенность = {confidence_score}%. Обоснование: {justification_text}")
        return {"confidence": confidence_score, "justification": justification_text}

    except Exception as e:
        logging.error(f"[{symbol}] Ошибка парсинга или запроса от AI (текст): {e}", exc_info=True)
        return {"confidence": 0, "justification": f"Ошибка AI: {e}"}


async def forecast_garch_volatility(symbol: str, market_data_store: Dict[str, Any], tf_key: str = '1h') -> Optional[float]:
    """
    Прогнозирует волатильность на следующий период с помощью модели GARCH(1,1).
    Возвращает прогнозируемое значение дневной волатильности в процентах.
    """
    logging.info(f"[{symbol}] Запуск расчета прогноза волатильности по GARCH...")
    try:
        from arch import arch_model
        
        df = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS[tf_key])
        if df is None or len(df) < 100:
            logging.warning(f"[{symbol}] Недостаточно данных ({len(df) if df is not None else 0} свечей) для GARCH-модели.")
            return None

        # Расчет логарифмических доходностей, умноженных на 100 для лучшей сходимости модели
        returns = 100 * np.log(df['close']).diff().dropna()
        
        # Если дисперсия почти нулевая (цена не менялась), нет смысла строить модель
        if returns.var() < 1e-8:
            return 0.0

        # Создание и обучение GARCH(1,1) модели
        model = arch_model(returns, p=1, q=1, vol='Garch')
        res = model.fit(disp='off', show_warning=False)

        # Прогноз на 1 шаг вперед
        forecast = res.forecast(horizon=1)
        predicted_variance = forecast.variance.values[-1, 0]
        
        # Прогнозируемая дневная волатильность = корень из прогнозируемой дисперсии
        daily_volatility_pct = np.sqrt(predicted_variance)
        
        logging.info(f"[{symbol}] Прогноз GARCH (дневная волатильность): {daily_volatility_pct:.2f}%")
        return daily_volatility_pct
        
    except ImportError:
        logging.error("Библиотека 'arch' не найдена. Невозможно рассчитать GARCH. Установите: pip install arch")
        return None
    except Exception as e:
        logging.error(f"[{symbol}] Ошибка при расчете GARCH-модели: {e}", exc_info=False)
        return None


# ##################################################################
# ## ЗАДАЧА 2: ПРОДВИНУТАЯ ИНЖЕНЕРИЯ ПРИЗНАКОВ                  ##
# ##################################################################

async def get_market_microstructure_features(symbol: str, client: "AsyncClient") -> Dict[str, Optional[float]]:
    """
    Получает микроструктурные признаки с кэшированием для снижения нагрузки на API.
    """
    global _micro_features_cache
    
    now = time.time()
    
    # 1. Проверяем, есть ли свежие данные в кэше
    if symbol in _micro_features_cache:
        cached_data, timestamp = _micro_features_cache[symbol]
        if now - timestamp < CACHE_TTL_SECONDS:
            # logging.info(f"[{symbol}] Используются кэшированные микроструктурные признаки.")
            return cached_data

    # 2. Если в кэше нет данных (или они устарели), делаем запрос к API
    features = {"order_book_imbalance": None, "bid_ask_spread_pct": None, "market_depth_usd": None}
    
    try:
        order_book = await client.futures_order_book(symbol=symbol, limit=20)
        
        if not order_book.get('bids') or not order_book.get('asks'):
            logging.warning(f"[{symbol}] Стакан ордеров пуст.")
            return features

        # --- Расчеты (остаются без изменений) ---
        best_bid = float(order_book['bids'][0][0])
        best_ask = float(order_book['asks'][0][0])
        
        if best_ask > 0:
            features['bid_ask_spread_pct'] = ((best_ask - best_bid) / best_ask) * 100

        bid_volume = sum(float(level[1]) for level in order_book['bids'][:10])
        ask_volume = sum(float(level[1]) for level in order_book['asks'][:10])
        total_volume = bid_volume + ask_volume
        
        if total_volume > 0:
            features['order_book_imbalance'] = (bid_volume - ask_volume) / total_volume

        mid_price = (best_bid + best_ask) / 2
        depth_usd = sum(p * q for p, q in ((float(price), float(qty)) for price, qty in order_book['bids']) if p > mid_price * 0.99) + \
                    sum(p * q for p, q in ((float(price), float(qty)) for price, qty in order_book['asks']) if p < mid_price * 1.01)
        features['market_depth_usd'] = depth_usd
        
        # 3. Сохраняем свежие данные в кэш
        _micro_features_cache[symbol] = (features, now)
        
        logging.info(f"[{symbol}] Микроструктурные признаки обновлены: OBI={features.get('order_book_imbalance'):.3f}")
        return features
        
    except Exception as e:
        # 4. Улучшенное логирование при ошибке
        logging.error(f"[{symbol}] Не удалось получить микроструктурные признаки: {e}", exc_info=True)
        # В случае ошибки возвращаем пустые данные, но не сохраняем их в кэш
        return features
async def get_derivatives_sentiment(symbol: str, client: "AsyncClient") -> Dict[str, Optional[float]]:
    """
    "Бронебойная" версия. Получает сантимент с рынка деривативов, пробуя несколько
    вариантов API-вызовов для L/S Ratio, чтобы обеспечить максимальную совместимость.
    """
    sentiment = {"open_interest_usd": None, "long_short_ratio": None}
    logging.info(f"[{symbol}] Получение сантимента с рынка деривативов...")
    
    try:
        # --- Получение Открытого Интереса (без изменений) ---
        oi_task = client.futures_open_interest(symbol=symbol)
        mark_price_task = client.futures_mark_price(symbol=symbol)
        
        results = await asyncio.gather(oi_task, mark_price_task, return_exceptions=True)
        oi_data, mark_price_data = results
        
        if isinstance(oi_data, dict) and isinstance(mark_price_data, dict):
            open_interest_amount = oi_data.get('openInterest')
            mark_price = mark_price_data.get('markPrice')
            if open_interest_amount and mark_price:
                sentiment['open_interest_usd'] = float(open_interest_amount) * float(mark_price)

        # --- ✅ НАДЕЖНЫЙ БЛОК ПОЛУЧЕНИЯ L/S RATIO ---
        ls_ratio_data = None
        error_log = ""
        
        # Список возможных функций для перебора
        possible_functions = [
            'futures_global_long_short_account_ratio',
            'futures_taker_long_short_ratio',
            'futures_top_long_short_account_ratio'
        ]
        
        for func_name in possible_functions:
            try:
                if hasattr(client, func_name):
                    # Если функция существует, вызываем ее
                    func = getattr(client, func_name)
                    ls_ratio_data = await func(symbol=symbol, period='1h', limit=1)
                    if ls_ratio_data:
                        logging.info(f"[{symbol}] L/S Ratio успешно получен через метод: {func_name}")
                        break # Выходим из цикла, если данные получены
            except Exception as e:
                error_log += f"Метод {func_name} не сработал: {e}. "
        
        if ls_ratio_data and isinstance(ls_ratio_data, list) and len(ls_ratio_data) > 0:
            if 'longShortRatio' in ls_ratio_data[0]:
                sentiment['long_short_ratio'] = float(ls_ratio_data[0]['longShortRatio'])
            elif 'buySellRatio' in ls_ratio_data[0]: # Для taker_long_short_ratio
                 sentiment['long_short_ratio'] = float(ls_ratio_data[0]['buySellRatio'])
        else:
            if not error_log: error_log = "Все известные методы не вернули данных."
            logging.warning(f"[{symbol}] Не удалось получить L/S Ratio. {error_log}Продолжаем без этих данных.")

        # --- ✅ БЕЗОПАСНОЕ ЛОГИРОВАНИЕ ---
        oi_val = sentiment.get('open_interest_usd')
        ls_val = sentiment.get('long_short_ratio')
        oi_str = f"${oi_val:,.0f}" if oi_val is not None else 'N/A'
        ls_str = f"{ls_val:.3f}" if ls_val is not None else 'N/A'
        logging.info(f"[{symbol}] Сантимент деривативов: OI={oi_str}, L/S Ratio={ls_str}")
        
        return sentiment
        
    except Exception as e:
        logging.error(f"[{symbol}] Критическая ошибка в get_derivatives_sentiment: {e}", exc_info=False)
        return sentiment

def estimate_slippage_bps(orderbook: dict, side: str, qty: float) -> Tuple[float, float]:
    """Оценивает среднее проскальзывание в базисных пунктах для данного объема."""
    levels = orderbook.get("asks") if side.upper()=="BUY" else orderbook.get("bids")
    if not levels: return 9999.0, 0.0
    
    remain, vwap, filled = qty, 0.0, 0.0
    for px, q in ((float(p), float(q)) for p,q in levels):
        take = min(remain, q)
        vwap += px * take
        filled += take
        remain -= take
        if remain <= 1e-12: break
            
    if filled <= 0: return 9999.0, 0.0
    vwap /= filled
    best_px = float(levels[0][0])
    
    bps = (vwap / best_px - 1.0) * 10_000.0 if side.upper()=="BUY" else (1.0 - vwap / best_px) * 10_000.0
    return max(0.0, bps), vwap

async def fetch_orderbook(symbol: str, depth: int = 50) -> Dict[str, List]:
    """Запрашивает и возвращает стакан ордеров."""
    ob = await client.futures_order_book(symbol=symbol, limit=depth)
    return {"bids": ob.get("bids", []), "asks": ob.get("asks", [])}

def kelly_fraction(p: float, R: float, cap: float = 0.3, fraction: float = 0.5) -> float:
    """
    p = winrate (0..1), R = avg reward/risk (>0)
    cap = макс. доля Келли, fraction = какую часть от оптимума использовать.
    """
    if R <= 0 or p <= 0 or p >= 1: return 0.0
    k_full = p - (1 - p) / R
    return max(0.0, min(cap, k_full)) * fraction

# --- ОБНОВЛЕННАЯ ФУНКЦИЯ РАСЧЕТА ОБЪЕМА V2.4 ---
async def calculate_smart_quantity_v2(
    symbol: str, entry_price: float, sl_price: float,
    unified_score: float, # Используем unified_score вместо ai_verdict напрямую
    base_risk_percent: float = BASE_RISK_PERCENT, # Берем из глобальной константы
    entry_threshold: float = SCORE_THRESHOLD_NORMAL_VOL # Используем порог
) -> float:
    """
    ФИНАЛЬНАЯ ВЕРСИЯ v2.4: Расчет объема с мин. дистанцией SL, клэмпингом риска/Келли,
    лимитами по нотионалу/количеству и улучшенным логированием.
    """
    global exchange_info_cache, MAX_RISK_MODIFIER, current_balance # Используем актуальный баланс

    log_prefix = f"[{symbol}] [Calc Qty V2.4]"

    # --- Базовые проверки ---
    available_balance = current_balance # Используем глобальный current_balance
    if available_balance <= 20.0:
        logging.warning(f"{log_prefix} Доступный баланс ({available_balance:.2f}) слишком мал.")
        return 0.0
    if not all([entry_price > 0, sl_price > 0, entry_price != sl_price]):
        logging.error(f"{log_prefix} Невалидные цены: Entry={entry_price}, SL={sl_price}")
        return 0.0

    # --- 1. Расчет минимальной дистанции риска ---
    raw_risk_distance = abs(entry_price - sl_price)
    min_risk_distance = 0.0
    try:
        # Получаем ATR и Шаг Цены
        df_atr = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS.get("15m")) # Используем 15м для ATR
        atr_val = _atr(df_atr, 14) if df_atr is not None else 0.0
        price_step = get_symbol_price_step(symbol) or 0.0

        # Рассчитываем компоненты минимальной дистанции
        min_dist_atr = MIN_SL_ATR_MULT * atr_val if atr_val > 0 else 0.0
        min_dist_step = MIN_SL_PRICE_STEP_MULT * price_step if price_step > 0 else 0.0
        min_dist_entry = MIN_SL_ENTRY_FRAC * entry_price

        # Выбираем максимум из них + добавляем сырую дистанцию в сравнение
        min_risk_distance = max(raw_risk_distance, min_dist_atr, min_dist_step, min_dist_entry, 1e-12) # 1e-12 чтобы избежать нуля

        logging.info(f"{log_prefix} Расчет мин. дистанции SL: Raw={raw_risk_distance:.5f}, ATR={min_dist_atr:.5f}, Step={min_dist_step:.5f}, Entry%={min_dist_entry:.5f} -> Итог={min_risk_distance:.5f}")

    except Exception as e:
        logging.error(f"{log_prefix} Ошибка расчета мин. дистанции SL: {e}. Использую raw_risk_distance.")
        min_risk_distance = max(raw_risk_distance, 1e-12)

    # Используем эффективную дистанцию (не меньше минимальной)
    effective_risk_per_coin = min_risk_distance

    # --- 2. Расчет и клэмпинг риска в % ---
    score_premium = unified_score - entry_threshold
    # Нормализуем премиум к диапазону [0, 1] относительно возможного максимума
    score_range = max(100 - entry_threshold, 1.0) # Защита от деления на ноль
    normalized_premium = score_premium / score_range
    # Модификатор уверенности теперь от 0.7 до MAX_RISK_MODIFIER
    confidence_modifier = 0.7 + normalized_premium * (MAX_RISK_MODIFIER - 0.7)
    confidence_modifier = min(MAX_RISK_MODIFIER, max(0.7, confidence_modifier))

    # Расчет и клэмпинг Келли (используем константы)
    p_winrate = 0.55 # Примерные значения, можно сделать адаптивными
    R_win_loss_ratio = 1.6
    kelly_mod = kelly_fraction(p_winrate, R_win_loss_ratio, cap=CLAMP_KELLY_MAX, fraction=1.0) # Используем полный Келли, т.к. риск зажмем ниже
    kelly_mod = max(CLAMP_KELLY_MIN, min(CLAMP_KELLY_MAX, kelly_mod)) # Клэмп Келли

    # Расчет и клэмпинг итогового риска в % (используем константы)
    final_risk_percent = base_risk_percent * confidence_modifier * kelly_mod
    final_risk_percent = max(CLAMP_FINAL_RISK_MIN_PCT, min(CLAMP_FINAL_RISK_MAX_PCT, final_risk_percent))

    risk_capital_usd = available_balance * (final_risk_percent / 100.0)

    logging.info(f"{log_prefix} Расчет риска: BaseRisk={base_risk_percent:.2f}%, ScoreMod={confidence_modifier:.2f}, KellyMod={kelly_mod:.2f} -> FinalRisk={final_risk_percent:.3f}% (${risk_capital_usd:.2f})")

    # --- 3. Расчет базового объема ---
    calculated_quantity = risk_capital_usd / effective_risk_per_coin

    # --- 4. Применение потолков по нотионалу и количеству ---
    final_quantity = calculated_quantity
    initial_notional = calculated_quantity * entry_price

    # Потолок по нотионалу (USD)
    if MAX_USD_NOTIONAL_PER_TRADE is not None and initial_notional > MAX_USD_NOTIONAL_PER_TRADE:
        cap_qty_notional = MAX_USD_NOTIONAL_PER_TRADE / max(entry_price, 1e-12)
        final_quantity = min(final_quantity, cap_qty_notional)
        logging.warning(f"{log_prefix} Объем ограничен потолком нотионала ${MAX_USD_NOTIONAL_PER_TRADE:.2f}. Новый объем: {final_quantity:.6f}")

    # Потолок по количеству монет
    if MAX_QTY_PER_SYMBOL is not None and final_quantity > MAX_QTY_PER_SYMBOL:
        final_quantity = min(final_quantity, MAX_QTY_PER_SYMBOL)
        logging.warning(f"{log_prefix} Объем ограничен потолком кол-ва {MAX_QTY_PER_SYMBOL}. Новый объем: {final_quantity:.6f}")

    # --- 5. Проверка и принудительное увеличение до мин. лота ($20) ---
    # (Логика без изменений, использует константу BINANCE_MIN_NOTIONAL_USD = 20.0)
    BINANCE_MIN_NOTIONAL_USD = 20.0
    min_notional_with_buffer = BINANCE_MIN_NOTIONAL_USD * 1.01
    current_notional = final_quantity * entry_price

    if current_notional < min_notional_with_buffer:
        logging.warning(f"{log_prefix} Объем после лимитов ($ {current_notional:.2f}) ниже минимума биржи (${BINANCE_MIN_NOTIONAL_USD:.2f}).")
        if available_balance * LEVERAGE >= min_notional_with_buffer: # Проверяем, хватает ли баланса с плечом
            logging.warning(f"{log_prefix} ⚠️ ПРИНУДИТЕЛЬНОЕ УВЕЛИЧЕНИЕ ОБЪЕМА до минимального.")
            final_quantity = min_notional_with_buffer / max(entry_price, 1e-12)
        else:
            logging.error(f"{log_prefix} Вход отменен: баланса ({available_balance:.2f}) недостаточно для минимального ордера.")
            return 0.0

    # --- 6. Форматирование и финальная проверка ---
    formatted_qty_str = format_quantity(final_quantity, symbol)
    if not formatted_qty_str:
        logging.error(f"{log_prefix} Ошибка форматирования итогового объема: {final_quantity}")
        return 0.0

    final_formatted_qty = float(formatted_qty_str)

    # Доп. проверка на старый minNotional из exchange_info (на всякий случай)
    min_notional_from_info = float(exchange_info_cache.get(symbol, {}).get('minNotional', '5.1'))
    final_notional_value = final_formatted_qty * entry_price
    if final_notional_value < min_notional_from_info * 0.99: # 0.99 буфер
        logging.error(
            f"{log_prefix} Вход отменен: объем после округления ({final_formatted_qty}, ${final_notional_value:.2f}) "
            f"ниже minNotional из exchange info (${min_notional_from_info:.2f})."
        )
        return 0.0

    logging.warning(f"{log_prefix} Итоговый расчетный объем: {final_formatted_qty:.6f} (Стоимость: ${final_notional_value:.2f})")
    return final_formatted_qty

# ЗАМЕНИТЕ ВАШУ СТАРУЮ ВЕРСИЮ ЭТОЙ ФУНКЦИИ

# --- ЗАМЕНИТЕ ВАШУ ФУНКЦИЮ find_and_execute_trade_v4 НА ЭТУ ВЕРСИЮ ---

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def adaptive_trend_state(
    df,                    # pd.DataFrame с колонками 'high','low','close'
    fast: int = 21,
    slow: int = 50,
    adx_soft: float = 17.0,
    adx_strict: float = 25.0,
    slope_lookback: int = 3,
    regime_id: int = 1     # 0=low vol, 1=mid, 2=high/trend (ваш HMM)
):
    """
    Возвращает (state, score) где state ∈ {'DOWN','EARLY_DOWN','SIDE','EARLY_UP','UP'}.
    Score ∈ [-100..+100] — сила направленного импульса.
    """
    close = df['close']
    high  = df['high']
    low   = df['low']
    
    # Базовые индикаторы
    ef = ema_indicator(close, window=fast)
    es = ema_indicator(close, window=slow)
    a  = adx(high, low, close, window=14)
    atr = average_true_range(high, low, close, window=14)
    
    # Текущие значения
    ef0, es0 = float(ef.iloc[-1]), float(es.iloc[-1])
    a0       = float(a.iloc[-1])
    c0       = float(close.iloc[-1])
    atr0     = float(atr.iloc[-1]) if float(atr.iloc[-1]) > 0 else 0.0
    
    # Производные признаки
    ema_dist = (ef0 - es0) / (es0 if es0 != 0 else 1.0)
    
    if len(ef) > slope_lookback and float(ef.iloc[-slope_lookback]) != 0:
        ema_slope = (ef0 - float(ef.iloc[-slope_lookback])) / float(ef.iloc[-slope_lookback])
    else:
        ema_slope = 0.0
        
    a_prev = float(a.iloc[-min(5, len(a)-1)]) if len(a) > 5 else a0
    adx_slope = a0 - a_prev
    vol_norm = atr0 / (c0 if c0 != 0 else 1.0)
    
    # Адаптация порогов
    k_vol = np.clip(vol_norm * 1000.0, 5.0, 25.0)
    soft = adx_soft + (2.0 if regime_id == 2 else -1.0 if regime_id == 0 else 0.0)
    strict = adx_strict + (2.0 if regime_id == 2 else -1.0 if regime_id == 0 else 0.0)
    
    # Считаем направленные "баллы"
    up_score = (
        0.40 * _sigmoid(ema_dist * 100.0) +
        0.30 * _sigmoid(ema_slope * 100.0 * (20.0/k_vol)) +
        0.20 * _sigmoid((a0 - soft) / 5.0) +
        0.10 * _sigmoid(adx_slope / 2.0) +
        0.10 * _sigmoid((c0 - ef0) / (atr0 + 1e-9))
    )
    down_score = (
        0.40 * _sigmoid(-ema_dist * 100.0) +
        0.30 * _sigmoid(-ema_slope * 100.0 * (20.0/k_vol)) +
        0.20 * _sigmoid((a0 - soft) / 5.0) +
        0.10 * _sigmoid(-adx_slope / 2.0) +
        0.10 * _sigmoid((ef0 - c0) / (atr0 + 1e-9))
    )
    
    score = float(np.clip((up_score - down_score) * 100.0, -100.0, 100.0))
    
    # Классификация состояний
    if ef0 > es0 and a0 >= strict:
        state = 'UP'
    elif ef0 < es0 and a0 >= strict:
        state = 'DOWN'
    elif score >= +20.0:
        state = 'EARLY_UP'
    elif score <= -20.0:
        state = 'EARLY_DOWN'
    else:
        state = 'SIDE'
        
    return state, score

async def get_btc_direction_signal(df_btc: pd.DataFrame) -> Tuple[int, float]:
    log_prefix = "[BTC Direction Filter]"
    try:
        rets = np.log(df_btc['close']).diff().dropna().values
        if len(rets) == 0:
            return (0, 0.5)

        # HMM-модель теперь не нужна для V2, так как он сам считает точность
        # Создаем "пустую" серию, так как V1 её требовал. V2 её проигнорирует.
        # В будущем можно будет убрать этот аргумент из функции.
        hmm_prob_series = pd.Series([0.5] * len(df_btc), index=df_btc.index)

        # Вызываем новый, умный метод
        final_direction, final_prob_long = btc_dir_filter.latest_signal(df_btc, hmm_prob_series)

        dir_map = {1: "LONG", -1: "SHORT", 0: "FLAT"}
        logging.warning(
            f"{log_prefix} РЕЗУЛЬТАТ: Направление={dir_map.get(final_direction, 'N/A')}, "
            f"Уверенность в лонге={final_prob_long:.2%}"
        )
        return final_direction, final_prob_long

    except Exception as e:
        logging.error(f"{log_prefix} Критическая ошибка при расчете сигнала: {e}", exc_info=True)
        return (0, 0.5)

def btc_alt_trend_momentum_gate_adaptive(symbol: str, trade_type: str, use_early: bool = True) -> Tuple[bool, str]:
    """
    ФИНАЛЬНАЯ ВЕРСИЯ. Проверяет тренд BTC на 1ч и тренд альта на 15м/1ч.
    --- ОБНОВЛЕНО: Добавлена проверка волатильности BTC ---
    """
    def strict_trend(df):
        try:
            ef = ta.trend.ema_indicator(df['close'], window=21).iloc[-1]
            es = ta.trend.ema_indicator(df['close'], window=50).iloc[-1]
            a  = ta.trend.adx(df['high'], df['low'], df['close'], window=14).iloc[-1]
            if a >= BTC_ADX_THRESHOLD:
                if ef > es: return 'UP'
                if ef < es: return 'DOWN'
            if ef > es: return 'WEAK_UP'
            if ef < es: return 'WEAK_DOWN'
            return 'SIDE'
        except Exception:
            return 'UNKNOWN'

    # --- Начало основной логики фильтра ---

    # 1) Проверка BTC на 1ч
    btc_1h = market_data_store.get('BTCUSDT', {}).get(ALLOWED_INTERVALS.get('1h'))
    if btc_1h is None or len(btc_1h) < 60: 
        return False, "Недостаточно данных BTC (1h)."
    
    st_btc_1h = strict_trend(btc_1h)
    
    # --- ✅ НОВЫЙ БЛОК: ПРОВЕРКА ВОЛАТИЛЬНОСТИ BTC ---
    try:
        bb = ta.volatility.BollingerBands(close=btc_1h['close'], window=20, window_dev=2)
        bbw = bb.bollinger_wband()
        # Считаем, что волатильность растет, если текущая ширина канала больше средней за 20 свечей
        is_vol_expanding = bbw.iloc[-1] > bbw.rolling(20).mean().iloc[-1]
    except Exception:
        is_vol_expanding = True # В случае ошибки, пропускаем проверку
    # --- КОНЕЦ НОВОГО БЛОКА ---

    if trade_type == 'long':
        if st_btc_1h in ('DOWN',): # ✅ Жёсткий бан только на 'DOWN'
            return False, f"ЛОНГ запрещён: BTC 1h={st_btc_1h}."
        # ✅ Новое условие: если тренд BTC слабый, требуем подтверждения волатильностью
        if st_btc_1h == 'WEAK_DOWN' and not is_vol_expanding:
            return False, f"ЛОНГ запрещён: слабый противотренд BTC (WEAK_DOWN) не подтвержден ростом волатильности."

    if trade_type == 'short':
        if st_btc_1h in ('UP',): # ✅ Жёсткий бан только на 'UP'
            return False, f"ШОРТ запрещён: BTC 1h={st_btc_1h}."
        # ✅ Новое условие: если тренд BTC слабый, требуем подтверждения волатильностью
        if st_btc_1h == 'WEAK_UP' and not is_vol_expanding:
            return False, f"ШОРТ запрещён: слабый противотренд BTC (WEAK_UP) не подтвержден ростом волатильности."

    # 2) Проверка тренда альта на 15м и 1ч (остается без изменений)
    alt_15m = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS.get('15m'))
    alt_1h  = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS.get('1h'))
    if any(df is None or len(df) < 60 for df in [alt_15m, alt_1h]):
        return False, f"Недостаточно данных {symbol} (15m/1h)."

    st15, st1h = strict_trend(alt_15m), strict_trend(alt_1h)
    reg = market_regimes.get(symbol, 1)
    e15, sc15 = adaptive_trend_state(alt_15m, regime_id=reg)
    e1h, sc1h = adaptive_trend_state(alt_1h,  regime_id=reg)
    
    # ✅ ЛОГИКА "ИЛИ" ВМЕСТО "И"
    if trade_type == 'long':
        ok_strict = (st15 in ('UP','WEAK_UP')) or (st1h in ('UP','WEAK_UP'))
        ok_early  = use_early and (e15 in ('EARLY_UP','UP') or e1h in ('EARLY_UP','UP'))
        if not (ok_strict or ok_early):
            return False, f"{symbol} тренд не подтверждён для LONG: 15m={st15}/{e15}, 1h={st1h}/{e1h}."
    else: # short
        ok_strict = (st15 in ('DOWN','WEAK_DOWN')) or (st1h in ('DOWN','WEAK_DOWN'))
        ok_early  = use_early and (e15 in ('EARLY_DOWN','DOWN') or e1h in ('EARLY_DOWN','DOWN'))
        if not (ok_strict or ok_early):
            return False, f"{symbol} тренд не подтверждён для SHORT: 15m={st15}/{e15}, 1h={st1h}/{e1h}."
            
    return True, "Тренд-фильтр пройден."

async def get_adaptive_strategy_parameters(symbol: str) -> Dict[str, float]:
    """
    Рассчитывает и возвращает адаптированные параметры для торговой стратегии
    на основе текущей волатильности рынка.
    """
    # --- Базовые (стандартные) параметры ---
    params = {
        'rsi_oversold': 38.0,
        'rsi_overbought': 65.0,
        'bb_std_dev': 2.0
    }

    try:
        # Анализируем волатильность на 15-минутном графике
        df = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS['15m'])
        if df is None or len(df) < 20:
            return params # Возвращаем стандартные параметры, если данных мало

        # Рассчитываем нормализованный ATR (NATR) как показатель волатильности в %
        atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'], 14).iloc[-1]
        price = df['close'].iloc[-1]
        natr_percent = (atr / price) * 100 if price > 0 else 0

        # --- Логика адаптации ---
        if natr_percent > 1.5: # Очень высокая волатильность
            logging.info(f"[{symbol}] Адаптация: РЫНОК ОЧЕНЬ ВОЛАТИЛЬНЫЙ (NATR={natr_percent:.2f}%)")
            params['rsi_oversold'] = 35.0 # Ищем более глубокую перепроданность
            params['rsi_overbought'] = 65.0 # Ищем более сильную перекупленность
            params['bb_std_dev'] = 2.2 # Расширяем канал Боллинджера
        
        elif natr_percent < 0.5: # Очень низкая волатильность
            logging.info(f"[{symbol}] Адаптация: РЫНОК СПОКОЙНЫЙ (NATR={natr_percent:.2f}%)")
            params['rsi_oversold'] = 35.0 # Реагируем на малейшие движения
            params['rsi_overbought'] = 67.0
            params['bb_std_dev'] = 2.0 # Сужаем канал Боллинджера
        
        else: # Нормальная волатильность
             logging.info(f"[{symbol}] Адаптация: РЫНОК В НОРМЕ (NATR={natr_percent:.2f}%)")
             # Используются стандартные параметры

        return params

    except Exception as e:
        logging.error(f"[{symbol}] Ошибка в get_adaptive_strategy_parameters: {e}")
        return params # В случае ошибки возвращаем стандартные параметры

async def last_chance_analysis(symbol: str, side: str) -> bool:
    """
    Проводит экстренный анализ на предмет возможного разворота, когда цена подходит к стоп-лоссу.
    Возвращает True, если есть веские основания отодвинуть стоп.
    """
    logging.warning(f"[{symbol}] Цена подошла близко к SL. Запуск анализа 'Последний Шанс'...")
    
    df = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS['5m'])
    if df is None or len(df) < 25:
        logging.info(f"[{symbol}] Недостаточно данных для анализа. Решение: НЕ двигать стоп.")
        return False

    try:
        # Критерий 1: Экстремальная перепроданность/перекупленность по RSI
        current_rsi = rsi(df['close'], window=14).iloc[-1]
        
        is_deeply_oversold = side == 'LONG' and current_rsi < 20
        is_deeply_overbought = side == 'SHORT' and current_rsi > 80

        if is_deeply_oversold or is_deeply_overbought:
            logging.warning(f"[{symbol}] ШАНС НАЙДЕН: Экстремальное значение RSI ({current_rsi:.1f}).")
            return True

        # Критерий 2: Касание сильной поддержки/сопротивления (например, нижняя/верхняя линия Боллинджера)
        bollinger = BollingerBands(close=df['close'], window=20, window_dev=2)
        lower_band = bollinger.bollinger_lband().iloc[-1]
        upper_band = bollinger.bollinger_hband().iloc[-1]
        last_low = df['low'].iloc[-1]
        last_high = df['high'].iloc[-1]

        touched_support = side == 'LONG' and last_low <= lower_band
        touched_resistance = side == 'SHORT' and last_high >= upper_band

        if touched_support or touched_resistance:
            logging.warning(f"[{symbol}] ШАНС НАЙДЕН: Цена коснулась границы Боллинджера.")
            return True

        logging.info(f"[{symbol}] Анализ не нашел веских причин для спасения. Решение: НЕ двигать стоп.")
        return False
        
    except Exception as e:
        logging.error(f"[{symbol}] Ошибка в 'last_chance_analysis': {e}")
        return False

# --- ✅ НАЧАЛО БЛОКА НОВОГО ГЛОБАЛЬНОГО ФИЛЬТРА V3 ---

# Глобальный кэш для хранения состояния и времени последнего обновления
GLOBAL_TREND_CACHE = {"state": "SIDEWAYS", "ts": 0}

def _natr_pct(df, period=14) -> Optional[float]:
    """Вспомогательная функция для расчета нормализованного ATR в процентах."""
    import numpy as np
    import pandas as pd
    if df is None or len(df) < period + 2:
        return None
    import ta
    try:
        atr = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=period).iloc[-1]
        close = float(df["close"].iloc[-1])
        if pd.isna(atr) or pd.isna(close) or close == 0:
            return None
        return float(atr / close * 100.0)
    except Exception:
        return None

def _features(df, fast, slow) -> Tuple[float, float, float]:
    """Вспомогательная функция для извлечения EMA и ADX."""
    import ta
    ema_fast = ta.trend.ema_indicator(df["close"], window=fast).iloc[-1]
    ema_slow = ta.trend.ema_indicator(df["close"], window=slow).iloc[-1]
    adx_val  = ta.trend.adx(df["high"], df["low"], df["close"], window=14).iloc[-1]
    return float(ema_fast), float(ema_slow), float(adx_val)

def _adx_threshold(natr_pct: Optional[float]) -> float:
    """Волатильно-адаптивный порог ADX."""
    if natr_pct is None:
        return 20.0
    if natr_pct < 0.8:  # Низкая волатильность -> требуем более сильного ADX
        return 22.0
    if natr_pct > 2.0:  # Высокая волатильность -> можем принять более низкий ADX
        return 18.0
    return 20.0


# --- КОНЕЦ БЛОКА ---

def is_breakout_volume_confirmed(df: pd.DataFrame, multiplier: float = 2.0) -> bool:
    """
    Проверяет, подтвержден ли пробой всплеском объема.
    Возвращает True, если объем последней свечи как минимум в `multiplier` раз выше среднего за 20 свечей.
    """
    if len(df) < 21:
        return False
    try:
        avg_volume = df['volume'].iloc[-21:-1].mean()
        breakout_volume = df['volume'].iloc[-1]
        
        if avg_volume > 0 and breakout_volume > avg_volume * multiplier:
            logging.info(f"    [VOL FILTER]: PASSED (Volume: {breakout_volume:.0f} > Avg: {avg_volume:.0f} * {multiplier})")
            return True
        else:
            logging.info(f"    [VOL FILTER]: FAILED (Volume: {breakout_volume:.0f} <= Avg: {avg_volume:.0f} * {multiplier})")
            return False
    except Exception:
        return False

def is_in_volatility_squeeze(df: pd.DataFrame, lookback: int = 20) -> bool:
    """
    Проверяет, находится ли рынок в состоянии 'сжатия волатильности' (узкие полосы Боллинджера).
    Возвращает True, если ширина полос Боллинджера находится в нижних 25% за последний `lookback` период.
    """
    if len(df) < lookback + 2:
        return False
    try:
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        bbw = bb.bollinger_wband() # Ширина полос
        
        # Находим минимальную ширину за lookback-период
        lowest_bbw = bbw.iloc[-lookback:-1].min()
        current_bbw = bbw.iloc[-1]
        
        # Пробой должен произойти из состояния низкой волатильности
        is_squeezing = bbw.iloc[-2] < lowest_bbw * 1.25
        if is_squeezing:
             logging.info(f"    [BBW SQUEEZE]: PASSED (Previous BBW was in a squeeze)")
             return True
        else:
             logging.info(f"    [BBW SQUEEZE]: FAILED (No volatility squeeze before breakout)")
             return False
    except Exception:
        return False

# --- ЗАМЕНИТЕ ВАШУ СТАРУЮ ФУНКЦИЮ НА ЭТУ ---

# ЗАМЕНИТЕ ВАШУ СТАРУЮ ФУНКЦИЮ find_and_execute_trade_v4 НА ЭТУ ВЕРСИЮ

def btc_4h_gate_with_override(
    symbol: str, 
    trade_type: str,
    ml_side_prob: float,   # Параметр сохранен для совместимости, но не используется
    alt_trend_ok: bool,    # Параметр сохранен для совместимости, но не используется
    adx_thr: float = BTC_ADX_THRESHOLD
) -> Tuple[bool, str]:
    """
    Строго проверяет тренд BTC на 4ч. Логика "пробития" (override) отключена,
    но сигнатура функции сохранена для совместимости вызовов.
    """
    btc_4h = market_data_store.get('BTCUSDT', {}).get(ALLOWED_INTERVALS.get('4h'))
    if btc_4h is None or len(btc_4h) < 60:
        return True, "BTC 4h недоступен — фильтр пропущен."

    try:
        ema_fast = ta.trend.ema_indicator(btc_4h['close'], window=21).iloc[-1]
        ema_slow = ta.trend.ema_indicator(btc_4h['close'], window=50).iloc[-1]
        adx_val = ta.trend.adx(btc_4h['high'], btc_4h['low'], btc_4h['close'], window=14).iloc[-1]
        price = float(btc_4h['close'].iloc[-1])
        
    except Exception as e:
        logging.warning(f"Не удалось рассчитать индикаторы BTC 4h: {e}")
        return True, "BTC 4h индикаторы недоступны — фильтр пропущен."

    is_trending = adx_val >= adx_thr
    strict_up = is_trending and (ema_fast > ema_slow)
    strict_down = is_trending and (ema_fast < ema_slow)

    # Смягчение: если цена уже корректируется к EMA, тренд не считаем строгим
    if strict_up and price <= ema_fast:
        strict_up = False
    if strict_down and price >= ema_fast:
        strict_down = False
        
    # --- Строгая проверка без override ---
    if trade_type == 'short' and strict_up:
        return False, f"БЛОК ШОРТ: BTC 4ч в сильном восходящем тренде (ADX: {adx_val:.1f})"
    
    if trade_type == 'long' and strict_down:
        return False, f"БЛОК ЛОНГ: BTC 4ч в сильном нисходящем тренде (ADX: {adx_val:.1f})"
        
    return True, f"ПРОЙДЕН: Тренд BTC 4ч ({'UP' if ema_fast > ema_slow else 'DOWN'}, ADX: {adx_val:.1f})"

async def get_authoritative_market_state(symbol: str, df: pd.DataFrame) -> str:
    """
    Определяет единое, главное направление рынка ('UP', 'DOWN', 'SIDE').
    Это единственная функция, которая решает, "куда дует ветер".
    """
    df_closed = df.iloc[:-1].copy() # Анализируем только последнюю закрытую свечу
    regime_id = market_regimes.get(symbol, 1)
    state, score = adaptive_trend_state(df_closed, regime_id=regime_id)
    
    # Упрощаем состояния для диспетчера
    if state in ('UP', 'EARLY_UP'):
        return 'UP'
    elif state in ('DOWN', 'EARLY_DOWN'):
        return 'DOWN'
    else:
        return 'SIDE'

# --- ИСПРАВЛЕННАЯ ФУНКЦИЯ find_and_execute_trade_v4 (ПОЛНАЯ ВЕРСИЯ V4.2 с AVWAP) ---
async def find_and_execute_trade_v4(symbol: str):
    """
    Стратегия Ranging V4.3: Использует AVWAP для SL/TP (с буфером 0.75*ATR),
    AI-вердикт для подтверждения, проверку лимита позиций.
    """
    logging.warning(f"--- ✅ [{symbol}] СТРАТЕГИЯ Ranging V4.3 (AVWAP Wide SL) ЗАПУЩЕНА ---")

    # Используем 15м таймфрейм и для сигналов, и для AVWAP
    df = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS['15m'])
    if df is None or len(df) < 60: # Увеличим требование к данным для AVWAP
        logging.debug(f"[{symbol}] Ranging V4.3: Недостаточно данных (нужно >= 60).")
        return

    # --- 1. Расчет индикаторов (RSI, BB, ADX) ---
    try:
        adaptive_params = await get_adaptive_strategy_parameters(symbol)
        rsi_oversold = adaptive_params['rsi_oversold']
        rsi_overbought = adaptive_params['rsi_overbought']
        bb_std_dev = adaptive_params['bb_std_dev']

        rsi_series = ta.momentum.rsi(df['close'], window=14)
        last_rsi = rsi_series.iloc[-1]
        prev_rsi = rsi_series.iloc[-2]

        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=bb_std_dev)
        bb_h = bb.bollinger_hband().iloc[-1]
        bb_l = bb.bollinger_lband().iloc[-1]
        bb_m = bb.bollinger_mavg().iloc[-1]

        last_close = df['close'].iloc[-1]
        last_low = df['low'].iloc[-1]
        last_high = df['high'].iloc[-1]
        adx_val = ta.trend.adx(df['high'], df['low'], df['close'], window=14).iloc[-1]

        # Проверка на NaN значения
        if any(pd.isna(x) for x in [last_rsi, prev_rsi, bb_h, bb_l, bb_m, last_close, last_low, last_high, adx_val]):
             logging.warning(f"[{symbol}] Ranging V4.3: Обнаружены NaN значения в индикаторах.")
             return

    except Exception as e:
        logging.error(f"[{symbol}] Ошибка расчета индикаторов для V4.3: {e}")
        return

    # --- 2. Логика поиска сигнала (RSI/BB) ---
    trade_type = None
    entry_context_detail = ""
    # Условия входа
    if (last_low <= bb_l and last_close < bb_m) or (prev_rsi < rsi_oversold and last_rsi >= rsi_oversold):
        trade_type = "long"
        entry_context_detail = f"BB Low Touch or RSI Cross Up ({last_rsi:.1f})"
    elif (last_high >= bb_h and last_close > bb_m) or (prev_rsi > rsi_overbought and last_rsi <= rsi_overbought):
        trade_type = "short"
        entry_context_detail = f"RSI Cross Down from Overbought ({last_rsi:.1f})"

    if not trade_type:
        return # Нет сигнала

    # --- ADX Фильтр ---
    ADX_RANGE_THRESHOLD = 25.0
    if adx_val >= ADX_RANGE_THRESHOLD:
        logging.info(f"[{symbol}] Ranging-стратегия V4.3 неактивна: ADX={adx_val:.1f} >= {ADX_RANGE_THRESHOLD}.")
        return

    # --- 3. Создание паспорта и верификация ---
    passport = TradePassport(symbol=symbol, strategy="Ranging V4.3 AVWAP WideSL", trigger=entry_context_detail, side=trade_type)
    logging.warning(f"🎯 [{symbol}] ОБНАРУЖЕН СИГНАЛ RANGING AVWAP: {trade_type.upper()} ({entry_context_detail}). ID: {passport.signal_id}. Верификация...")

    try:
        # --- Блок проверок под глобальной блокировкой ---
        async with GLOBAL_ENTRY_LOCK:
            # Проверка лимита позиций, уже открытой позиции, реверса
            dynamic_limit = get_dynamic_max_positions()
            if len(current_positions) >= dynamic_limit: raise ValueError(f"Достигнут лимит позиций ({len(current_positions)}/{dynamic_limit})")
            if symbol in current_positions: raise ValueError("Позиция уже открыта")
            was_reversed = await handle_position_reversal(symbol, trade_type)
            if was_reversed: raise ValueError("Позиция была реверсирована")

            # AI Вердикт
            ai_verdict = await get_ai_model_verdict_async(symbol, trade_type, "ranging")
            ai_confidence = ai_verdict.get("confidence", 0)
            entry_threshold = await get_adaptive_entry_threshold(symbol)
            passport.ai_confidence = ai_confidence
            passport.required_threshold = entry_threshold
            if ai_confidence < entry_threshold: raise ValueError(f"Низкая уверенность AI: {ai_confidence}% < {entry_threshold:.1f}%")

            logging.warning(f"✅ [{symbol}] ВХОД Ranging AVWAP ОДОБРЕН AI ({ai_confidence}%). Финальные расчеты...")

            # Фильтр качества сетапа
            regime = await train_and_predict_hmm_regime(symbol) or 1
            is_valid, reason = await is_trade_setup_valid_v2(symbol, df, regime)
            if not is_valid: raise ValueError(f"Отклонено фильтром качества: {reason}")

            # --- ❗ Расчет SL/TP на основе AVWAP с УВЕЛИЧЕННЫМ БУФЕРОМ ---
            entry_price = float(df['close'].iloc[-1])
            final_sl_price = None
            final_tp1_price = None # Медиана
            final_tp2_price = None # Граница

            avwap_levels = await calculate_avwap_levels(symbol, df)

            if avwap_levels:
                atr_val = _atr(df, 14) or entry_price * 0.005 # Fallback ATR
                # --- ✅ ИЗМЕНЕНИЕ ЗДЕСЬ: Увеличиваем буфер ATR ---
                sl_buffer = atr_val * 0.75 # Стало: Буфер в 75% ATR
                # --- КОНЕЦ ИЗМЕНЕНИЯ ---

                if trade_type == "long":
                    final_sl_price = avwap_levels['lower'] - sl_buffer
                else: # short
                    final_sl_price = avwap_levels['upper'] + sl_buffer

                final_tp1_price = avwap_levels['median'] # TP1 = Медиана
                final_tp2_price = avwap_levels['upper'] if trade_type == "long" else avwap_levels['lower'] # TP2 = Граница
                logging.info(f"[{symbol}] Уровни SL/TP рассчитаны по AVWAP с буфером {sl_buffer:.5f}.")
            else:
                # Fallback на ATR
                logging.warning(f"[{symbol}] Не удалось рассчитать AVWAP. Использую ATR fallback для SL/TP.")
                sltp_fallback = calculate_atr_sltp(symbol, trade_type, entry_price, df)
                if sltp_fallback and sltp_fallback.get('stop_loss') and sltp_fallback.get('take_profit'):
                    final_sl_price = sltp_fallback['stop_loss']
                    final_tp1_price = sltp_fallback['take_profit']
                    final_tp2_price = None
                else:
                    raise ValueError("Не удалось рассчитать SL/TP ни по AVWAP, ни по ATR.")

            # Валидация геометрии SL/TP1
            if final_sl_price is None or final_tp1_price is None:
                 raise ValueError("Ошибка расчета SL/TP1.")
            final_sl_price, final_tp1_price, norm_notes = normalize_sltp_for_side(
                 side="BUY" if trade_type == "long" else "SELL", entry=entry_price, sl=final_sl_price, tp=final_tp1_price,
                 min_rr=MIN_RR_RATIO, price_step=get_symbol_price_step(symbol)
            )
            if norm_notes: logging.warning(f"[{symbol}] Уровни SL/TP1 нормализованы: {', '.join(norm_notes)}")
            # Нормализация TP2
            if final_tp2_price and abs(final_tp2_price - final_tp1_price) > 1e-9:
                 _, final_tp2_price, _ = normalize_sltp_for_side(
                     side="BUY" if trade_type == "long" else "SELL", entry=entry_price, sl=final_sl_price, tp=final_tp2_price,
                     min_rr=1.0, price_step=get_symbol_price_step(symbol)
                 )

            # --- Расчет объема ---
            quantity = await calculate_smart_quantity_v2(symbol, entry_price, final_sl_price, ai_confidence, BASE_RISK_PERCENT, entry_threshold)
            if quantity <= 0:
                raise ValueError("Расчетный объем равен нулю")

            # --- Исполнение входа ---
            passport.status = "EXECUTED"
            passport.entry_price = entry_price
            passport.sl_price = final_sl_price
            passport.tp_price = final_tp1_price # TP1

            execution_result = await adaptive_execute_entry(
                symbol=symbol, side_str="BUY" if trade_type == "long" else "SELL", quantity=quantity,
                sl_price=final_sl_price, tp_price=final_tp1_price,
                entry_context=f"AI Conf: {ai_confidence}%, Trigger: {entry_context_detail}, ID: {passport.signal_id}",
                signal_price=entry_price
            )
            if not execution_result: raise ValueError("Исполнение сделки не удалось")

            # --- Сохранение состояния ---
            execution_result['meta']['strategy_type'] = 'ranging'
            execution_result['meta']['bars_since_entry'] = 0
            execution_result['meta']['tp1_price'] = final_tp1_price # Медиана (или ATR fallback)
            execution_result['meta']['tp2_price'] = final_tp2_price # Граница (или None)
            current_positions[symbol] = PositionManager(execution_result)
            await save_state_async()

    # --- Обработка отказов ---
    except ValueError as e:
        passport.status = "REJECTED"; passport.reject_reason = str(e)
        logging.info(f"[{symbol}] Сигнал Ranging AVWAP {passport.signal_id} ОТКЛОНЕН. Причина: {passport.reject_reason}")
    # --- Обработка критических ошибок ---
    except Exception as e:
        if passport: passport.status = "REJECTED"; passport.reject_reason = f"КРИТИЧЕСКАЯ ОШИБКА: {e}"
        logging.error(f"[{symbol}] КРИТИЧЕСКАЯ ОШИБКА Ranging AVWAP: {e}", exc_info=True)
    # --- Логирование и уведомление ---
    finally:
        if passport and passport.status != "PENDING": await log_and_notify_async(passport, df)
# --- КОНЕЦ ИСПРАВЛЕННОЙ ФУНКЦИИ ---
# --- КОНЕЦ ИСПРАВЛЕННОЙ ФУНКЦИИ ---

# --- ИСПРАВЛЕННАЯ RANGING СТРАТЕГИЯ V4.3 (AVWAP + Wide SL Buffer) ---
# Переименована из find_and_execute_trade_v4
# --- ИСПРАВЛЕННАЯ RANGING СТРАТЕГИЯ V4.3 (AVWAP + Wide SL Buffer + Regime Detector) ---
# --- ОБНОВЛЕННАЯ RANGING СТРАТЕГИЯ V4.3 (Ложный Пробой AVWAP + Wide SL Buffer) ---
# --- НОВАЯ ФУНКЦИЯ ДЛЯ ДЕТЕКТОРА СВЕЧНЫХ ПАТТЕРНОВ ---
def _detect_pa_patterns(df: pd.DataFrame) -> Dict[str, int]:
    """
    Использует Talib для обнаружения ключевых разворотных паттернов на ПОСЛЕДНЕЙ свече.
    Возвращает словарь с флагами: 100 (сигнал), 0 (нет), -100 (противоп. сигнал).
    """
    if df is None or len(df) < 5: # 5 - безопасный минимум для некоторых паттернов
        return {"bullish": 0, "bearish": 0}
        
    try:
        # Импортируем talib, если он доступен
        import talib as ta_lib
    except ImportError:
        logging.warning("[PA Detect] Библиотека 'talib' не найдена. Детектор паттернов отключен.")
        return {"bullish": 0, "bearish": 0}

    try:
        # Берем "сырые" numpy массивы для Talib
        o, h, l, c = df['open'].values, df['high'].values, df['low'].values, df['close'].values

        # --- Бычьи Паттерны ---
        # Бычье поглощение (Engulfing)
        eng_bull = ta_lib.CDLENGULFING(o, h, l, c)[-1]
        # Молот (Hammer)
        hammer = ta_lib.CDLHAMMER(o, h, l, c)[-1]
        # Утренняя звезда (Morning Star)
        morning_star = ta_lib.CDLMORNINGSTAR(o, h, l, c, penetration=0)[-1]
        
        # --- Медвежьи Паттерны ---
        # Медвежье поглощение (Engulfing)
        eng_bear = ta_lib.CDLENGULFING(o, h, l, c)[-1] # ta_lib.CDLENGULFING возвращает 100 или -100
        # Падающая звезда (Shooting Star)
        shooting_star = ta_lib.CDLSHOOTINGSTAR(o, h, l, c)[-1]
        # Вечерняя звезда (Evening Star)
        evening_star = ta_lib.CDLEVENINGSTAR(o, h, l, c, penetration=0)[-1]

        # --- Агрегация ---
        # Engulfing возвращает 100 для бычьего и -100 для медвежьего
        is_bullish = (eng_bull == 100) or (hammer == 100) or (morning_star == 100)
        is_bearish = (eng_bear == -100) or (shooting_star == -100) or (evening_star == 100) # Evening Star == 100

        return {
            "bullish": 100 if is_bullish else 0,
            "bearish": 100 if is_bearish else 0 # Возвращаем 100 (сигнал есть)
        }
        
    except Exception as e:
        logging.error(f"[PA Detect] Ошибка расчета паттернов Talib: {e}")
        return {"bullish": 0, "bearish": 0}
# --- КОНЕЦ НОВОЙ ФУНКЦИИ ---

# --- ОБНОВЛЕННАЯ RANGING СТРАТЕГИЯ V4.3 (Патч "Флет 5.0") ---
# (Использует Ложный Пробой и Свечные Паттерны)
async def find_and_execute_ranging_trade_v4_3(symbol: str):
    """
    Стратегия Ranging V4.3 (Патч "Флет 5.0"):
    Вход 1: Ложный пробой границ AVWAP.
    Вход 2: Свечной паттерн у границ AVWAP.
    Управляется через manage() с целями по AVWAP.
    """
    log_prefix = f"[{symbol}] [Ranging V5.0 Patch]"
    logging.warning(f"--- ✅ {log_prefix} СТРАТЕГИЯ ЗАПУЩЕНА ---")

    df = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS['15m'])
    if df is None or len(df) < 60:
        logging.debug(f"{log_prefix} Недостаточно данных.")
        return

    passport = None

    try:
        # --- 1. Проверка Режима Флэта ---
        if not is_range_regime_v4_1(df, flat_adx=FLAT_ADX_THRESHOLD):
            return # Рынок не во флэте

        # --- 2. Расчет Уровней и Индикаторов ---
        avwap_levels = await calculate_avwap_levels(symbol, df)
        if not avwap_levels:
            logging.warning(f"{log_prefix} Не удалось рассчитать AVWAP.")
            return
        
        support = avwap_levels['lower']
        resistance = avwap_levels['upper']
        median = avwap_levels['median']
        atr_val = _atr(df, 14)
        if atr_val <= 0:
            logging.warning(f"{log_prefix} Невалидный ATR ({atr_val}).")
            return
        
        # --- 3. Поиск Сигналов (Ложный Пробой или Свечной Паттерн) ---
        if len(df) < 3: return # Нужно 3 свечи (prev2, prev1, last)

        prev_candle = df.iloc[-2] # Предпоследняя свеча (n-1)
        last_candle = df.iloc[-1] # Последняя (только что закрывшаяся) свеча (n)

        trade_type = None
        entry_context_detail = ""
        entry_price = float(last_candle['close']) # Вход по закрытию последней свечи
        final_sl_price = None
        final_tp1_price = None
        final_tp2_price = None
        
        # --- ТРИГГЕР 1: Ложный Пробой (анализируем свечу n-1) ---
        fb_long = (prev_candle['low'] < support) and (prev_candle['close'] > support)
        fb_short = (prev_candle['high'] > resistance) and (prev_candle['close'] < resistance)

        if fb_long:
            logging.info(f"{log_prefix} Обнаружен [Триггер 1: Ложный пробой ВНИЗ].")
            trade_type = "long"
            entry_context_detail = f"False Breakout Support ({support:.4f})"
            sl_buffer = atr_val * 0.75
            final_sl_price = prev_candle['low'] - sl_buffer
            final_tp1_price = median
            final_tp2_price = resistance
        elif fb_short:
            logging.info(f"{log_prefix} Обнаружен [Триггер 1: Ложный пробой ВВЕРХ].")
            trade_type = "short"
            entry_context_detail = f"False Breakout Resistance ({resistance:.4f})"
            sl_buffer = atr_val * 0.75
            final_sl_price = prev_candle['high'] + sl_buffer
            final_tp1_price = median
            final_tp2_price = support
        
        # --- ТРИГГЕР 2: Свечной Паттерн (анализируем свечу n) ---
        if not trade_type: # Ищем, только если не было ложного пробоя
            pa_patterns = _detect_pa_patterns(df)
            
            # Находимся ли мы у границы? (в пределах 20% ATR от нее)
            is_near_support = last_candle['low'] <= (support + atr_val * 0.2)
            is_near_resistance = last_candle['high'] >= (resistance - atr_val * 0.2)

            if pa_patterns['bullish'] == 100 and is_near_support:
                logging.info(f"{log_prefix} Обнаружен [Триггер 2: Бычий PA] у поддержки.")
                trade_type = "long"
                entry_context_detail = f"Bullish PA (Hammer/Engulf) at Support ({support:.4f})"
                sl_buffer = atr_val * 0.75
                final_sl_price = last_candle['low'] - sl_buffer
                final_tp1_price = median
                final_tp2_price = resistance
            elif pa_patterns['bearish'] == 100 and is_near_resistance:
                logging.info(f"{log_prefix} Обнаружен [Триггер 2: Медвежий PA] у сопротивления.")
                trade_type = "short"
                entry_context_detail = f"Bearish PA (ShootingStar/Engulf) at Resistance ({resistance:.4f})"
                sl_buffer = atr_val * 0.75
                final_sl_price = last_candle['high'] + sl_buffer
                final_tp1_price = median
                final_tp2_price = support

        if not trade_type:
            return # Сигнала нет

        # --- 4. Создание Паспорта и Верификация ---
        passport = TradePassport(symbol=symbol, strategy="Ranging V5.0 (FB/PA)", trigger=entry_context_detail, side=trade_type)
        logging.warning(f"🎯 {log_prefix} ОБНАРУЖЕН СИГНАЛ: {trade_type.upper()} ({entry_context_detail}). ID: {passport.signal_id}. Верификация...")

        # --- Блок проверок под глобальной блокировкой ---
        async with GLOBAL_ENTRY_LOCK:
            # Проверки: Лимит, Открытые позиции, Реверс, Фильтр BTC
            dynamic_limit = get_dynamic_max_positions()
            if len(current_positions) >= dynamic_limit: raise ValueError(f"Лимит позиций ({len(current_positions)}/{dynamic_limit})")
            if symbol in current_positions: raise ValueError("Позиция уже открыта")
            was_reversed = await handle_position_reversal(symbol, trade_type)
            if was_reversed: raise ValueError("Позиция была реверсирована")
            btc_state, _ = await get_btc_trend_state('1h')
            if trade_type == "long" and btc_state == "STRONG_DOWN": raise ValueError(f"Отклонено: LONG против STRONG_DOWN BTC")
            if trade_type == "short" and btc_state == "STRONG_UP": raise ValueError(f"Отклонено: SHORT против STRONG_UP BTC")
            logging.info(f"{log_prefix} Фильтр сильного тренда BTC пройден (BTC: {btc_state}).")

            # AI Вердикт (Контекст 'ranging')
            ai_verdict = await get_ai_model_verdict_async(symbol, trade_type, "ranging")
            ai_confidence = ai_verdict.get("confidence", 0)
            base_ranging_threshold = AUTOCONFIDENCE_RANGING_THRESHOLD * 100.0 if AUTOCONFIDENCE_RANGING_THRESHOLD < 10 else AUTOCONFIDENCE_RANGING_THRESHOLD
            entry_threshold = await get_adaptive_entry_threshold(symbol, ai_confidence, base_ranging_threshold)
            passport.ai_confidence = ai_confidence; passport.required_threshold = entry_threshold
            if ai_confidence < entry_threshold: raise ValueError(f"Низкая уверенность AI: {ai_confidence}% < {entry_threshold:.1f}%")
            logging.warning(f"✅ {log_prefix} ВХОД ОДОБРЕН AI ({ai_confidence}%). Финальные расчеты...")

            # Фильтр качества сетапа
            regime = await train_and_predict_hmm_regime(symbol) or 1
            is_valid, reason = await is_trade_setup_valid_v2(symbol, df, regime)
            if not is_valid: raise ValueError(f"Отклонено фильтром качества: {reason}")

            # Валидация геометрии SL/TP1 (R:R)
            if final_sl_price is None or final_tp1_price is None: raise ValueError("Ошибка расчета SL/TP1.")
            if 'normalize_sltp_for_side' in globals():
                final_sl_price, final_tp1_price, norm_notes = normalize_sltp_for_side(
                     side="BUY" if trade_type == "long" else "SELL", entry=entry_price, sl=final_sl_price, tp=final_tp1_price,
                     min_rr=MIN_RR_RATIO, price_step=get_symbol_price_step(symbol)
                )
                if norm_notes: logging.warning(f"{log_prefix} SL/TP1 нормализованы: {', '.join(norm_notes)}")
                if final_tp2_price:
                     _, final_tp2_price, _ = normalize_sltp_for_side(
                         side="BUY" if trade_type == "long" else "SELL", entry=entry_price, sl=final_sl_price, tp=final_tp2_price,
                         min_rr=1.0, price_step=get_symbol_price_step(symbol)
                     )
            else: raise NameError("Функция normalize_sltp_for_side не найдена!")

            # Расчет объема (v2.4)
            quantity = await calculate_smart_quantity_v2(symbol, entry_price, final_sl_price, ai_confidence, BASE_RISK_PERCENT, entry_threshold)
            if quantity <= 0: raise ValueError("Расчетный объем равен нулю")

            # Исполнение входа (v4.9)
            passport.status = "EXECUTED"; passport.entry_price = entry_price
            passport.sl_price = final_sl_price; passport.tp_price = final_tp1_price # TP1
            execution_result = await adaptive_execute_entry(
                symbol=symbol, side_str="BUY" if trade_type == "long" else "SELL", quantity=quantity,
                sl_price=final_sl_price, tp_price=final_tp1_price,
                entry_context=f"AI Conf: {ai_confidence}%, Trigger: {entry_context_detail}, ID: {passport.signal_id}",
                signal_price=entry_price
            )
            if not execution_result: raise ValueError("Исполнение сделки не удалось")

            # --- Сохранение состояния ---
            execution_result['meta']['strategy_type'] = 'ranging'
            execution_result['meta']['bars_since_entry'] = 0
            execution_result['meta']['tp1_price'] = final_tp1_price # Медиана
            execution_result['meta']['tp2_price'] = final_tp2_price # Граница
            current_positions[symbol] = PositionManager(execution_result)
            await save_state_async()

    # --- Обработка отказов и ошибок ---
    except ValueError as e:
        if passport: passport.status = "REJECTED"; passport.reject_reason = str(e)
        logging.info(f"{log_prefix} Сигнал {passport.signal_id if passport else ''} ОТКЛОНЕН. Причина: {e}")
    except Exception as e:
        if passport: passport.status = "REJECTED"; passport.reject_reason = f"КРИТИЧЕСКАЯ ОШИБКА: {e}"
        logging.error(f"{log_prefix} КРИТИЧЕСКАЯ ОШИБКА: {e}", exc_info=True)
    # --- Логирование и уведомление ---
    finally:
        if passport and passport.status != "PENDING": await log_and_notify_async(passport, df)
# --- КОНЕЦ ОБНОВЛЕННОЙ RANGING СТРАТЕГИИ ---
# --- КОНЕЦ ОБНОВЛЕННОЙ RANGING СТРАТЕГИИ ---
# --- КОНЕЦ НОВОЙ RANGING СТРАТЕГИИ ---

# --- НОВАЯ ФУНКЦИЯ ДЛЯ РАСЧЕТА AVWAP УРОВНЕЙ ---
async def calculate_avwap_levels(symbol: str, df: pd.DataFrame, lookback: int = 60) -> Optional[Dict[str, float]]:
    """
    Рассчитывает уровни Anchored VWAP (AVWAP) от недавних максимума и минимума.

    Args:
        symbol (str): Символ для логирования.
        df (pd.DataFrame): DataFrame с данными OHLCV (например, 15 минут).
        lookback (int): Количество свечей назад для поиска якорей (High/Low).

    Returns:
        Dict[str, float]: Словарь с 'upper' (AVWAP от High), 'lower' (AVWAP от Low),
                          и 'median' (среднее между upper и lower), или None при ошибке.
    """
    if df is None or len(df) < lookback + 1:
        logging.warning(f"[{symbol}] Недостаточно данных для расчета AVWAP (нужно > {lookback}).")
        return None

    log_prefix = f"[{symbol}] [AVWAP]"

    try:
        # 1. Находим индексы якорей (самый высокий High и самый низкий Low за lookback)
        recent_data = df.iloc[-lookback:]
        anchor_high_idx = recent_data['high'].idxmax()
        anchor_low_idx = recent_data['low'].idxmin()

        # Проверка, что якоря найдены и различны
        if anchor_high_idx == anchor_low_idx:
             logging.warning(f"{log_prefix} Якоря High и Low совпадают. Невозможно рассчитать AVWAP коридор.")
             return None

        # 2. Функция для расчета AVWAP от якоря
        def calculate_single_avwap(anchor_idx):
            data_from_anchor = df.loc[anchor_idx:]
            # Используем цену закрытия для расчета
            pv = data_from_anchor['close'] * data_from_anchor['volume']
            cumulative_pv = pv.cumsum()
            cumulative_volume = data_from_anchor['volume'].cumsum()
            # Избегаем деления на ноль
            avwap = (cumulative_pv / cumulative_volume.replace(0, pd.NA)).iloc[-1]
            return float(avwap) if pd.notna(avwap) else None

        # 3. Рассчитываем AVWAP от High и Low
        avwap_from_high = calculate_single_avwap(anchor_high_idx)
        avwap_from_low = calculate_single_avwap(anchor_low_idx)

        if avwap_from_high is None or avwap_from_low is None:
            logging.error(f"{log_prefix} Ошибка расчета одного из AVWAP.")
            return None

        # 4. Определяем верхний, нижний и медиану
        upper_avwap = max(avwap_from_high, avwap_from_low)
        lower_avwap = min(avwap_from_high, avwap_from_low)
        median_avwap = (upper_avwap + lower_avwap) / 2

        logging.info(f"{log_prefix} Уровни рассчитаны: Low={lower_avwap:.4f}, Median={median_avwap:.4f}, High={upper_avwap:.4f}")

        return {
            "upper": upper_avwap,
            "lower": lower_avwap,
            "median": median_avwap
        }

    except Exception as e:
        logging.error(f"{log_prefix} Непредвиденная ошибка при расчете AVWAP: {e}", exc_info=True)
        return None
# --- КОНЕЦ НОВОЙ ФУНКЦИИ ---

async def get_btc_trend_state(timeframe_key: str = None) -> Tuple[str, Dict[str, float]]:
    """
    Возвращает ('STRONG_UP' | 'WEAK_UP' | 'SIDEWAYS' | 'WEAK_DOWN' | 'STRONG_DOWN' | 'UNKNOWN', details).
    Использует EMA(FAST/SLOW) и ADX(14) на выбранном таймфрейме.
    Совместимо с конфигом TC_TREND_FILTER_TIMEFRAME и ALLOWED_INTERVALS.
    """
    try:
        tf_key = timeframe_key or TC_TREND_FILTER_TIMEFRAME
        tf_api_val = ALLOWED_INTERVALS.get(tf_key) or ALLOWED_INTERVALS.get('1h')
        df_btc = market_data_store.get('BTCUSDT', {}).get(tf_api_val)
        # Минимум данных: чтобы посчитать ADX(14) и обе EMA + 1 бар для стабильности
        min_len = max(BTC_EMA_SLOW_PERIOD + 1, 50)
        if df_btc is None or len(df_btc) < min_len:
            logging.warning(f"[BTC Trend] Недостаточно данных на {tf_key}: {len(df_btc) if df_btc is not None else 0}/{min_len}")
            return 'UNKNOWN', {}
        # Индикаторы
        ema_fast_series = ema_indicator(df_btc['close'], window=BTC_EMA_FAST_PERIOD)
        ema_slow_series = ema_indicator(df_btc['close'], window=BTC_EMA_SLOW_PERIOD)
        adx_series = adx(df_btc['high'], df_btc['low'], df_btc['close'], window=14)
        last_price = df_btc['close'].iloc[-1]
        ef = float(ema_fast_series.iloc[-1])
        es = float(ema_slow_series.iloc[-1])
        a = float(adx_series.iloc[-1])
        # Состояние тренда
        state = 'SIDEWAYS'
        # Сильный тренд только при a >= порога
        if a >= BTC_ADX_THRESHOLD:
            if last_price > ef > es:
                state = 'STRONG_UP'
            elif last_price < ef < es:
                state = 'STRONG_DOWN'
            else:
                # EMA не согласованы — не форсируем "сильный" статус
                state = 'SIDEWAYS'
        else:
            # Переходная зона (слабый тренд)
            if a >= 20.0:
                if ef > es:
                    state = 'WEAK_UP'
                elif ef < es:
                    state = 'WEAK_DOWN'
                else:
                    state = 'SIDEWAYS'
            else:
                state = 'SIDEWAYS'
        details = {
            "timeframe": tf_key,
            "price": round(last_price, 6),
            "ema_fast": round(ef, 6),
            "ema_slow": round(es, 6),
            "adx": round(a, 2),
            "threshold_adx": float(BTC_ADX_THRESHOLD)
        }
        logging.info(f"[BTC Trend] State={state} | TF={tf_key} | ADX={a:.2f} | EMA{BTC_EMA_FAST_PERIOD}={ef:.2f} | EMA{BTC_EMA_SLOW_PERIOD}={es:.2f}")
        return state, details
    except Exception as e:
        logging.error(f"[BTC Trend] Ошибка расчёта тренда: {e}", exc_info=True)
        return 'UNKNOWN', {}

# --- Вспомогательные функции для BTC-фильтра ---

def _ema_slope(series: pd.Series, period: int, lb: int = 3) -> float:
    """Рассчитывает наклон EMA."""
    if series is None or len(series) < period + lb + 1:
        return 0.0
    ema = ta.trend.ema_indicator(series, window=period).iloc[-(lb+1):].to_numpy(dtype=float)
    diffs = np.diff(ema)
    return float(np.nanmean(diffs[-lb:]))

def _atr(df: pd.DataFrame, period: int = 14) -> float:
    """Рассчитывает ATR."""
    if df is None or len(df) < period + 1:
        return 0.0
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    prev_close = close[:-1]
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - prev_close), np.abs(low[1:] - prev_close)))
    if len(tr) < period:
        return 0.0
    return float(np.nanmean(tr[-period:]))

def _nowcast_up(current_price: float, ema_fast_15m: float, atr15: float, buf_mult: float = 0.15) -> bool:
    """Проверяет, находится ли цена выше быстрой EMA с буфером."""
    thresh = ema_fast_15m + max(atr15 * buf_mult, ema_fast_15m * 0.0008)
    return current_price > thresh

def _nowcast_down(current_price: float, ema_fast_15m: float, atr15: float, buf_mult: float = 0.15) -> bool:
    """Проверяет, находится ли цена ниже быстрой EMA с буфером."""
    thresh = ema_fast_15m - max(atr15 * buf_mult, ema_fast_15m * 0.0008)
    return current_price < thresh

def _classify_btc_trend_realtime() -> Tuple[str, Dict[str, float]]:
    """
    Основной классификатор тренда BTC, использующий несколько таймфреймов,
    мгновенную оценку (nowcast) и гистерезис.
    """
    global market_data_store, btc_filter_state
    
    # --- 1. Сбор данных ---
    try:
        df1h = market_data_store.get("BTCUSDT", {}).get(ALLOWED_INTERVALS["1h"])
        df15 = market_data_store.get("BTCUSDT", {}).get(ALLOWED_INTERVALS["15m"])
        df1m = market_data_store.get("BTCUSDT", {}).get(ALLOWED_INTERVALS["1m"])
        
        if df1h is None or df15 is None or len(df1h) < 52 or len(df15) < 22:
            return "SIDEWAYS", {"reason": "Недостаточно данных для анализа"}
    except (KeyError, AttributeError):
        return "SIDEWAYS", {"reason": "Ошибка доступа к данным"}

    # --- 2. Расчет индикаторов ---
    ef1h = ta.trend.ema_indicator(df1h["close"], window=BTC_EMA_FAST_PERIOD).iloc[-1]
    es1h = ta.trend.ema_indicator(df1h["close"], window=BTC_EMA_SLOW_PERIOD).iloc[-1]
    ef15 = ta.trend.ema_indicator(df15["close"], window=BTC_EMA_FAST_PERIOD).iloc[-1]
    slope1h = _ema_slope(df1h["close"], BTC_EMA_FAST_PERIOD, lb=3)
    adx1h = ta.trend.adx(df1h["high"], df1h["low"], df1h["close"], window=14).iloc[-1]
    atr15 = _atr(df15, period=14)
    last1m = float(df1m["close"].iloc[-1]) if (df1m is not None and not df1m.empty) else float(df15["close"].iloc[-1])

    # --- 3. Определение условий ---
    up_struct = (ef1h > es1h) and (slope1h > 0) and (adx1h >= float(BTC_ADX_THRESHOLD))
    dn_struct = (ef1h < es1h) and (slope1h < 0) and (adx1h >= float(BTC_ADX_THRESHOLD))
    up_now = _nowcast_up(last1m, float(ef15), float(atr15))
    dn_now = _nowcast_down(last1m, float(ef15), float(atr15))

    # --- 4. Логика гистерезиса (защита от "мигания") ---
    last_dir = btc_filter_state.get("direction", 0)
    last_ts = btc_filter_state.get("last_updated", 0.0)
    now_ts = time.time()

    # --- 5. Принятие решения ---
    state = "SIDEWAYS"
    if up_struct and (up_now or (now_ts - last_ts < 90 and last_dir == 1)):
        state = "STRONGUP"
    elif dn_struct and (dn_now or (now_ts - last_ts < 90 and last_dir == -1)):
        state = "STRONGDOWN"
    else:
        if ef1h > es1h and slope1h >= 0:
            state = "WEAKUP"
        elif ef1h < es1h and slope1h <= 0:
            state = "WEAKDOWN"

    # Обновление глобального состояния для гистерезиса и быстрого доступа
    btc_filter_state["direction"] = 1 if "UP" in state else (-1 if "DOWN" in state else 0)
    btc_filter_state["last_updated"] = now_ts
    
    return state, {}

# --- Основная функция фильтра, которую вызывает остальной код ---

async def btc_trend_filter(trade_type: str) -> tuple[bool, str]:
    """
    Жёсткий блок только при сильном конфликте. 
    WEAK/SIDEWAYS разрешены. Сохранён «last‑chance» по касанию BB (BTC 1h).
    """
    try:
        # ИСПРАВЛЕНО: Более явная и читаемая логика
        state, details = await get_btc_trend_state()
        t = trade_type.lower()

        if state == "STRONG_DOWN":
            if t == "short":
                return True, f"Разрешен ШОРТ: BTC в сильном нисходящем тренде ({state})"
            else: # long
                return False, f"Заблокирован ЛОНГ: BTC в сильном нисходящем тренде ({state})"
        
        if state == "STRONG_UP":
            if t == "long":
                return True, f"Разрешен ЛОНГ: BTC в сильном восходящем тренде ({state})"
            else: # short
                return False, f"Заблокирован ШОРТ: BTC в сильном восходящем тренде ({state})"

        # Для WEAK и SIDEWAYS состояний - проверяем "последний шанс"
        btc_df = market_data_store.get("BTCUSDT", {}).get(ALLOWED_INTERVALS.get("1h"))
        if btc_df is not None and len(btc_df) > 55:
            try:
                bb = ta.volatility.BollingerBands(close=btc_df["close"], window=20, window_dev=2.0)
                lower = float(bb.bollinger_lband().iloc[-1])
                upper = float(bb.bollinger_hband().iloc[-1])
                last_low = float(btc_df["low"].iloc[-1])
                last_high = float(btc_df["high"].iloc[-1])
                
                touched_support = t == "long" and last_low <= lower
                touched_resistance = t == "short" and last_high >= upper
                
                if touched_support or touched_resistance:
                    logging.warning(f"[{t.upper()}] Фильтр BTC пройден по правилу 'последнего шанса' (касание BB).")
                    return True, "BTC BB touch override"
            except Exception:
                pass
                
        return True, f"Фильтр пройден: BTC в состоянии {state}"
        
    except Exception as e:
        logging.error(f"BTC trend filter error: {e}", exc_info=True)
        return True, "BTC filter error -> allow"


# main.py - Блок 2

async def get_market_context_for_strategy_selection(symbol: str) -> str:
    """
    Анализирует рыночный контекст и возвращает 'trend' или 'flat'.
    """
    global market_data_store, ALLOWED_INTERVALS, FLAT_ADX_THRESHOLD, ML_MODEL_TF_SHORT

    primary_tf_api = ALLOWED_INTERVALS.get(ML_MODEL_TF_SHORT)
    if not primary_tf_api:
        logging.error(f"Неверный primary_timeframe '{ML_MODEL_TF_SHORT}' в конфиге.")
        return 'trend'
        
    df = market_data_store.get(symbol, {}).get(primary_tf_api)

    if df is None or len(df) < 50:
        logging.warning(f"[{symbol}] Недостаточно данных для определения контекста, используется 'trend' по умолчанию.")
        return 'trend'

    try:
        adx_val = ta.trend.adx(df['high'], df['low'], df['close'], window=14).iloc[-1]
        
        # Если ADX низкий и модель для флэта существует, выбираем её
        if adx_val < FLAT_ADX_THRESHOLD and 'flat' in ml_models_manager.get(symbol, {}):
            logging.info(f"[{symbol}] Контекст: ФЛЭТ (ADX={adx_val:.1f}). Выбрана стратегия: 'flat'")
            return 'flat'

        # Во всех остальных случаях используется трендовая стратегия
        logging.info(f"[{symbol}] Контекст: ТРЕНД (ADX={adx_val:.1f}). Выбрана стратегия: 'trend'")
        return 'trend'

    except Exception as e:
        logging.error(f"[{symbol}] Ошибка в get_market_context: {e}. Используется 'trend' по умолчанию.")
        return 'trend'

async def btc_momentum_filter(trade_type: str) -> Tuple[bool, str]:
    """
    Фильтр краткосрочного импульса BTC по двум свечам.
    Блокирует лонги при сильном падении BTC и шорты при сильном росте.
    """
    try:
        # Используем 15-минутный таймфрейм для анализа импульса
        tf_key = '15m'
        btc_df = market_data_store.get('BTCUSDT', {}).get(ALLOWED_INTERVALS[tf_key])
        
        if btc_df is None or len(btc_df) < 20:
            logging.warning("[BTC Momentum Filter] Недостаточно данных для анализа, пропускаю фильтр.")
            return True, "Недостаточно данных по BTC для анализа импульса."

        # Нам нужны две последние ПОЛНОСТЬЮ ЗАКРЫТЫЕ свечи
        last_two_candles = btc_df.iloc[-3:-1]
        if len(last_two_candles) < 2:
            return True, "Недостаточно свечей для анализа."

        candle1 = last_two_candles.iloc[0]
        candle2 = last_two_candles.iloc[1]

        # Рассчитываем ATR для определения "значимости" свечи
        atr = ta.volatility.average_true_range(btc_df['high'], btc_df['low'], btc_df['close'], 14).iloc[-2]
        if pd.isna(atr) or atr == 0:
            return True, "Не удалось рассчитать ATR."
            
        significance_threshold = atr * 0.3 # Тело свечи должно быть > 30% от ATR

        # --- Проверка для ЛОНГ-позиции ---
        if trade_type == 'long':
            is_c1_red_significant = (candle1['open'] > candle1['close']) and (abs(candle1['open'] - candle1['close']) > significance_threshold)
            is_c2_red_significant = (candle2['open'] > candle2['close']) and (abs(candle2['open'] - candle2['close']) > significance_threshold)
            is_price_falling = candle2['close'] < candle1['open']

            if is_c1_red_significant and is_c2_red_significant and is_price_falling:
                reason = "ЛОНГ ЗАБЛОКИРОВАН: Обнаружен сильный медвежий импульс по BTC (2 красные свечи)."
                logging.warning(f"[BTC Momentum Filter] {reason}")
                return False, reason

        # --- Проверка для ШОРТ-позиции ---
        elif trade_type == 'short':
            is_c1_green_significant = (candle1['close'] > candle1['open']) and (abs(candle1['close'] - candle1['open']) > significance_threshold)
            is_c2_green_significant = (candle2['close'] > candle2['open']) and (abs(candle2['close'] - candle2['open']) > significance_threshold)
            is_price_rising = candle2['close'] > candle1['open']

            if is_c1_green_significant and is_c2_green_significant and is_price_rising:
                reason = "ШОРТ ЗАБЛОКИРОВАН: Обнаружен сильный бычий импульс по BTC (2 зеленые свечи)."
                logging.warning(f"[BTC Momentum Filter] {reason}")
                return False, reason

        return True, "Импульс BTC позволяет открыть сделку."

    except Exception as e:
        logging.error(f"[BTC Momentum Filter] Ошибка при проверке импульса: {e}", exc_info=True)
        return True, "Ошибка в работе фильтра импульса BTC."



# --- ДОБАВЬТЕ ЭТИ ФУНКЦИИ ИЛИ ЗАМЕНИТЕ ИМИ СТАРЫЕ АНАЛОГИ ---

def calculate_fixed_risk_quantity(symbol: str, entry_price: float, sl_price: float, base_risk_percent: float) -> float:
    """
    Рассчитывает размер позиции на основе фиксированного риска с проверкой на минимальную стоимость ордера $20.
    """
    global current_balance, exchange_info_cache

    BINANCE_MIN_NOTIONAL_USD = 20.0

    if not all([entry_price > 0, sl_price > 0, entry_price != sl_price]):
        return 0.0
    
    risk_capital_usd = current_balance * (base_risk_percent / 100.0)
    risk_per_coin = abs(entry_price - sl_price)
    if risk_per_coin == 0:
        return 0.0

    calculated_quantity = risk_capital_usd / risk_per_coin

    # --- БЛОК ПРОВЕРКИ И УВЕЛИЧЕНИЯ ОБЪЕМА ---
    final_quantity = calculated_quantity
    min_notional_with_buffer = BINANCE_MIN_NOTIONAL_USD * 1.01
    calculated_notional = calculated_quantity * entry_price
    
    if calculated_notional < min_notional_with_buffer:
        logging.warning(
            f"[{symbol}] Расчетный объем ($ {calculated_notional:.2f}) ниже минимума биржи (${BINANCE_MIN_NOTIONAL_USD:.2f})."
        )
        if current_balance * LEVERAGE >= min_notional_with_buffer:
            logging.warning(f"[{symbol}] ⚠️ ПРИНУДИТЕЛЬНОЕ УВЕЛИЧЕНИЕ ОБЪЕМА до минимального.")
            final_quantity = min_notional_with_buffer / entry_price
        else:
            logging.error(f"[{symbol}] Вход отменен: баланса недостаточно для минимального ордера.")
            return 0.0

    # --- Форматирование ---
    formatted_qty = format_quantity(final_quantity, symbol)
    logging.info(f"[{symbol}] Расчет Qty (Fixed Risk): Риск={base_risk_percent}%, Итоговый объем={formatted_qty}")
    return float(formatted_qty) if formatted_qty else 0.0

def calculate_atr_sltp(symbol: str, side: str, entry_price: float, df: pd.DataFrame) -> Dict[str, Optional[float]]:
    """
    Надежный расчет Stop Loss и Take Profit на основе ATR.
    """
    if df is None or len(df) < COMMON_ATR_PERIOD_FOR_SLTP:
        logging.warning(f"[{symbol}] Недостаточно данных для расчета ATR SL/TP.")
        return {"stop_loss": None, "take_profit": None}
        
    try:
        atr_value = ta.volatility.average_true_range(
            df['high'], df['low'], df['close'], window=COMMON_ATR_PERIOD_FOR_SLTP
        ).iloc[-1]
        
        if pd.isna(atr_value) or atr_value <= 0:
            raise ValueError(f"Невалидное значение ATR: {atr_value}")

        # Используем простые, но эффективные множители
        # SL ставим относительно близко, TP - дальше, чтобы обеспечить хороший R:R
        sl_multiplier = 2.0
        tp_multiplier = 4.0 # Обеспечивает R:R = 2.0

        risk_distance = atr_value * sl_multiplier
        reward_distance = atr_value * tp_multiplier

        if side.upper() == "LONG":
            stop_loss = entry_price - risk_distance
            take_profit = entry_price + reward_distance
        else: # SHORT
            stop_loss = entry_price + risk_distance
            take_profit = entry_price - reward_distance
            
        logging.info(f"[{symbol}] Расчет SL/TP по ATR: ATR={atr_value:.5f}, SL={stop_loss:.4f}, TP={take_profit:.4f}")

        return {"stop_loss": stop_loss, "take_profit": take_profit}
        
    except Exception as e:
        logging.error(f"[{symbol}] Ошибка в расчете ATR SL/TP: {e}")
        return {"stop_loss": None, "take_profit": None}

def compute_dynamic_adx_threshold(
    adx_series: pd.Series,    # pd.Series со значениями ADX
    btc_state: str,           # 'STRONG_UP', 'WEAK_UP', 'SIDEWAYS', etc.
    regime_id: int,           # 0=Low Vol, 1=Medium, 2=High/Trend (из HMM)
    p: float = 0.35,          # Перцентиль для базового уровня
    lookback: int = 200       # Окно для анализа перцентиля
) -> float:
    """
    Возвращает адаптивный порог ADX с учётом локальной истории, BTC-состояния и HMM-режима.
    """
    s = adx_series.dropna()
    window = s.values[-lookback:] if len(s) >= lookback else s.values
    
    # Расчет базового порога на основе истории ADX
    if window.size == 0 or np.all(~np.isfinite(window)):
        base = 15.0 # Безопасное значение по умолчанию
    else:
        base = float(np.nanquantile(window, p))

    # Ограничиваем базовый порог разумными пределами
    base = float(np.clip(base, ADAPTIVE_ADX_THRESHOLD_LOW, ADAPTIVE_ADX_THRESHOLD_HIGH_VOL))
    
    # Поправка от состояния тренда BTC
    if 'UP' in btc_state or 'DOWN' in btc_state:
        base += 2.0 # Если есть тренд BTC, требуем более сильного подтверждения
    else: # SIDEWAYS
        base -= 1.0 # Если BTC во флэте, можем быть менее строгими

    # Поправка от HMM-режима монеты
    if regime_id == 2:  # High Vol/Trend
        base += 2.0
    elif regime_id == 0: # Low Vol/Flat
        base -= 2.0
        
    # Финальные границы безопасности, чтобы порог не был слишком низким или высоким
    final_threshold = float(np.clip(base, 10.0, 30.0))
    
    return final_threshold

def find_rsi_divergence(
    df: pd.DataFrame,
    rsi_period: int = 14,
    peak_lookback: int = 50, # За сколько свечей назад ищем пики/впадины
    peak_prominence: float = 1.5 # Насколько "выдающимся" должен быть пик (в % от RSI)
) -> tuple[int, int]:
    """
    Находит классическую бычью или медвежью дивергенцию RSI.

    Args:
        df: DataFrame с колонками 'high', 'low', 'close'.
        rsi_period: Период для расчета RSI.
        peak_lookback: Окно в свечах для поиска последних 2-х экстремумов.
        peak_prominence: Минимальная "заметность" пика RSI для его учета.

    Returns:
        Кортеж (signal, age):
        - signal: 1 (бычья дивергенция), -1 (медвежья дивергенция), 0 (нет дивергенции).
        - age: Сколько свечей назад была обнаружена дивергенция.
    """
    if len(df) < peak_lookback + rsi_period:
        return 0, 0 # Недостаточно данных для анализа

    # 1. Рассчитываем RSI
    rsi = ta.momentum.rsi(df['close'], window=rsi_period)
    
    # Ограничиваем данные последними `peak_lookback` свечами для поиска
    price_data_high = df['high'].tail(peak_lookback)
    price_data_low = df['low'].tail(peak_lookback)
    rsi_data = rsi.tail(peak_lookback)

    # 2. Находим пики (для медвежьей дивергенции) и впадины (для бычьей)
    # prominence - это мера того, насколько пик выделяется над окружающим ландшафтом
    rsi_peaks, _ = find_peaks(rsi_data, prominence=peak_prominence)
    rsi_troughs, _ = find_peaks(-rsi_data, prominence=peak_prominence)

    # --- Проверка на медвежью дивергенцию (Higher Highs на цене, Lower Highs на RSI) ---
    if len(rsi_peaks) >= 2:
        # Берем два последних пика RSI
        p1_idx, p2_idx = rsi_peaks[-2], rsi_peaks[-1]
        
        # Получаем их значения и соответствующие им значения цены
        rsi_p1, rsi_p2 = rsi_data.iloc[p1_idx], rsi_data.iloc[p2_idx]
        price_p1, price_p2 = price_data_high.iloc[p1_idx], price_data_high.iloc[p2_idx]

        # Условие дивергенции
        if price_p2 > price_p1 and rsi_p2 < rsi_p1:
            age = len(rsi_data) - 1 - p2_idx # Сколько свечей прошло с момента фиксации
            return -1, age # Медвежий сигнал

    # --- Проверка на бычью дивергенцию (Lower Lows на цене, Higher Lows на RSI) ---
    if len(rsi_troughs) >= 2:
        # Берем две последние впадины RSI
        t1_idx, t2_idx = rsi_troughs[-2], rsi_troughs[-1]

        # Получаем их значения и соответствующие им значения цены
        rsi_t1, rsi_t2 = rsi_data.iloc[t1_idx], rsi_data.iloc[t2_idx]
        price_t1, price_t2 = price_data_low.iloc[t1_idx], price_data_low.iloc[t2_idx]
        
        # Условие дивергенции
        if price_t2 < price_t1 and rsi_t2 > rsi_t1:
            age = len(rsi_data) - 1 - t2_idx # Сколько свечей прошло с момента фиксации
            return 1, age # Бычий сигнал

    return 0, 0 # Дивергенция не найдена

def decide_entry_with_dynamic_adx(
    symbol: str,
    adx_series: pd.Series,
    current_adx: float,
    btc_state: str,
    regime_id: int,
    active_strategy: str
) -> dict:
    """
    Решает, разрешать ли вход и какие стратегии допускать при текущем ADX.
    """
    thr = compute_dynamic_adx_threshold(adx_series, btc_state, regime_id)
    
    # Трендовые стратегии разрешены только если ADX выше динамического порога
    allow_trend = current_adx >= thr
    
    # Контртрендовые и пробойные стратегии разрешены всегда (их фильтрует сама ML-модель)
    allow_reversion = True
    allow_breakout  = True

    # Итоговый допуск для конкретной стратегии, которую выбрала ML-модель
    if active_strategy == 'trend':
        allow_entry = allow_trend
    else: # 'reversion' или 'breakout'
        allow_entry = True

    logging.info(f"[{symbol}] ADX Check: Current={current_adx:.1f}, DynamicThreshold={thr:.1f} | Strategy='{active_strategy}' -> Entry Allowed: {allow_entry}")

    return {
        'allow_entry': bool(allow_entry),
        'dynamic_threshold': float(thr),
    }

async def is_trade_setup_valid_v2(symbol: str, df_short: pd.DataFrame, regime: int) -> Tuple[bool, str]:
    """
    Улучшенная проверка качества сетапа с АДАПТИВНЫМ фильтром объема и расширенным коридором NATR.
    """
    if df_short is None or len(df_short) < 21:
        return False, "Недостаточно данных для анализа."

    try:
        # 1. ✅ Адаптивная проверка поддержки объемом
        avg_volume = df_short['volume'].rolling(window=20).mean().iloc[-2]
        last_volume = df_short['volume'].iloc[-1]

        # Устанавливаем порог в зависимости от режима рынка
        if regime == 2: # Режим "Тренд"
            volume_threshold = 0.8
            regime_name = "Тренд"
        else: # Режимы "Флэт" / "Низкая волатильность"
            volume_threshold = 0.6 # Требуем всего 60%
            regime_name = "Флэт"

        if last_volume < avg_volume * volume_threshold:
            return False, f"Низкий объем для режима '{regime_name}' ({last_volume:.0f} < {int(volume_threshold*100)}% от avg {avg_volume:.0f})."

        # 2. ✅ Проверка на аномальную волатильность с расширенным коридором
        atr_val = average_true_range(df_short['high'], df_short['low'], df_short['close'], 14).iloc[-1]
        last_close = df_short['close'].iloc[-1]

        if last_close <= 0: return False, "Нулевая цена закрытия."

        natr_percent = (atr_val / last_close) * 100

        if natr_percent > 5.0: # Раньше было 4.0
            return False, f"Аномальная волатильность (NATR={natr_percent:.1f}%)."
        if natr_percent < 0.15: # Раньше было 0.2
            return False, f"Слишком низкая волатильность (NATR={natr_percent:.1f}%)."

        return True, "Сетап прошел проверку качества."

    except Exception as e:
        logging.error(f"[{symbol}] Ошибка в is_trade_setup_valid_v2: {e}")
        return False, "Внутренняя ошибка анализа."

# --- ДОБАВЬТЕ ЭТИ ФУНКЦИИ ИЛИ ЗАМЕНИТЕ ИМИ СТАРЫЕ АНАЛОГИ ---

# ЗАМЕНИТЕ ВАШУ СТАРУЮ ВЕРСИЮ ЭТОЙ ФУНКЦИИ

# --- Сначала добавьте эту новую вспомогательную функцию ---

async def fetch_fill_info(symbol: str, order_id: int) -> dict:
    """
    Гарантированно получает данные об исполнении ордера (среднюю цену и объем),
    опрашивая биржу до получения статуса FILLED.
    """
    for _ in range(10): # Делаем до 10 попыток в течение ~2 секунд
        try:
            od = await client.futures_get_order(symbol=symbol, orderId=order_id)
            if od and od.get("status") == "FILLED":
                avg_price = float(od.get("avgPrice", 0))
                filled_qty = float(od.get("executedQty", 0))
                
                # Дополнительная проверка через account_trades для максимальной точности
                if avg_price <= 0 or filled_qty <= 0:
                    trades = await client.futures_account_trades(symbol=symbol, orderId=order_id, limit=5)
                    if trades:
                        qty = sum(float(t["qty"]) for t in trades)
                        notional = sum(float(t["quoteQty"]) for t in trades)
                        avg_price = (notional / qty) if qty > 0 else 0
                        filled_qty = qty
                
                if avg_price > 0 and filled_qty > 0:
                    return {"filled_qty": filled_qty, "entry_price": avg_price}
        except Exception as e:
            logging.warning(f"[{symbol}] Ожидание fill для ордера {order_id}: {e}")
        await asyncio.sleep(0.2)
    
    logging.error(f"[{symbol}] Не удалось получить данные об исполнении для ордера {order_id}!")
    return {"filled_qty": 0.0, "entry_price": 0.0}

# --- Затем полностью замените вашу старую adaptive_execute_entry на эту ---

# --- ОБНОВЛЕННАЯ ФУНКЦИЯ ИСПОЛНЕНИЯ ВХОДА V4.9 ---
# --- ОБНОВЛЕННАЯ ФУНКЦИЯ ИСПОЛНЕНИЯ ВХОДА V5.0 ---
async def adaptive_execute_entry(
    symbol: str, side_str: str, quantity: float,
    sl_price: float, tp_price: float, # tp_price - это TP1
    entry_context: str,
    signal_price: float
) -> Optional[Dict[str, Any]]:
    """
    АТОМАРНЫЙ ИСПОЛНИТЕЛЬ (v5.0): Отказ от closePosition=True.
    Ставит SL (полный объем) + TP1 (частичный) + TP2 (частичный).
    """
    side = 'LONG' if side_str == 'BUY' else 'SHORT'
    log_prefix = f"[{symbol}] [Entry V5.0]"
    logging.warning(f"{log_prefix} АДАПТИВНОЕ ИСПОЛНЕНИЕ (3 ордера): {side} {quantity} ед.")

    total_qty_filled = 0.0
    actual_entry_price = 0.0
    all_executed_order_ids = []
    
    # --- 1. Логика входа (IOC/MARKET) - Без изменений ---
    try:
        ob = await fetch_orderbook(symbol)
        if not (ob and ob.get("bids") and ob.get("asks")): raise ValueError("Не удалось получить стакан ордеров.")
        remaining_qty = quantity
        if remaining_qty > 0:
            logging.info(f"{log_prefix} Попытка входа через IOC...")
            try:
                limit_price_ioc = float(ob["asks"][0][0] if side_str == "BUY" else ob["bids"][0][0])
                ioc_order = await place_order_async( symbol=symbol, side=side_str, order_type="LIMIT", quantity=remaining_qty, timeInForce="IOC", price=limit_price_ioc, use_smart_entry=True, slip_threshold_bps=SLIP_BP, limit_off_bps=LIMIT_OFFSET_BPS )
                if ioc_order and ioc_order.get('orderId'):
                    executed_qty = float(ioc_order.get('executedQty', 0))
                    if executed_qty > 0:
                        all_executed_order_ids.append(int(ioc_order['orderId']))
                        remaining_qty -= executed_qty
                        logging.info(f"{log_prefix} IOC исполнил {executed_qty:.4f}. Осталось: {remaining_qty:.4f}")
            except Exception as e: logging.warning(f"{log_prefix} Ошибка IOC: {e}")
        if remaining_qty > 0:
            logging.warning(f"{log_prefix} Остаток {remaining_qty:.4f}. Вход по рынку.")
            market_order_ack = await place_order_async( symbol=symbol, side=side_str, order_type="MARKET", quantity=remaining_qty, use_smart_entry=True, slip_threshold_bps=SLIP_BP, limit_off_bps=LIMIT_OFFSET_BPS )
            if market_order_ack and market_order_ack.get('orderId'):
                all_executed_order_ids.append(int(market_order_ack['orderId']))
                logging.info(f"{log_prefix} Market ордер {market_order_ack['orderId']} размещен.")
        if not all_executed_order_ids: raise ValueError("Ни один ордер (IOC/MARKET) не был размещен/исполнен.")

        # --- 2. Получение фактической цены входа - Без изменений ---
        await asyncio.sleep(1.5)
        total_notional = 0.0
        for order_id in set(all_executed_order_ids):
             avg_fill_price_part = await await_filled_avg_price(symbol, order_id, signal_price)
             filled_qty_part = 0.0
             try:
                 od_part = await client.futures_get_order(symbol=symbol, orderId=order_id)
                 filled_qty_part = float(od_part.get("executedQty", 0))
                 if filled_qty_part <= 0 and od_part.get("status") == "FILLED":
                     trades = await client.futures_account_trades(symbol=symbol, orderId=order_id, limit=5)
                     if trades: filled_qty_part = sum(float(t.get("qty", 0)) for t in trades)
             except Exception as e: logging.error(f"{log_prefix} Ошибка получения executedQty для {order_id}: {e}")
             if filled_qty_part > 0:
                  if avg_fill_price_part > 0: total_qty_filled += filled_qty_part; total_notional += avg_fill_price_part * filled_qty_part
                  else: logging.warning(f"{log_prefix} Цена для ордера {order_id} неизвестна. Использую signal_price."); total_qty_filled += filled_qty_part; total_notional += signal_price * filled_qty_part
        if total_qty_filled <= 0: raise ValueError("Не удалось получить объем входа.")
        actual_entry_price = total_notional / total_qty_filled if total_qty_filled > 0 else 0
        if actual_entry_price <= 0: logging.error(f"{log_prefix} Рассчитанная цена входа <= 0. Использую signal_price."); actual_entry_price = signal_price

        # --- 3. Пересчет SL/TP от фактической цены входа ---
        # (Используем TP1_RR и TP2_RR из констант для расчета)
        original_risk_distance = abs(signal_price - sl_price)
        if side == 'LONG':
            final_sl_price = actual_entry_price - original_risk_distance
            final_tp1_price = actual_entry_price + (original_risk_distance * TP1_RR) # TP1
            final_tp2_price = actual_entry_price + (original_risk_distance * TP2_RR) # TP2
        else:
            final_sl_price = actual_entry_price + original_risk_distance
            final_tp1_price = actual_entry_price - (original_risk_distance * TP1_RR) # TP1
            final_tp2_price = actual_entry_price - (original_risk_distance * TP2_RR) # TP2
        
        logging.info(f"{log_prefix} Уровни пересчитаны: Факт. цена={actual_entry_price:.5f}. SL={final_sl_price:.5f}, TP1={final_tp1_price:.5f}, TP2={final_tp2_price:.5f}")

        # --- ✅ 4. РАСЧЕТ ОБЪЕМОВ ДЛЯ TP1 и TP2 ---
        # Используем total_qty_filled как "initial_amount"
        qty_tp1_raw = total_qty_filled * TP1_CLOSE_FRAC
        qty_tp2_raw = total_qty_filled * TP2_CLOSE_FRAC
        
        qty_tp1_str = format_quantity(qty_tp1_raw, symbol)
        qty_tp2_str = format_quantity(qty_tp2_raw, symbol)
        
        qty_tp1 = float(qty_tp1_str) if qty_tp1_str else 0.0
        qty_tp2 = float(qty_tp2_str) if qty_tp2_str else 0.0

        # Сумма TP не должна превышать общий объем
        if (qty_tp1 + qty_tp2) > total_qty_filled:
            logging.warning(f"{log_prefix} Сумма TP ({qty_tp1+qty_tp2}) > общего объема ({total_qty_filled}). Уменьшаю TP2.")
            qty_tp2 = max(0.0, total_qty_filled - qty_tp1)
        
        logging.info(f"{log_prefix} Объемы TP: TP1={qty_tp1} (доля {TP1_CLOSE_FRAC}), TP2={qty_tp2} (доля {TP2_CLOSE_FRAC})")
        
        # --- 5. УСТАНОВКА 3 ОРДЕРОВ (SL, TP1, TP2) ---
        opposite_side_str = "SELL" if side_str == "BUY" else "BUY"

        # --- SL (на ПОЛНЫЙ объем) ---
        formatted_sl_price = format_price(final_sl_price, symbol, side=opposite_side_str)
        sl_order = await place_order_async(
            symbol=symbol, side=opposite_side_str, order_type="STOP_MARKET",
            quantity=total_qty_filled, # SL ставится на полный объем
            stopPrice=formatted_sl_price,
            reduce_only=True, # Важно
            workingType='MARK_PRICE',
            priceProtect=True
        )
        if not sl_order or not sl_order.get('orderId'):
            raise ValueError("КРИТИЧЕСКАЯ ОШИБКА: Не удалось установить Stop Loss. Запускаю аварийное закрытие.")
        
        sl_order_id = sl_order.get('orderId')
        logging.warning(f"{log_prefix} ✅ СТОП-ЛОСС УСПЕШНО УСТАНОВЛЕН (ID: {sl_order_id})")

        # --- TP1 (на частичный объем) ---
        tp1_order_id = None
        if qty_tp1 > 0:
            formatted_tp1_price = format_price(final_tp1_price, symbol, side=opposite_side_str)
            tp1_order = await place_order_async(
                symbol=symbol, side=opposite_side_str, order_type="TAKE_PROFIT_MARKET",
                quantity=qty_tp1,
                stopPrice=formatted_tp1_price,
                reduce_only=True
            )
            if tp1_order and tp1_order.get('orderId'):
                tp1_order_id = tp1_order.get('orderId')
                logging.info(f"{log_prefix} ✅ TP1 (Частичный) УСПЕШНО УСТАНОВЛЕН (ID: {tp1_order_id})")
            else:
                logging.warning(f"{log_prefix} ⚠️ Не удалось установить TP1. Управление через 'manage'.")

        # --- TP2 (на частичный объем) ---
        tp2_order_id = None
        if qty_tp2 > 0:
            formatted_tp2_price = format_price(final_tp2_price, symbol, side=opposite_side_str)
            tp2_order = await place_order_async(
                symbol=symbol, side=opposite_side_str, order_type="TAKE_PROFIT_MARKET",
                quantity=qty_tp2,
                stopPrice=formatted_tp2_price,
                reduce_only=True
            )
            if tp2_order and tp2_order.get('orderId'):
                tp2_order_id = tp2_order.get('orderId')
                logging.info(f"{log_prefix} ✅ TP2 (Частичный) УСПЕШНО УСТАНОВЛЕН (ID: {tp2_order_id})")
            else:
                logging.warning(f"{log_prefix} ⚠️ Не удалось установить TP2. Управление через 'manage'.")


        # --- 6. Возвращаем данные для создания PositionManager ---
        return {
            "symbol": symbol, "side": side, "entry_price": actual_entry_price,
            "amount": total_qty_filled, "initial_amount": total_qty_filled,
            "sl_price": final_sl_price,
            "initial_sl_price": final_sl_price,
            "tp1_price": final_tp1_price, # Сохраняем виртуальную цель 1
            "tp2_price": final_tp2_price, # Сохраняем виртуальную цель 2
            "entry_context": entry_context,
            "meta": {
                'sl_order_id': sl_order_id, # ID УСПЕШНО установленного SL
                'tp1_order_id': tp1_order_id, # ID TP1
                'tp2_order_id': tp2_order_id, # ID TP2
                'scaling_plan': {}, # Оставляем пустым, как в v4.9
                'atr_trail_active': False # Трейлинг по умолчанию выключен
            }
        }
    
    except Exception as e:
        # --- 7. Блок обработки критических ошибок (вызывает close_position_emergency) ---
        logging.critical(f"{log_prefix} КРИТИЧЕСКАЯ ОШИБКА ИСПОЛНЕНИЯ ВХОДА ИЛИ УСТАНОВКИ SL: {e}. Запускаю аварийное закрытие...", exc_info=True)
        qty_to_close_on_error = quantity
        if total_qty_filled > 0: qty_to_close_on_error = total_qty_filled
        
        # Отменяем SL, если он вдруг успел поставиться
        if 'sl_order_id' in locals() and locals()['sl_order_id']:
             await cancel_single_order_async(symbol, locals()['sl_order_id'])

        if qty_to_close_on_error > 0:
             await close_position_emergency(
                  symbol,
                  {'symbol': symbol, 'side': side, 'amount': qty_to_close_on_error, 'entry_price': actual_entry_price or signal_price},
                  f"Откат из-за крит. ошибки при входе/установке SL (v5.0): {e}"
             )
        return None
# --- КОНЕЦ ОБНОВЛЕННОЙ ФУНКЦИИ ---
#

def get_symbol_price_step(symbol: str) -> Optional[float]:
    """Извлекает шаг цены (tickSize) из кэша биржевой информации."""
    global exchange_info_cache
    try:
        symbol_info = exchange_info_cache.get(symbol)
        if symbol_info and 'tickSize' in symbol_info:
            return float(symbol_info['tickSize'])
        return None
    except (ValueError, KeyError, TypeError):
        logging.warning(f"[{symbol}] Не удалось получить tickSize из exchange_info_cache.")
        return None

def normalize_sltp_for_side(
    side: str,
    entry: float,
    sl: float,
    tp: float,
    *,
    min_rr: float = 1.2,
    price_step: Optional[float] = None
) -> tuple[float, float, list[str]]:
    """
    Гарантирует корректную геометрию брекетов:
    - LONG: SL < entry < TP и TP >= entry + min_rr * (entry - SL)
    - SHORT: TP < entry < SL и TP <= entry - min_rr * (SL - entry)
    Возвращает (sl_fixed, tp_fixed, warnings)
    """
    notes = []
    order_side = side.upper()
    risk = abs(entry - sl) if sl is not None else None

    def rnd(x: float) -> float:
        if price_step and price_step > 0:
            k = round(x / price_step)
            return float(f"{k * price_step:.8f}")
        return float(f"{x:.8f}")

    if order_side == "BUY": # Для LONG сделок
        if sl is None or sl >= entry:
            sl = entry - (risk if risk else max(entry * 0.002, 1e-6))
            notes.append("Fixed SL for LONG below entry")
        
        risk_dist = entry - sl
        min_tp = entry + max(min_rr * risk_dist, risk_dist)
        
        if tp is None or tp <= entry or tp < min_tp:
            tp = min_tp
            notes.append(f"Fixed TP for LONG above entry with min RR >={min_rr}")

    else:  # SELL, для SHORT сделок
        if sl is None or sl <= entry:
            sl = entry + (risk if risk else max(entry * 0.002, 1e-6))
            notes.append("Fixed SL for SHORT above entry")

        risk_dist = sl - entry
        max_tp = entry - max(min_rr * risk_dist, risk_dist)

        if tp is None or tp >= entry or tp > max_tp:
            tp = max_tp
            notes.append(f"Fixed TP for SHORT below entry with min RR >={min_rr}")
            
    return rnd(sl), rnd(tp), notes


async def get_btc_trend_strength() -> str:
    """
    Определяет силу и направление тренда BTC на основе ADX и положения цены относительно EMA.
    Возвращает одно из состояний: 'Strong Up', 'Weak Up', 'Sideways', 'Weak Down', 'Strong Down'.
    """
    try:
        # Используем данные по BTCUSDT с часового таймфрейма для анализа тренда
        btc_df = market_data_store.get('BTCUSDT', {}).get(ALLOWED_INTERVALS['1h'])
        
        if btc_df is None or len(btc_df) < 50:
            logging.warning("[BTC Trend] Недостаточно данных для определения тренда BTC.")
            return 'Sideways' # Возвращаем нейтральное значение по умолчанию

        # Расчет индикаторов
        from ta.trend import adx, ema_indicator
        adx_indicator = adx(btc_df['high'], btc_df['low'], btc_df['close'], window=14)
        ema_200 = ema_indicator(btc_df['close'], window=200)
        
        last_price = btc_df['close'].iloc[-1]
        last_adx = adx_indicator.iloc[-1]
        last_ema = ema_200.iloc[-1]

        # Логика определения состояния
        is_uptrend = last_price > last_ema
        
        if last_adx > 25: # Сильный тренд
            if is_uptrend:
                return 'Strong Up'
            else:
                return 'Strong Down'
        elif last_adx >= 20: # Слабый, но заметный тренд
             if is_uptrend:
                return 'Weak Up'
             else:
                return 'Weak Down'
        else: # Отсутствие тренда (флэт)
            return 'Sideways'
            
    except Exception as e:
        logging.error(f"[BTC Trend] Ошибка при определении тренда BTC: {e}")
        return 'Sideways' # Безопасное значение в случае ошибки

def get_dynamic_risk_percent(symbol: str) -> float:
    """
    Рассчитывает динамический процент риска на основе текущего рыночного режима HMM.
    """
    global market_regimes, REGIME_CONFIG, BASE_RISK_PERCENT

    # Получаем текущий режим для символа
    regime = market_regimes.get(symbol)

    if regime is not None:
        # Получаем модификатор риска из конфигурации для этого режима
        regime_config = REGIME_CONFIG.get(regime, {})
        risk_modifier = regime_config.get('risk_modifier', 1.0) # 1.0 - значение по умолчанию
        
        # Рассчитываем и возвращаем скорректированный риск
        adjusted_risk = BASE_RISK_PERCENT * risk_modifier
        logging.info(f"[{symbol}] Динамический риск: Режим={regime}, Модификатор={risk_modifier:.2f} -> Итоговый риск={adjusted_risk:.2f}%")
        return adjusted_risk
    
    # Если режим не определен, возвращаем базовый риск
    return BASE_RISK_PERCENT

def calculate_atr_trailing_stop(symbol: str, side: str, current_price: float, current_sl: float, multiplier: float) -> Optional[float]:
    """
    Рассчитывает новую позицию для трейлинг-стопа на основе ATR с динамическим множителем.
    """
    if not ENABLE_ATR_TRAILING_STOP:
        return None

    df = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS.get('5m'))
    if df is None or len(df) < TRAILING_STOP_ATR_PERIOD:
        return None
        
    try:
        atr_value = ta.volatility.average_true_range(
            df['high'], df['low'], df['close'], window=TRAILING_STOP_ATR_PERIOD
        ).iloc[-1]
        
        if pd.isna(atr_value) or atr_value <= 0:
            return None

        distance = atr_value * multiplier # Используем переданный множитель
        new_potential_sl = 0.0

        if side.upper() == 'LONG':
            new_potential_sl = current_price - distance
            if new_potential_sl > current_sl:
                return new_potential_sl
                
        elif side.upper() == 'SHORT':
            new_potential_sl = current_price + distance
            if new_potential_sl < current_sl:
                return new_potential_sl
                
        return None

    except Exception as e:
        logging.error(f"[{symbol}] Ошибка в расчете ATR Trailing Stop: {e}")
        return None

def calculate_unified_entry_score_v2(
    symbol: str, side: str, ml_confidence: float, llm_confidence: float,
    derivatives_sentiment: Dict[str, Optional[float]],
    microstructure: Dict[str, Optional[float]],
    garch_forecast: Optional[float], btc_trend_strength: str
) -> float:
    """
    Рассчитывает единый взвешенный скоринг-балл v3 с правом вето.
    """
    # === Право Вето: Блокируем заведомо плохие сделки ===
    # 1. Если базовая ML-модель совсем не уверена.
    MIN_ML_CONFIDENCE = 15 # Минимальная уверенность от ML-модели
    if ml_confidence < MIN_ML_CONFIDENCE:
        logging.warning(f"[{symbol}] СКОРИНГ ПРЕРВАН (ВЕТО): Уверенность ML ({ml_confidence:.1f}) ниже порога ({MIN_ML_CONFIDENCE}).")
        return 0.0

    # 2. Если мы пытаемся торговать против сильного тренда BTC.
    btc_trend_scores_map = {'Strong Up': 100, 'Weak Up': 70, 'Sideways': 50, 'Weak Down': 30, 'Strong Down': 0}
    btc_score_raw = btc_trend_scores_map.get(btc_trend_strength, 50)
    
    is_fighting_trend = (side == 'LONG' and btc_score_raw < 40) or (side == 'SHORT' and btc_score_raw > 60)
    if is_fighting_trend:
        logging.warning(f"[{symbol}] СКОРИНГ ПРЕРВАН (ВЕТО): Попытка входа в {side} против тренда BTC ({btc_trend_strength}).")
        return 0.0

    # === Если вето не сработало, продолжаем расчет ===
    weights = {
        "ml": 0.25,         # Увеличили вес ML, т.к. теперь мы доверяем ему больше
        "llm": 0.25,
        "obi": 0.15,
        "ls_ratio": 0.10,
        "btc_trend": 0.15,  # BTC тренд все еще очень важен
        "garch_vol": 0.10
    }
    
    scores = {}
    
    scores['ml'] = ml_confidence if ml_confidence is not None else 50
    scores['llm'] = llm_confidence if llm_confidence is not None else 50
    
    order_book_imbalance = microstructure.get('order_book_imbalance')
    scores['obi'] = (order_book_imbalance * 50 + 50) if side == 'LONG' else (-order_book_imbalance * 50 + 50) if order_book_imbalance is not None else 50

    long_short_ratio = derivatives_sentiment.get('long_short_ratio')
    if long_short_ratio and long_short_ratio > 0:
        score_val = 50 * math.log2(long_short_ratio) + 50
        scores['ls_ratio'] = max(0, min(100, score_val)) if side == 'LONG' else max(0, min(100, 100 - score_val))
    else:
        scores['ls_ratio'] = 50

    scores['btc_trend'] = btc_score_raw if side == 'LONG' else 100 - btc_score_raw
    
    if garch_forecast is not None:
        if garch_forecast < 1.5: scores['garch_vol'] = 90
        elif garch_forecast < 4.0: scores['garch_vol'] = 60
        else: scores['garch_vol'] = 20
    else:
        scores['garch_vol'] = 50

    final_score = sum(scores.get(key, 50) * weights[key] for key in weights)
    
    logging.info(f"[{symbol}] Расчет скоринга V3 для {side}: ML={scores['ml']:.0f}, LLM={scores['llm']:.0f}, OBI={scores['obi']:.0f}, LS={scores['ls_ratio']:.0f}, BTC={scores['btc_trend']:.0f}, GARCH={scores['garch_vol']:.0f} -> Итог: {final_score:.2f}")
    return final_score


# ##################################################################
# ## ЗАДАЧА 3: ПРОДВИНУТОЕ УПРАВЛЕНИЕ КАПИТАЛОМ                 ##
# ##################################################################

# --- БЛОК КОДА ДЛЯ УРОВНЯ 1: ЭВОЛЮЦИЯ ---

async def manage_donchian_trailing_stop(manager: PositionManager):
    """
    Управляет трейлинг-стопом на основе средней линии Канала Дончиана и ATR.
    """
    pos = manager.state
    df = market_data_store.get(manager.symbol, {}).get(ALLOWED_INTERVALS['1h'])
    
    if df is None or len(df) < 41: return

    try:
        _, _, middle_line = calculate_donchian_channels(df, period=40)
        atr_value = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14).iloc[-1]
        
        last_middle_line = middle_line.iloc[-1]
        if pd.isna(last_middle_line) or pd.isna(atr_value): return

        volatility_buffer = atr_value * 0.5
        
        if pos.side.upper() == 'LONG' and pos.sl_price is not None:
            potential_new_sl = last_middle_line - volatility_buffer
            if potential_new_sl > pos.sl_price:
                logging.warning(f"📈 [{manager.symbol}] TRAILING STOP (Donchian): SL перемещается с {pos.sl_price:.4f} на {potential_new_sl:.4f}")
                # --- ✅ ИСПРАВЛЕНИЕ: Убран лишний аргумент manager.symbol ---
                await manager._update_sl_callback(potential_new_sl, "Donchian Trail")

        elif pos.side.upper() == 'SHORT' and pos.sl_price is not None:
            potential_new_sl = last_middle_line + volatility_buffer
            if potential_new_sl < pos.sl_price:
                logging.warning(f"📉 [{manager.symbol}] TRAILING STOP (Donchian): SL перемещается с {pos.sl_price:.4f} на {potential_new_sl:.4f}")
                # --- ✅ ИСПРАВЛЕНИЕ: Убран лишний аргумент manager.symbol ---
                await manager._update_sl_callback(potential_new_sl, "Donchian Trail")

    except Exception as e:
        # Теперь здесь будет выводиться полная ошибка, если она не TypeError
        logging.error(f"[{manager.symbol}] Ошибка в логике трейлинга по Дончиану: {e}", exc_info=True)

def calculate_donchian_channels(df: pd.DataFrame, period: int = 40) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Рассчитывает верхнюю, нижнюю и среднюю линии Канала Дончиана."""
    upper = df['high'].rolling(window=period).max()
    lower = df['low'].rolling(window=period).min()
    middle = (upper + lower) / 2
    return upper, lower, middle

def find_next_structural_level(df: pd.DataFrame, side: str, lookback: int = 50) -> Optional[float]:
    """Находит ближайший уровень сопротивления (для лонга) или поддержки (для шорта)."""
    if df is None or len(df) < lookback:
        return None
    recent_data = df.iloc[-lookback:-1]
    try:
        if side.upper() == 'LONG':
            return recent_data['high'].max() # Ближайшее сопротивление
        else: # SHORT
            return recent_data['low'].min() # Ближайшая поддержка
    except Exception as e:
        logging.error(f"Ошибка в find_next_structural_level: {e}")
        return None

# 2. Функция продвинутого многоуровневого трейлинга
# Замените вашу старую функцию на эту новую версию
async def advanced_trailing_stop(manager: "PositionManager", current_price: float, df: pd.DataFrame):
    """
    Управляет позицией с помощью многоступенчатого "протектора прибыли":
    1. Перевод в безубыток при RR >= 1.0.
    2. Частичная фиксация прибыли на уровнях RR >= 1.5 и RR >= 3.0.
    3. Прогрессивный ATR-трейлинг для защиты оставшейся позиции.
    """
    pos = manager.state
    if pos.initial_sl_price is None or pos.sl_price is None:
        return

    # --- Шаг 1: Расчет ключевых метрик (RR и ATR) ---
    # Используем ваши существующие вспомогательные функции
    current_rr = _rr(pos.side, pos.entry_price, pos.initial_sl_price, current_price)
    atr_value = _atr(df, period=14)
    if atr_value <= 0:
        # Логирование может быть полезно для отладки, если ATR не считается
        # logging.warning(f"[{manager.symbol}] Некорректное значение ATR: {atr_value}")
        return

    # --- Шаг 2: Жесткий перевод в безубыток (один раз) ---
    meta = getattr(pos, "meta", {}) # Получаем метаданные позиции или создаем пустой dict
    if not meta.get("be_done", False) and current_rr >= 1.0:
        # Добавляем небольшой буфер, чтобы покрыть комиссию/проскальзывание
        breakeven_buffer = 0.25 * atr_value
        if pos.side.upper() == "LONG":
            be_sl_price = pos.entry_price + breakeven_buffer
            # Обновляем, только если новый стоп-лосс выгоднее
            if be_sl_price > pos.sl_price:
                await manager._update_sl_callback(manager.symbol, be_sl_price, "BE lock + ATR buffer")
                meta["be_done"] = True
        elif pos.side.upper() == "SHORT":
            be_sl_price = pos.entry_price - breakeven_buffer
            # Обновляем, только если новый стоп-лосс выгоднее
            if be_sl_price < pos.sl_price:
                await manager._update_sl_callback(manager.symbol, be_sl_price, "BE lock + ATR buffer")
                meta["be_done"] = True
        setattr(pos, "meta", meta) # Сохраняем флаг, чтобы не делать это снова

    # --- Шаг 3: Частичная фиксация прибыли (один раз для каждого уровня) ---
    # Первая цель: RR >= 1.5
    if not meta.get("tp1_done", False) and current_rr >= 1.5:
        try:
            # Закрываем 25% позиции. Убедитесь, что у вас есть функция partial_close!
            await partial_close(manager.symbol, pos, 0.25, "TP1 @1.5R")
            meta["tp1_done"] = True
            setattr(pos, "meta", meta)
        except Exception as e:
            logging.error(f"[{manager.symbol}] Ошибка частичного закрытия TP1: {e}")

    # Вторая цель: RR >= 3.0
    if not meta.get("tp2_done", False) and current_rr >= 3.0:
        try:
            # Закрываем еще 25% позиции.
            await partial_close(manager.symbol, pos, 0.25, "TP2 @3R")
            meta["tp2_done"] = True
            setattr(pos, "meta", meta)
        except Exception as e:
            logging.error(f"[{manager.symbol}] Ошибка частичного закрытия TP2: {e}")

    # --- Шаг 4: Ужесточение ATR-трейлинга (ваша исходная логика) ---
    # Этот блок остается почти без изменений
    atr_multiplier = None
    if current_rr >= 4.0:
        atr_multiplier = 1.0   # Агрессивный
    elif current_rr >= 2.5:
        atr_multiplier = 1.5   # Средний
    elif current_rr >= 1.5:
        atr_multiplier = 2.0   # Консервативный
    else:
        return # Не трейлим до 1.5R (кроме безубытка)

    distance = atr_value * atr_multiplier
    
    if pos.side.upper() == 'LONG':
        new_sl = current_price - distance
        # Обновляем, только если новый SL выше текущего
        if new_sl > pos.sl_price and new_sl < current_price:
            reason = f"Adv Trail (RR={current_rr:.1f}, ATRx{atr_multiplier})"
            await manager._update_sl_callback(manager.symbol, new_sl, reason)
    
    elif pos.side.upper() == 'SHORT':
        new_sl = current_price + distance
        # Обновляем, только если новый SL ниже текущего
        if new_sl < pos.sl_price and new_sl > current_price:
            reason = f"Adv Trail (RR={current_rr:.1f}, ATRx{atr_multiplier})"
            await manager._update_sl_callback(manager.symbol, new_sl, reason)

async def get_regime_and_trend_adjusted_sltp(symbol: str, side: str, entry_price: float, regime: int) -> Dict[str, Optional[float]]:
    """
    Рассчитывает SL/TP на основе режима HMM и дополнительно корректирует TP на основе силы тренда ADX.
    
    ИНТЕГРАЦИЯ:
    В вашей основной логике принятия решений (например, внутри `find_and_execute_trade_v4`),
    когда вы рассчитываете SL/TP, вызовите эту функцию ВМЕСТО `calculate_quantitative_sltp`.
    """
    # Шаг 1: Получаем базовые расчеты из вашей существующей функции
    base_sltp = await calculate_quantitative_sltp(symbol, side, entry_price, regime)
    
    if not base_sltp or not base_sltp.get("take_profit"):
        logging.warning(f"[{symbol}] Не удалось получить базовый SL/TP от 'calculate_quantitative_sltp'. Коррекция отменена.")
        return {"stop_loss": None, "take_profit": None}

    # Шаг 2: Получаем данные для расчета ADX
    # Используем средний таймфрейм для более стабильной оценки тренда
    df_medium = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS[ML_MODEL_TF_MEDIUM])
    if df_medium is None or len(df_medium) < 25: # ADX(14) требует ~25 периодов для разогрева
        logging.info(f"[{symbol}] Недостаточно данных для ADX-коррекции. Используются базовые SL/TP.")
        return base_sltp

    try:
        # Шаг 3: Рассчитываем ADX и применяем логику коррекции
        adx_value = ta.trend.adx(df_medium['high'], df_medium['low'], df_medium['close'], window=14).iloc[-1]
        
        base_tp = base_sltp["take_profit"]
        reward_distance = abs(base_tp - entry_price)
        
        tp_adjustment_factor = 1.0
        if adx_value > 30:  # Сильный тренд
            tp_adjustment_factor = 1.4  # Увеличиваем цель на 40%
            logging.info(f"[{symbol}] ADX={adx_value:.1f} (сильный тренд). Цель TP УВЕЛИЧЕНА.")
        elif adx_value < 20:  # Слабый тренд или флэт
            tp_adjustment_factor = 0.8  # Уменьшаем цель на 20%
            logging.info(f"[{symbol}] ADX={adx_value:.1f} (слабый тренд). Цель TP УМЕНЬШЕНА.")
            
        # Если фактор изменился, пересчитываем TP
        if tp_adjustment_factor != 1.0:
            if side.upper() == "LONG":
                base_sltp["take_profit"] = entry_price + (reward_distance * tp_adjustment_factor)
            else:  # SHORT
                base_sltp["take_profit"] = entry_price - (reward_distance * tp_adjustment_factor)
            
            logging.warning(f"[{symbol}] Скорректированный TP: {base_sltp['take_profit']:.5f}")

    except Exception as e:
        logging.error(f"[{symbol}] Ошибка при ADX-коррекции SL/TP: {e}", exc_info=True)
        # В случае ошибки возвращаем базовые, не измененные значения
    
    return base_sltp

# --- БЛОК КОДА ДЛЯ УРОВНЯ 2: ГИБРИДИЗАЦИЯ ---



async def execute_ranging_strategy(symbol: str, df: pd.DataFrame) -> None:
    """
    Реализует торговую логику для флэтового рынка (возврат к среднему).
    --- ОБНОВЛЕНО: Добавлено детальное логирование причин отсутствия сигнала. ---
    """

    # --- Параметры Стратегии ---
    RSI_OVERSOLD = 32.0
    RSI_OVERBOUGHT = 68.0
    RSI_PERIOD = 14
    BB_PERIOD = 20
    BB_STD_DEV = 2.0
    RISK_PERCENT = 0.75

    try:
        # --- Расчет индикаторов ---
        rsi_value = ta.momentum.rsi(df['close'], window=RSI_PERIOD).iloc[-1]
        bollinger = ta.volatility.BollingerBands(df['close'], window=BB_PERIOD, window_dev=BB_STD_DEV)
        bb_high = bollinger.bollinger_hband().iloc[-1]
        bb_low = bollinger.bollinger_lband().iloc[-1]
        entry_price = df['close'].iloc[-1]

        trade_type = None
        sl_price = None
        tp_price = None

        # --- Условия для входа ---
        is_long_rsi = rsi_value < RSI_OVERSOLD
        is_short_rsi = rsi_value > RSI_OVERBOUGHT

        if is_long_rsi:
            trade_type = 'long'
            side_str = 'BUY'
            sl_price = bb_low * 0.998
            tp_price = bb_high
            logging.warning(f"📈 [{symbol}] RANGING СИГНАЛ: ЛОНГ. RSI ({rsi_value:.2f}) < {RSI_OVERSOLD}")

        elif is_short_rsi:
            trade_type = 'short'
            side_str = 'SELL'
            sl_price = bb_high * 1.002
            tp_price = bb_low
            logging.warning(f"📉 [{symbol}] RANGING СИГНАЛ: ШОРТ. RSI ({rsi_value:.2f}) > {RSI_OVERBOUGHT}")
        
        else:
            # --- НОВЫЙ БЛОК ЛОГИРОВАНИЯ ---
            log_msg = (
                f"[{symbol}] Нет сигнала RANGING. "
                f"Long Check (RSI<{RSI_OVERSOLD}): FAIL (RSI:{rsi_value:.2f}). "
                f"Short Check (RSI>{RSI_OVERBOUGHT}): FAIL (RSI:{rsi_value:.2f})."
            )
            logging.info(log_msg)
            # --- КОНЕЦ НОВОГО БЛОКА ---

        # --- Исполнение сделки ---
        if trade_type and sl_price and tp_price:
            risk_dist = abs(entry_price - sl_price)
            reward_dist = abs(tp_price - entry_price)
            
            if risk_dist == 0 or (reward_dist / risk_dist) < 1.2:
                logging.warning(f"[{symbol}] Ranging-сделка отменена: низкое R:R ({reward_dist/risk_dist if risk_dist > 0 else 0 :.2f})")
                return

            quantity = calculate_fixed_risk_quantity(symbol, entry_price, sl_price, RISK_PERCENT)
            if quantity <= 0:
                logging.warning(f"[{symbol}] Ranging-сделка отменена: расчетный объем равен нулю.")
                return

            logging.info(f"[{symbol}] Все проверки для Ranging-стратегии пройдены. Вход в сделку.")
            await place_robust_market_entry(
                symbol=symbol,
                side_str=side_str,
                quantity=qty,
                entry_price=entry_price,
                sl_price=sl_price,
                tp_price=tp_price,
                entry_context=f"Strategy: Ranging (RSI: {rsi_value:.1f})"
            )

    except Exception as e:
        logging.error(f"[{symbol}] Ошибка в execute_ranging_strategy: {e}", exc_info=True)



def find_structural_level(df: pd.DataFrame, side: str, lookback: int = 25) -> Optional[float]:
    """
    Находит ближайший значимый локальный экстремум для установки структурного стопа.
    """
    if df is None or len(df) < lookback:
        return None
    
    # Берем данные за последние `lookback` свечей, ИСКЛЮЧАЯ последнюю (текущую)
    recent_data = df.iloc[-lookback:-1]
    
    try:
        if side.upper() == 'LONG':
            # Ищем самый низкий 'low' за период
            return recent_data['low'].min()
        else:  # SHORT
            # Ищем самый высокий 'high' за период
            return recent_data['high'].max()
    except Exception as e:
        logging.error(f"Ошибка в find_structural_level: {e}")
        return None

async def calculate_hybrid_stop_loss(symbol: str, side: str, entry_price: float, regime: int) -> Optional[float]:
    """
    Рассчитывает гибридный SL, выбирая более безопасный вариант
    --- ВЕРСИЯ 2: УВЕЛИЧЕННЫЙ БУФЕР для структурного уровня ---
    """
    # 1. Получаем стоп по волатильности (ATR) - Используем новую версию V2
    sltp_atr = await calculate_quantitative_sltp(symbol, side, entry_price, regime) # <-- Вызываем V2
    stop_loss_atr = sltp_atr.get("stop_loss")

    if not stop_loss_atr:
        logging.error(f"[{symbol}] Гибридный SL невозможен: не удалось рассчитать ATR-стоп.")
        return None # Невозможно рассчитать гибридный стоп без ATR

    # 2. Получаем структурный стоп
    # Используем 15m TF для поиска структуры
    df_short = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS[ML_MODEL_TF_SHORT])
    stop_loss_structure_raw = find_structural_level(df_short, side, lookback=30) # Увеличим lookback до 30

    if not stop_loss_structure_raw:
        logging.info(f"[{symbol}] Не найден структурный уровень. Используется только ATR-стоп.")
        return stop_loss_atr

    # --- ✅ ИЗМЕНЕНИЕ: УВЕЛИЧЕННЫЙ БУФЕР ---
    atr_val = _atr(df_short, period=14) # ATR для буфера
    if atr_val <= 0:
        buffer = abs(entry_price * 0.001) # Минимальный буфер, если ATR=0
    else:
        buffer = atr_val * STRUCTURAL_SL_BUFFER_ATR_MULT # Используем новую константу (1.0 * ATR)

    stop_loss_structure_buffered = stop_loss_structure_raw - buffer if side.upper() == 'LONG' else stop_loss_structure_raw + buffer
    # --- КОНЕЦ ИЗМЕНЕНИЯ ---

    logging.info(f"[{symbol}] Сравнение SL: ATR-стоп={stop_loss_atr:.5f}, Структурный-стоп={stop_loss_structure_buffered:.5f} (raw: {stop_loss_structure_raw:.5f}, buffer:{buffer:.5f})")

    # 3. Выбираем самый безопасный (дальний) стоп
    if side.upper() == 'LONG':
        final_stop = min(stop_loss_atr, stop_loss_structure_buffered)
    else: # SHORT
        final_stop = max(stop_loss_atr, stop_loss_structure_buffered)

    reason = "ATR" if abs(final_stop - stop_loss_atr) < 1e-9 else "Структура+Буфер"
    logging.warning(f"[{symbol}] Выбран гибридный SL (V2): {final_stop:.5f} (на основе: {reason})")

    return final_stop

# --- НОВЫЕ ФУНКЦИИ ДЛЯ RANGING V4.3 ---

# --- Детектор Флэтового Режима ---
def is_range_regime_v4_1(df15: pd.DataFrame, flat_adx=22.0, donch=55) -> bool:
    """
    Проверяет, соответствует ли рынок критериям флэта: низкий ADX, узкий BBW, нет недавних пробоев Дончиана.
    """
    if df15 is None or len(df15) < max(donch + 1, 60): # Увеличим запас для MA в BBW
        return False
    try:
        h, l, c = df15["high"].values, df15["low"].values, df15["close"].values
        # Используем talib для скорости, если доступен
        try:
            import talib as ta_lib
            adx_val = float(ta_lib.ADX(h, l, c, timeperiod=14)[-1])
            mbb, ubb, lbb = ta_lib.BBANDS(c, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
            mbb_val, ubb_val, lbb_val = float(mbb[-1]), float(ubb[-1]), float(lbb[-1])
            atr_val = float(ta_lib.ATR(h, l, c, timeperiod=14)[-1])
        except ImportError:
            # Fallback на библиотеку ta
            adx_val = float(ta.trend.adx(df15['high'], df15['low'], df15['close'], window=14).iloc[-1])
            bb = ta.volatility.BollingerBands(df15['close'], window=20, window_dev=2.0)
            mbb_val = float(bb.bollinger_mavg().iloc[-1])
            ubb_val = float(bb.bollinger_hband().iloc[-1])
            lbb_val = float(bb.bollinger_lband().iloc[-1])
            atr_val = _atr(df15, 14) # Используем твой helper _atr

        bbw_abs = ubb_val - lbb_val
        bbw_rel = (bbw_abs / max(mbb_val, 1e-9)) if mbb_val > 0 else 0.0 # Относительная ширина

        # Проверка пробоя Дончиана (на ПРЕДПОСЛЕДНЕЙ свече, чтобы пробой был уже ФАКТОМ)
        donch_high = df15['high'].iloc[-(donch+1):-1].max()
        donch_low = df15['low'].iloc[-(donch+1):-1].min()
        last_high = h[-1]
        last_low = l[-1]
        broke_out = (last_high > donch_high) or (last_low < donch_low)

        # Критерии: ADX ниже порога, ОТНОСИТЕЛЬНАЯ ширина BBW умеренная, ПРОБОЯ НЕТ
        is_range = (adx_val < flat_adx) and (bbw_rel < 0.08) and (not broke_out) # 8% ширина BBW

        logging.info(f"[{df15.index.name}] Range Regime Check: ADX={adx_val:.1f}<{flat_adx}? {'Да' if adx_val < flat_adx else 'Нет'}, "
                     f"BBW%={bbw_rel:.3f}<0.08? {'Да' if bbw_rel < 0.08 else 'Нет'}, "
                     f"BrokeOut? {'Да' if broke_out else 'Нет'} -> Regime Range? {'Да' if is_range else 'Нет'}")
        return is_range

    except Exception as e_regime:
        logging.error(f"[Range Regime Check] Ошибка: {e_regime}")
        return False

# --- Функция Расчета Сигнала Ranging ---
async def compute_ranging_signal_v4_1(symbol: str) -> dict:
    """
    Ищет сигнал на вход во флэте от границ AVWAP с подтверждением RSI и OFI/OBI.
    Рассчитывает SL за AVWAP + буфер ATR, TP1=медиана, TP2=противоположная граница.
    """
    global market_data_store, ALLOWED_INTERVALS, client # Доступ к глобальным переменным

    df15 = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS.get('15m'))
    if df15 is None or len(df15) < 60:
        return {"ok": False, "reason": "Insufficient 15m data"}

    # 1. Проверяем, находимся ли мы во флэтовом режиме
    if not is_range_regime_v4_1(df15, flat_adx=FLAT_ADX_THRESHOLD): # Используем глобальный FLAT_ADX_THRESHOLD
        return {"ok": False, "reason": "Not in Range regime"}

    # 2. Рассчитываем уровни AVWAP
    avwap_levels = await calculate_avwap_levels(symbol, df15, lookback=60)
    if not avwap_levels:
        return {"ok": False, "reason": "Failed to calculate AVWAP levels"}

    lbb_avwap = avwap_levels['lower']
    mbb_avwap = avwap_levels['median']
    ubb_avwap = avwap_levels['upper']

    # 3. Расчет RSI и ATR
    try:
        rsi_val = float(ta.momentum.rsi(df15['close'], window=14).iloc[-1])
        atr_val = _atr(df15, 14)
        if atr_val <= 0: raise ValueError("Invalid ATR")
    except Exception as e_ind:
        return {"ok": False, "reason": f"Indicator calculation error: {e_ind}"}

    # 4. Получение OFI/OBI и текущей цены
    try:
        price = float(df15['close'].iloc[-1])
        ofi_score, _ = await calculate_order_flow_imbalance(symbol)
        micro = await get_market_microstructure_features(symbol, client)
        obi_val = float(micro.get("order_book_imbalance", 0.0)) if micro else 0.0
    except Exception as e_context:
         return {"ok": False, "reason": f"Context data error: {e_context}"}

    # 5. Проверка условий входа
    # Используем глобальные RANGING_RSI_OVERSOLD/OVERBOUGHT
    long_edge = (price <= lbb_avwap * 1.001) and (rsi_val <= RANGING_RSI_OVERSOLD) and (ofi_score >= 5 or obi_val >= 0.10)
    short_edge = (price >= ubb_avwap * 0.999) and (rsi_val >= RANGING_RSI_OVERBOUGHT) and (ofi_score <= -5 or obi_val <= -0.10)

    # 6. Расчет SL/TP и возврат сигнала
    if long_edge:
        # SL = Нижняя граница AVWAP - Буфер ATR
        sl_buffer = atr_val * 0.75 # Используем увеличенный буфер
        sl = lbb_avwap - sl_buffer
        tp1 = mbb_avwap # Медиана
        tp2 = ubb_avwap # Верхняя граница
        logging.info(f"[{symbol}] Ranging Signal Found: LONG at {price:.5f}")
        return {"ok": True, "side": "LONG", "entry": price, "sl": sl, "tp1": tp1, "tp2": tp2, "why": f"AVWAP Low {lbb_avwap:.4f} + RSI {rsi_val:.1f}"}

    if short_edge:
        # SL = Верхняя граница AVWAP + Буфер ATR
        sl_buffer = atr_val * 0.75 # Используем увеличенный буфер
        sl = ubb_avwap + sl_buffer
        tp1 = mbb_avwap # Медиана
        tp2 = lbb_avwap # Нижняя граница
        logging.info(f"[{symbol}] Ranging Signal Found: SHORT at {price:.5f}")
        return {"ok": True, "side": "SHORT", "entry": price, "sl": sl, "tp1": tp1, "tp2": tp2, "why": f"AVWAP High {ubb_avwap:.4f} + RSI {rsi_val:.1f}"}

    return {"ok": False, "reason": "No entry edge"}

# --- Атомарная Перестановка Ордеров ---
async def replace_order_safe(pm: PositionManager, key: str, create_fn, reason: str):
    """
    АТОМАРНАЯ ВЕРСИЯ: Сначала ставит новый ордер, ждет подтверждения, потом отменяет старый.
    """
    symbol = pm.symbol
    pos = pm.state
    old_oid = pos.meta.get(key)
    log_prefix = f"[{symbol}] [Replace '{key}']"

    logging.info(f"{log_prefix} Попытка установки нового ордера ({reason})...")
    new_order = await create_fn() # Выполняем функцию создания нового ордера

    if new_order and new_order.get("orderId"):
        new_oid = new_order.get("orderId")
        logging.info(f"{log_prefix} Новый ордер {new_oid} размещен. Ожидание статуса NEW...")
        # --- ОЖИДАНИЕ СТАТУСА ---
        if await wait_order_new(symbol, new_oid, timeout_s=4.0):
            logging.info(f"{log_prefix} Статус NEW для {new_oid} подтвержден.")
            # --- ТОЛЬКО ТЕПЕРЬ ОТМЕНЯЕМ СТАРЫЙ ---
            if old_oid and old_oid != new_oid: # Не отменяем, если ID совпали (маловероятно)
                logging.info(f"{log_prefix} Отмена старого ордера {old_oid}...")
                await cancel_single_order_async(symbol, old_oid)
                # Удаляем старый ID из meta только если он там был под этим ключом
                if pos.meta.get(key) == old_oid:
                     pos.meta.pop(key, None)

            # Сохраняем ID нового ордера и обновляем состояние
            pos.meta[key] = new_oid
            await save_state_async()
            logging.warning(f"{log_prefix} ✅ Ордер успешно заменен на {new_oid} ({reason}).")
            return True # Успех
        else:
            # Если новый ордер не получил статус NEW - КРИТИЧЕСКАЯ ОШИБКА
            logging.critical(f"{log_prefix} ❌ КРИТИЧЕСКАЯ ОШИБКА: Новый ордер {new_oid} не получил статус NEW! Попытка отмены...")
            await cancel_single_order_async(symbol, new_oid) # Пытаемся отменить "плохой" ордер
            # Старый ордер НЕ отменяем!
            await send_telegram_message_safe(f"🚨 **{symbol}**: Крит. ошибка при замене ордера ({reason}). Проверьте вручную!")
            return False # Неудача
    else:
        # Если даже разместить не удалось
        logging.error(f"{log_prefix} ❌ НЕ УДАЛОСЬ разместить новый ордер ({reason}). Старый ордер (если был) остается активным.")
        return False # Неудача
# --- КОНЕЦ НОВЫХ ФУНКЦИЙ ---

async def check_btc_for_emergency_exit():
    """
    v3.1: Исправлен вызов обновления SL, добавлено детальное логирование метрик.
    Мульти-валидированный импульс BTC с NATR, BBW, OBI, корреляцией и ступенями действий.
    """
    log_main_prefix = "[BTC Flash Guard v3.1]"
    logging.info(f"{log_main_prefix} 🚨 Запущен.")

    # --- Пороговые настройки (остаются прежними) ---
    TFS = ['1m', '5m']
    NATR_THR = {'L1': 0.70, 'L2': 1.00, 'L3': 1.50}  # % (ATR/Close*100)
    BODY_ATR = {'L1': 1.20, 'L2': 1.80, 'L3': 2.50}  # тело/ATR
    VOL_SPIKE = {'L1': 1.8,  'L2': 2.2,  'L3': 2.8}   # volume / MA20
    BBW_MULT  = {'L1': 1.20, 'L2': 1.30, 'L3': 1.50} # BBW / MA50(BBW)
    OBI_LIFT  = 0.35
    CORR_THR  = 0.60
    CHECK_EVERY_SEC = 10
    COOLDOWN_SEC = 600
    BLOCK_NEW_ENTRIES_SEC = 300
    PARTIAL_CLOSE_L2 = 0.25

    global trading_pause_until, skip_new_entries_until
    trading_pause_until = 0
    skip_new_entries_until = 0

    # --- Вспомогательные функции (без изменений) ---
    def last_closed(df):
        return df.iloc[-2] if df is not None and len(df) >= 2 else None

    def bbw_and_ma(df, w=20, dev=2, ma=50):
        bb = ta.volatility.BollingerBands(close=df['close'], window=w, window_dev=dev)
        bbw = bb.bollinger_wband()
        # Добавим проверку на NaN перед расчетом среднего
        bbw_ma_val = bbw.rolling(ma).mean().iloc[-2]
        return float(bbw.iloc[-2]), float(bbw_ma_val) if pd.notna(bbw_ma_val) else 0.0

    def tf_metrics(df):
        if df is None or len(df) < 60: return None
        c = last_closed(df)
        if c is None: return None
        try:
            atr = average_true_range(df['high'], df['low'], df['close'], 14).iloc[-2]
            close = float(c['close']); open_ = float(c['open'])
            vol = float(c['volume'])
            # Убедимся, что индекс не выходит за границы при расчете MA
            vol_ma20_series = df['volume'].rolling(20).mean()
            vol_ma20 = float(vol_ma20_series.iloc[-3]) if len(vol_ma20_series) > 2 else 0.0

            bbw, bbw_ma = bbw_and_ma(df)
            body = abs(close - open_)
            natr_pct = float(atr / close * 100) if close > 0 and atr > 0 else 0.0
            body_atr = float(body / atr) if atr > 0 else 0.0
            vol_spike = float(vol / vol_ma20) if vol_ma20 > 0 else 1.0
            bbw_mult = float(bbw / bbw_ma) if bbw_ma > 0 else 1.0
            direction = 'UP' if close > open_ else 'DOWN'

            metrics_dict = dict(natr_pct=natr_pct, body_atr=body_atr, vol_spike=vol_spike, bbw_mult=bbw_mult, direction=direction)
            # Логируем рассчитанные метрики
            # logging.info(f"{log_main_prefix} Метрики BTC ({df.index.name}): " + ", ".join([f"{k}={v:.2f}" for k,v in metrics_dict.items() if isinstance(v, float)]) + f", Dir={direction}")
            return metrics_dict
        except Exception as e_metrics:
            logging.error(f"{log_main_prefix} Ошибка расчета метрик: {e_metrics}")
            return None

    def meets(level, m) -> bool:
        """Проверяет, соответствуют ли метрики заданному уровню L1/L2/L3."""
        # --- ✅ ДОБАВЛЕНО ЛОГИРОВАНИЕ СРАВНЕНИЯ ---
        passed = (m['natr_pct'] >= NATR_THR[level] and
                  m['body_atr'] >= BODY_ATR[level] and
                  m['vol_spike'] >= VOL_SPIKE[level] and
                  m['bbw_mult']  >= BBW_MULT[level])
        # --- ЗАКОММЕНТИРУЙ ИЛИ УДАЛИ ЭТУ СТРОКУ ---
        # logging.info(f"{log_main_prefix} Проверка уровня {level}: "
        #              f"NATR={m['natr_pct']:.2f}>={NATR_THR[level]}? {'Да' if m['natr_pct'] >= NATR_THR[level] else 'Нет'}, "
        #              f"BodyATR={m['body_atr']:.2f}>={BODY_ATR[level]}? {'Да' if m['body_atr'] >= BODY_ATR[level] else 'Нет'}, "
        #              f"VolSpike={m['vol_spike']:.2f}>={VOL_SPIKE[level]}? {'Да' if m['vol_spike'] >= VOL_SPIKE[level] else 'Нет'}, "
        #              f"BBWMult={m['bbw_mult']:.2f}>={BBW_MULT[level]}? {'Да' if m['bbw_mult'] >= BBW_MULT[level] else 'Нет'} "
        #              f"-> Результат: {'ПРОЙДЕНО' if passed else 'НЕ ПРОЙДЕНО'}")
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---
        return passed
        # --- КОНЕЦ ЛОГИРОВАНИЯ ---

    def aggregate_level(mets):
        hits = []
        for tf, m in mets.items():
            if not m: continue # Пропускаем, если метрики не рассчитались
            # logging.info(f"{log_main_prefix} --- Проверка порогов для BTC {tf} ---")
            # Проверяем уровни от L1 до L3
            if meets('L1', m): hits.append('L1')
            if meets('L2', m): hits.append('L2')
            if meets('L3', m): hits.append('L3')
            # logging.info(f"{log_main_prefix} --- Конец проверки {tf} ---")
        if not hits:
            return None
        # Возвращаем максимальный сработавший уровень
        return max(hits, key=lambda x: {'L1':1, 'L2':2, 'L3':3}[x])

    async def get_obi_abs():
        try:
            # Используем futures_order_book
            ob = await client.futures_order_book(symbol='BTCUSDT', limit=20)
            bids = ob.get('bids', [])
            asks = ob.get('asks', [])
            if not bids or not asks: return None
            bid_vol = sum(float(q) for _, q in bids[:10])
            ask_vol = sum(float(q) for _, q in asks[:10])
            tot = bid_vol + ask_vol
            if tot <= 0: return None
            return abs((bid_vol - ask_vol) / tot)
        except Exception as e_obi:
            logging.error(f"{log_main_prefix} Ошибка получения OBI: {e_obi}")
            return None

    # Вспомогательные функции partial_close и close_position_emergency остаются без изменений
    async def partial_close(symbol, pos, frac, note):
        side = 'SELL' if pos['side'] == 'LONG' else 'BUY'
        qty = float(format_quantity(pos['amount'] * frac, symbol) or 0.0)
        if qty <= 0: return
        logging.warning(f"{log_main_prefix} [{symbol}] Частичное закрытие {frac*100:.0f}% ({qty}). Причина: {note}")
        await place_order_async(symbol=symbol, side=side, order_type="MARKET", quantity=qty, reduce_only=True)
        await send_telegram_message_safe(f"⚠️ **{symbol}**: частичное сокращение {frac*100:.0f}% ({note})")

    # --- Основной Цикл Мониторинга ---
    while True:
        try:
            await asyncio.sleep(CHECK_EVERY_SEC)
            now = time.time()
            if now < trading_pause_until:
                logging.info(f"{log_main_prefix} На паузе до {datetime.datetime.fromtimestamp(trading_pause_until).strftime('%H:%M:%S')}")
                continue

            # 1. Собираем метрики для BTC на 1м и 5м
            btc_mets = {}
            directions = []
            for tf in TFS:
                tf_api = ALLOWED_INTERVALS.get(tf)
                df = market_data_store.get('BTCUSDT', {}).get(tf_api)
                m = tf_metrics(df) # Расчет метрик
                if m:
                    btc_mets[tf] = m
                    directions.append(m['direction'])

            if not btc_mets:
                logging.debug(f"{log_main_prefix} Нет метрик BTC для анализа.")
                continue

            # 2. Определяем максимальный уровень опасности L1/L2/L3
            level = aggregate_level(btc_mets)
            if not level:
                logging.debug(f"{log_main_prefix} Уровни опасности не достигнуты.")
                continue # Ни один порог не пройден

            # 3. Проверяем согласованность направления импульса
            if len(set(directions)) > 1:
                logging.warning(f"{log_main_prefix} Направления импульса на {TFS} расходятся. Пропуск.")
                continue
            direction = directions[0]  # 'UP' / 'DOWN'
            logging.warning(f"{log_main_prefix} Обнаружен импульс BTC {direction} уровня {level}!")

            # 4. Проверяем OBI для возможного усиления уровня
            obi_abs = await get_obi_abs()
            if obi_abs is not None and obi_abs >= OBI_LIFT and level in ('L1', 'L2'):
                new_level_num = int(level[-1]) + 1
                level = f'L{new_level_num}'
                logging.warning(f"{log_main_prefix} Уровень усилен до {level} из-за OBI={obi_abs:.2f} >= {OBI_LIFT}")

            # 5. Определяем целевые позиции для закрытия/управления
            side_to_close = 'LONG' if direction == 'DOWN' else 'SHORT'
            targets = []
            for symbol, mgr in list(current_positions.items()):
                if symbol == 'BTCUSDT': continue # Не трогаем сам BTC
                if not isinstance(mgr, PositionManager): continue # Пропускаем, если не менеджер

                pos_state_dict = mgr.get_state_dict() # Получаем словарь состояния
                if pos_state_dict.get('side') != side_to_close: continue # Пропускаем попутные

                # Проверяем корреляцию
                try:
                    corr = await get_correlation_snapshot(symbol, 'BTCUSDT')
                except Exception:
                    corr = 1.0 # Считаем 1.0 при ошибке для безопасности

                if abs(corr) >= CORR_THR:
                    targets.append(mgr) # Добавляем объект менеджера
                else:
                    logging.info(f"{log_main_prefix} [{symbol}] Пропуск: Низкая корреляция ({corr:.2f} < {CORR_THR})")

            if not targets:
                logging.info(f"{log_main_prefix} Нет целевых позиций ({side_to_close}) для реакции на импульс BTC.")
                continue

            # --- 6. Выполнение Действий по Уровням ---
            logging.warning(f"{log_main_prefix} Обнаружено {len(targets)} позиций для реакции на BTC {level} {direction}...")

            if level == 'L1':
                # Ужесточаем трейлы
                l1_tasks = []
                for mgr in targets:
                    symbol = mgr.symbol
                    pos_state = mgr.state # Доступ к состоянию
                    tf5 = ALLOWED_INTERVALS.get('5m')
                    df5 = market_data_store.get(symbol, {}).get(tf5)
                    if df5 is None or len(df5) < 20: continue

                    current_price = float(df5['close'].iloc[-1])
                    cur_sl = pos_state.sl_price # Текущий SL из состояния

                    if not cur_sl: continue # Не можем трейлить без текущего SL

                    # Используем более агрессивный множитель 1.0
                    new_sl = calculate_atr_trailing_stop(symbol, pos_state.side, current_price, cur_sl, 1.0)
                    if new_sl:
                        # --- ✅ ИСПРАВЛЕННЫЙ ВЫЗОВ ---
                        # Создаем lambda, чтобы передать new_sl в replace_order_safe
                        sl_update_func = partial(mgr._place_updated_sl, new_sl)
                        l1_tasks.append(replace_order_safe(mgr, "sl_order_id", sl_update_func, f"BTC L1 Guard Tighten"))
                        # Обновляем SL в состоянии сразу, до выполнения задачи (оптимистично)
                        pos_state.sl_price = new_sl
                        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

                if l1_tasks:
                     await asyncio.gather(*l1_tasks, return_exceptions=True)
                     await save_state_async() # Сохраняем обновленные SL

                skip_new_entries_until = now + BLOCK_NEW_ENTRIES_SEC
                await send_telegram_message_safe(f"⛔ **BTC L1**: Импульс {direction}! Ужесточены стопы для {len(targets)} поз., новые входы временно блокированы ({BLOCK_NEW_ENTRIES_SEC}с).")

            elif level == 'L2':
                # Частичное сокращение
                l2_tasks = [partial_close(mgr.symbol, mgr.get_state_dict(), PARTIAL_CLOSE_L2, f"BTC L2 {direction}") for mgr in targets]
                await asyncio.gather(*l2_tasks, return_exceptions=True)
                skip_new_entries_until = now + BLOCK_NEW_ENTRIES_SEC
                await send_telegram_message_safe(f"🚨 **BTC L2**: Импульс {direction}! Частичное сокращение {PARTIAL_CLOSE_L2*100:.0f}% для {len(targets)} поз.")

            elif level == 'L3':
                # Полное закрытие и пауза
                await send_telegram_message_safe(f"🟥 **BTC L3**: Аномальный импульс {direction}! Закрываю {len(targets)} позиций.")
                l3_tasks = [close_position_emergency(mgr.symbol, mgr.get_state_dict(), f"BTC L3 {direction}") for mgr in targets]
                await asyncio.gather(*l3_tasks, return_exceptions=True)
                trading_pause_until = now + COOLDOWN_SEC
                skip_new_entries_until = trading_pause_until
                await send_telegram_message_safe(f"⏸️ Торговля приостановлена на {COOLDOWN_SEC // 60} минут.")

        except asyncio.CancelledError:
             logging.info(f"{log_main_prefix} Задача отменена.")
             break # Выходим из цикла при отмене
        except Exception as e:
            logging.error(f"{log_main_prefix} Критическая ошибка: {e}", exc_info=True)
            await asyncio.sleep(CHECK_EVERY_SEC * 2) # Увеличиваем паузу при ошибке

async def get_llm_strategic_verdict(symbol: str, side: str, entry_price: float, context: Dict) -> Dict:
    """
    Запрашивает у LLM не просто уровни, а целую СТРАТЕГИЮ управления позицией.
    Возвращает словарь с выбранной стратегией и параметрами.
    """
    # Собираем контекст в читаемый формат для промпта
    base_sl = context.get('base_sl_atr')
    base_tp = context.get('base_tp_atr')
    structural_sl = context.get('structural_sl')
    
    prompt = f"""
Вы — главный риск-аналитик в квантовом хедж-фонде. Ваша задача — разработать оптимальный план управления риском (Take Profit и Stop Loss) для новой сделки.

**АНАЛИТИЧЕСКАЯ СВОДКА:**
- **Инструмент:** {symbol}
- **Направление:** {side.upper()}
- **Цена входа:** {entry_price:.5f}
- **Сигнал от ML-модели:** Уверенность {context.get('ml_confidence', 'N/A'):.1f}%
- **Режим рынка (HMM):** '{context.get('hmm_regime_name', 'N/A')}'
- **Сила тренда (ADX):** {context.get('adx', 'N/A'):.1f}

**ПРЕДВАРИТЕЛЬНЫЕ РАСЧЕТЫ УРОВНЕЙ:**
- **SL по волатильности (ATR):** {base_sl:.5f}
- **TP по волатильности (ATR):** {base_tp:.5f}
- **SL по структуре рынка (ближайший экстремум):** {structural_sl:.5f}

**ВАША ЗАДАЧА:**
Проанализируйте все данные и выберите ОДНУ из следующих стратегий управления позицией. При выборе SL отдавайте предпочтение более безопасному (дальнему) варианту между ATR и Структурой. Обоснуйте свой выбор.

**ВОЗМОЖНЫЕ СТРАТЕГИИ:**
1.  `aggressive_trend`: Для сильных трендовых рынков (ADX > 28). Использовать широкий Take Profit (например, TP * 1.5) и самый безопасный SL. Позицию вести активным трейлинг-стопом.
2.  `conservative_range`: Для флэтовых или неясных рынков (ADX < 20). Использовать близкий Take Profit (например, TP * 0.75) и самый безопасный SL. Цель - забрать быструю прибыль.
3.  `standard_balanced`: Сбалансированный подход для умеренного рынка. Использовать базовый TP по ATR и самый безопасный SL.
4.  `reject_trade`: Отклонить сделку. Если риски слишком высоки (например, сигнал ML < 20% или сигнал идет против очевидного сильного тренда).

**Формат ответа (строго JSON, без лишнего текста):**
{{
  "chosen_strategy": "название_стратегии",
  "final_sl_price": <float_цена_стоп_лосса>,
  "final_tp_price": <float_цена_тейк_профита>,
  "justification": "Краткое обоснование вашего выбора на 1-2 предложения."
}}
"""

    loop = asyncio.get_running_loop()
    # Используем вашу существующую функцию для отправки запроса
    response_data = await loop.run_in_executor(
        None, partial(sync_deepseek_request, prompt, DEEPSEEK_API_URL, True)
    )
    
    # --- Парсинг ответа ---
    default_strategy = {
        "chosen_strategy": "standard_balanced",
        "final_sl_price": min(base_sl, structural_sl) if side.upper() == 'LONG' else max(base_sl, structural_sl),
        "final_tp_price": base_tp,
        "justification": "Ошибка LLM, используется стандартный сбалансированный подход."
    }
    
    if not response_data or 'choices' not in response_data:
        logging.warning(f"[{symbol}] LLM не вернул ответ. Используется стратегия по умолчанию.")
        return default_strategy

    try:
        content = response_data['choices'][0]['message']['content']
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_match:
            raise ValueError("JSON не найден в ответе LLM")
        
        strategy_data = json.loads(json_match.group(0))
        # Валидация ответа
        if not all(k in strategy_data for k in ["chosen_strategy", "final_sl_price", "final_tp_price"]):
             raise ValueError("Ответ LLM не содержит всех необходимых ключей.")
             
        logging.warning(f"[{symbol}] Стратегический вердикт от LLM: {strategy_data}")
        return strategy_data
    except Exception as e:
        logging.error(f"[{symbol}] Ошибка парсинга вердикта LLM: {e}. Используется стратегия по умолчанию.")
        return default_strategy

def calculate_fractional_kelly_stake(
    win_probability: float, 
    win_loss_ratio: float, 
    fraction: float = 0.5
) -> float:
    """
    Рассчитывает долю капитала для ставки по дробному критерию Келли.

    Использование дробного Келли (например, 0.5) снижает агрессивность и волатильность
    роста капитала по сравнению с полным критерием Келли.

    Args:
        win_probability (float): Вероятность выигрыша (от 0.0 до 1.0).
        win_loss_ratio (float): Среднее соотношение прибыли к убытку (R).
        fraction (float): Доля от полного Келли (0.1 до 1.0).

    Returns:
        float: Рекомендуемый процент риска на сделку (например, 0.02 для 2%).
    """
    if not (0 < win_probability <= 1 and win_loss_ratio > 0 and 0 < fraction <= 1):
        return 0.0

    try:
        # Формула Келли: K% = W - [(1 - W) / R]
        kelly_percentage = win_probability - ((1 - win_probability) / win_loss_ratio)
        
        # Если Келли отрицательный (математическое ожидание отрицательное), не рискуем
        if kelly_percentage <= 0:
            return 0.0
            
        return kelly_percentage * fraction

    except ZeroDivisionError:
        return 0.0

def calculate_portfolio_risk_adjustment(
    new_trade_symbol: str,
    open_positions: Dict[str, Any],
    market_data_store: Dict[str, Dict[str, pd.DataFrame]],
    tf_key: str = '1h',
    lookback_period: int = 50
) -> float:
    """
    Рассчитывает коэффициент снижения риска на основе корреляции с открытыми позициями.
    
    Цель - уменьшить риск, если мы добавляем в портфель сильно скоррелированный актив,
    чтобы избежать концентрации риска в одном движении рынка.

    Args:
        new_trade_symbol (str): Символ новой сделки.
        open_positions (Dict): Словарь с текущими открытыми позициями.
        market_data_store (Dict): Глобальное хранилище данных.
        tf_key (str): Таймфрейм для расчета корреляции.
        lookback_period (int): Период для расчета корреляции.

    Returns:
        float: Коэффициент-модификатор риска (от 0.0 до 1.0). 1.0 = нет корреляции, <1.0 = риск снижается.
    """
    if not open_positions:
        return 1.0 # Если нет открытых позиций, коррекция не нужна

    symbols_in_portfolio = list(open_positions.keys()) + [new_trade_symbol]
    
    returns_df = pd.DataFrame()
    for symbol in set(symbols_in_portfolio): # Используем set для уникальности
        df = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS[tf_key])
        if df is not None and len(df) >= lookback_period:
            returns_df[symbol] = df['close'].pct_change().tail(lookback_period)
    
    if len(returns_df.columns) < 2 or new_trade_symbol not in returns_df.columns:
        return 1.0 # Недостаточно данных для корреляции

    # Расчет корреляционной матрицы
    corr_matrix = returns_df.corr()
    
    # Находим максимальную абсолютную корреляцию нового актива с уже открытыми
    open_symbols_in_matrix = [s for s in open_positions.keys() if s in corr_matrix.columns]
    if not open_symbols_in_matrix:
        return 1.0
        
    max_correlation = corr_matrix.loc[new_trade_symbol, open_symbols_in_matrix].abs().max()

    if pd.isna(max_correlation):
        return 1.0

    # Простая нелинейная формула для снижения риска: (1 - corr^2)
    # При corr=0.5, коэф.=0.75. При corr=0.9, коэф.=0.19.
    risk_adjustment_factor = 1.0 - (max_correlation ** 2)
    
    logging.info(f"Макс. корреляция для {new_trade_symbol}: {max_correlation:.2f}. Коэф. коррекции риска: {risk_adjustment_factor:.2f}")
    
    return risk_adjustment_factor


# ##################################################################
# ## ЗАДАЧА 4: ПРОДВИНУТОЕ УПРАВЛЕНИЕ ПОЗИЦИЯМИ                 ##
# ##################################################################

def calculate_chandelier_exit_stop(
    symbol: str,
    side: str,
    market_data_store: Dict[str, Dict[str, pd.DataFrame]],
    tf_key: str = '1h',
    period: int = 22,
    multiplier: float = 3.0
) -> Optional[float]:
    """
    Рассчитывает стоп-лосс по волатильности "Выход по Люстре" (Chandelier Exit).
    """
    df = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS[tf_key])
    if df is None or len(df) < period:
        logging.warning(f"[{symbol}] Недостаточно данных ({len(df) if df is not None else 0}/{period}) для Chandelier Exit.")
        return None
        
    try:
        # Используем последние `period` свечей для расчета
        recent_df = df.iloc[-period:]
        
        # Расчет ATR на всем доступном датафрейме для стабильности
        atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=period).iloc[-1]
        
        if side.upper() == 'LONG':
            # Находим максимальный максимум за период
            highest_high = recent_df['high'].max()
            # Стоп = Максимум - (ATR * Множитель)
            stop_price = highest_high - (atr * multiplier)
        elif side.upper() == 'SHORT':
            # Находим минимальный минимум за период
            lowest_low = recent_df['low'].min()
            # Стоп = Минимум + (ATR * Множитель)
            stop_price = lowest_low + (atr * multiplier)
        else:
            return None
        
        logging.info(f"[{symbol}] Chandelier Exit ({side}): Стоп = {stop_price:.4f} (ATR={atr:.4f})")
        return stop_price
        
    except Exception as e:
        logging.error(f"[{symbol}] Ошибка расчета Chandelier Exit: {e}", exc_info=False)
        return None

# **Интеграция:** Эту функцию следует вызывать внутри `advanced_position_manager`
# для динамического обновления стоп-лосса. Пример:
#
# async def advanced_position_manager(symbol: str, pos_data: Dict[str, Any]):
#     ...
#     # Внутри цикла while:
#     new_sl_price_ce = calculate_chandelier_exit_stop(symbol, pos_data['side'], market_data_store)
#     if new_sl_price_ce:
#          # Логика сравнения new_sl_price_ce с текущим стопом pos_data['sl_price']
#          # и его обновление, если новый стоп выгоднее (выше для лонга, ниже для шорта)
#     ...

# ##################################################################
# ## ЗАДАЧА 5: ЕДИНАЯ СКОРИНГОВАЯ МОДЕЛЬ ВХОДА                   ##
# ##################################################################

def calculate_unified_entry_score(
    symbol: str,
    side: str, # 'LONG' или 'SHORT'
    ml_confidence: float,
    llm_confidence: float,
    order_book_imbalance: float,
    long_short_ratio: float,
    regime_favorability: float
) -> float:
    """
    Рассчитывает единый взвешенный скоринг-балл для принятия решения о входе.
    Объединяет сигналы от разных систем в один показатель.

    Args:
        symbol (str): Символ для логирования.
        side (str): Направление сделки ('LONG' или 'SHORT').
        ml_confidence (float): Уверенность ML-модели (0-100).
        llm_confidence (float): Уверенность LLM (0-100).
        order_book_imbalance (float): OBI (-1.0 до 1.0).
        long_short_ratio (float): Соотношение лонг/шорт (например, от 0.5 до 2.0).
        regime_favorability (float): Благоприятность режима для данного направления (0.0=плохо, 0.5=нейтрально, 1.0=хорошо).

    Returns:
        float: Итоговый балл от 0 до 100.
    """
    # Веса компонентов (сумма должна быть равна 1.0)
    weights = {
        "ml": 0.40,
        "llm": 0.25,
        "obi": 0.15,
        "ls_ratio": 0.10,
        "regime": 0.10
    }

    # --- Нормализация всех входов к шкале 0-100 ---
    scores = {}
    scores['ml'] = ml_confidence if ml_confidence is not None else 50
    scores['llm'] = llm_confidence if llm_confidence is not None else 50
    
    # OBI: Преобразуем -1..1 в 0..100. OBI > 0 хорошо для лонга, < 0 для шорта.
    if order_book_imbalance is not None:
        if side == 'LONG':
            scores['obi'] = (order_book_imbalance + 1) * 50
        else: # SHORT
            scores['obi'] = (-order_book_imbalance + 1) * 50
    else:
        scores['obi'] = 50

    # LS Ratio: >1 хорошо для лонга, <1 для шорта.
    if long_short_ratio is not None:
        if side == 'LONG':
            # Чем выше ratio, тем ближе к 100
            scores['ls_ratio'] = 100 / (1 + (1 / long_short_ratio)**2) if long_short_ratio > 0 else 50
        else: # SHORT
            # Чем ниже ratio, тем ближе к 100
            scores['ls_ratio'] = 100 - (100 / (1 + (1 / long_short_ratio)**2)) if long_short_ratio > 0 else 50
    else:
        scores['ls_ratio'] = 50
        
    # Regime Favorability: Уже в шкале 0..1, переводим в 0..100
    scores['regime'] = regime_favorability * 100 if regime_favorability is not None else 50

    # Расчет итогового взвешенного балла
    final_score = sum(scores[key] * weights[key] for key in weights)
    
    logging.info(f"[{symbol}] Расчет скоринга для {side}: ML={scores['ml']:.0f}, LLM={scores['llm']:.0f}, OBI={scores['obi']:.0f}, LS={scores['ls_ratio']:.0f}, Regime={scores['regime']:.0f} -> Итог: {final_score:.2f}")

    return final_score

async def get_deepseek_verdict_async_v3(symbol: str, trade_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Продвинутая версия с обогащенным контекстом и более детальным промптом для LLM.
    """
    logging.info(f"[{symbol}] Запрос вердикта от LLM v3 (Расширенный контекст)...")
    
    # Форматируем переменные для большей читабельности в промпте
    side_rus = "ПОКУПКА (LONG)" if trade_type == "long" else "ПРОДАЖА (SHORT)"
    ml_conf = context.get('ml_confidence', 0)
    
    prompt = f"""
Вы — главный риск-менеджер и аналитик в хедж-фонде. Вам поступило предложение на открытие сделки. Ваша задача — провести всесторонний анализ, взвесить все "ЗА" и "ПРОТИВ" и вынести вердикт с оценкой уверенности.

**АНАЛИТИЧЕСКАЯ СВОДКА ПО СДЕЛКЕ**
- **Инструмент:** {symbol}
- **Предлагаемое действие:** {side_rus}
- **Сигнал от ML-модели:** Уверенность {(ml_conf or 0):.1f}%
- **Текущий режим рынка (HMM):** {context.get('market_regime_name', 'Не определен')}

**ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ:**
- **RSI (14) на 15m:** {(context.get('rsi_15m') or 50):.1f} (Зоны: <30 перепроданность, >70 перекупленность)
- **ADX (14) на 1h:** {(context.get('adx_1h') or 0):.1f} (Значения: >25 сильный тренд, <20 флэт)
- **MACD Hist на 15m:** {(context.get('macd_hist_15m') or 0):.6f} (Значения: >0 бычий моментум, <0 медвежий)

**АНАЛИЗ РЫНКА:**
- **Общий тренд BTC:** {context.get('btc_trend_strength', 'Нейтральный')}
- **Давление в стакане (OBI):** {(context.get('microstructure', {}).get('order_book_imbalance') or 0):.3f} (Значения: >0 давление покупателей, <0 давление продавцов)
- **Настроения на фьючерсах (L/S Ratio):** {(context.get('derivatives_sentiment', {}).get('long_short_ratio') or 0):.2f} (Значения: >1 преобладают лонги, <1 преобладают шорты)
- **Прогноз волатильности (GARCH):** {context.get('garch_forecast_text', 'Стабильная')}

---
**ВАША ЗАДАЧА:**

1.  **Пошаговый анализ (Chain of Thought):**
    * **Соответствие тренду:** Сигнал ({side_rus}) соответствует общему тренду BTC и силе тренда по ADX?
    * **Сила сигнала:** Насколько силен базовый сигнал от ML? Являются ли значения RSI и MACD подтверждающими или противоречащими?
    * **Поддержка рынка:** Подтверждают ли OBI и L/S Ratio направление сделки?
    * **Риски:** Какие факторы явно противоречат сделке? Низкая уверенность ML, сигнал против тренда, отсутствие поддержки рынка?

2.  **Итоговый вердикт (строго в формате JSON):**
    * `confidence`: Ваша итоговая уверенность в сделке (целое число от 0 до 100). Оценивайте строго: 50 - нейтрально, >70 - сильная уверенность.
    * `explanation`: Краткое (1-2 предложения) резюме вашего анализа с ключевыми факторами "ЗА" и "ПРОТИВ".
"""

    loop = asyncio.get_running_loop()
    response_data = await loop.run_in_executor(
        None, partial(sync_deepseek_request, prompt, DEEPSEEK_API_URL, True)
    )

    # ... (остальная часть функции get_deepseek_verdict_async_v2 без изменений)
    default_response = {"confidence": 50, "explanation": "Ошибка ответа или парсинга от LLM."}
    if not response_data or 'choices' not in response_data:
        logging.warning(f"[{symbol}] LLM не вернул валидный ответ.")
        return default_response
    try:
        content = response_data['choices'][0]['message']['content']
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_match:
            logging.error(f"[{symbol}] Не найден JSON в ответе LLM: {content}")
            return default_response
        ai_data = json.loads(json_match.group(0))
        if 'confidence' not in ai_data or 'explanation' not in ai_data:
            raise ValueError("Ответ ИИ не содержит полей 'confidence' или 'explanation'")
        logging.info(f"[{symbol}] Вердикт LLM (v3): Уверенность={ai_data['confidence']}, Обоснование='{ai_data['explanation']}'")
        return ai_data
    except Exception as e:
        logging.error(f"[{symbol}] Ошибка парсинга вердикта LLM: {e}", exc_info=False)
        return default_response 


# ##################################################################
# ## ЗАДАЧА 6: ПРОДВИНУТЫЙ ПРОМПТ ДЛЯ LLM (CHAIN-OF-THOUGHT)     ##
# ##################################################################

# Эта функция является ЗАМЕНОЙ для вашей существующей `get_deepseek_verdict_async`
# Замените эту функцию в main.py
async def get_deepseek_verdict_async_v2(symbol: str, trade_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Получает вердикт от LLM, используя продвинутый промпт "Chain-of-Thought" (CoT).
    *** ВЕРСИЯ С ЗАЩИТОЙ ОТ None ***
    """
    logging.info(f"[{symbol}] Запрос вердикта от LLM v2 (Chain-of-Thought)...")
    
    prompt = f"""
Вы — элитный количественный трейдинг-аналитик. Проведите пошаговый анализ и дайте итоговый скоринг для сделки.

**АНАЛИЗ СДЕЛКИ**
- **Инструмент:** {symbol}
- **Направление:** {trade_type.upper()}
- **Общий Контекст BTC:** {context.get('btc_trend_strength', 'N/A')}
- **Прогноз волатильности (GARCH):** {"Растет" if (context.get('garch_forecast') or 0) > 2.5 else "Снижается"}

**КЛЮЧЕВЫЕ ДАННЫЕ:**
- **ML Модель (Тренд/Флэт):** Уверенность {(context.get('ml_confidence') or 0):.1f}%
- **Микроструктура (Давление в стакане):** OBI = {(context.get('microstructure', {}).get('order_book_imbalance') or 0):.3f}
- **Сантимент (Соотношение Long/Short):** L/S Ratio = {(context.get('derivatives_sentiment', {}).get('long_short_ratio') or 0):.2f}

---
**ЗАДАЧА: ПРОВЕДИТЕ АНАЛИЗ ПО ШАГАМ**

**Шаг 1: Анализ базового сигнала.**
Соответствует ли сигнал ({trade_type.upper()}) общему тренду BTC? Сигнал от ML-модели сильный?

**Шаг 2: Анализ подтверждающих факторов.**
Подтверждает ли давление в стакане (OBI) и сантимент на рынке фьючерсов (L/S Ratio) направление сделки? (Для LONG OBI и L/S Ratio должны быть положительными/больше 1. Для SHORT - наоборот).

**Шаг 3: Анализ рисков.**
Какие факторы противоречат сделке? Является ли прогнозируемая волатильность слишком высокой или низкой для текущей стратегии?

**Шаг 4: Синтез и итоговый вердикт (только JSON).**
На основе вашего анализа, предоставьте ответ в формате JSON.
- `confidence`: Ваша общая уверенность в успехе сделки от 0 до 100.
- `explanation`: Краткое обоснование, суммирующее главные факторы "ЗА" и "ПРОТИВ".
"""
    
    loop = asyncio.get_running_loop()
    response_data = await loop.run_in_executor(
        None, partial(sync_deepseek_request, prompt, DEEPSEEK_API_URL, True)
    )

    default_response = {"confidence": 50, "explanation": "Ошибка ответа или парсинга от LLM."}
    if not response_data or 'choices' not in response_data:
        logging.warning(f"[{symbol}] LLM не вернул валидный ответ.")
        return default_response

    try:
        content = response_data['choices'][0]['message']['content']
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_match:
            logging.error(f"[{symbol}] Не найден JSON в ответе LLM: {content}")
            return default_response
            
        ai_data = json.loads(json_match.group(0))
        if 'confidence' not in ai_data or 'explanation' not in ai_data:
            raise ValueError("Ответ ИИ не содержит полей 'confidence' или 'explanation'")
        
        logging.info(f"[{symbol}] Вердикт LLM (CoT): Уверенность={ai_data['confidence']}, Обоснование='{ai_data['explanation']}'")
        return ai_data
    except Exception as e:
        logging.error(f"[{symbol}] Ошибка парсинга вердикта LLM: {e}", exc_info=False)
        return default_response

async def execute_tp(symbol: str, pos_data: Dict[str, Any], tp_level: int):
    logging.warning(f"✅ [{symbol}] Сработал TP{tp_level}. Фиксация части прибыли...")
    
    order_ids_to_cancel = [pos_data.get(key) for key in ['sl_order_id', 'tp1_order_id', 'tp2_order_id'] if pos_data.get(key)]
    await asyncio.gather(*[cancel_single_order_async(symbol, oid) for oid in order_ids_to_cancel])

    original_amount = pos_data['amount']
    initial_amount = pos_data.get('initial_amount', original_amount)
    close_fraction = 0.25 
    amount_to_close = float(format_quantity(initial_amount * close_fraction, symbol))
    amount_to_close = min(amount_to_close, original_amount)
    remaining_amount = original_amount - amount_to_close
    
    close_side = 'SELL' if pos_data['side'] == 'LONG' else 'BUY'
    await place_order_async(symbol=symbol, side=close_side, order_type="MARKET", quantity=amount_to_close, reduce_only=True)
    
    await send_telegram_message(f"✅ <b>{symbol}</b>: {close_fraction*100:.0f}% прибыли зафиксировано на TP{tp_level}!")
    
    if remaining_amount > 0:
        entry_price = pos_data['entry_price']
        new_sl_price = entry_price
        
        next_tp_price = pos_data.get('tp2_price') if tp_level == 1 else None
        
        sl_task = place_order_async(symbol=symbol, side=close_side, order_type="STOP_MARKET", quantity=remaining_amount, stopPrice=new_sl_price, reduce_only=True)
        tp_task = asyncio.sleep(0)
        if next_tp_price:
             tp_qty = float(format_quantity(initial_amount * 0.25, symbol))
             tp_task = place_order_async(symbol=symbol, side=close_side, order_type="TAKE_PROFIT_MARKET", quantity=tp_qty, stopPrice=next_tp_price, reduce_only=True)

        sl_res, tp_res = await asyncio.gather(sl_task, tp_task)
        
        current_positions[symbol].update({
            'amount': remaining_amount, 'sl_price': new_sl_price,
            'sl_order_id': sl_res.get('orderId') if sl_res and isinstance(sl_res, dict) else None,
            'tp1_order_id': None, 
            'tp2_order_id': tp_res.get('orderId') if tp_res and isinstance(tp_res, dict) else None,
            'state': 'trailing_after_partial'
        })
        await send_telegram_message(f"   -> Стоп переведен в безубыток. Оставшийся объем: {remaining_amount}")
    else:
        if symbol in current_positions:
            del current_positions[symbol]
        await send_telegram_message(f"   -> Позиция полностью закрыта.")
    
    await save_state_async()

async def handle_portfolio_reversal(new_trade_type: str):
    """
    Проверяет все открытые позиции. Если есть позиции в направлении,
    противоположном новому сигналу, закрывает их все. (ИСПРАВЛЕННАЯ ВЕРСИЯ)
    """
    if not ENABLE_PORTFOLIO_REVERSAL or not current_positions:
        return

    opposite_side = 'SHORT' if new_trade_type == 'long' else 'LONG'
    positions_to_close = []
    
    # Находим все позиции, которые нужно закрыть
    for symbol, manager in current_positions.items():
        pos_data = manager.get_state_dict() if isinstance(manager, PositionManager) else manager
        if pos_data.get('side') == opposite_side:
            
            # --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
            # Убеждаемся, что в словаре с данными о позиции всегда есть ключ 'symbol'
            pos_data['symbol'] = symbol
            positions_to_close.append(pos_data)

    if not positions_to_close:
        return

    # Если нашли, что закрывать - отправляем уведомление и запускаем закрытие
    symbols_str = ", ".join([p['symbol'] for p in positions_to_close])
    logging.warning(
        f"🚨 РЕВЕРСИРОВАНИЕ ПОРТФЕЛЯ! Новый сигнал: {new_trade_type.upper()}. "
        f"Закрываю все {opposite_side} позиции: {symbols_str}"
    )
    await send_telegram_message(
        f"🚨 **Реверс Портфеля** 🚨\n"
        f"Получен новый сигнал на **{new_trade_type.upper()}**.\n"
        f"Принудительно закрываю все противоположные позиции: <b>{symbols_str}</b>"
    )

    # Асинхронно закрываем все найденные позиции
    closing_tasks = [
        close_position_emergency(
            pos['symbol'], 
            pos, 
            reason=f"Реверс портфеля из-за нового сигнала на {new_trade_type.upper()}"
        ) 
        for pos in positions_to_close
    ]
    await asyncio.gather(*closing_tasks)

    # Даем бирже секунду на обработку всех закрытий
    await asyncio.sleep(1)


# ЗАМЕНИТЕ СТАРУЮ ФУНКЦИЮ НА ЭТУ В ВАШЕМ СКРИПТЕ (находится в ЧАСТИ 7)

async def place_market_entry_with_tp_sl(signal_params: Dict[str, Any], ai_verdict: Dict[str, Any], final_threshold: float):
    """
    ОТКАЗОУСТОЙЧИВАЯ функция для входа в рынок с последовательной установкой ордеров.
    (Полная версия для замены)
    """
    symbol = signal_params['symbol']
    side_str = signal_params['side']
    side = 'LONG' if side_str == 'BUY' else 'SHORT'
    signal_price = signal_params['entry_price_signal']

    # --- ШАГ 1: Получение и валидация торгового плана ---
    # (Этот блок использует ваши новые функции, которые вы уже добавили)
    logging.info(f"[{symbol}] Начало процесса входа. Сигнальная цена: {signal_price:.4f}. Запрашиваю SL/TP...")
    
    sltp_prices_ai = await get_sl_tp_from_ai_async(symbol, side, signal_price)
    final_sl_price = sltp_prices_ai.get('stop_loss')

    if not final_sl_price:
        logging.error(f"[{symbol}] Вход отменен: AI не вернул валидный уровень SL.")
        return

    dynamic_tp_levels = await get_dynamic_tp_levels(symbol, signal_price, final_sl_price, signal_params['market_env'], side)
    final_tp1_price = dynamic_tp_levels.get('take_profit_1')
    
    if not final_tp1_price:
        logging.error(f"[{symbol}] Вход отменен: не удалось рассчитать уровень TP.")
        return

    risk_distance = abs(signal_price - final_sl_price)
    if risk_distance == 0:
        logging.warning(f"[{symbol}] Вход отменен: Дистанция до SL равна нулю.")
        return
        
    reward_distance_1 = abs(final_tp1_price - signal_price)
    rr_ratio = reward_distance_1 / risk_distance

    # ИСПРАВЛЕННАЯ ПРОВЕРКА R:R
    if (side == 'LONG' and final_tp1_price <= final_sl_price) or \
       (side == 'SHORT' and final_tp1_price >= final_sl_price):
        logging.error(f"[{symbol}] Вход отменен: Нелогичные уровни SL/TP. SL={final_sl_price}, TP={final_tp1_price}")
        return

    if rr_ratio < (MIN_RR_RATIO - 0.01):
        logging.warning(f"[{symbol}] Вход отменен: Низкое R:R ({rr_ratio:.2f} < {MIN_RR_RATIO}).")
        return

    qty_to_trade = calculate_smart_quantity(
        symbol,
        entry_price=signal_price,
        sl_price=final_sl_price,
        ai_verdict=ai_verdict,
        risk_percent=get_dynamic_risk_percent(symbol), # <-- Теперь вызываем с символом
        final_threshold=final_threshold
    )

    if qty_to_trade <= 0:
        logging.warning(f"[{symbol}] Вход отменен: расчетный объем равен нулю или меньше.")
        return

    # --- ШАГ 2: Вход в рынок ---
    logging.warning(f"[{symbol}] ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ. Отправка рыночного ордера...")
    entry_order_info = await place_order_async(symbol=symbol, side=side_str, order_type="MARKET", quantity=qty_to_trade)
    if not entry_order_info or not entry_order_info.get('orderId'):
        logging.error(f"[{symbol}] Не удалось разместить рыночный ордер на вход.")
        return

    # --- ШАГ 3: Получение точной цены входа ---
    actual_entry_price = 0.0
    try:
        await asyncio.sleep(POST_ENTRY_DELAY_SECONDS)
        filled_order = await client.futures_get_order(symbol=symbol, orderId=entry_order_info['orderId'])
        if filled_order and filled_order.get('status') == 'FILLED' and float(filled_order.get('avgPrice', 0)) > 0:
            actual_entry_price = float(filled_order['avgPrice'])
    except Exception as e:
        logging.error(f"[{symbol}] Не удалось получить цену исполнения: {e}")

    if actual_entry_price <= 0:
        await close_position_emergency(symbol, {'side': side, 'amount': qty_to_trade}, "Крит. ошибка: не удалось подтвердить цену входа.")
        return

    logging.info(f"[{symbol}] Позиция открыта по цене {actual_entry_price:.4f}. Установка защитных ордеров.")
    
    pos_to_save = {'side': side, 'amount': qty_to_trade, 'entry_price': actual_entry_price, 'state': 'placing_brackets'}
    current_positions[symbol] = pos_to_save
    
    # --- ШАГ 4: ПОСЛЕДОВАТЕЛЬНАЯ установка защитных ордеров ---
    opposite_side_str = "SELL" if side_str == "BUY" else "BUY"
    
    # СНАЧАЛА СТАВИМ ГЛАВНЫЙ ОРДЕР - STOP LOSS
    sl_order = await place_order_async(symbol=symbol, side=opposite_side_str, order_type="STOP_MARKET", quantity=qty_to_trade, stopPrice=final_sl_price, reduce_only=True)
    
    if not sl_order or not sl_order.get('orderId'):
        # Если НЕ удалось поставить SL - это КРИТИЧЕСКАЯ ОШИБКА. Немедленно закрываем позицию.
        await close_position_emergency(symbol, pos_to_save, "Критическая ошибка установки SL ордера.")
        return
        
    # Если SL установлен, обновляем состояние и готовим сообщение
    current_positions[symbol]['sl_order_id'] = sl_order.get('orderId')
    current_positions[symbol]['sl_price'] = final_sl_price
    msg_tg = f"✅ Вход {'Лонг' if side_str == 'BUY' else 'Шорт'} <b>{symbol}</b>\nЦена: {actual_entry_price:.4f}\nSL: {format_price(final_sl_price, symbol)}"
    
    # ТЕПЕРЬ СТАВИМ TAKE PROFIT
    tp_order = await place_order_async(symbol=symbol, side=opposite_side_str, order_type="TAKE_PROFIT_MARKET", quantity=qty_to_trade, stopPrice=final_tp1_price, reduce_only=True)
    
    if tp_order and tp_order.get('orderId'):
        current_positions[symbol]['tp1_order_id'] = tp_order.get('orderId')
        current_positions[symbol]['tp1_price'] = final_tp1_price
        msg_tg += f"\nTP: {format_price(final_tp1_price, symbol)}"
    else:
        # Если TP не установился - это НЕ критично. Позиция защищена SL.
        msg_tg += "\n⚠️ ОШИБКА установки TP! Позиция защищена SL."
        logging.warning(f"[{symbol}] Не удалось установить TP. Позиция остается открытой со стоп-лоссом.")
        
    # Все готово, переводим позицию в рабочий режим
    current_positions[symbol]['state'] = 'initial'
    asyncio.create_task(advanced_position_manager(symbol, current_positions[symbol]))
    await save_state_async()
    await send_telegram_message(msg_tg)

async def close_position_emergency(symbol: str, pos_data: Dict[str, Any], reason: str) -> bool:
    """
    Аварийно закрывает позицию V2: Проверяет статус на бирже перед отправкой
    критического сообщения об ошибке.
    """
    global TRADE_HISTORY, current_positions, client # Доступ к глобальным переменным

    # --- 1. Базовая проверка и логирование начала ---
    log_prefix = f"[{symbol}] [Emergency Close]"
    logging.warning(f"{log_prefix} Запуск аварийного закрытия. Причина: {reason}")

    # Получаем актуальные данные о позиции перед закрытием
    # Используем get() с '0.0' по умолчанию для amount
    current_amount_str = pos_data.get('amount', '0.0')
    try:
        current_amount = float(current_amount_str)
    except (ValueError, TypeError):
        logging.error(f"{log_prefix} Ошибка: Неверное значение amount '{current_amount_str}'. Закрытие невозможно.")
        return False

    if current_amount <= 0:
        logging.info(f"{log_prefix} Позиция уже закрыта или объем равен нулю. Удаляю из памяти, если есть.")
        if symbol in current_positions:
            del current_positions[symbol]
            await save_state_async()
        return True # Считаем успешным, так как позиции нет

    # --- 2. Отмена связанных ордеров (SL/TP) ---
    order_keys = ['sl_order_id', 'tp1_order_id', 'tp2_order_id', 'near_tp_partial_oid', 'near_tp_guard_oid']
    order_ids_to_cancel = [pos_data.get(key) for key in order_keys if pos_data.get(key)]
    if order_ids_to_cancel:
        logging.info(f"{log_prefix} Отмена {len(order_ids_to_cancel)} связанных ордеров...")
        await asyncio.gather(*[cancel_single_order_async(symbol, oid) for oid in order_ids_to_cancel])
        # Очищаем ID в pos_data на всякий случай
        for key in order_keys: pos_data.pop(key, None)

    # --- 3. Попытка закрытия рыночным ордером ---
    close_side = 'SELL' if pos_data.get('side') == 'LONG' else 'BUY'
    logging.info(f"{log_prefix} Отправка рыночного ордера на закрытие ({close_side} {current_amount})...")
    closing_order = await place_order_async(
        symbol=symbol, side=close_side, order_type="MARKET",
        quantity=current_amount, reduce_only=True
    )

    # --- 4. Обработка результата ---
    if closing_order and closing_order.get('orderId'):
        # --- Сценарий 1: Ордер УСПЕШНО отправлен ---
        closing_order_id = closing_order['orderId']
        logging.info(f"{log_prefix} Ордер на закрытие {closing_order_id} успешно отправлен.")

        # Расчет PNL и обновление истории (остается без изменений)
        try:
            await asyncio.sleep(1.5) # Даем время на исполнение
            trades = await client.futures_account_trades(symbol=symbol, orderId=closing_order_id, limit=1)
            if trades:
                exit_price = float(trades[0]['price'])
                entry_price = pos_data.get('entry_price', 0.0) # Безопасное извлечение
                quantity = current_amount # Используем объем перед закрытием
                pnl = (exit_price - entry_price) * quantity if pos_data.get('side') == 'LONG' else (entry_price - exit_price) * quantity
                trade_info = {'symbol': symbol, 'pnl': pnl, 'side': pos_data.get('side')}
                TRADE_HISTORY.append(trade_info)
                await track_trade_result(pnl)
                logging.info(f"{log_prefix} Сделка закрыта ордером {closing_order_id}. PNL: {pnl:.2f} USDT.")
        except Exception as e:
            logging.error(f"{log_prefix} Не удалось рассчитать PNL после закрытия ордером {closing_order_id}: {e}")

        # Удаление из памяти и уведомление
        escaped_reason = html.escape(reason)
        await send_telegram_message_safe(f"✅ <b>Позиция по {symbol} успешно закрыта ордером бота.</b>\nПричина: <i>{escaped_reason}</i>")
        if symbol in current_positions:
            del current_positions[symbol]
        await save_state_async()
        return True

    else:
        # --- Сценарий 2: Ошибка отправки ордера ---
        logging.error(f"{log_prefix} НЕ УДАЛОСЬ отправить ордер на закрытие! Проверяю статус на бирже...")
        await asyncio.sleep(2.0) # Пауза перед проверкой

        try:
            position_info = await client.futures_position_information(symbol=symbol)
            current_pos_on_api = next((p for p in position_info if p.get('symbol') == symbol), None)
            api_pos_qty = Decimal(current_pos_on_api.get('positionAmt', '0')) if current_pos_on_api else Decimal('0')

            if api_pos_qty == 0:
                # Позиция уже закрыта (вероятно, сработал SL/TP)
                logging.warning(f"{log_prefix} Ордер не прошел, НО позиция на бирже уже закрыта (вероятно SL/TP). Удаляю из памяти.")
                escaped_reason = html.escape(reason)
                await send_telegram_message_safe(f"⚠️ <b>Позиция по {symbol} закрыта (вероятно SL/TP до попытки бота).</b>\nПричина попытки закрытия: <i>{escaped_reason}</i>")
                if symbol in current_positions:
                    del current_positions[symbol]
                await save_state_async()
                return True # Считаем успешным, т.к. цель достигнута
            else:
                # Позиция все еще открыта! КРИТИЧЕСКАЯ ОШИБКА
                logging.critical(f"{log_prefix} КРИТИЧЕСКАЯ ОШИБКА! Не удалось закрыть позицию ({api_pos_qty} {symbol} все еще открыто).")
                await send_telegram_message_safe(f"🚨 <b>КРИТИЧЕСКАЯ ОШИБКА!</b>\nНе удалось закрыть позицию по {symbol} ({api_pos_qty} открыто). Проверьте вручную!")
                # НЕ удаляем из current_positions, оставляем для следующей синхронизации
                return False

        except Exception as e_check:
            # Ошибка при проверке статуса - тоже критично
            logging.critical(f"{log_prefix} КРИТИЧЕСКАЯ ОШИБКА! Не удалось ни закрыть, ни проверить статус позиции: {e_check}")
            await send_telegram_message_safe(f"🚨 <b>КРИТИЧЕСКАЯ ОШИБКА!</b>\nНе удалось закрыть И ПРОВЕРИТЬ позицию по {symbol}. Проверьте вручную!")
            return False


# --- Остальные функции Части 7 ---
async def get_long_term_trend(df_long: pd.DataFrame) -> str:
    return "Bullish" if df_long['close'].iloc[-1] > ema_indicator(close=df_long['close'], window=21).iloc[-1] else "Bearish"

async def get_btc_context() -> Dict[str, Any]:
    symbol = 'BTCUSDT'
    tf_api_val = ALLOWED_INTERVALS[SAFETY_FILTER_TIMEFRAME]
    df = market_data_store.get(symbol, {}).get(tf_api_val)
    if df is None or len(df) < 21: return {"trend": "unknown", "adx": 0, "momentum": 0}
    try:
        adx_val = adx(df['high'], df['low'], df['close'], 14).iloc[-1]
        momentum = ta.trend.cci(df['high'], df['low'], df['close'], window=20).iloc[-1]
        trend = "Bullish" if df['close'].iloc[-1] > ema_indicator(close=df['close'], window=21).iloc[-1] else "Bearish"
        if any(pd.isna(v) for v in [adx_val, momentum]): return {"trend": "unknown", "adx": 0, "momentum": 0}
        return {"trend": trend, "adx": round(adx_val, 2), "momentum": round(momentum, 2)}
    except Exception: return {"trend": "unknown", "adx": 0, "momentum": 0}

async def get_eth_context() -> Dict[str, Any]:
    symbol = 'ETHUSDT'
    tf_api_val = ALLOWED_INTERVALS[SAFETY_FILTER_TIMEFRAME]
    df = market_data_store.get(symbol, {}).get(tf_api_val)
    if df is None or len(df) < 21: return {"trend": "unknown", "adx": 0, "rsi": 50, "volatility": 0}
    try:
        adx_val = adx(df['high'], df['low'], df['close'], 14).iloc[-1]
        rsi_val = rsi(df['close'], 14).iloc[-1]
        atr_val = average_true_range(df['high'], df['low'], df['close'], 14).iloc[-1]
        last_close = df['close'].iloc[-1]
        volatility = (atr_val / last_close) * 100 if last_close > 0 else 0
        trend = "Bullish" if last_close > ema_indicator(close=df['close'], window=21).iloc[-1] else "Bearish"
        if any(pd.isna(v) for v in [adx_val, rsi_val, atr_val]): return {"trend": "unknown", "adx": 0, "rsi": 50, "volatility": 0}
        return {"trend": trend, "adx": round(adx_val, 2), "rsi": round(rsi_val, 2), "volatility": round(volatility, 2)}
    except Exception: return {"trend": "unknown", "adx": 0, "rsi": 50, "volatility": 0}

async def get_volume_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or len(df) < 21: return {"spike": False, "obv_trend": "flat", "ratio": 1.0, "avg_volume": 0}
    try:
        volume_ma = df['volume'].rolling(window=20).mean().iloc[-2]
        last_volume = df['volume'].iloc[-1]
        spike = last_volume > (volume_ma * 2.0) if volume_ma > 0 else False
        ratio = last_volume / volume_ma if volume_ma > 0 else 1.0
        obv = ta.volume.on_balance_volume(df['close'], df['volume'])
        obv_trend = "up" if obv.iloc[-1] > ema_indicator(obv, window=5).iloc[-1] else "down"
        return {"spike": spike, "obv_trend": obv_trend, "ratio": round(ratio, 2), "avg_volume": volume_ma}
    except Exception: return {"spike": False, "obv_trend": "flat", "ratio": 1.0, "avg_volume": 0}

async def get_correlation_snapshot(symbol_a: str, symbol_b: str) -> float:
    try:
        tf_api_val = ALLOWED_INTERVALS[ML_MODEL_TF_MEDIUM]
        df_a = market_data_store.get(symbol_a, {}).get(tf_api_val)
        df_b = market_data_store.get(symbol_b, {}).get(tf_api_val)
        if any(df is None or len(df) < 50 for df in [df_a, df_b]): return 0.0
        returns = pd.DataFrame({'symbol_a': df_a['close'].pct_change(), 'symbol_b': df_b['close'].pct_change()}).dropna()
        if len(returns) < 2: return 0.0
        correlation = returns['symbol_a'].corr(returns['symbol_b'])
        return correlation if pd.notna(correlation) else 0.0
    except Exception as e:
        logging.error(f"Ошибка при расчете корреляции между {symbol_a} и {symbol_b}: {e}")
        return 0.0

def format_price(price: Any, symbol: str) -> Optional[str]:
    symbol_info = exchange_info_cache.get(symbol)
    if not symbol_info: return f"{Decimal(str(price)):.8f}"
    tick_size = Decimal(symbol_info['tickSize'])
    
    if '.' in str(tick_size):
        price_precision = len(str(tick_size).split('.')[1].rstrip('0'))
    else:
        price_precision = 0
        
    rounded_price = (Decimal(str(price)) / tick_size).quantize(Decimal('0'), rounding=ROUND_DOWN) * tick_size
    return f"{rounded_price:.{price_precision}f}"

def format_quantity(quantity: Any, symbol: str) -> Optional[str]:
    symbol_info = exchange_info_cache.get(symbol)
    if not symbol_info: return f"{Decimal(str(quantity)):.8f}"
    step_size = Decimal(symbol_info['stepSize'])
    return f"{(Decimal(str(quantity)) / step_size).quantize(Decimal('0'), rounding=ROUND_DOWN) * step_size:.{symbol_info.get('quantityPrecision', 8)}f}"

def calculate_smart_quantity(symbol: str, entry_price: float, sl_price: float, ai_verdict: Dict[str, Any], risk_percent: float, final_threshold: float) -> float:
    global current_balance, MAX_RISK_MODIFIER
    
    base_score = final_threshold
    
    if entry_price == 0 or sl_price == 0 or entry_price == sl_price: return 0.0
    ai_confidence = ai_verdict.get('confidence', 50)
    
    confidence_modifier = 1.0 + (ai_confidence - base_score) / base_score * 2.0
    confidence_modifier = min(MAX_RISK_MODIFIER, max(0.5, confidence_modifier))
    
    adjusted_risk_percent = risk_percent * confidence_modifier
    risk_capital_usd = current_balance * (adjusted_risk_percent / 100.0)
    
    logging.info(f"[{symbol}] Динамический риск: База={risk_percent}%, Модификатор={confidence_modifier:.2f}, Итог={adjusted_risk_percent:.2f}% (${risk_capital_usd:.2f})")
    
    risk_per_coin_usd = abs(entry_price - sl_price)
    if risk_per_coin_usd == 0: return 0.0
    
    quantity = risk_capital_usd / risk_per_coin_usd
    return float(format_quantity(quantity, symbol) or 0.0)

# --- ОБНОВЛЕННАЯ ФУНКЦИЯ РАЗМЕЩЕНИЯ ОРДЕРОВ V-FINAL ---
async def place_order_async(symbol: str, side: str, order_type: str, quantity: Optional[float] = None, **kwargs) -> Optional[Dict[str, Any]]:
    """
    Финальная версия v-final: quantity необязателен (для closePosition=True),
    исправлена передача reduceOnly, добавлена защита от низкой стоимости, ретраи.
    """
    if not client:
        logging.error(f"[{symbol}] place_order_async: Клиент Binance не инициализирован.")
        return None

    # Проверка и переименование reduce_only
    if 'reduce_only' in kwargs:
        kwargs['reduceOnly'] = kwargs.pop('reduce_only')

    use_close_position = kwargs.get('closePosition', False)

    # --- Валидация Quantity ---
    qty_str: Optional[str] = None
    if quantity is not None and quantity > 0:
        qty_str = format_quantity(quantity, symbol)
        if not qty_str or float(qty_str) <= 0:
            logging.error(f"[{symbol}] Ошибка: Неверное количество для ордера: {quantity}")
            return None
    elif not use_close_position and quantity is None:
        # Quantity обязателен, если НЕ используется closePosition
        logging.error(f"[{symbol}] Ошибка: Количество (quantity) обязательно, если не используется closePosition=True.")
        return None

    # --- Защита от низкой стоимости для условных ордеров ---
    # (Оставляем эту логику, она полезна для обычных STOP/TAKE_PROFIT с quantity)
    if not use_close_position and qty_str and kwargs.get('reduceOnly') and order_type in ['LIMIT', 'STOP_MARKET', 'TAKE_PROFIT_MARKET']:
        try:
            df = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS.get("5m"))
            if df is not None and not df.empty:
                current_price = float(df['close'].iloc[-1])
                notional = float(qty_str) * current_price
                min_notional = float(exchange_info_cache.get(symbol, {}).get("minNotional", 5.0))
                if notional < min_notional * 0.95:
                    logging.warning(f"[{symbol}] ReduceOnly: стоимость ордера ${notional:.2f} < минимума ${min_notional:.2f}. Меняю тип на MARKET.")
                    order_type = 'MARKET'
                    kwargs.pop('price', None); kwargs.pop('stopPrice', None); kwargs.pop('timeInForce', None)
        except Exception as e:
            logging.error(f"[{symbol}] Ошибка в блоке проверки notional: {e}")

    # --- Подготовка параметров для API ---
    params: Dict[str, Any] = {
        'symbol': symbol,
        'side': side,
        'type': order_type,
        'recvWindow': 10000,
        **kwargs  # Включает reduceOnly, closePosition и т.д.
    }

    # Добавляем quantity только если он нужен
    if qty_str and not use_close_position:
        params['quantity'] = qty_str
    elif use_close_position and 'quantity' in params:
         # Убедимся, что quantity не передается с closePosition=True
         del params['quantity']


    # Форматирование цен
    for price_key in ['price', 'stopPrice']:
        if price_key in params and params[price_key]:
            try:
                # Используем новую _formatprice_side
                price_str = _formatprice_side(float(params[price_key]), symbol, side)
                if not price_str:
                    logging.error(f"[{symbol}] Не удалось отформатировать цену '{price_key}': {params[price_key]}")
                    return None
                params[price_key] = price_str
            except (ValueError, TypeError):
                 logging.error(f"[{symbol}] Неверное значение для '{price_key}': {params[price_key]}")
                 return None

    # --- Отправка ордера с ретраями ---
    retries, base_retry_delay = 3, 5
    for attempt in range(retries):
        try:
            logging.info(f"[{symbol}] Попытка #{attempt+1} размещения ордера: {params}")
            order_result = await client.futures_create_order(**params)
            # Логируем успешный ордер
            await log_order_to_json(order_result, f"place_order_{order_type}")
            return order_result

        except BinanceAPIException as e:
            logging.error(f"[{symbol}] API Ошибка (попытка #{attempt+1}): Code={e.code}, Msg='{e.message}'")
            # Не повторяем при логических ошибках или ошибках времени
            if e.code in [-1021, -2021, -4164, -2011, -1111]: # Добавил -1111 (Precision issue), -4164 (closePosition conflict)
                 # Логируем детали для отладки
                 logging.error(f"[{symbol}] Детали ошибки API: {params}")
                 return None
            if attempt >= retries - 1: return None # Превышен лимит ретраев

        except Exception as e:
            logging.error(f"[{symbol}] Неожиданная ошибка (попытка #{attempt+1}): {e}", exc_info=True)
            if attempt >= retries - 1: return None # Превышен лимит ретраев

        delay = base_retry_delay * (2 ** attempt) + random.uniform(0, 1)
        logging.info(f"[{symbol}] Пауза {delay:.1f} сек перед ретраем...")
        await asyncio.sleep(delay)

    return None # Если все ретраи не удались

def _formatprice_side(price: float, symbol: str, side: str) -> Optional[str]:
    """
    Обертка над основной функцией format_price для безопасной передачи
    аргумента 'side', обеспечивая обратную совместимость.
    """
    try:
        # Пытаемся вызвать новую версию format_price с тремя аргументами
        return format_price(price, symbol, side=side)
    except TypeError:
        # Если старая версия (с двумя аргументами), вызываем ее
        return format_price(price, symbol)

async def cancel_single_order_async(symbol: str, order_id: Optional[Any]) -> bool:
    if not client or order_id is None: return True
    try:
        order_id_int = int(order_id)
        await client.futures_cancel_order(symbol=symbol, orderId=order_id_int)
        logging.info(f"Ордер [{symbol}] ID:{order_id_int} успешно отменен.")
        return True
    except BinanceAPIException as e:
        if e.code == -2011:
            logging.info(f"Ордер [{symbol}] ID:{order_id} уже не существует (исполнен/отменен).")
            return True
        logging.error(f"API Ошибка отмены ордера [{symbol}] ID:{order_id}: {e.message}")
        return False
    except (ValueError, TypeError):
        return True
    except Exception as e:
        logging.error(f"Неизвестная ошибка отмены ордера [{symbol}] ID:{order_id}: {e}", exc_info=True)
        return False

# --- КОНЕЦ СЕДЬМОЙ ЧАСТИ ---
# --- НАЧАЛО ВОСЬМОЙ ЧАСТИ (ПОЛНАЯ ФИНАЛЬНАЯ ВЕРСИЯ) ---

async def calculate_market_activity_score_async(symbol: str) -> Tuple[float, Dict[str, Any]]:
    """
    Рассчитывает единый 'Индекс Активности Рынка' (0-10) на основе нескольких факторов.
    """
    scores = {}
    
    # 1. Волатильность BTC (NATR) - Вес: 40%
    try:
        btc_df = market_data_store.get('BTCUSDT', {}).get(ALLOWED_INTERVALS[VOLATILITY_CHECK_TIMEFRAME])
        if btc_df is not None and len(btc_df) >= VOLATILITY_CHECK_PERIOD:
            atr = average_true_range(btc_df['high'], btc_df['low'], btc_df['close'], window=VOLATILITY_CHECK_PERIOD).iloc[-1]
            last_price = btc_df['close'].iloc[-1]
            natr_percent = (atr / last_price) * 100 if last_price > 0 else 0
            scores['btc_volatility'] = min(10, (natr_percent / 0.5) * 10) 
        else:
            scores['btc_volatility'] = 3.0
    except Exception:
        scores['btc_volatility'] = 3.0

    # 2. Торговая сессия - Вес: 30%
    time_context = get_time_context()
    session_scores = {"Asian": 2.0, "European": 7.0, "American": 9.0}
    scores['session'] = session_scores.get(time_context['session'], 5.0)

    # 3. Ликвидность символа - Вес: 20%
    liquidity = await estimate_market_liquidity(symbol)
    scores['liquidity'] = min(10, liquidity * 2)

    # 4. Сила тренда BTC (ADX) - Вес: 10%
    try:
        btc_df_adx = market_data_store.get('BTCUSDT', {}).get(ALLOWED_INTERVALS['1h'])
        if btc_df_adx is not None and len(btc_df_adx) >= 20:
            adx_val = adx(btc_df_adx['high'], btc_df_adx['low'], btc_df_adx['close'], 14).iloc[-1]
            scores['btc_trend_strength'] = min(10, max(0, (adx_val - 20) / 2))
        else:
            scores['btc_trend_strength'] = 4.0
    except Exception:
        scores['btc_trend_strength'] = 4.0

    total_score = (
        scores['btc_volatility'] * 0.4 +
        scores['session'] * 0.3 +
        scores['liquidity'] * 0.2 +
        scores['btc_trend_strength'] * 0.1
    )
    
    return total_score, scores

async def get_adaptive_entry_threshold(
    symbol: str, 
    ai_confidence: Optional[int] = None, 
    required_threshold: Optional[float] = None
) -> float:
    """Адаптивный порог, дополнительно сниженный для флэта."""
    base = float(required_threshold) if required_threshold is not None else SCORE_THRESHOLD_NORMAL_VOL
    
    adx_val = None
    try:
        tf_key = ALLOWED_INTERVALS.get(ML_MODEL_TF_SHORT, ALLOWED_INTERVALS.get("15m"))
        df = market_data_store.get(symbol, {}).get(tf_key)
        if df is not None and len(df) > 30:
            adx_val = float(ta.trend.adx(df["high"], df["low"], df["close"], window=14).iloc[-1])
    except Exception:
        adx_val = None

    if adx_val is not None:
        if adx_val < 18.0:
            base -= 3.0  # ✅ Было -2.0, стало -3.0 для большего смягчения
        elif adx_val >= 28.0:
            base += 2.0

    try:
        btc_state, _ = await get_btc_trend_state()
        if btc_state in ("SIDEWAYS", "WEAK_UP", "WEAK_DOWN"):
            base -= 1.0
    except Exception:
        pass

    base = max(58.0, min(68.0, base))
    if required_threshold is not None:
        base = max(base, float(required_threshold))
        
    return float(round(base, 1))


async def get_global_market_environment() -> str:
    """
    Получает состояние рынка по BTC, кэширует его на 1 минуту.
    """
    global global_market_environment, global_market_env_last_checked
    
    if time.time() - global_market_env_last_checked < 60:
        return global_market_environment

    async with market_env_lock:
        if time.time() - global_market_env_last_checked < 60:
            return global_market_environment
            
        logging.info("Обновляю глобальное состояние рынка по BTC...")
        new_env = await get_market_environment_from_ai('BTCUSDT')
        global_market_environment = new_env
        global_market_env_last_checked = time.time()
        
        return global_market_environment

def get_time_context() -> Dict[str, Any]:
    now_utc = datetime.datetime.now(dt_timezone.utc)
    hour = now_utc.hour
    session = "Asian" if 0 <= hour < 8 else "European" if 8 <= hour < 16 else "American"
    return {"session": session, "hour_utc": hour}

async def estimate_market_liquidity(symbol: str) -> float:
    """
    Оценивает ликвидность символа по объему торгов (0-5).
    Возвращает оценку от 0 до 5, где 5 - максимальная ликвидность.
    """
    try:
        tf_val = ALLOWED_INTERVALS.get('15m')
        df = market_data_store.get(symbol, {}).get(tf_val)
        
        if df is None or len(df) < 20:
            return 2.5  # Средняя оценка по умолчанию
        
        # Средний объем за последние 20 периодов в USDT
        avg_volume_usdt = (df['volume'] * df['close']).tail(20).mean()
        
        # Шкала: до 100k = низкая, 100k-500k = средняя, 500k+ = высокая
        if avg_volume_usdt > 500000:
            return 5.0
        elif avg_volume_usdt > 250000:
            return 4.0
        elif avg_volume_usdt > 100000:
            return 3.0
        elif avg_volume_usdt > 50000:
            return 2.0
        else:
            return 1.0
            
    except Exception as e:
        logging.error(f"Ошибка оценки ликвидности для {symbol}: {e}")
        return 2.5

# --- БЛОК 1: НОВЫЙ ГЛОБАЛЬНЫЙ ФИЛЬТР BTC (ДЛЯ ЗАМЕНЫ) ---

# Глобальный кэш для хранения состояния и времени последнего обновления
GLOBAL_TREND_CACHE = {"state": "SIDEWAYS", "ts": 0.0}


def _safe(df, need=60):
    """Проверяет, что DataFrame существует и содержит достаточно данных."""
    return df is not None and len(df) >= need

def _natr_pct(df, period=14) -> float:
    """Вспомогательная функция для расчета нормализованного ATR в процентах."""
    atr = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=period).iloc[-1]
    close = float(df["close"].iloc[-1])
    return float(atr / max(close, 1e-9) * 100.0)

def _ema_adx(df, fast, slow, adx_win=14):
    """Вспомогательная функция для извлечения EMA и ADX."""
    ef = float(ta.trend.ema_indicator(df["close"], window=fast).iloc[-1])
    es = float(ta.trend.ema_indicator(df["close"], window=slow).iloc[-1])
    av = float(ta.trend.adx(df["high"], df["low"], df["close"], window=adx_win).iloc[-1])
    return ef, es, av

def _adx_thr(natr):
    """Волатильно-адаптивный порог ADX."""
    if natr is None: return 20.0
    if natr < 0.8: return 22.0  # В тихом рынке требуем более сильного ADX
    if natr > 2.0: return 18.0  # В высокой волатильности можем принять более низкий ADX
    return 20.0

async def get_global_trend_direction() -> str:
    """
    Возвращает: UPTREND / DOWNTREND / SIDEWAYS.
    --- ВЕРСИЯ V3.1: Улучшенная логика с взвешенным MTF, подтверждениями,
                     динамическим гистерезисом и нормализацией факторов. ---
    """
    symbol = "BTCUSDT"
    log_prefix = "[Global Trend V3.1]"

    # --- 1. Получение данных и базовых индикаторов ---
    tf15, tf1h, tf4h = ALLOWED_INTERVALS.get("15m"), ALLOWED_INTERVALS.get("1h"), ALLOWED_INTERVALS.get("4h")
    df15 = market_data_store.get(symbol, {}).get(tf15)
    df1h = market_data_store.get(symbol, {}).get(tf1h)
    df4h = market_data_store.get(symbol, {}).get(tf4h)

    need = max(BTC_EMA_SLOW_PERIOD + 5, 60)
    if not (_safe(df15, need) and _safe(df1h, need) and _safe(df4h, need)):
        logging.warning(f"{log_prefix} Недостаточно данных. Возвращаем кэш: {GLOBAL_TREND_CACHE['state']}")
        return GLOBAL_TREND_CACHE.get("state", "SIDEWAYS")

    try:
        n15, n1h, n4h = _natr_pct(df15), _natr_pct(df1h), _natr_pct(df4h)
        th15, th1h, th4h = _adx_thr(n15), _adx_thr(n1h), _adx_thr(n4h)
        ef15, es15, adx15 = _ema_adx(df15, BTC_EMA_FAST_PERIOD, BTC_EMA_SLOW_PERIOD)
        ef1h, es1h, adx1h = _ema_adx(df1h, BTC_EMA_FAST_PERIOD, BTC_EMA_SLOW_PERIOD)
        ef4h, es4h, adx4h = _ema_adx(df4h, BTC_EMA_FAST_PERIOD, BTC_EMA_SLOW_PERIOD)
        GLOBAL_TREND_CACHE.update({"last_adx1h": adx1h, "last_natr1h": n1h or 1.0}) # Сохраняем для гистерезиса
    except Exception as e:
        logging.error(f"{log_prefix} Ошибка расчета индикаторов: {e}. Возвращаем кэш.")
        return GLOBAL_TREND_CACHE.get("state", "SIDEWAYS")

    # --- 2. Взвешенное MTF голосование ---
    # Веса: 15m=1, 1h=2 (ядро), 4h=1.5 (подтверждение)
    score = 0.0
    max_score = 1 + 2 + 1.5
    tf_details = []

    for tf_name, ef, es, av, thr, weight in [("15m", ef15, es15, adx15, th15, 1.0),
                                            ("1h", ef1h, es1h, adx1h, th1h, 2.0),
                                            ("4h", ef4h, es4h, adx4h, th4h, 1.5)]:
        state = "SIDE"
        if av >= thr:
            if ef > es: state = "UP"; score += weight
            elif ef < es: state = "DOWN"; score -= weight
        tf_details.append(f"{tf_name}={state}({av:.1f})")

    # Нормализуем скор к [-1, 1]
    norm_score = score / max_score if max_score > 0 else 0.0
    logging.info(f"{log_prefix} MTF Score: {norm_score:.2f} [{', '.join(tf_details)}]")

    # --- 3. Определение базового состояния по скору ---
    # Пороги для определения состояния (можно калибровать)
    UP_THRESHOLD = 0.40
    DOWN_THRESHOLD = -0.40
    if norm_score >= UP_THRESHOLD: base_state = "UPTREND"
    elif norm_score <= DOWN_THRESHOLD: base_state = "DOWNTREND"
    else: base_state = "SIDEWAYS"

    # --- 4. Применение Фильтра 4h (как обязательное условие для ТРЕНДА) ---
    state_4h = "UP" if adx4h >= th4h and ef4h > es4h else ("DOWN" if adx4h >= th4h and ef4h < es4h else "SIDE")
    if base_state == "UPTREND" and state_4h == "DOWN":
        logging.warning(f"{log_prefix} UPTREND заблокирован фильтром 4h (DOWN). Установлен SIDEWAYS.")
        base_state = "SIDEWAYS"
    if base_state == "DOWNTREND" and state_4h == "UP":
        logging.warning(f"{log_prefix} DOWNTREND заблокирован фильтром 4h (UP). Установлен SIDEWAYS.")
        base_state = "SIDEWAYS"

    # --- 5. Подтверждение смены тренда (Volume/Squeeze) ---
    # Применяется только при переходе из SIDEWAYS в UPTREND/DOWNTREND
    prev_state = GLOBAL_TREND_CACHE.get("state", "SIDEWAYS")
    confirmed_change = True
    if prev_state == "SIDEWAYS" and base_state != "SIDEWAYS":
        # Проверяем подтверждения (используем ваши существующие функции)
        try:
            vol_ok = is_breakout_volume_confirmed(df1h, multiplier=1.5) # Используем 1h для большей значимости
            # Squeeze проверяем на 15m, так как он обычно предшествует пробою
            squeeze_ok = is_in_volatility_squeeze(df15.iloc[:-1], lookback=20) # Проверяем НАКАНУНЕ пробоя
            if not (vol_ok or squeeze_ok):
                confirmed_change = False
                logging.warning(f"{log_prefix} Смена {prev_state} -> {base_state} НЕ ПОДТВЕРЖДЕНА (Vol OK: {vol_ok}, Squeeze OK: {squeeze_ok}). Сохраняем {prev_state}.")
                base_state = prev_state # Отменяем смену состояния
        except Exception as e_confirm:
            logging.error(f"{log_prefix} Ошибка проверки подтверждений: {e_confirm}. Смена разрешена по умолчанию.")

    # --- 6. Нормализация и проверка вторичных факторов (OFI, Funding, Basis, Liqs) ---
    # (Функции normalize_ofi, normalize_funding_basis, get_liquidation_signal_v2 - ПРЕДПОЛАГАЮТСЯ РЕАЛИЗОВАННЫМИ)
    veto_reason = None
    try:
        ofi_norm = normalize_ofi(symbol) # Возвращает z-score или процентиль
        funding_norm, basis_norm = normalize_funding_basis(symbol) # Возвращают z-scores/процентили
        liq_signal = get_liquidation_signal(symbol) # Ваша существующая функция

        # Экстремальные значения (например, |z-score| > 2.0 или >95%/ <5% процентиль)
        EXTREME_THRESHOLD = 2.0

        is_liq_spike = liq_signal.get('spike', False)
        liq_bias = liq_signal.get('bias', 'NEUTRAL')

        if base_state == "UPTREND":
            if ofi_norm < -EXTREME_THRESHOLD and is_liq_spike and liq_bias == "SELL": veto_reason = "Экстремально негативный OFI + Ликв. продавцов"
            if funding_norm > EXTREME_THRESHOLD and basis_norm > EXTREME_THRESHOLD: veto_reason = "Экстремально перегретый Funding/Basis"
        elif base_state == "DOWNTREND":
            if ofi_norm > EXTREME_THRESHOLD and is_liq_spike and liq_bias == "BUY": veto_reason = "Экстремально позитивный OFI + Ликв. покупателей"
            if funding_norm < -EXTREME_THRESHOLD and basis_norm < -EXTREME_THRESHOLD: veto_reason = "Экстремально переохлажденный Funding/Basis"

        if veto_reason:
            logging.warning(f"{log_prefix} Состояние '{base_state}' отменено из-за вето вторичных факторов: {veto_reason}. Установлен SIDEWAYS.")
            base_state = "SIDEWAYS"

    except Exception as e_secondary:
        logging.error(f"{log_prefix} Ошибка проверки вторичных факторов: {e_secondary}. Проверка пропущена.")

    # --- 7. Динамический Гистерезис и Обновление Кэша ---
    now = time.time()
    prev_ts = GLOBAL_TREND_CACHE.get("ts", 0.0)
    last_adx = GLOBAL_TREND_CACHE.get("last_adx1h", 20.0)
    last_natr = GLOBAL_TREND_CACHE.get("last_natr1h", 1.0)

    # Динамическое время удержания
    if last_adx >= 30: hold_sec = 1200 # Сильный тренд - держим дольше
    elif last_adx < 20: hold_sec = 450 # Флэт - меняем быстрее
    else: hold_sec = 750 # Умеренный тренд
    hold_sec *= max(0.8, min(1.5, last_natr / 1.0)) # Коррекция на волатильность

    # Пороги для смены состояния (двухпороговый гистерезис на основе norm_score)
    P_HIGH = 0.55 # Порог для входа в UPTREND
    P_LOW = -0.55 # Порог для входа в DOWNTREND
    P_EXIT = 0.15 # |norm_score| < P_EXIT для возврата в SIDEWAYS

    final_state = prev_state

    if prev_state == "SIDEWAYS":
        if base_state == "UPTREND" and norm_score >= P_HIGH and confirmed_change: final_state = "UPTREND"
        elif base_state == "DOWNTREND" and norm_score <= P_LOW and confirmed_change: final_state = "DOWNTREND"
    elif prev_state == "UPTREND":
        if base_state == "DOWNTREND" and norm_score <= P_LOW: final_state = "DOWNTREND" # Прямой переворот
        elif abs(norm_score) < P_EXIT or base_state == "SIDEWAYS": final_state = "SIDEWAYS" # Возврат во флэт
    elif prev_state == "DOWNTREND":
        if base_state == "UPTREND" and norm_score >= P_HIGH: final_state = "UPTREND" # Прямой переворот
        elif abs(norm_score) < P_EXIT or base_state == "SIDEWAYS": final_state = "SIDEWAYS" # Возврат во флэт

    # Обновляем кэш, только если прошло время удержания ИЛИ состояние изменилось на SIDEWAYS
    if final_state != prev_state:
        if (now - prev_ts) >= hold_sec or final_state == "SIDEWAYS":
            # --- !!! ПРОВЕРКА 4h GATE ПЕРЕД АПДЕЙТОМ КЭША !!! ---
            gate_ok, gate_reason = btc_4h_gate_with_override(symbol, 'long' if final_state == "UPTREND" else 'short', 0.5, True) # Параметры для совместимости
            if not gate_ok and final_state != "SIDEWAYS":
                 logging.warning(f"{log_prefix} Финальное обновление КЭША {prev_state} -> {final_state} ЗАБЛОКИРОВАНО 4h Gate: {gate_reason}. Сохраняем {prev_state}.")
                 # Не обновляем кэш, возвращаем старое состояние
                 return prev_state
            # --- КОНЕЦ ПРОВЕРКИ 4h GATE ---

            logging.critical(f"🔥🔥🔥 {log_prefix} СМЕНА ГЛОБАЛЬНОГО НАПРАВЛЕНИЯ: {prev_state} -> {final_state} 🔥🔥🔥 (Hold: {hold_sec:.0f}s)")
            GLOBAL_TREND_CACHE.update({"state": final_state, "ts": now})
            return final_state
        else:
            # Время удержания не прошло, возвращаем старое состояние
            logging.info(f"{log_prefix} Удержание состояния {prev_state} (осталось {hold_sec - (now - prev_ts):.0f}s). Сигнал на смену: {base_state} (Score: {norm_score:.2f})")
            return prev_state
    else:
        # Состояние не изменилось, обновляем метку времени
        GLOBAL_TREND_CACHE.update({"ts": now})
        return final_state

# --- НЕОБХОДИМЫЕ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (заглушки, требуют реализации) ---
def normalize_ofi(symbol: str) -> float:
    # TODO: Реализовать нормализацию OFI (z-score или процентиль за окно)
    # Заглушка:
    try: score, _ = asyncio.run(calculate_order_flow_imbalance(symbol)) # Запускаем синхронно для простоты
    except: score = 0.0
    return np.clip(score / 50.0, -3.0, 3.0) # Примерная нормализация к z-score

def normalize_funding_basis(symbol: str) -> Tuple[float, float]:
    # TODO: Реализовать нормализацию Funding/Basis (z-score или процентиль за окно)
    # Заглушка:
    try: data = asyncio.run(get_basis_and_funding(symbol))
    except: data = {}
    funding_pct = data.get("funding_pct", 0.0)
    basis_bps = data.get("basis_bps", 0.0)
    # Примерная нормализация к z-score
    funding_norm = np.clip(funding_pct / 0.05, -3.0, 3.0)
    basis_norm = np.clip(basis_bps / 15.0, -3.0, 3.0)
    return funding_norm, basis_norm

# --- БЛОК 2: УМНОЕ УПРАВЛЕНИЕ TAKE-PROFIT ---

def rr(side: str, entry: float, sl: float, price: float) -> float:
    """Рассчитывает текущее соотношение Риск/Прибыль."""
    if side.upper() == "LONG":
        risk = max(entry - sl, 1e-9)
        reward = max(price - entry, 0.0)
    else:
        risk = max(sl - entry, 1e-9)
        reward = max(entry - price, 0.0)
    return float(reward / risk)

def atr(df, period=14):
    """Рассчитывает ATR."""
    import ta
    if df is None or len(df) < period + 1: return 0.0
    try:
        return float(ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=period).iloc[-1])
    except Exception:
        return 0.0


async def sync_server_time():
    """Периодически синхронизирует время с сервером Binance для контроля."""
    global SERVER_TIME_OFFSET
    try:
        if not client: return
        server_time = await client.futures_time()
        server_ms = int(server_time["serverTime"])
        local_ms = int(time.time() * 1000)
        SERVER_TIME_OFFSET = server_ms - local_ms
        logging.info(f"Смещение времени с сервером: {SERVER_TIME_OFFSET} мс.")
    except Exception as e:
        logging.error(f"Ошибка синхронизации времени: {e}")

async def manage_near_tp_band(pm, band_pct=0.12, atr_band_mult=0.25, slip_bps=8, min_rr=1.3) -> bool:
    """
    Исправленная версия: использует GTC-ордера для надёжности и отслеживает
    их по orderId, чтобы избежать "одноразовых" попыток.
    """
    pos = pm.state
    df = market_data_store.get(pm.symbol, {}).get(ALLOWED_INTERVALS["5m"])
    if df is None or df.empty or pos.tp1_price is None or pos.initial_sl_price is None:
        return False

    price = float(df["close"].iloc[-1])
    rr_now = rr(pos.side, pos.entry_price, pos.initial_sl_price, price)
    target = float(pos.tp1_price)
    dist_total = abs(target - pos.entry_price)
    dist_curr  = abs(target - price)
    atr_val = atr(df, 14) or 0.0
    band_abs = max(dist_total * band_pct, atr_val * atr_band_mult)
    
    if rr_now < min_rr or dist_curr > band_abs:
        # Если вышли из зоны цели, отменяем старые лимитные ордера
        partial_oid = pos.meta.get("near_tp_partial_oid")
        if partial_oid:
            await cancel_single_order_async(pm.symbol, partial_oid)
            del pos.meta["near_tp_partial_oid"]
        return False

    # --- ИСПРАВЛЕННАЯ ЛОГИКА ЧАСТИЧНОЙ ФИКСАЦИИ ---
    # Выставляем ордер, только если его ещё нет
    if not pos.meta.get("near_tp_partial_oid"):
        frac = 0.30 if not pos.meta.get("tp1_done", False) else 0.20
        qty = float(pos.current_quantity) * frac
        if qty > 0:
            off = (slip_bps / 10000.0) * price
            side_str = "SELL" if pos.side.upper() == "LONG" else "BUY"
            lim_price = target - off if side_str == "SELL" else target + off
            pstr = format_price(lim_price, pm.symbol, side=side_str)
            if pstr:
                # Ставим GTC ордер, который будет "жить" до исполнения
                order_info = await place_order_async(
                    symbol=pm.symbol, side=side_str, order_type="LIMIT",
                    timeInForce="GTC", reduce_only=True, quantity=qty, price=pstr
                )
                if order_info and order_info.get("orderId"):
                    # Сохраняем ID ордера, а не просто флаг
                    pos.meta["near_tp_partial_oid"] = order_info.get("orderId")
                    await save_state_async()

    # --- Логика страховочного TP-MARKET и трейлинга остаётся прежней ---
    if not pos.meta.get("near_tp_guard_oid"):
        trig_str = format_price(target, pm.symbol, side="SELL" if pos.side.upper()=="LONG" else "BUY")
        if trig_str:
            order_info = await place_order_async(
                symbol=pm.symbol, side="SELL" if pos.side.upper()=="LONG" else "BUY",
                order_type="TAKE_PROFIT_MARKET", reduce_only=True,
                quantity=float(pos.current_quantity) * 0.50, stopPrice=trig_str
            )
            if order_info and order_info.get("orderId"):
                pos.meta["near_tp_guard_oid"] = order_info.get("orderId")
                await save_state_async()
    
    # Ужесточаем ATR-трейлинг в зоне
    if atr_val > 0:
        atr_mult = 1.0 if rr_now >= 2.5 else (1.5 if rr_now >= 1.5 else 2.0)
        new_sl = price - atr_val * atr_mult if pos.side.upper() == "LONG" else price + atr_val * atr_mult
        if (pos.side.upper()=="LONG" and new_sl > pos.sl_price) or \
           (pos.side.upper()=="SHORT" and new_sl < pos.sl_price):
            await pm._update_sl_callback(float(new_sl), f"Near TP trail RR={rr_now:.2f}, ATRx{atr_mult}")
            await save_state_async()
            
    return True

async def global_direction_filter(trade_type: str, symbol_for_logging: str) -> tuple[bool, str]:
    """
    Простая обёртка над новым вероятностным агрегатором BTC_DIRECTION.
    ИСПРАВЛЕНА ОШИБКА ЛОГИРОВАНИЯ.
    """
    # Вызываем основной вердикт (эта часть работает правильно)
    ok, reason, snap = await global_direction_verdict(trade_type)
    
    # ✅ ИСПРАВЛЕНИЕ: Используем if/else для вызова правильной функции логирования
    if ok:
        logging.info(f"[{symbol_for_logging}] Global Direction Verdict for {trade_type.upper()}: PASSED. Reason: {reason}")
    else:
        logging.warning(f"[{symbol_for_logging}] Global Direction Verdict for {trade_type.upper()}: BLOCKED. Reason: {reason}")
    # --- КОНЕЦ ИСПРАВЛЕНИЯ ---
    
    return ok, reason

async def get_market_sentiment_filter(symbol: str, trade_type: str) -> Tuple[bool, str]:
    """
    Проверяет рыночные настроения по индексу "Страха и жадности".
    Возвращает (True/False, "причина").
    """
    if not globals().get('MARKET_SENTIMENT_FILTER_ENABLED', False):
        return True, "Фильтр настроений отключен."

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("https://api.alternative.me/fng/?limit=1")
            response.raise_for_status()
            data = response.json().get('data')
            
            if not data:
                logging.warning("[Sentiment Filter] Не удалось получить данные индекса страха и жадности.")
                return True, "Не удалось получить данные F&G."

            sentiment_value = int(data[0]['value'])
            sentiment_class = data[0]['value_classification']
            
            if trade_type == 'long' and sentiment_value <= 15: # Extreme Fear
                reason = f"ЛОНГ заблокирован. Экстремальный страх (индекс: {sentiment_value})."
                logging.warning(f"[{symbol}] {reason}")
                return False, reason
            
            if trade_type == 'short' and sentiment_value >= 85: # Extreme Greed
                reason = f"ШОРТ заблокирован. Экстремальная жадность (индекс: {sentiment_value})."
                logging.warning(f"[{symbol}] {reason}")
                return False, reason

            return True, f"Настроения рынка в норме (индекс: {sentiment_value})."

    except Exception as e:
        logging.error(f"[Sentiment Filter] Ошибка при проверке индекса страха и жадности: {e}")
        return True, "Ошибка при проверке F&G."


async def multi_tf_trend_check(symbol: str, trade_type: str) -> bool:
    # Убираем ML_MODEL_TF_LONG, так как новая модель его не использует
    tfs_to_check = [ML_MODEL_TF_SHORT, ML_MODEL_TF_MEDIUM]
    
    agreeing_trends = 0
    for tf_key in tfs_to_check:
        df = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS[tf_key])
        if df is None or len(df) < 21: continue
        try:
            ema = ema_indicator(df['close'], window=21).iloc[-1]
            current_price = df['close'].iloc[-1]
            if (trade_type == 'long' and current_price > ema) or (trade_type == 'short' and current_price < ema):
                agreeing_trends += 1
        except Exception:
            continue
            
    # Для новой модели достаточно совпадения на 2-х таймфреймах
    is_confirmed = agreeing_trends >= 2
    
    if not is_confirmed:
        logging.info(f"[{symbol}] Фильтр Multi-TF: Тренд не подтвержден. Совпадений: {agreeing_trends}/{len(tfs_to_check)} для {trade_type}.")
    return is_confirmed

async def worker(worker_id: int):
    logging.info(f"[Worker-{worker_id}] Запущен.")
    while True:
        try:
            msg = await message_queue.get()
            await process_message(msg)
            message_queue.task_done()
        except Exception as e:
            logging.error(f"[Worker-{worker_id}] Критическая ошибка: {e}", exc_info=True)


async def log_order_to_json(order_data: Dict[str, Any], purpose: str):
    log_entry = {"timestamp_utc": datetime.datetime.now(dt_timezone.utc).isoformat(), "purpose": purpose, "order_details": {k: str(v) for k, v in order_data.items()}}
    async with order_log_lock:
        try:
            _all_orders: List[Dict] = []
            if aiofiles and os.path.exists(ORDER_LOG_FILE):
                async with aiofiles.open(ORDER_LOG_FILE, mode='r', encoding='utf-8') as f:
                    content = await f.read()
                    if content: _all_orders = json.loads(content)
            _all_orders.append(log_entry)
            if aiofiles:
                async with aiofiles.open(ORDER_LOG_FILE, mode='w', encoding='utf-8') as f:
                    await f.write(json.dumps(_all_orders, indent=2))
            else:
                if loop: await loop.run_in_executor(None, sync_log_order_fallback, log_entry)
        except Exception as e:
            logging.error(f"Не удалось записать лог ордера в {ORDER_LOG_FILE}: {e}", exc_info=True)


def sync_log_order_fallback(log_entry):
    _all_orders = []
    if os.path.exists(ORDER_LOG_FILE):
        with open(ORDER_LOG_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
            if content: _all_orders = json.loads(content)
    _all_orders.append(log_entry)
    with open(ORDER_LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(_all_orders, f, indent=2)


async def set_leverage_async(symbol: str, leverage: int) -> bool:
    global client
    if not client: return False
    try:
        await client.futures_change_leverage(symbol=symbol, leverage=leverage)
        return True
    except BinanceAPIException as e:
        if e.code != -4046: logging.error(f"API Ошибка уст. плеча {symbol}: {e.message}")
        return e.code == -4046


async def apply_global_leverage_async(target_leverage: int):
    global exchange_info_cache, SYMBOLS
    symbols_to_set=[s for s in SYMBOLS if s in exchange_info_cache and exchange_info_cache[s].get('status')=='TRADING']
    tasks = [set_leverage_async(s, target_leverage) for s in symbols_to_set]
    results = await asyncio.gather(*tasks)
    logging.info(f"Применение плеча завершено. Успешно: {sum(1 for r in results if r)}/{len(results)}.")


async def initialize_clients():
    """Создает асинхронный клиент Binance с увеличенным таймаутом."""
    global client, loop, API_KEY, API_SECRET
    if not API_KEY or not API_SECRET:
        raise ValueError("API ключи для Binance не установлены")

    if not loop:
        loop = asyncio.get_running_loop()

    # Создаем словарь с параметрами для http-клиента, включая таймаут
    # Устанавливаем общий таймаут в 60 секунд.
    requests_params = {"timeout": 60}

    # Передаем этот словарь при создании клиента
    client = await AsyncClient.create(API_KEY, API_SECRET, loop=loop, requests_params=requests_params)
    
    # Проверяем соединение
    await client.ping()
    logging.warning("✅ AsyncClient для Binance успешно создан (с таймаутом 60с).")




async def on_closed_candle_analysis(symbol: str):
    """
    Флэт-логика с пре-фильтрами, финальным вердиктом от Gemini и полным логированием.
    --- ВЕРСИЯ С ГЛОБАЛЬНЫМ ФИЛЬТРОМ И СЕТКОЙ ГРАФИКОВ 4-В-1 ---
    """
    df = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS.get("15m"))
    if df is None or len(df) < 80:
        return

    # --- Диагностика "Пульс бота" ---
    try:
        rsi_series = ta.momentum.rsi(df["close"], window=14)
        last_rsi = float(rsi_series.iloc[-1])
        adx_val = float(ta.trend.adx(df["high"], df["low"], df["close"], window=14).iloc[-1])
        params = await get_adaptive_strategy_parameters(symbol)
        rsi_oversold = float(params.get("rsi_oversold", 30))
        rsi_overbought = float(params.get("rsi_overbought", 70))
        logging.info(f"~ [{symbol}] Проверка флэта: RSI={last_rsi:.1f} (Пороги: {rsi_oversold:.1f}/{rsi_overbought:.1f}), ADX={adx_val:.1f}")
    except Exception:
        return

    passport = None
    try:
        # --- Этап 1: Поиск первичного триггера ---
        prev_rsi = float(rsi_series.iloc[-2])
        trade_type, trigger_reason = None, None
        
        if prev_rsi < rsi_oversold and last_rsi > rsi_oversold:
            trade_type, trigger_reason = "long", f"RSI Выход из перепроданности ({last_rsi:.1f})"
        elif prev_rsi > rsi_overbought and last_rsi < rsi_overbought:
            trade_type, trigger_reason = "short", f"RSI Выход из перекупленности ({last_rsi:.1f})"
        
        if not trade_type:
            return

        # --- Этап 2: Быстрые локальные пре-фильтры ---
        if adx_val >= 25.0:
            # Для флэтовой стратегии это не ошибка, а просто пропуск
            logging.info(f"~ [{symbol}] Ranging сигнал пропущен: рынок в тренде (ADX={adx_val:.1f})")
            return

        atr_val = _atr(df, 14)
        natr_percent = (atr_val / df['close'].iloc[-1]) * 100 if df['close'].iloc[-1] > 0 else 0
        if natr_percent < 0.2:
            logging.info(f"~ [{symbol}] Ranging сигнал пропущен: рынок 'мертвый' (NATR={natr_percent:.2f}%)")
            return

        # --- Этап 3: Создание паспорта и глубокий анализ AI ---
        passport = TradePassport(symbol=symbol, strategy="Ranging_AI_Filtered", trigger=trigger_reason, side=trade_type)
        logging.warning(f"🎯 [{symbol}] НАЙДЕН СИГНАЛ RANGING: {passport.side.upper()}. Прошел пре-фильтры. Отправляю на анализ в Gemini...")

        async with GLOBAL_ENTRY_LOCK:
            # --- ✅ ИЗМЕНЕНИЕ: ФИЛЬТР ГЛОБАЛЬНОГО НАПРАВЛЕНИЯ ---
            ok, reason = await global_direction_filter(trade_type, symbol)
            if not ok:
                raise ValueError(reason)

            # --- ПРОВЕРКА ЛИМИТА ПОЗИЦИЙ ---
            dynamic_limit = get_dynamic_max_positions()
            if len(current_positions) >= dynamic_limit:
                logging.info(f"[{symbol}] Сигнал 'Ranging' пропущен: достигнут лимит позиций ({len(current_positions)}/{dynamic_limit})")
                return

            # Сбор базового контекста
            basis_data = await get_basis_and_funding(symbol)
            context_text = f"Funding: {basis_data.get('funding_pct', 0):.4f}%. Basis: {basis_data.get('basis_bps', 0):.1f} bps."
            
            # --- ✅ ИЗМЕНЕНИЕ: СОЗДАЕМ СЕТКУ 4-В-1 ---
            df_sym_1h = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS['1h'])
            df_btc_15m = market_data_store.get('BTCUSDT', {}).get(ALLOWED_INTERVALS['15m'])
            df_btc_1h = market_data_store.get('BTCUSDT', {}).get(ALLOWED_INTERVALS['1h'])
            
            combined_chart_base64 = await create_multi_timeframe_chart_grid(
                symbol, df, df_sym_1h, df_btc_15m, df_btc_1h
            )
            if not combined_chart_base64: raise ValueError("Не удалось создать сетку графиков")

            # --- ✅ ИЗМЕНЕНИЕ: ОБНОВЛЯЕМ ПРОМПТ ДЛЯ AI ---
            prompt = (
                f"Вы — аналитик, оцениваете контртрендовый сигнал '{trade_type.upper()}' (выход из зоны перепроданности/перекупленности RSI) по {symbol}.\n"
                f"Перед вами 4 графика: {symbol} (15м/1ч) и BTC (15м/1ч).\n\n"
                f"**ЗАДАЧА:** Проведите мульти-таймфрейм анализ. Является ли этот сигнал на 15м графике хорошей возможностью для входа против локального движения в рамках боковика? "
                f"Не указывает ли старший тренд на 1ч или общая ситуация на BTC на то, что это начало сильного тренда, а не просто коррекция?\n\n"
                f"Ответ строго в формате JSON: {{\"confidence\": 0-100, \"explanation\": \"...\"}}"
            )
            
            ai_verdict = await analyze_chart_with_gemini_vision(symbol, combined_chart_base64, trade_type, prompt)
            ai_confidence = ai_verdict.get("confidence", 0)
            
            entry_threshold = (await get_adaptive_entry_threshold(symbol)) - 5 
            if ai_confidence < entry_threshold:
                raise ValueError(f"Gemini отклонил сделку (уверенность {ai_confidence}% < порога {entry_threshold:.1f}%)")

            # --- Этап 4: Исполнение сделки ---
            entry_price = float(df["close"].iloc[-1])
            sltp = await get_sltp_from_ai_async(symbol, trade_type, entry_price, context={})
            if not (sltp and sltp.get("stop_loss") and sltp.get("take_profit")): raise ValueError("AI не предоставил SL/TP")
            final_sl_price, final_tp_price = float(sltp["stop_loss"]), float(sltp["take_profit"])
            
            qty = await calculate_smart_quantity_v2(symbol, entry_price, final_sl_price, ai_confidence, BASE_RISK_PERCENT, entry_threshold)
            if qty <= 0: raise ValueError("Объем равен нулю")
            
            execution_result = await adaptive_execute_entry(
                symbol=symbol, 
                side_str="BUY" if trade_type == "long" else "SELL", 
                quantity=qty, 
                sl_price=final_sl_price, 
                tp_price=final_tp_price, 
                entry_context=f"AI Conf: {ai_confidence}%, Trigger: {trigger_reason}, ID: {passport.signal_id}",
                signal_price=entry_price
            )

            if not execution_result: raise ValueError("Исполнение сделки не удалось")

            passport.status = "EXECUTED"
            passport.entry_price = execution_result['entry_price']
            passport.sl_price = final_sl_price
            passport.tp_price = final_tp_price
            passport.ai_confidence = ai_confidence
            passport.required_threshold = entry_threshold
            current_positions[symbol] = PositionManager(execution_result)
            await save_state_async()

    except ValueError as e:
        if passport:
            passport.status = "REJECTED"
            passport.reject_reason = str(e)
            logging.warning(f"🚫 [{symbol}] Сигнал ОТКЛОНЕН. Причина: {passport.reject_reason}")
    except Exception as e:
        if passport:
            passport.status = "REJECTED"
            passport.reject_reason = f"КРИТИЧЕСКАЯ ОШИБКА: {e}"
        logging.error(f"[{symbol}] КРИТИЧЕСКАЯ ОШИБКА при проверке сигнала: {e}", exc_info=True)
            
    finally:
        if passport and passport.status != "PENDING":
            await log_and_notify_async(passport, df)

async def pyramiding_entry(symbol: str, existing_manager: PositionManager, new_signal_confidence: float) -> bool:
    """
    Пирамидинг: добавление к прибыльной позиции в направлении тренда.
    """
    pos = existing_manager.state
    log_prefix = f"🔺 [{symbol}]"

    df = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS['5m'])
    if df is None or pos.initial_sl_price is None or pos.initial_quantity is None:
        return False

    current_price = df['close'].iloc[-1]
    current_rr = _rr(pos.side, pos.entry_price, pos.initial_sl_price, current_price)

    if current_rr < 1.0:
        logging.info(f"{log_prefix} Пирамидинг недоступен (RR={current_rr:.1f} < 1.0)")
        return False
        
    # <<< ИЗМЕНЕНИЕ: Повышен порог уверенности для пирамидинга >>>
    if new_signal_confidence < 80: # Раньше было 75, теперь строже
        logging.info(f"{log_prefix} Пирамидинг отклонен (уверенность AI={new_signal_confidence:.1f} < 80)")
        return False
    # <<< КОНЕЦ ИЗМЕНЕНИЯ >>>

    pyramids_count = pos.meta.get('pyramids_count', 0)
    if pyramids_count >= 2:
        logging.info(f"{log_prefix} Достигнут лимит пирамидинга (2)")
        return False

    add_quantity_raw = pos.initial_quantity * 0.5
    add_quantity = float(format_quantity(add_quantity_raw, symbol))
    if add_quantity <= 0: return False

    logging.warning(f"{log_prefix} ПИРАМИДИНГ #{pyramids_count + 1} ОДОБРЕН | "
                    f"Добавление {add_quantity:.4f} @ ${current_price:.4f}")

    side_str = 'BUY' if pos.side.upper() == 'LONG' else 'SELL'
    entry_order = await place_order_async(
        symbol=symbol, side=side_str, order_type='MARKET', quantity=add_quantity
    )

    if not (entry_order and entry_order.get('orderId')):
        logging.error(f"{log_prefix} Ошибка ордера на добавление.")
        return False

    new_avg_entry = (pos.entry_price * pos.current_quantity + current_price * add_quantity) / (pos.current_quantity + add_quantity)
    total_quantity = pos.current_quantity + add_quantity
    new_sl = new_avg_entry

    pos.current_quantity = total_quantity
    pos.entry_price = new_avg_entry
    pos.meta['pyramids_count'] = pyramids_count + 1

    await existing_manager._update_sl_callback(symbol, new_sl, f"Пирамидинг #{pyramids_count + 1}")

    await save_state_async()
    await send_telegram_message(
        f"<b>🔺 {symbol} ПИРАМИДИНГ #{pyramids_count + 1}</b>\n"
        f"  - Добавлено: {add_quantity:.4f} @ ${current_price:.4f}\n"
        f"  - Новая средняя: ${new_avg_entry:.4f}\n"
        f"  - SL передвинут в Б/У: ${new_sl:.4f}"
    )
    return True

def format_price(price: Any, symbol: str, side: str = 'BUY') -> Optional[str]:
    """
    Форматирует цену с учетом размера тика и направления ордера
    (округляет ВВЕРХ для стопов на шорт и ВНИЗ для стопов на лонг).
    """
    symbol_info = exchange_info_cache.get(symbol)
    if not symbol_info:
        # Фолбэк, если нет информации о символе
        return f"{Decimal(str(price)):.8f}"

    try:
        price_dec = Decimal(str(price))
        tick_size = Decimal(symbol_info['tickSize'])

        # --- ГЛАВНОЕ ИСПРАВЛЕНИЕ ЗДЕСЬ ---
        # Выбираем метод округления в зависимости от стороны
        # Для ордеров на ПОКУПКУ (включая стоп-лосс шорта и тейк-профит шорта) округляем ВВЕРХ
        # Для ордеров на ПРОДАЖУ (включая стоп-лосс лонга и тейк-профит лонга) округляем ВНИЗ
        rounding_method = ROUND_UP if side.upper() == 'BUY' else ROUND_DOWN
        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

        rounded_price = (price_dec / tick_size).quantize(Decimal('1'), rounding=rounding_method) * tick_size
        
        # Определяем количество знаков после запятой для форматирования строки
        price_precision = abs(tick_size.as_tuple().exponent)
        
        return f"{rounded_price:.{price_precision}f}"

    except (InvalidOperation, TypeError):
        logging.error(f"[{symbol}] Не удалось отформатировать неверное значение цены: {price}")
        return None

async def find_and_execute_trend_trade(symbol: str, expected_side: str):
    """
    Трендовая стратегия "Pullback to EMA" с быстрыми пре-фильтрами, финальным вердиктом от Gemini
    и сохранением типа стратегии.
    --- ВЕРСИЯ 1.1 (Патч для ключей TP) ---
    """
    # Используем таймфрейм из константы
    df = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS.get(TRENDING_SIGNAL_TF))
    # Проверяем достаточность данных
    if df is None or len(df) < max(TRENDING_SLOW_EMA + 1, 50): # +1 для EMA
        logging.debug(f"[{symbol}] Trend: Недостаточно данных.")
        return

    passport = None
    try:
        # --- Этап 1: Поиск первичного триггера (откат к быстрой EMA) ---
        fast_ema = ta.trend.ema_indicator(df["close"], window=TRENDING_FAST_EMA)
        slow_ema = ta.trend.ema_indicator(df["close"], window=TRENDING_SLOW_EMA)
        adx_val = float(ta.trend.adx(df["high"], df["low"], df["close"], window=14).iloc[-1])

        trade_type, trigger_reason = None, None

        # Проверяем тренд на ПРЕДПОСЛЕДНЕЙ свече, а откат на ПОСЛЕДНЕЙ
        is_uptrend = fast_ema.iloc[-2] > slow_ema.iloc[-2]
        is_downtrend = fast_ema.iloc[-2] < slow_ema.iloc[-2]

        last_low = df['low'].iloc[-1]
        last_high = df['high'].iloc[-1]
        last_close = df['close'].iloc[-1]
        ema_f_last = fast_ema.iloc[-1] # Быстрая EMA на последней свече

        # Условие для LONG: Ожидаем long, тренд был UP, последняя свеча коснулась/пробила быструю EMA снизу и закрылась выше нее
        if expected_side == 'long' and is_uptrend:
            if last_low <= ema_f_last and last_close >= ema_f_last:
                trade_type, trigger_reason = "long", f"Pullback to EMA{TRENDING_FAST_EMA}"
        # Условие для SHORT: Ожидаем short, тренд был DOWN, последняя свеча коснулась/пробила быструю EMA сверху и закрылась ниже нее
        elif expected_side == 'short' and is_downtrend:
            if last_high >= ema_f_last and last_close <= ema_f_last:
                trade_type, trigger_reason = "short", f"Pullback to EMA{TRENDING_FAST_EMA}"

        if not trade_type:
            # logging.debug(f"[{symbol}] Trend: Нет сигнала отката к EMA.")
            return

        # --- Этап 2: Быстрые локальные пре-фильтры ---
        # Проверяем силу тренда по ADX
        if adx_val < TRENDING_ADX_THRESHOLD:
            logging.info(f"~ [{symbol}] Trend сигнал пропущен: недостаточная сила тренда (ADX={adx_val:.1f} < {TRENDING_ADX_THRESHOLD})")
            return

        # --- Этап 3: Создание паспорта и глубокий анализ AI ---
        passport = TradePassport(symbol=symbol, strategy="Trending_AI_Filtered", trigger=trigger_reason, side=trade_type)
        logging.warning(f"🎯 [{symbol}] НАЙДЕН СИГНАЛ TREND: {passport.side.upper()}. Прошел пре-фильтры. Отправляю на анализ в Gemini...")

        async with GLOBAL_ENTRY_LOCK:
            # --- ФИЛЬТР ГЛОБАЛЬНОГО НАПРАВЛЕНИЯ ---
            ok, reason = await global_direction_filter(trade_type, symbol)
            if not ok:
                raise ValueError(reason)

            # --- ПРОВЕРКА ЛИМИТА ПОЗИЦИЙ ---
            dynamic_limit = get_dynamic_max_positions()
            if len(current_positions) >= dynamic_limit:
                raise ValueError(f"Достигнут лимит позиций ({len(current_positions)}/{dynamic_limit})")
            if symbol in current_positions:
                 raise ValueError("Позиция уже открыта") # Добавим проверку на уже открытую

            # --- Проверка реверса ---
            was_reversed = await handle_position_reversal(symbol, trade_type)
            if was_reversed:
                raise ValueError("Позиция была реверсирована, пропуск нового входа")


            # --- Сбор контекста для AI ---
            btc_state, _ = await get_btc_trend_state('1h')
            basis_data = await get_basis_and_funding(symbol)
            mtf_ok, mtf_score = await multi_timeframe_trend_confirmation(symbol, trade_type)

            context_text = (
                f"BTC Trend: {btc_state}. "
                f"Multi-TF Score: {mtf_score:.2f} ({'Confirmed' if mtf_ok else 'Not Confirmed'}). "
                f"Funding: {basis_data.get('funding_pct', 0):.4f}%. "
                f"Basis: {basis_data.get('basis_bps', 0):.1f} bps."
            )
            logging.info(f"[{symbol}] Контекст для AI: {context_text}")

            # --- Создание сетки 4-в-1 для Gemini ---
            df_sym_1h = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS['1h'])
            df_btc_15m = market_data_store.get('BTCUSDT', {}).get(ALLOWED_INTERVALS['15m'])
            df_btc_1h = market_data_store.get('BTCUSDT', {}).get(ALLOWED_INTERVALS['1h'])

            # Передаем df (15м) как df_sym_15m
            combined_chart_base64 = await create_multi_timeframe_chart_grid(
                symbol, df, df_sym_1h, df_btc_15m, df_btc_1h
            )
            if not combined_chart_base64: raise ValueError("Не удалось создать сетку графиков")

            # --- Промпт для AI ---
            prompt = (
                f"Вы — главный аналитик. Перед вами 4 графика для оценки сигнала '{trade_type.upper()}' по {symbol}.\n"
                f"Верхний ряд: 15-минутные графики ({symbol} слева, BTC справа).\n"
                f"Нижний ряд: часовые графики ({symbol} слева, BTC справа).\n\n"
                f"**Дополнительный контекст:**\n{context_text}\n\n"
                f"**ЗАДАЧА:** Проведите мульти-таймфрейм анализ. Соответствует ли краткосрочный сигнал (15м) старшему тренду (1ч)? "
                f"Поддерживает ли общая динамика BTC открытие этой сделки? Оцените все риски и вынесите вердикт.\n\n"
                f"Ответ строго в формате JSON: {{\"confidence\": 0-100, \"explanation\": \"...\"}}"
            )

            ai_verdict = await analyze_chart_with_gemini_vision(symbol, combined_chart_base64, trade_type, prompt)
            ai_confidence = ai_verdict.get("confidence", 0)

            entry_threshold = await get_adaptive_entry_threshold(symbol)
            if ai_confidence < entry_threshold:
                raise ValueError(f"Gemini отклонил сделку (уверенность {ai_confidence}% < порога {entry_threshold:.1f}%)")

            logging.warning(f"✅ [{symbol}] ВХОД Trend ОДОБРЕН AI ({ai_confidence}%). Финальные расчеты...")

            # --- Этап 4: Исполнение сделки ---
            entry_price = float(df["close"].iloc[-1]) # Цена закрытия последней свечи
            sltp_levels = await get_sltp_from_ai_async(symbol, trade_type, entry_price, context={}) # Получаем SL/TP от AI
            
            # --- ✅ ИСПРАВЛЕНИЕ: Проверяем "take_profit_1" ---
            if not (sltp_levels and sltp_levels.get("stop_loss") and sltp_levels.get("take_profit_1")):
                # Оставляем старое сообщение об ошибке для ясности, но причина - в ключе
                raise ValueError("AI не предоставил SL/TP (ошибка ключа 'take_profit_1')")

            final_sl_price = float(sltp_levels["stop_loss"])
            final_tp_price = float(sltp_levels["take_profit_1"]) # Используем TP1
            # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

            # Валидация геометрии
            final_sl_price, final_tp_price, norm_notes = normalize_sltp_for_side(
                 side="BUY" if trade_type == "long" else "SELL", entry=entry_price, sl=final_sl_price, tp=final_tp_price,
                 min_rr=MIN_RR_RATIO, price_step=get_symbol_price_step(symbol)
            )
            if norm_notes: logging.warning(f"[{symbol}] Уровни SL/TP нормализованы: {', '.join(norm_notes)}")

            # Расчет объема
            quantity = await calculate_smart_quantity_v2(symbol, entry_price, final_sl_price, ai_confidence, BASE_RISK_PERCENT, entry_threshold)
            if quantity <= 0: raise ValueError("Расчетный объем равен нулю")

            # Исполнение входа
            passport.status = "EXECUTED"
            passport.entry_price = entry_price
            passport.sl_price = final_sl_price
            passport.tp_price = final_tp_price # TP1
            passport.ai_confidence = ai_confidence
            passport.required_threshold = entry_threshold

            execution_result = await adaptive_execute_entry(
                symbol=symbol,
                side_str="BUY" if trade_type == "long" else "SELL",
                quantity=quantity,
                sl_price=final_sl_price,
                tp_price=final_tp_price, # Передаем TP1
                entry_context=f"AI Conf: {ai_confidence}%, Trigger: {trigger_reason}, ID: {passport.signal_id}",
                signal_price=entry_price
            )
            if not execution_result: raise ValueError("Исполнение сделки не удалось")

            # --- ✅ СОХРАНЕНИЕ СОСТОЯНИЯ С ТИПОМ СТРАТЕГИИ ---
            execution_result['meta']['strategy_type'] = 'trend' # Указываем тип стратегии
            current_positions[symbol] = PositionManager(execution_result)
            await save_state_async()
            # --- КОНЕЦ ИЗМЕНЕНИЯ ---

    # Обработка отказов
    except ValueError as e:
        if passport:
            passport.status = "REJECTED"; passport.reject_reason = str(e)
            # Логируем только реальные отказы, а не отсутствие сигнала
            logging.warning(f"🚫 [{symbol}] Сигнал Trend ОТКЛОНЕН. Причина: {passport.reject_reason}")
    # Обработка критических ошибок
    except Exception as e:
        if passport: passport.status = "REJECTED"; passport.reject_reason = f"КРИТИЧЕСКАЯ ОШИБКА: {e}"
        logging.error(f"[{symbol}] Критическая ошибка в find_and_execute_trend_trade: {e}", exc_info=True)
    # Логирование и уведомление
    finally:
        if passport and passport.status != "PENDING":
            await log_and_notify_async(passport, df)

async def should_close_dead_position_ai_verdict(symbol: str, side: str, holding_time_minutes: int, pnl_percent: float) -> Dict[str, Any]:
    """
    Запрашивает у AI, стоит ли закрывать "мертвую" позицию.
    """
    global DEEPSEEK_CHAT_COMPLETIONS_URL

    prompt = (
        f"Вы — риск-менеджер. Оцените ситуацию по 'зависшей' сделке и дайте рекомендацию.\n\n"
        f"**ДАННЫЕ ПО СДЕЛКЕ:**\n"
        f"- Инструмент: {symbol}\n"
        f"- Направление: {side.upper()}\n"
        f"- Время в позиции: {holding_time_minutes:.0f} минут\n"
        f"- Текущий PnL: {pnl_percent:.2f}%\n\n"
        f"**СИТУАЦИЯ:**\n"
        f"Позиция открыта уже долгое время, но цена практически не изменилась. Рынок находится в боковом движении. "
        f"Капитал 'заморожен' в этой сделке.\n\n"
        f"**ЗАДАЧА:**\n"
        f"Проанализируй, стоит ли принудительно закрыть эту позицию сейчас, чтобы освободить капитал для более перспективных сделок, "
        f"или же есть вероятность скорого движения цены в нужную сторону и стоит подождать?\n\n"
        f"**Формат ответа (строго JSON):** {{\"should_close\": <true_or_false>, \"reason\": \"<краткое_обоснование_решения>\"}}"
    )
    
    default_response = {"should_close": False, "reason": "Ошибка AI, решено не закрывать позицию."}
    
    try:
        loop = asyncio.get_running_loop()
        response_data = await loop.run_in_executor(None, partial(sync_deepseek_request, prompt, DEEPSEEK_CHAT_COMPLETIONS_URL, True))
        
        if not response_data or 'choices' not in response_data:
            return default_response
            
        content = response_data['choices'][0]['message']['content']
        ai_data = json.loads(content)
        
        if 'should_close' not in ai_data or 'reason' not in ai_data:
            raise ValueError("Неверный формат ответа ИИ")
            
        logging.info(f"[{symbol}] Вердикт AI по 'мертвой' позиции: Закрывать? {ai_data['should_close']}. Причина: {ai_data['reason']}")
        return ai_data
        
    except Exception as e:
        logging.error(f"[{symbol}] Ошибка при получении вердикта по 'мертвой' позиции: {e}")
        return default_response


# ============================================================================
# НАЧАЛО БЛОКА: АНАЛИЗ SL/TP ЧЕРЕЗ GEMINI И СТАКАН ЗАЯВОК (V-GEMINI-OB)
# ============================================================================

async def get_orderbook_snapshot(symbol: str, depth: int = 50) -> Dict[str, Any]:
    """
    Получает срез стакана заявок с Binance Futures.
    
    Args:
        symbol: Торговая пара (например, 'BTCUSDT')
        depth: Глубина стакана - ТОЛЬКО [5, 10, 20, 50, 100, 500, 1000]
    
    Returns:
        Dict с bids, asks и метриками
    """
    try:
        # Валидация depth - только допустимые значения для Binance Futures
        valid_depths = [5, 10, 20, 50, 100, 500, 1000]
        if depth not in valid_depths:
            # Округляем до ближайшего допустимого значения
            depth = min(valid_depths, key=lambda x: abs(x - depth))
            logging.warning(f"{symbol} | Depth скорректирован до {depth} (допустимые: {valid_depths})")
        
        # Получаем стакан через Binance API
        orderbook = await client.futures_order_book(symbol=symbol, limit=depth)
        
        bids = orderbook['bids'][:depth]  # [[price, qty], ...]
        asks = orderbook['asks'][:depth]
        
        # Расчет базовых метрик
        bid_volume = sum(float(b[1]) for b in bids)
        ask_volume = sum(float(a[1]) for a in asks)
        
        # Защита от деления на ноль
        total_volume = bid_volume + ask_volume
        imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
        
        best_bid = float(bids[0][0]) if bids else 0
        best_ask = float(asks[0][0]) if asks else 0
        mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
        spread_pct = ((best_ask - best_bid) / mid_price * 100) if mid_price > 0 else 0
        
        return {
            'bids': bids,
            'asks': asks,
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'imbalance': imbalance,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'mid_price': mid_price,
            'spread_pct': spread_pct,
            'timestamp': orderbook.get('lastUpdateId', 0)
        }
        
    except Exception as e:
        logging.error(f"{symbol} | get_orderbook_snapshot error: {e}", exc_info=True)
        return None


def detect_liquidity_walls(orderbook_data: Dict[str, Any], threshold: float = 2.5) -> Dict[str, list]:
    """
    Обнаруживает крупные стенки ликвидности (walls) в стакане.
    
    Args:
        orderbook_data: Данные стакана из get_orderbook_snapshot
        threshold: Множитель среднего объема для определения "стенки" (по умолчанию 2.5x)
    
    Returns:
        Dict с bid_walls и ask_walls
    """
    try:
        bids = orderbook_data['bids']
        asks = orderbook_data['asks']
        
        if not bids or not asks:
            return {'bid_walls': [], 'ask_walls': []}
        
        # Средние объемы
        avg_bid_vol = sum(float(b[1]) for b in bids) / len(bids)
        avg_ask_vol = sum(float(a[1]) for a in asks) / len(asks)
        
        # Находим стенки (объем > threshold * средний)
        bid_walls = [
            {
                'price': float(b[0]),
                'volume': float(b[1]),
                'distance_pct': (orderbook_data['mid_price'] - float(b[0])) / orderbook_data['mid_price'] * 100,
                'type': 'BID_WALL'
            }
            for b in bids if float(b[1]) > avg_bid_vol * threshold
        ]
        
        ask_walls = [
            {
                'price': float(a[0]),
                'volume': float(a[1]),
                'distance_pct': (float(a[0]) - orderbook_data['mid_price']) / orderbook_data['mid_price'] * 100,
                'type': 'ASK_WALL'
            }
            for a in asks if float(a[1]) > avg_ask_vol * threshold
        ]
        
        return {
            'bid_walls': sorted(bid_walls, key=lambda x: x['distance_pct']),
            'ask_walls': sorted(ask_walls, key=lambda x: x['distance_pct'])
        }
            
    except Exception as e:
        logging.error(f"detect_liquidity_walls error: {e}")
        return {'bid_walls': [], 'ask_walls': []}

def format_orderbook_for_prompt(orderbook_data: Dict[str, Any], levels: int = 10) -> str:
    """Форматирует стакан в читаемый текст для промпта"""
    try:
        lines = []
        
        # BID сторона
        lines.append("**BID (покупатели):**")
        for i, (price, qty) in enumerate(orderbook_data['bids'][:levels]):
            lines.append(f"  {i+1}. {float(price):.5f} | {float(qty):.4f}")
        
        lines.append("")
        
        # ASK сторона
        lines.append("**ASK (продавцы):**")
        for i, (price, qty) in enumerate(orderbook_data['asks'][:levels]):
            lines.append(f"  {i+1}. {float(price):.5f} | {float(qty):.4f}")
        
        return "\n".join(lines)
            
    except Exception as e:
        logging.error(f"format_orderbook_for_prompt error: {e}")
        return ""

def format_walls_for_prompt(walls: Dict[str, list]) -> str:
    """Форматирует обнаруженные стенки для промпта"""
    try:
        lines = []
        
        if walls['bid_walls']:
            lines.append("**Крупные BID стенки (поддержка):**")
            for w in walls['bid_walls'][:5]:  # Топ 5
                lines.append(f"  • {w['price']:.5f} | Vol: {w['volume']:.4f} | Дистанция: {w['distance_pct']:.2f}%")
        else:
            lines.append("**Крупные BID стенки:** не обнаружены")
        
        lines.append("")
        
        if walls['ask_walls']:
            lines.append("**Крупные ASK стенки (сопротивление):**")
            for w in walls['ask_walls'][:5]:
                lines.append(f"  • {w['price']:.5f} | Vol: {w['volume']:.4f} | Дистанция: {w['distance_pct']:.2f}%")
        else:
            lines.append("**Крупные ASK стенки:** не обнаружены")
        
        return "\n".join(lines)
            
    except Exception as e:
        logging.error(f"format_walls_for_prompt error: {e}")
        return ""

async def send_to_gemini_with_json_schema(prompt: str) -> Optional[Dict[str, Any]]:
    """
    Отправляет промпт в Gemini и получает структурированный JSON ответ.
    
    Args:
        prompt: Текст запроса
    
    Returns:
        Dict с stoploss, takeprofit1, takeprofit2, reasoning
    """
    # Используем существующий AI_REQUEST_SEMAPHORE
    global AI_REQUEST_SEMAPHORE, GEMINI_MODEL
    
    async with AI_REQUEST_SEMAPHORE:
        try:
            # Создаем модель с JSON schema
            model = genai.GenerativeModel(GEMINI_MODEL) # Используем существующую GEMINI_MODEL
            
            # Явно определяем JSON schema для ответа
            generation_config = genai.GenerationConfig(
                temperature=0.5, # Установлено значение по умолчанию
                response_mime_type="application/json",
                response_schema={
                    "type": "object",
                    "properties": {
                        "stoploss": {
                            "type": "number",
                            "description": "Цена Stop Loss"
                        },
                        "takeprofit1": {
                            "type": "number",
                            "description": "Цена первого Take Profit (TP1)"
                        },
                        "takeprofit2": {
                            "type": "number",
                            "description": "Цена второго Take Profit (TP2)"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Краткое объяснение выбора уровней"
                        }
                    },
                    "required": ["stoploss", "takeprofit1", "takeprofit2", "reasoning"]
                }
            )
            
            # Отправляем запрос
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
            )
            
            # Парсим JSON
            result = json.loads(response.text)
            
            return result
                    
        except json.JSONDecodeError as e:
            logging.error(f"Gemini JSON parse error: {e}. Response: {response.text[:500]}")
            return None
                    
        except Exception as e:
            logging.error(f"send_to_gemini_with_json_schema error: {e}", exc_info=True)
            return None

async def get_sltp_fallback_atr(
    symbol: str,
    side: str,
    entryprice: float) -> Dict[str, Optional[float]]:
    """
    Резервная функция расчета SL/TP через ATR.
    Используется если Gemini не доступен или вернул ошибку.
    """
    try:
        # Исправлена опечатка marketdatastore -> market_data_store
        df_short = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS.get("5m"))
        
        if df_short is None or len(df_short) < 20:
            logging.error(f"{symbol} | Нет данных для fallback ATR расчета.")
            return {'stop_loss': None, 'take_profit_1': None, 'take_profit_2': None}
        
        # Используем существующую функцию _atr
        atr_val = _atr(df_short, 14)
        
        if atr_val <= 0:
            raise ValueError("ATR = 0")
        
        # Расчет через ATR (используем существующие глобальные константы)
        if side.upper() == "LONG":
            sl_atr = entryprice - (atr_val * DEFAULT_ATR_MULTIPLIER)
            tp1_atr = entryprice + (atr_val * DEFAULT_ATR_MULTIPLIER * PARTIAL_TP_RR)
            tp2_atr = entryprice + (atr_val * DEFAULT_ATR_MULTIPLIER * (PARTIAL_TP_RR * 2))
        else:  # SHORT
            sl_atr = entryprice + (atr_val * DEFAULT_ATR_MULTIPLIER)
            tp1_atr = entryprice - (atr_val * DEFAULT_ATR_MULTIPLIER * PARTIAL_TP_RR)
            tp2_atr = entryprice - (atr_val * DEFAULT_ATR_MULTIPLIER * (PARTIAL_TP_RR * 2))
        
        logging.warning(f"{symbol} | Fallback ATR: SL={sl_atr:.5f}, TP1={tp1_atr:.5f}, TP2={tp2_atr:.5f}")
        
        return {
            'stop_loss': sl_atr,
            'take_profit_1': tp1_atr,
            'take_profit_2': tp2_atr
        }
            
    except Exception as e:
        logging.error(f"{symbol} | get_sltp_fallback_atr error: {e}")
        return {'stop_loss': None, 'take_profit_1': None, 'take_profit_2': None}

# ============================================================================
# 6. ГЛАВНАЯ ФУНКЦИЯ - ЗАМЕНА get_sltp_from_ai_async
# ============================================================================
async def get_sltp_from_ai_async(
    symbol: str,
    side: str,
    entry_price: float,
    context: dict = {}) -> Dict[str, Optional[float]]:
    """
    НОВАЯ ФУНКЦИЯ: Анализирует стакан через Gemini и возвращает SL/TP.
    (Заменяет старую get_sltp_from_ai_async с DeepSeek).
    
    Args:
        symbol: Торговая пара
        side: 'LONG' или 'SHORT'
        entry_price: Цена входа
        context: Дополнительный контекст (HMM режим, ADX и т.д.)
    
    Returns:
        Dict {'stop_loss': float, 'take_profit_1': float, 'take_profit_2': float}
    """
    try:
        # 1. Получаем стакан
        logging.info(f"{symbol} | Получаем стакан для анализа через Gemini...")
        orderbook_data = await get_orderbook_snapshot(symbol, depth=50)
        
        if not orderbook_data:
            logging.error(f"{symbol} | Не удалось получить стакан. Используем fallback ATR.")
            return await get_sltp_fallback_atr(symbol, side, entry_price)
        
        # 2. Обнаруживаем крупные стенки
        walls = detect_liquidity_walls(orderbook_data, threshold=2.5)
        
        # 3. Получаем дополнительные данные для контекста
        # Исправлена опечатка marketdatastore -> market_data_store
        df_short = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS.get("5m"))
        
        if df_short is None or len(df_short) < 20:
            logging.warning(f"{symbol} | Недостаточно данных для контекста.")
            context_indicators = "Нет данных по индикаторам"
        else:
            # Рассчитываем индикаторы для контекста
            atr_val = _atr(df_short, 14) # Используем _atr
            adx_val = adx(df_short['high'], df_short['low'], df_short['close'], 14).iloc[-1]
            rsi_val = rsi(df_short['close'], 14).iloc[-1]
            
            context_indicators = (
                f"ATR(14): {atr_val:.5f}, "
                f"ADX(14): {adx_val:.1f}, "
                f"RSI(14): {rsi_val:.1f}"
            )
        
        # 4. Формируем промпт для Gemini
        prompt = f"""Ты - профессиональный трейдер криптовалют. Проанализируй стакан заявок и определи оптимальные уровни Stop Loss и Take Profit.
**ДАННЫЕ ПОЗИЦИИ:**
- Symbol: {symbol}
- Side: {side.upper()}
- Entry Price: {entry_price:.5f}
- Context: {context}
- Indicators: {context_indicators}
**СТАКАН ЗАЯВОК (TOP 10 LEVELS):**
{format_orderbook_for_prompt(orderbook_data, levels=10)}
**МЕТРИКИ СТАКАНА:**
- Bid/Ask Imbalance: {orderbook_data['imbalance']:.2%}
- Spread: {orderbook_data['spread_pct']:.3%}
- Total Bid Volume (30 levels): {orderbook_data['bid_volume']:.2f}
- Total Ask Volume (30 levels): {orderbook_data['ask_volume']:.2f}
- Mid Price: {orderbook_data['mid_price']:.5f}
**ОБНАРУЖЕННЫЕ СТЕНКИ ЛИКВИДНОСТИ:**
{format_walls_for_prompt(walls)}
**ТВОЯ ЗАДАЧА:**
Основываясь на анализе стакана, определи безопасные уровни SL/TP которые учитывают:
1. **Stop Loss**:
   - Для LONG: ставь SL ЗА ближайшей крупной BID стенкой (поддержкой), чтобы защититься от охоты за стопами
   - Для SHORT: ставь SL ЗА ближайшей крупной ASK стенкой (сопротивлением)
   - Минимальное расстояние от входа: 0.8-1.5%
   - Учитывай spread и проскальзывание
2. **Take Profit 1 (TP1)**:
   - Первый уровень фиксации ПЕРЕД первой крупной противоположной стенкой
   - Risk/Reward минимум 1.5:1
   - Расстояние примерно 1.2-2% от входа
3. **Take Profit 2 (TP2)**:
   - Второй уровень ЗА второй стенкой, если imbalance положительный для направления
   - Risk/Reward минимум 2.5-3:1
   - Расстояние примерно 2.5-4% от входа
4. **Дополнительные факторы**:
   - Если spread > 0.15%, увеличь расстояние SL на 10-15%
   - Если imbalance сильно против позиции (>30%), сделай SL ближе, а TP агрессивнее
   - Если стенок мало или они далеко, используй ATR-based расчет как ориентир
**ВАЖНО**: Все цены должны быть числами (float), без markdown, без дополнительного текста.
Верни ТОЛЬКО валидный JSON в формате:
{{
  "stoploss": <число>,
  "takeprofit1": <число>,
  "takeprofit2": <число>,
  "reasoning": "<краткое объяснение в 1-2 предложения>"
}}"""
        
        # 5. Отправляем в Gemini
        logging.info(f"{symbol} | Отправляем стакан в Gemini для анализа...")
        gemini_response = await send_to_gemini_with_json_schema(prompt)
        
        if not gemini_response:
            logging.error(f"{symbol} | Gemini не вернул валидный ответ. Используем fallback.")
            return await get_sltp_fallback_atr(symbol, side, entry_price)
        
        # 6. Валидация ответа
        sl = float(gemini_response['stoploss'])
        tp1 = float(gemini_response['takeprofit1'])
        tp2 = float(gemini_response['takeprofit2'])
        reasoning = gemini_response.get('reasoning', 'N/A')
        
        # Проверка корректности SL
        if side.upper() == "LONG":
            if sl >= entry_price:
                raise ValueError(f"SL для LONG должен быть < entry. SL={sl}, Entry={entry_price}")
        else:  # SHORT
            if sl <= entry_price:
                raise ValueError(f"SL для SHORT должен быть > entry. SL={sl}, Entry={entry_price}")
        
        # Проверка минимального R:R
        risk_dist = abs(entry_price - sl)
        reward_dist_tp1 = abs(tp1 - entry_price)
        rr = reward_dist_tp1 / risk_dist if risk_dist > 0 else 0
        
        if rr < 1.2:
            logging.warning(f"{symbol} | R:R слишком низкий ({rr:.2f}). Корректируем TP1...")
            # Корректируем TP1 для R:R минимум 1.5
            if side.upper() == "LONG":
                tp1 = entry_price + (risk_dist * 1.5)
            else:
                tp1 = entry_price - (risk_dist * 1.5)
        
        logging.warning(f"{symbol} | ✅ Gemini SL/TP установлены: "
                       f"SL={sl:.5f}, TP1={tp1:.5f}, TP2={tp2:.5f}")
        logging.info(f"{symbol} | Reasoning: {reasoning}")
        
        return {
            'stop_loss': sl,
            'take_profit_1': tp1,
            'take_profit_2': tp2,
            'reasoning': reasoning
        }
            
    except Exception as e:
        logging.error(f"{symbol} | get_sltp_from_ai_async (Gemini OB) error: {e}", exc_info=True)
        logging.warning(f"{symbol} | Используем fallback ATR-based расчет...")
        return await get_sltp_fallback_atr(symbol, side, entry_price)

# ============================================================================
# 10. ТЕСТИРОВАНИЕ
# ============================================================================
# ============================================================================
# 10. ТЕСТИРОВАНИЕ (ИСПРАВЛЕННАЯ ВЕРСИЯ V1.1)
# ============================================================================
async def test_gemini_orderbook_analysis():
    """Функция для тестирования нового функционала"""
    
    test_symbol = "BTCUSDT"
    test_side = "LONG"
    
    print(f"\n{'='*60}")
    print(f"ТЕСТИРОВАНИЕ GEMINI ORDERBOOK ANALYSIS")
    print(f"{'='*60}\n")
    
    print(f"Symbol: {test_symbol}")
    print(f"Side: {test_side}")
    
    # Тест 1: Получение стакана
    print("1. Получение стакана...")
    # Используем исправленный depth=20
    orderbook = await get_orderbook_snapshot(test_symbol, depth=20)
    if orderbook:
        print(f"   ✅ Успешно. Imbalance: {orderbook['imbalance']:.2%}, Spread: {orderbook['spread_pct']:.3%}")
    else:
        print("   ❌ Ошибка получения стакана")
        return
        
    # --- ✅ ИСПРАВЛЕНИЕ: УСТАНОВКА РЕАЛИСТИЧНОЙ ЦЕНЫ ВХОДА ---
    # Используем mid_price из стакана как нашу "тестовую" цену входа
    test_entry = orderbook.get('mid_price', 0.0)
    if test_entry == 0.0:
        print("   ❌ Ошибка: не удалось получить mid_price из стакана.")
        return
    print(f"Entry Price (Live): {test_entry}\n") # <--- Печатаем новую цену
    # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

    # Тест 2: Обнаружение стенок
    print("\n2. Обнаружение стенок...")
    walls = detect_liquidity_walls(orderbook, threshold=2.5)
    print(f"   BID walls: {len(walls['bid_walls'])}")
    print(f"   ASK walls: {len(walls['ask_walls'])}")
    if walls['bid_walls']:
        print(f"   Ближайшая BID стенка: {walls['bid_walls'][0]['price']:.5f} "
              f"({walls['bid_walls'][0]['distance_pct']:.2f}% от mid)")
    if walls['ask_walls']:
        print(f"   Ближайшая ASK стенка: {walls['ask_walls'][0]['price']:.5f} "
              f"({walls['ask_walls'][0]['distance_pct']:.2f}% от mid)")
    
    # Тест 3: Gemini анализ (теперь с правильной test_entry)
    print("\n3. Отправка в Gemini для анализа...")
    # Используем имя главной функции get_sltp_from_ai_async
    sltp = await get_sltp_from_ai_async(test_symbol, test_side, test_entry, context={"test": "Test context"})
    
    if sltp and sltp.get('stop_loss'):
        sl = sltp['stop_loss']
        tp1 = sltp['take_profit_1']
        tp2 = sltp['take_profit_2']
        
        risk_dist = abs(test_entry - sl)
        reward_dist = abs(tp1 - test_entry)
        rr = reward_dist / risk_dist if risk_dist > 0 else 0
        
        print(f"\n   ✅ РЕЗУЛЬТАТ:")
        print(f"   Stop Loss:     {sl:.5f} ({((sl - test_entry) / test_entry * 100):.2f}%)")
        print(f"   Take Profit 1: {tp1:.5f} ({((tp1 - test_entry) / test_entry * 100):.2f}%)")
        print(f"   Take Profit 2: {tp2:.5f} ({((tp2 - test_entry) / test_entry * 100):.2f}%)")
        print(f"   Risk/Reward:   {rr:.2f}:1")
        print(f"\n   Reasoning: {sltp.get('reasoning', 'N/A')}")
    else:
        print("   ❌ Gemini не вернул валидный результат (или сработал fallback, проверь логи выше)")
    
    print(f"\n{'='*60}\n")
    
# ============================================================================
# КОНЕЦ БЛОКА V-GEMINI-OB
# ============================================================================

async def start_websockets(symbols_to_stream_list: List[str]):
    """
    Более надежный обработчик WebSocket с использованием aiohttp.
    """
    global websocket_connected, message_queue
    
    # Формируем список стримов (без изменений)
    tfs_to_subscribe = {'1m', '5m', '15m', '1h'} # Оставляем только нужные для стратегий
    streams = {f"{s.lower()}@kline_{tf}" for s in symbols_to_stream_list for tf in tfs_to_subscribe}
    # Убедимся, что BTC на месте
    streams.add("btcusdt@kline_15m")
    streams.add("btcusdt@kline_1h")
    
    url = f"wss://fstream.binance.com/stream?streams={'/'.join(streams)}"
    logging.warning(f"WS: Подписка на {len(streams)} стримов...")

    while True:
        websocket_connected = False
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(url, timeout=30) as ws:
                    logging.warning("✅ WebSocket (aiohttp) успешно подключен.")
                    websocket_connected = True
                    
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                data = json.loads(msg.data)
                                if 'data' in data:
                                    # Отправляем сообщение в очередь для обработки воркерами
                                    message_queue.put_nowait(data['data'])
                            except json.JSONDecodeError:
                                logging.warning(f"WS: Не удалось декодировать JSON: {msg.data}")
                            except asyncio.QueueFull:
                                logging.warning("!!! Основная очередь обработки переполнена! Пропускаем сообщение.")
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logging.error(f"WS Ошибка (aiohttp): {ws.exception()}")
                            break
                        
        except Exception as e:
            logging.error(f"WS Критическая ошибка (aiohttp): {e}. Переподключение через {API_RETRY_DELAY_SECONDS}с...")
            await asyncio.sleep(API_RETRY_DELAY_SECONDS)

# --- КОНЕЦ ВОСЬМОЙ ЧАСТИ ---
# --- НАЧАЛО ДЕВЯТОЙ ЧАСТИ (ПОЛНАЯ ФИНАЛЬНАЯ ВЕРСИЯ) ---

def load_state_sync():
    """Синхронно загружает состояние позиций из файла при старте."""
    global current_positions
    if os.path.exists("bot_state.json"):
        try:
            with open("bot_state.json", "r", encoding='utf-8') as f:
                content = f.read()
                if not content:
                    logging.warning("Файл состояния bot_state.json пуст.")
                    return
                data = json.loads(content)
                current_positions = data
            logging.warning("!!! Состояние бота успешно загружено из файла. Позиций в памяти: %d", len(current_positions))
        except json.JSONDecodeError:
            logging.error("Ошибка декодирования JSON в файле bot_state.json. Файл может быть поврежден.")
        except Exception as e:
            logging.error(f"Неизвестная ошибка загрузки состояния: {e}")

async def log_trade_to_csv(trade_data: Dict[str, Any]):
    """Записывает информацию о закрытой сделке в CSV-файл."""
    global trade_csv_log_lock
    
    file_path = "trade_history.csv"
    headers = [
        "timestamp_utc", "symbol", "side", "pnl", "entry_price", "exit_price", 
        "amount", "market_env_entry", "ai_confidence", "ai_rr_score", "strategy_group"
    ]
    
    ai_verdict_data = trade_data.get('ai_verdict') or {}
    log_entry = {
        "timestamp_utc": datetime.datetime.now(dt_timezone.utc).isoformat(),
        "symbol": trade_data.get('symbol'),
        "side": trade_data.get('side'),
        "pnl": trade_data.get('pnl'),
        "entry_price": trade_data.get('entry_price'),
        "exit_price": trade_data.get('exit_price'),
        "amount": trade_data.get('initial_amount'), 
        "market_env_entry": trade_data.get('entry_context'),
        "ai_confidence": ai_verdict_data.get('confidence'),
        "ai_rr_score": ai_verdict_data.get('rr_score'),
        "strategy_group": trade_data.get('strategy_group', 'A')
    }

    async with trade_csv_log_lock:
        try:
            file_exists = os.path.exists(file_path)
            if aiofiles:
                async with aiofiles.open(file_path, mode='a', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    if not file_exists:
                        await f.write(','.join(headers) + '\n')
                    row_values = [str(log_entry.get(h, '')) for h in headers]
                    await f.write(','.join(row_values) + '\n')
            else:
                with open(file_path, mode='a', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(log_entry)
        except Exception as e:
            logging.error(f"Не удалось записать лог сделки в {file_path}: {e}")

async def should_close_dead_position_with_vision(symbol: str, side: str, holding_time_minutes: int, pnl_percent: float, entry_price: float) -> Dict[str, Any]:
    """
    Запрашивает у Gemini Vision, стоит ли закрывать "мертвую" позицию, анализируя график.
    """
    default_response = {"should_close": False, "reason": "Ошибка AI, решено не закрывать позицию."}

    try:
        # 1. Получаем данные и создаем график
        df_chart = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS.get("15m"))
        if df_chart is None or len(df_chart) < 50:
            logging.warning(f"[{symbol}] Недостаточно данных для создания графика для 'мертвой' позиции.")
            return default_response

        fig = await create_enhanced_chart(symbol, df_chart, side, entry_price)
        if fig is None:
            return default_response

        chart_base64 = await save_chart_to_base64(symbol, fig)
        plt.close(fig)

        if not chart_base64:
            return default_response

        # 2. Формируем специальный промпт для Gemini
        prompt = (
            f"Вы — риск-менеджер. Оцените ситуацию по 'зависшей' сделке и дайте рекомендацию.\n\n"
            f"**СИТУАЦИЯ:**\n"
            f"- Инструмент: {symbol}\n"
            f"- Направление: {side.upper()}\n"
            f"- Время в позиции: {holding_time_minutes:.0f} минут\n"
            f"- Текущий PnL: ~{pnl_percent:.2f}%\n"
            f"Позиция открыта уже долгое время, но цена практически не движется (флэт).\n\n"
            f"**ЗАДАЧА:**\n"
            f"Проанализируй приложенный график. Есть ли на нем признаки скорого импульсного движения (например, сильное сужение полос Боллинджера, формирование паттерна 'флаг'/'вымпел')? "
            f"Или же график подтверждает 'болото' и позицию лучше закрыть, чтобы освободить капитал?\n\n"
            f"**Формат ответа (строго JSON):** {{\"should_close\": <true_or_false>, \"reason\": \"<краткое_обоснование_решения>\"}}"
        )

        # 3. Вызываем Gemini Vision (похоже на analyze_chart_with_gemini_vision)
        if not GEMINI_API_KEY:
            logging.error("[Gemini] API-ключ не найден. Анализ 'мертвой' позиции невозможен.")
            return default_response
            
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        image_part = {"mime_type": "image/png", "data": base64.b64decode(chart_base64)}
        
        async with AI_REQUEST_SEMAPHORE:
            response = await model.generate_content_async(
                [prompt, image_part],
                generation_config=types.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.5
                )
            )

        ai_data = json.loads(response.text)
        
        if 'should_close' not in ai_data or 'reason' not in ai_data:
            raise ValueError("Неверный формат ответа ИИ")

        logging.info(f"[{symbol}] Вердикт Gemini Vision по 'мертвой' позиции: Закрывать? {ai_data['should_close']}. Причина: {ai_data['reason']}")
        return ai_data

    except Exception as e:
        logging.error(f"[{symbol}] Ошибка при получении вердикта по 'мертвой' позиции: {e}")
        return default_response

async def cleanup_dangling_orders():
    """Периодически проверяет и отменяет "зависшие" SL/TP ордера."""
    if not client: return
    log_prefix = "[Order-Cleanup]"
    logging.info(f"{log_prefix} Запуск проверки зависших ордеров...")
    try:
        open_orders = await client.futures_get_open_orders()
        symbols_with_open_positions = set(current_positions.keys())
        
        for order in open_orders:
            order_symbol = order['symbol']
            order_type = order['type']
            
            if order_type in ["STOP_MARKET", "TAKE_PROFIT_MARKET", "TRAILING_STOP_MARKET"]:
                if order_symbol not in symbols_with_open_positions:
                    logging.warning(f"{log_prefix} Найден ордер-сирота! Символ: {order_symbol}, ID: {order['orderId']}. Отменяю...")
                    await cancel_single_order_async(order_symbol, order['orderId'])

    except Exception as e:
        logging.error(f"{log_prefix} Критическая ошибка во время очистки ордеров: {e}", exc_info=True)

def get_session_label(now_utc: datetime.datetime | None = None) -> str:
    """Определяет текущую торговую сессию по времени UTC."""
    now = now_utc or datetime.datetime.utcnow()
    h = now.hour
    # Упрощенная сессионная модель для крипторынка (UTC):
    # Азия (ночь) 0–6, Европа 7–12, США 13–20, Тихий океан 21–23
    if 0 <= h <= 6: return "ASIA_NIGHT"
    if 7 <= h <= 12: return "EUROPE"
    if 13 <= h <= 20: return "US"
    return "QUIET"

def is_quiet_session() -> bool:
    """Возвращает True, если сейчас "тихая" торговая сессия."""
    return get_session_label() in ("ASIA_NIGHT", "QUIET")

async def create_multi_chart_for_gemini(symbol: str, df_symbol: pd.DataFrame, df_btc: pd.DataFrame) -> Optional[str]:
    """
    Создаёт единое изображение с двумя графиками (основной символ и BTC) для анализа в Gemini Vision.
    Возвращает изображение в формате base64.
    """
    if df_symbol is None or df_btc is None or len(df_symbol) < 80 or len(df_btc) < 80:
        logging.warning(f"[{symbol}] Недостаточно данных для создания двойного графика.")
        return None

    try:
        # 1. Создаем фигуру с двумя областями для графиков (1 ряд, 2 колонки)
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Настраиваем стиль для mplfinance
        mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350', inherit=True)
        style = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc, gridstyle=':')

        # --- 2. Рисуем первый график (основной символ) ---
        df_plot_symbol = df_symbol.tail(80).copy()
        df_plot_symbol['EMA20'] = ta.trend.ema_indicator(df_plot_symbol['close'], window=20)
        df_plot_symbol['EMA50'] = ta.trend.ema_indicator(df_plot_symbol['close'], window=50)
        apds_symbol = [
            mpf.make_addplot(df_plot_symbol['EMA20'], ax=axes[0], color='cyan', width=1.0),
            mpf.make_addplot(df_plot_symbol['EMA50'], ax=axes[0], color='orange', width=1.0),
        ]
        mpf.plot(df_plot_symbol, ax=axes[0], type='candle', style=style, addplot=apds_symbol, axtitle=f'{symbol} / USDT')

        # --- 3. Рисуем второй график (BTC) ---
        df_plot_btc = df_btc.tail(80).copy()
        df_plot_btc['EMA20'] = ta.trend.ema_indicator(df_plot_btc['close'], window=20)
        df_plot_btc['EMA50'] = ta.trend.ema_indicator(df_plot_btc['close'], window=50)
        apds_btc = [
            mpf.make_addplot(df_plot_btc['EMA20'], ax=axes[1], color='cyan', width=1.0),
            mpf.make_addplot(df_plot_btc['EMA50'], ax=axes[1], color='orange', width=1.0),
        ]
        mpf.plot(df_plot_btc, ax=axes[1], type='candle', style=style, addplot=apds_btc, axtitle='BTC / USDT')

        # 4. Сохраняем результат в память
        buf = io.BytesIO()
        fig.tight_layout() # Оптимизируем расположение
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        
        # 5. Кодируем в base64 и закрываем фигуру, чтобы освободить память
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        logging.info(f"[{symbol}] Двойной график для Gemini успешно создан.")
        return image_base64

    except Exception as e:
        logging.error(f"[{symbol}] Ошибка при создании двойного графика: {e}", exc_info=True)
        # Убедимся, что фигура закрыта даже в случае ошибки
        if 'fig' in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)
        return None

async def periodic_tasks():
    """
    Исправленная версия v2.0.
    Эта функция больше НЕ ищет новые сделки. Она отвечает только за:
    1. Ежеминутное управление ОТКРЫТЫМИ позициями (трейлинг, DCA, "мертвые" сделки).
    2. Периодическую синхронизацию состояния (баланс, ордера, время).
    """
    global current_balance, current_positions, client, loop
    
    # Таймер для периодической синхронизации времени с сервером
    last_time_sync = 0
    
    await asyncio.sleep(20)
    logging.info("✅ Запуск цикла периодических задач (только УПРАВЛЕНИЕ и СИНХРОНИЗАЦИЯ)...")

    while True:
        try:
            current_time = time.time()
            
            # --- БЛОК 1: СИНХРОНИЗАЦИЯ И ОБСЛУЖИВАНИЕ (каждую минуту) ---
            
            # Синхронизируем время раз в час для предотвращения ошибки -1021
            if current_time - last_time_sync > 3600:
                await sync_server_time()
                last_time_sync = current_time
            
            # Обновляем баланс и кривую эквити
            await update_balance_async()
            update_equity_curve(current_balance)
            
            # Синхронизируем позиции с биржей
            positions_from_api = (await loop.run_in_executor(None, get_open_positions_sync) if loop else {}) or {}
            
            # Удаляем из памяти локально закрытые позиции
            closed_symbols = set(current_positions.keys()) - set(positions_from_api.keys())
            for symbol in closed_symbols:
                if symbol in current_positions: 
                    del current_positions[symbol]
                    logging.info(f"[{symbol}] Позиция закрыта на бирже, удалена из памяти.")
            
            # Восстанавливаем "бездомные" позиции, открытые вручную или при сбое
            for symbol, api_pos_data in positions_from_api.items():
                if symbol not in current_positions:
                    logging.warning(f"[{symbol}] Обнаружена 'бездомная' позиция. Воссоздаю менеджер...")
                    api_pos_data['symbol'] = symbol
                    current_positions[symbol] = PositionManager(api_pos_data)
            
            # Очищаем "зависшие" SL/TP ордера от уже закрытых позиций
            await cleanup_dangling_orders()

            # --- БЛОК 2: УПРАВЛЕНИЕ АКТИВНЫМИ ПОЗИЦИЯМИ ---
            active_symbols = list(current_positions.keys())
            for symbol in active_symbols:
                manager = current_positions.get(symbol)
                if not isinstance(manager, PositionManager) or manager.state.current_quantity <= 0:
                    continue

                df_5m = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS.get('5m'))
                if df_5m is not None and not df_5m.empty:
                    # Основная логика (трейлинг, near-TP, % PnL)
                    await manager.manage(df_work_tf=df_5m)
                    
                    # Логика добавления к позиции (DCA / Пирамидинг)
                    await check_and_execute_scaling_v2(symbol)

                # Проверка "зависших" позиций (например, дольше 4 часов без движения)
                pos_age_minutes = (time.time() - manager.state.entry_timestamp) / 60
                details = await manager.get_detailed_status()
                pnl_pct = details.get("pnl_percent", 0.0)

                if pos_age_minutes > 240 and abs(pnl_pct) < 5.0: # > 4 часов и PnL < 5%
                    verdict = await should_close_dead_position_with_vision(
                        symbol, manager.state.side, pos_age_minutes, pnl_pct, manager.state.entry_price
                    )
                    if verdict.get("should_close"):
                        await close_position_emergency(symbol, manager.get_state_dict(), f"Закрыто по AI: {verdict.get('reason')}")

            logging.info(f"✅ Периодическая проверка завершена. Баланс: {current_balance:.2f}, Позиций: {len(current_positions)}.")

        except asyncio.CancelledError:
            logging.info("Цикл периодических задач отменен."); break
        except Exception as e:
            logging.error(f"Критическая ошибка в periodic_tasks: {e}", exc_info=True)
            
        await asyncio.sleep(PERIODIC_CHECK_INTERVALS)  # Пауза 60 секунд

async def initialize_market_data(symbols_to_init: List[str]):
    global market_data_store, loop
    logging.info(f"Загрузка исторических данных (Лимит: {MAX_DATAFRAME_ROWS})...");
    
    symbols_for_data = set(symbols_to_init) | {'BTCUSDT', 'ETHUSDT'}
    # Загружаем все таймфреймы, которые могут понадобиться боту
    tfs_to_load = {'5m', '15m', '1h', '4h', '1d'}

    for symbol in symbols_for_data:
        for tf in tfs_to_load:
            if tf not in ALLOWED_INTERVALS: continue

            interval_api_value = ALLOWED_INTERVALS[tf]
            
            # --- НОВЫЙ БЛОК: ЛОГИКА ПОВТОРНЫХ ПОПЫТОК ---
            for attempt in range(3): # Делаем 3 попытки
                logging.info(f"Загрузка {symbol}/{tf}, попытка {attempt + 1}/3...")
                df = await loop.run_in_executor(
                    None, partial(get_ohlcv_sync, symbol, interval_api_value, MAX_DATAFRAME_ROWS)
                )
                
                if df is not None and not df.empty:
                    if symbol not in market_data_store: market_data_store[symbol] = {}
                    market_data_store[symbol][interval_api_value] = df
                    logging.info(f"✅ Успешно: Данные для {symbol} ({tf}) загружены ({len(df)} строк)")
                    break # Выходим из цикла попыток, если успешно
                else:
                    logging.warning(f"❌ Неудача: Не удалось загрузить данные для {symbol} ({tf}), попытка {attempt + 1}/3.")
                    if attempt < 2: # Если это не последняя попытка
                        await asyncio.sleep(5) # Ждем 5 секунд перед повтором
            else: # Этот блок выполнится, если все 3 попытки провалились
                logging.critical(f"!!! КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить данные для {symbol} ({tf}) после 3 попыток. Бот может работать некорректно.")
    
    logging.info("✅ Загрузка исторических данных завершена.")

async def emergency_close_all_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Аварийно закрывает ВСЕ открытые позиции по рынку."""
    if not is_authorized(update.effective_chat.id): return
    global current_positions

    if not current_positions:
        await update.message.reply_text("✅ Открытых позиций для закрытия нет.")
        return

    await update.message.reply_text(f"🚨 ИНИЦИИРОВАНО АВАРИЙНОЕ ЗАКРЫТИЕ {len(current_positions)} ПОЗИЦИЙ! 🚨")
    
    # Создаем копию, так как словарь будет изменяться во время итерации
    positions_to_close = list(current_positions.values())
    
    closing_tasks = []
    for manager in positions_to_close:
        pos_data = manager.get_state_dict() if isinstance(manager, PositionManager) else manager
        # Создаем асинхронную задачу на закрытие для каждой позиции
        task = asyncio.create_task(
            close_position_emergency(
                pos_data['symbol'], 
                pos_data, 
                reason="Ручная команда /emergency_close_all"
            )
        )
        closing_tasks.append(task)

    # Запускаем все задачи на закрытие параллельно
    results = await asyncio.gather(*closing_tasks, return_exceptions=True)
    
    successful_closes = sum(1 for r in results if r is True)
    failed_closes = len(results) - successful_closes
    
    final_message = f"🏁 Процесс аварийного закрытия завершен.\n"
    final_message += f"✅ Успешно закрыто: {successful_closes}\n"
    if failed_closes > 0:
        final_message += f"❌ Не удалось закрыть: {failed_closes}. Проверьте логи и терминал!"

    await update.message.reply_html(f"<b>{final_message}</b>")

async def send_telegram_message_safe(message: str):
    if telegram_bot and TELEGRAM_CHAT_ID:
        await send_telegram_message(message)

async def get_available_margin() -> float:
    """Запрашивает у Binance доступный баланс для новых позиций."""
    global client, COLLATERAL_ASSET
    try:
        # Используем futures_account(), так как он содержит поле availableBalance
        account_info = await client.futures_account()
        # Находим наш основной актив (например, USDT)
        for asset in account_info.get('assets', []):
            if asset.get('asset') == COLLATERAL_ASSET:
                # 'availableBalance' - это то, что можно использовать для открытия НОВЫХ позиций
                available_bal = float(asset.get('availableBalance', 0.0))
                logging.info(f"Доступный баланс для новых сделок: {available_bal:.2f} {COLLATERAL_ASSET}")
                return available_bal
        return 0.0
    except Exception as e:
        logging.error(f"Не удалось получить доступный баланс: {e}")
        return 0.0

async def update_balance_async():
    """Получает и обновляет глобальный баланс фьючерсного кошелька."""
    global current_balance
    try:
        # ✅ ИСПРАВЛЕНИЕ: Добавлен параметр recvWindow=10000 (10 секунд)
        account_info = await client.futures_account(recvWindow=10000)
        for asset in account_info['assets']:
            if asset['asset'] == 'USDT':
                current_balance = float(asset['walletBalance'])
                logging.warning(f"Баланс фьючерсного кошелька обновлен: {current_balance:.2f} USDT")
                return
    except BinanceAPIException as e:
        logging.error(f"Не удалось обновить баланс кошелька: {e}")
        if e.code == -1021:
            logging.warning("Обнаружена ошибка времени (-1021) при обновлении баланса.")
    except Exception as e:
        logging.error(f"Критическая ошибка в update_balance_async: {e}")


async def run_bot():
    global loop, client, telegram_app, telegram_bot, ws_task_handle, periodic_task, current_balance, NUM_WORKERS
    # <<< ИСПРАВЛЕНИЕ ЗДЕСЬ >>>
    # Явно указываем, что мы работаем с глобальной переменной current_positions
    global current_positions
    
    worker_tasks = []
    stop_event = asyncio.Event()
    def signal_handler(*args):
        logging.warning("Получен сигнал ОС. Инициирую грациозную остановку...")
        if not stop_event.is_set(): stop_event.set()
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        loop = asyncio.get_running_loop()
        await initialize_clients()
        recalculate_max_rows()

        # 1. Сначала загружаем состояние, чтобы знать об открытых позициях.
        load_state_sync()

        # 2. Затем запрашиваем актуальный баланс с биржи.
        await update_balance_async()

        if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
            telegram_app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
            telegram_bot = telegram_app.bot
            handlers = [
                CommandHandler("start", start_command), CommandHandler("help", help_command),
                CommandHandler("balance", balance_command), CommandHandler("status", status_command),
                CommandHandler("get_settings", get_settings_command), CommandHandler("positions", positions_command),
                CommandHandler("dailystats", daily_stats_command), CommandHandler("ai_status", ai_status_command),
                CommandHandler("debug_positions", debug_positions_command),
                CommandHandler("set_leverage", set_leverage_command),
                CommandHandler("set_risk", set_risk_percent_command), CommandHandler("set_min_rr", set_min_rr_command),
                CommandHandler("set_normal_score", set_normal_score_threshold_command),
                CommandHandler("set_low_vol_score", set_low_vol_score_threshold_command),
                CommandHandler("set_volume_mult", set_volume_mult_command),
                CommandHandler("toggle_btc_filter", toggle_btc_filter_command),
                CommandHandler("toggle_consolidation_filter", toggle_consolidation_filter_command),
                CommandHandler("toggle_correlation_filter", toggle_correlation_filter_command),
                CommandHandler("toggle_ab_testing", toggle_ab_testing_command),
                CommandHandler("toggle_one_trade", toggle_unidirectional_trades_command),
                CommandHandler("toggle_multi_tf", toggle_multi_tf_filter_command),
                CommandHandler("toggle_reversal", toggle_portfolio_reversal_command),
                CommandHandler("toggle_momentum_entry", toggle_momentum_entry_command),
                CommandHandler("set_momentum_increase", set_momentum_increase_command),
                CommandHandler("set_momentum_level", set_momentum_level_command),
                CommandHandler("set_max_positions", set_max_positions_command),
                CommandHandler("close_partial", close_partial_command),
                CommandHandler("trend_chart", trend_chart_command),
                CommandHandler("monthlystats", monthly_stats_command),
                CommandHandler("emergency_close_all", emergency_close_all_command),
                CommandHandler("pnl", pnl_command),
                MessageHandler(filters.TEXT & ~filters.COMMAND, unknown_command)
            ]
            telegram_app.add_handlers(handlers)
            telegram_app.add_error_handler(error_handler)
            await telegram_app.bot.set_my_commands([BotCommand("status", "📊 Статус"), BotCommand("positions", "📈 Позиции"), BotCommand("get_settings", "⚙️ Настройки"), BotCommand("help", "ℹ️ Справка")])
            await telegram_app.initialize()
            await telegram_app.start()
            await telegram_app.updater.start_polling() 
            await send_telegram_message_safe(f"🤖 Бот запущен. Баланс: {current_balance:.2f} {COLLATERAL_ASSET}")


        await loop.run_in_executor(None, initialize_exchange_info_sync)
        await loop.run_in_executor(None, partial(set_leverage_sync, SYMBOLS, LEVERAGE))
        await initialize_market_data(SYMBOLS)


        
        load_state_sync()

        # Теперь этот блок будет работать корректно
        if current_positions:
            logging.warning("Восстановление объектов-менеджеров из сохраненного состояния...")
            rehydrated_positions = {}
            for symbol, pos_data in current_positions.items():
                if isinstance(pos_data, dict):
                    pos_data['symbol'] = symbol
                    # Создаем новый экземпляр менеджера на основе сохраненных данных
                    manager_instance = PositionManager(pos_data)
                    rehydrated_positions[symbol] = manager_instance
                    logging.info(f" -> Для {symbol} восстановлен менеджер позиций.")
            
            current_positions = rehydrated_positions

        # --- ✅ ЗАПУСК ПРОВЕРКИ СТРАТЕГИЙ СРАЗУ ПОСЛЕ СТАРТА ---
        logging.warning("🚀 Запуск первоначальной проверки стратегий для всех символов...")
        initial_check_tasks = []
        # Убедимся, что ta импортирован (обычно он импортируется в начале файла)
        import ta
        for symbol in SYMBOLS:
            # Проверяем только если нет открытой позиции по этому символу
            #  и символ есть в загруженных данных
            if symbol not in current_positions and symbol in market_data_store:
                logging.debug(f" -> Первоначальная проверка для {symbol}...")
                # Собираем задачи для асинхронного выполнения
                # Добавьте сюда вызовы ВСЕХ ваших функций поиска входа
                #initial_check_tasks.append(find_and_execute_ranging_trade_v4_3(symbol)) # Ranging (вызывает on_closed_candle_analysis)
                initial_check_tasks.append(find_and_execute_pullback_trade(symbol)) # Pullback
                initial_check_tasks.append(find_and_execute_impulse_trade(symbol)) # Impulse
                # Breakout
                # Trend (вызывается с определением стороны)
                df_trend_check = market_data_store.get(symbol, {}).get(ALLOWED_INTERVALS.get(TRENDING_SIGNAL_TF))
                if df_trend_check is not None and len(df_trend_check) > TRENDING_SLOW_EMA:
                    try:
                        fast_ema_init = ta.trend.ema_indicator(df_trend_check["close"], window=TRENDING_FAST_EMA).iloc[-1]
                        slow_ema_init = ta.trend.ema_indicator(df_trend_check["close"], window=TRENDING_SLOW_EMA).iloc[-1]
                        initial_check_tasks.append(find_and_execute_trend_trade(symbol, expected_side='long' if fast_ema_init > slow_ema_init else 'short'))
                    except Exception as e:
                        logging.error(f"[{symbol}] Ошибка расчета EMA для первоначальной проверки тренда: {e}")

        # Асинхронно выполняем все проверки
        if initial_check_tasks:
            await asyncio.gather(*initial_check_tasks, return_exceptions=True)
        logging.warning("✅ Первоначальная проверка стратегий завершена.")
        # --- КОНЕЦ БЛОКА ПРОВЕРКИ СТРАТЕГИЙ ---

        await send_telegram_message_safe("✅✅✅ Бот полностью готов и запущен! ✅✅✅")
        
        for i in range(NUM_WORKERS):
            task = asyncio.create_task(worker(i))
            worker_tasks.append(task)
        logging.warning(f"Запущено {NUM_WORKERS} воркеров для обработки данных.")

        ws_task_handle = asyncio.create_task(start_websockets(SYMBOLS), name="WebSocketHandlerTask")
        periodic_task = asyncio.create_task(periodic_tasks(), name="PeriodicChecksTask")
        btc_monitor_task = asyncio.create_task(check_btc_for_emergency_exit(), name="BTCEmergencyMonitor")
        # <--- ИСПРАВЛЕНИЕ 1: ЗАПУСКАЕМ СТРИМ ЛИКВИДАЦИЙ ---
        liq_stream_task = asyncio.create_task(stream_liquidations(SYMBOLS), name="LiquidationStreamTask")
        
        await test_gemini_orderbook_analysis()
        await stop_event.wait()



    except Exception as e:
        logging.critical(f"Критическая ошибка в run_bot: {e}", exc_info=True)
        await send_telegram_message_safe(f"❌ Крит. ошибка бота: {e}. Остановка...")
        
    finally:
        logging.warning("--- НАЧАЛО ПРОЦЕДУРЫ ГРАЦИОЗНОГО ЗАВЕРШЕНИЯ БОТА ---")
        
        tasks_to_cancel = [ws_task_handle, periodic_task] + worker_tasks
        for task in tasks_to_cancel:
            if task and not task.done():
                task.cancel()
        await asyncio.gather(*[t for t in tasks_to_cancel if t], return_exceptions=True)
        
        if telegram_app:
            if telegram_app.updater and telegram_app.updater.running:
                logging.info("Останавливаю Telegram Updater...")
                await telegram_app.updater.stop()

            if telegram_app.running:
                logging.info("Останавливаю Telegram Application...")
                await telegram_app.stop()

            logging.info("Выполняю очистку ресурсов Telegram...")
            await telegram_app.shutdown()
        
        if client:
            await client.close_connection()
            
        logging.warning("--- БОТ ПОЛНОСТЬЮ ОСТАНОВЛЕН ---")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    print("Запуск торгового бота...")
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        print("\nПрограмма завершена пользователем.")
    except Exception as e:
        print(f"\nФатальная ошибка в __main__: {e}")
        traceback.print_exc()