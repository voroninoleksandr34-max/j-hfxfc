# =======================================================================================
# УЛЬТИМАТИВНАЯ ВЕРСЯ ТОРГОВОГО БОТА v7.9.1 - ДВУХРЕЖИМНАЯ АРХИТЕКТУРА (ТРЕНД/ФЛЭТ)
#
# Ключевые улучшения:
# 1. ФОКУСИРОВКА НА ДВУХ СТРАТЕГИЯХ: Логика "пробоя" (breakout) полностью удалена
#    из конвейера обучения и работы бота. Модель обучается и работает только
#    в двух режимах: торговля по тренду и возврат к среднему (флэт).
# 2. ИНТЕГРАЦИЯ BTC-КОНТЕКСТА: Сохранена ключевая архитектура, при которой модель
#    для каждого альткоина получает признаки от BTCUSDT для учета состояния рынка.
# 3. УПРОЩЕННАЯ ЛОГИКА БОТА: Механизм выбора стратегии в реальном времени
#    упрощен и теперь переключается только между 'trend' и 'reversion'.
# =======================================================================================

import optuna
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
import joblib
import talib as ta
import warnings
import logging
import os
import time
from binance.client import Client
import datetime
from typing import Optional, Dict, Any, List
import threading
import re

# --- ИМПОРТЫ ДЛЯ УЛУЧШЕНИЙ ---
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from alibi_detect.cd import KSDrift
from prometheus_client import Counter, Gauge
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE

# --- НАСТРОЙКА ЛОГГИРОВАНИЯ ---
ml_logger = logging.getLogger("ML_Trading_Bot_v7_9_1_Dual_Mode")
if not ml_logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s [%(funcName)s:%(lineno)d]')
    handler.setFormatter(formatter)
    ml_logger.addHandler(handler)
    ml_logger.setLevel(logging.INFO)

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# --- МЕТРИКИ ДЛЯ PROMETHEUS ---
TRADES_COUNT = Counter('trades_total', 'Total trades', ['symbol', 'side'])
MODEL_F1_SCORE = Gauge('model_f1_score', 'Model F1-Score on test data', ['symbol', 'model_type'])
DATA_DRIFT_DETECTED = Counter('data_drift_detected_total', 'Total data drift detections', ['symbol'])

# --- МЕХАНИЗМ CIRCUIT BREAKER ---
class CircuitBreaker:
    def __init__(self, max_errors: int = 5, reset_timeout: int = 300):
        self.error_count = 0
        self.last_error_time = 0
        self.max_errors = max_errors
        self.reset_timeout = reset_timeout

    def should_stop(self) -> bool:
        if time.time() - self.last_error_time > self.reset_timeout:
            self.error_count = 0
        return self.error_count >= self.max_errors

    def record_error(self):
        self.error_count += 1
        self.last_error_time = time.time()
        ml_logger.warning(f"Circuit Breaker: ошибка зафиксирована. Всего ошибок: {self.error_count}/{self.max_errors}.")


# --- БЛОК 1: ЗАГРУЗКА ДАННЫХ ---
def load_historical_data_binance(client_obj: Client, symbol: str, interval_str: str, start_date_str: str, end_date_str: Optional[str] = None) -> pd.DataFrame:
    ml_logger.info(f"Загрузка данных: {symbol} [{interval_str}] с {start_date_str}")
    try:
        klines = client_obj.get_historical_klines(symbol, interval_str, start_date_str, end_date_str)
        if not klines:
            ml_logger.warning(f"Данные для {symbol} не загружены.")
            return pd.DataFrame()

        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        cols_to_numeric = ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base']
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)
        ml_logger.info(f"Загружено {len(df)} свечей для {symbol}.")
        return df
    except Exception as e:
        ml_logger.error(f"Ошибка при загрузке данных для {symbol}: {e}")
        return pd.DataFrame()


# --- БЛОК 2: ИНЖЕНЕРИЯ ПРИЗНАКОВ (С ПОДДЕРЖКОЙ BTC-КОНТЕКСТА) ---
class BaseFeatureEngineer:
    def __init__(self, lookback_windows: Optional[Dict[str, int]] = None):
        if lookback_windows is None:
            self.lookback_windows = {'5m': 50, '15m': 50, '1h': 50}
        else:
            self.lookback_windows = lookback_windows
        self.tfs: List[str] = sorted(list(self.lookback_windows.keys()), key=self._get_interval_minutes)
        self.feature_names_final_order: List[str] = []

    @staticmethod
    def _get_interval_minutes(tf_str: str) -> float:
        num_match = re.search(r'(\d+)', tf_str)
        unit_match = re.search(r'([mhd])', tf_str.lower())
        if not num_match or not unit_match: return float('inf')
        num = int(num_match.group(1))
        unit = unit_match.group(1)
        if unit == 'm': return float(num)
        if unit == 'h': return float(num * 60)
        if unit == 'd': return float(num * 60 * 24)
        return float('inf')

    @staticmethod
    def _choppiness_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
        tr = pd.DataFrame(index=df.index)
        tr['h-l'] = df['high'] - df['low']
        tr['h-pc'] = abs(df['high'] - df['close'].shift(1))
        tr['l-pc'] = abs(df['low'] - df['close'].shift(1))
        true_range = tr[['h-l', 'h-pc', 'l-pc']].max(axis=1)
        
        atr_sum = true_range.rolling(window=period).sum()
        hh = df['high'].rolling(window=period).max()
        ll = df['low'].rolling(window=period).min()
        
        range_hl = (hh - ll).replace(0, np.nan)
        chop = 100 * np.log10(atr_sum / range_hl) / np.log10(period)
        return chop.fillna(50)

    def _calculate_base_tf_features(self, df_history: pd.DataFrame, tf_suffix: str) -> pd.DataFrame:
        df = df_history.copy()
        features = pd.DataFrame(index=df.index)
        
        features[f'rsi_14_{tf_suffix}'] = ta.RSI(df['close'], timeperiod=14)
        _, _, features[f'macd_hist_{tf_suffix}'] = ta.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        features[f'adx_14_{tf_suffix}'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        features[f'cci_14_{tf_suffix}'] = ta.CCI(df['high'], df['low'], df['close'], timeperiod=14)
        features[f'roc_10_{tf_suffix}'] = ta.ROC(df['close'], timeperiod=10)
        features[f'obv_{tf_suffix}'] = ta.OBV(df['close'], df['volume'])
        if 'taker_buy_base' in df.columns:
            features[f'volume_delta_{tf_suffix}'] = 2.0 * df['taker_buy_base'] - df['volume']
        
        features[f'atr_14_{tf_suffix}'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        bb_upper, bb_middle, bb_lower = ta.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
        features[f'bb_width_{tf_suffix}'] = (bb_upper - bb_lower) / bb_middle.replace(0, 1)
        
        features[f'williams_r_{tf_suffix}'] = ta.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
        stoch_k, stoch_d = ta.STOCH(df['high'], df['low'], df['close'], fastk_period=14, slowk_period=3, slowd_period=3)
        features[f'stoch_k_{tf_suffix}'] = stoch_k
        features[f'stoch_d_{tf_suffix}'] = stoch_d
        features[f'hour_of_day_{tf_suffix}'] = df.index.hour
        features[f'day_of_week_{tf_suffix}'] = df.index.dayofweek
        
        ema_fast = ta.EMA(df['close'], timeperiod=12)
        ema_slow = ta.EMA(df['close'], timeperiod=26)
        features[f'price_dist_ema_fast_{tf_suffix}'] = (df['close'] - ema_fast) / ema_fast
        features[f'price_dist_ema_slow_{tf_suffix}'] = (df['close'] - ema_slow) / ema_slow
        
        vol_sma = ta.SMA(df['volume'], timeperiod=30)
        features[f'volume_norm_{tf_suffix}'] = df['volume'] / vol_sma.replace(0, np.nan)
        
        atr_series = pd.Series(features[f'atr_14_{tf_suffix}'], index=features.index)
        features[f'atr_roc_{tf_suffix}'] = atr_series.pct_change(periods=5)

        sma_200 = ta.SMA(df['close'], timeperiod=200)
        features[f'price_vs_sma200_{tf_suffix}'] = (df['close'] - sma_200) / sma_200
        
        adx_val = features[f'adx_14_{tf_suffix}']
        features[f'trend_strength_{tf_suffix}'] = pd.cut(adx_val, bins=[0, 20, 25, 100], labels=[0, 1, 2], right=False)
        features[f'trend_direction_sma200_{tf_suffix}'] = np.sign(df['close'] - sma_200).fillna(0)
        features[f'momentum_3_{tf_suffix}'] = df['close'].pct_change(3)
        features[f'momentum_10_{tf_suffix}'] = df['close'].pct_change(10)
        
        price_position_in_bb = (df['close'] - bb_middle) / (bb_upper - bb_lower).replace(0, np.nan)
        features[f'price_pos_in_bb_{tf_suffix}'] = price_position_in_bb.replace([np.inf, -np.inf], np.nan).fillna(0)
        features[f'is_touching_bb_upper_{tf_suffix}'] = (df['high'] >= bb_upper).astype(int)
        features[f'is_touching_bb_lower_{tf_suffix}'] = (df['low'] <= bb_lower).astype(int)
        
        rsi_val = features[f'rsi_14_{tf_suffix}']
        features[f'rsi_zone_{tf_suffix}'] = pd.cut(rsi_val, bins=[0, 30, 70, 100], labels=[0, 1, 2], right=False)
        
        rolling_avg_bbw = features[f'bb_width_{tf_suffix}'].rolling(50).mean()
        features[f'squeeze_factor_{tf_suffix}'] = features[f'bb_width_{tf_suffix}'] / rolling_avg_bbw.replace(0, np.nan)
        features[f'normalized_atr_{tf_suffix}'] = features[f'atr_14_{tf_suffix}'] / df['close']
        
        avg_volume_20 = ta.SMA(df['volume'], timeperiod=20)
        features[f'volume_breakout_spike_{tf_suffix}'] = (df['volume'] > (avg_volume_20 * 2.0)).astype(int)

        features[f'chop_14_{tf_suffix}'] = self._choppiness_index(df, period=14)

        kc_middle = ta.EMA(df['close'], timeperiod=20)
        kc_atr = ta.ATR(df['high'], df['low'], df['close'], timeperiod=10)
        kc_upper = kc_middle + kc_atr * 2
        kc_lower = kc_middle - kc_atr * 2
        features[f'kc_width_{tf_suffix}'] = (kc_upper - kc_lower) / kc_middle.replace(0, np.nan)
        features[f'ttm_squeeze_{tf_suffix}'] = ((bb_lower > kc_lower) & (bb_upper < kc_upper)).astype(int)
        
        features[f'donchian_h_20_{tf_suffix}'] = df['high'].rolling(20).max()
        features[f'donchian_l_20_{tf_suffix}'] = df['low'].rolling(20).min()
        features[f'price_dist_donchian_h_{tf_suffix}'] = (features[f'donchian_h_20_{tf_suffix}'] - df['close']) / df['close']
        features[f'price_dist_donchian_l_{tf_suffix}'] = (df['close'] - features[f'donchian_l_20_{tf_suffix}']) / df['close']

        safe_volume = df['volume'].replace(0, 1)
        features[f'vol_roc_5_{tf_suffix}'] = ta.ROC(safe_volume, timeperiod=5)
        
        atr_safe = features[f'atr_14_{tf_suffix}'].replace(0, np.nan)
        features[f'roc_10_atr_norm_{tf_suffix}'] = features[f'roc_10_{tf_suffix}'] / atr_safe
        
        features[f'rsi_slope_3_{tf_suffix}'] = features[f'rsi_14_{tf_suffix}'].diff(3)
        
        kc_width_val = (kc_upper - kc_lower).replace(0, np.nan)
        features[f'price_pos_in_kc_{tf_suffix}'] = (df['close'] - kc_middle) / kc_width_val
        features[f'price_pos_in_kc_{tf_suffix}'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return features

    def calculate_multi_tf_features(self, dfs_altcoin: Dict[str, pd.DataFrame], dfs_btc: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        ml_logger.info(f"Создание комбинированных МТФ признаков (Альткоин + BTC)...")
        
        # 1. Рассчитать признаки для Альткоина
        altcoin_features_by_tf = {}
        for tf in self.tfs:
            if tf in dfs_altcoin and not dfs_altcoin[tf].empty:
                altcoin_features_by_tf[tf] = self._calculate_base_tf_features(dfs_altcoin[tf], tf)

        # 2. Рассчитать признаки для Bitcoin и добавить префикс 'btc_'
        btc_features_by_tf = {}
        for tf in self.tfs:
            if tf in dfs_btc and not dfs_btc[tf].empty:
                btc_features = self._calculate_base_tf_features(dfs_btc[tf], tf)
                btc_features.columns = ['btc_' + col for col in btc_features.columns]
                btc_features_by_tf[tf] = btc_features

        primary_tf = self.tfs[0]
        if primary_tf not in altcoin_features_by_tf:
            ml_logger.error("Нет признаков для основного ТФ альткоина. Возвращается пустой DF.")
            return pd.DataFrame()

        # 3. Объединить признаки Альткоина по таймфреймам
        merged_altcoin = altcoin_features_by_tf[primary_tf]
        for i in range(1, len(self.tfs)):
            longer_tf = self.tfs[i]
            if longer_tf in altcoin_features_by_tf:
                merged_altcoin = pd.merge_asof(
                    left=merged_altcoin.sort_index(), right=altcoin_features_by_tf[longer_tf].sort_index(),
                    left_index=True, right_index=True, direction='backward'
                )

        # 4. Объединить признаки BTC по таймфреймам
        if primary_tf not in btc_features_by_tf:
             ml_logger.error("Нет BTC признаков для основного ТФ. Возвращается пустой DF.")
             return pd.DataFrame()
        merged_btc = btc_features_by_tf[primary_tf]
        for i in range(1, len(self.tfs)):
            longer_tf = self.tfs[i]
            if longer_tf in btc_features_by_tf:
                merged_btc = pd.merge_asof(
                    left=merged_btc.sort_index(), right=btc_features_by_tf[longer_tf].sort_index(),
                    left_index=True, right_index=True, direction='backward'
                )

        # 5. Финальное объединение: Совместить признаки альткоина с признаками BTC
        final_merged_df = pd.merge_asof(
            left=merged_altcoin, right=merged_btc,
            left_index=True, right_index=True, direction='backward'
        )
        
        ml_logger.info(f"Комбинированные признаки созданы. Итоговое кол-во: {len(final_merged_df.columns)}")
        self.feature_names_final_order = final_merged_df.columns.tolist()
        return final_merged_df

# --- БЛОК 3: ЦЕЛЕВЫЕ ПЕРЕМЕННЫЕ (СТРАТЕГИИ) ---
def create_aggressive_target(df: pd.DataFrame, horizon: int) -> pd.Series:
    close = pd.to_numeric(df['close'], errors='coerce')
    momentum_threshold = 0.015
    future_momentum = ta.ROC(close, 5).shift(-horizon)
    ema_filter = ta.EMA(close, timeperiod=50)
    adx = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    buy_cond = ((future_momentum > momentum_threshold) & (close > ema_filter) & (adx > 25))
    sell_cond = ((future_momentum < -momentum_threshold) & (close < ema_filter) & (adx > 25))
    y = pd.Series(0, index=df.index, dtype=int, name="target")
    y[buy_cond] = 1
    y[sell_cond] = -1
    return y.iloc[:-horizon]

def create_mean_reversion_target(df: pd.DataFrame, horizon: int, bb_period: int = 20, bb_std: float = 2.0) -> pd.Series:
    close = pd.to_numeric(df['close'], errors='coerce')
    bb_upper, bb_middle, bb_lower = ta.BBANDS(close, timeperiod=bb_period, nbdevup=bb_std, nbdevdn=bb_std)
    y = pd.Series(0, index=df.index, name="target", dtype=int)
    channel_width = bb_upper - bb_lower
    buy_success_threshold = bb_middle + (channel_width * 0.1)
    sell_success_threshold = bb_middle - (channel_width * 0.1)
    buy_entry_condition = (df['low'] <= bb_lower)
    future_high_series = df['high'].rolling(window=horizon).max().shift(-horizon)
    buy_success_condition = (future_high_series >= buy_success_threshold)
    y[buy_entry_condition & buy_success_condition] = 1
    sell_entry_condition = (df['high'] >= bb_upper)
    future_low_series = df['low'].rolling(window=horizon).min().shift(-horizon)
    sell_success_condition = (future_low_series <= sell_success_threshold)
    y[sell_entry_condition & sell_success_condition] = -1
    return y.iloc[:-horizon]


# --- БЛОК 4: КОНВЕЙЕР ОБУЧЕНИЯ (ОБНОВЛЕННЫЙ) ---
def train_and_save_model(symbol: str, config: Dict, strategy: str) -> Optional[str]:
    strategy_type_rus = {'trend': "ТРЕНД", 'reversion': "КОНТРТРЕНД"}.get(strategy, "НЕИЗВЕСТНАЯ")
    ml_logger.info(f"--- Начало обучения ({strategy_type_rus}) для {symbol} с BTC-контекстом ---")

    try:
        timeframes = config['timeframes']
        primary_tf = config['primary_timeframe']
        
        model_suffix = f"_{strategy}_btc_context"
        model_path = os.path.join(config['models_dir'], f"{symbol}{model_suffix}_model.joblib")
        fe_path = os.path.join(config['models_dir'], f"{symbol}{model_suffix}_fe.joblib")
        os.makedirs(config['models_dir'], exist_ok=True)

        client = Client(config['api_key'], config['api_secret'])
        
        # Загрузка данных для альткоина
        all_dfs_altcoin = {tf: load_historical_data_binance(client, symbol, tf, config['data_start_date']) for tf in timeframes}
        if any(df.empty for df in all_dfs_altcoin.values()):
            ml_logger.error(f"Не удалось загрузить данные для {symbol}.")
            return None

        # Загрузка данных для BTC
        all_dfs_btc = {tf: load_historical_data_binance(client, 'BTCUSDT', tf, config['data_start_date']) for tf in timeframes}
        if any(df.empty for df in all_dfs_btc.values()):
            ml_logger.error("Не удалось загрузить данные для BTCUSDT.")
            return None

        feature_engineer = BaseFeatureEngineer(lookback_windows={tf: 50 for tf in timeframes})
        features_df = feature_engineer.calculate_multi_tf_features(all_dfs_altcoin, all_dfs_btc)

        primary_df_altcoin = all_dfs_altcoin[primary_tf]
        
        if strategy == 'reversion':
            target_series = create_mean_reversion_target(primary_df_altcoin, config['horizon'])
        else: # 'trend'
            target_series = create_aggressive_target(primary_df_altcoin, config['horizon'])

        full_df = pd.concat([features_df, target_series], axis=1).dropna()
        X = full_df.drop(columns='target')
        y = full_df['target']
        y = y + 1 # Преобразуем классы -1, 0, 1 в 0, 1, 2

        if len(X) < 500:
            ml_logger.error(f"Недостаточно данных ({len(X)}) для обучения {symbol}.")
            return None
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        imputer = IterativeImputer(random_state=42, max_iter=10)
        X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)

        try:
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
            X_train_final, y_train_final = X_train_resampled, y_train_resampled
        except ValueError as e:
            ml_logger.warning(f"SMOTE не применен для {symbol} ({strategy}): {e}.")
            X_train_final, y_train_final = X_train_scaled, y_train

        def optimize_model(trial, model_name):
            if model_name == 'lgbm':
                return {'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss', 'random_state': 42,
                        'device': 'gpu', 'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2), 'num_leaves': trial.suggest_int('num_leaves', 20, 80)}
            if model_name == 'xgb':
                return {'objective': 'multi:softprob', 'num_class': 3, 'eval_metric': 'mlogloss', 'random_state': 42,
                        'tree_method': 'gpu_hist', 'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2), 'max_depth': trial.suggest_int('max_depth', 3, 8)}
            if model_name == 'cat':
                return {'objective': 'MultiClass', 'random_state': 42, 'verbose': 0, 'task_type': 'GPU',
                        'iterations': trial.suggest_int('iterations', 200, 1000), 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                        'depth': trial.suggest_int('depth', 4, 8)}

        best_params = {}
        for model_name in ['lgbm', 'xgb', 'cat']:
            def objective(trial):
                params = optimize_model(trial, model_name)
                if model_name == 'lgbm': model = lgb.LGBMClassifier(**params)
                elif model_name == 'xgb': model = XGBClassifier(**params, use_label_encoder=False)
                else: model = CatBoostClassifier(**params)
                model.fit(X_train_final, y_train_final)
                preds = model.predict(X_test_scaled)
                return f1_score(y_test, preds, average='weighted')

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=5)
            best_params[model_name] = study.best_params
        
        models = {
            'lgbm': lgb.LGBMClassifier(objective='multiclass', num_class=3, random_state=42, device='gpu', **best_params['lgbm']),
            'xgb': XGBClassifier(objective='multi:softprob', num_class=3, use_label_encoder=False, random_state=42, tree_method='gpu_hist', **best_params['xgb']),
            'cat': CatBoostClassifier(objective='MultiClass', verbose=0, random_state=42, task_type='GPU', **best_params['cat'])
        }
        
        weights = []
        for name, model in models.items():
            model.fit(X_train_final, y_train_final)
            preds = model.predict(X_test_scaled)
            score = f1_score(y_test, preds, average='weighted')
            weights.append(score)
            ml_logger.info(f"Модель {name} ({strategy}) F1-score: {score:.4f}.")
        
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights] if total_weight > 0 else [1/3, 1/3, 1/3]
        
        ensemble_model = VotingClassifier(estimators=list(models.items()), voting='soft', weights=normalized_weights)
        ensemble_model.fit(X_train_final, y_train_final)

        y_pred = ensemble_model.predict(X_test_scaled)
        report = classification_report(y_test, y_pred, target_names=['sell', 'hold', 'buy'], zero_division=0)
        final_f1_score = f1_score(y_test, y_pred, average='weighted')
        ml_logger.info(f"Итоговый отчет для АНСАМБЛЯ ({strategy_type_rus}) для {symbol}:\n{report}")
        MODEL_F1_SCORE.labels(symbol=symbol, model_type=strategy).set(final_f1_score)

        x_train_sample = X_train_imputed.sample(n=min(1000, len(X_train_imputed)), random_state=42).to_numpy()
        joblib.dump({'model': ensemble_model, 'scaler': scaler, 'imputer': imputer, 'x_train_sample': x_train_sample}, model_path)
        joblib.dump(feature_engineer, fe_path)
        ml_logger.info(f"Артефакты ансамбля ({strategy_type_rus}) сохранены: {model_path}, {fe_path}")
        return model_path
    except Exception as e:
        ml_logger.error(f"Критическая ошибка в обучении для {symbol} ({strategy_type_rus}): {e}", exc_info=True)
        return None


# --- БЛОК 5: ЛОГИКА БОТА (ОБНОВЛЕННАЯ) ---
class AdvancedBinanceTradingBot:
    def __init__(self, symbol: str, client: Client, config: Dict[str, Any]):
        self.symbol = symbol
        self.client = client
        self.config = config
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.circuit_breaker = CircuitBreaker()
        self.models = {}
        self.feature_engineer: Optional[BaseFeatureEngineer] = None
        self._load_all_artifacts()
        self.initial_data_load()

    def _load_all_artifacts(self):
        strategies = ['trend', 'reversion']
        for strategy in strategies:
            try:
                model_suffix = f"_{strategy}_btc_context"
                path_model = os.path.join(self.config['models_dir'], f"{self.symbol}{model_suffix}_model.joblib")
                artifacts = joblib.load(path_model)
                self.models[strategy] = artifacts
                ml_logger.info(f"Артефакты для стратегии '{strategy}' ({self.symbol}) успешно загружены.")
                
                # Загружаем один экземпляр feature_engineer
                if self.feature_engineer is None:
                    path_fe = os.path.join(self.config['models_dir'], f"{self.symbol}{model_suffix}_fe.joblib")
                    self.feature_engineer = joblib.load(path_fe)

            except Exception as e:
                ml_logger.warning(f"Не удалось загрузить артефакты для стратегии '{strategy}' ({self.symbol}): {e}")

    def initial_data_load(self):
        tfs = self.config['timeframes']
        # Загрузка данных для альткоина
        for tf in tfs:
            df = load_historical_data_binance(self.client, self.symbol, tf, "3 days ago")
            if not df.empty: self.data_cache[f"{self.symbol}_{tf}"] = df
        # Загрузка данных для BTC
        for tf in tfs:
            df = load_historical_data_binance(self.client, 'BTCUSDT', tf, "3 days ago")
            if not df.empty: self.data_cache[f"BTCUSDT_{tf}"] = df
        ml_logger.info(f"Начальные данные для {self.symbol} и BTCUSDT загружены в кеш.")

    def get_market_context(self) -> str:
        primary_tf = self.config['primary_timeframe']
        df_key = f"{self.symbol}_{primary_tf}"
        if df_key not in self.data_cache or len(self.data_cache[df_key]) < 50:
            return 'trending' # Значение по умолчанию
        
        df = self.data_cache[df_key]
        adx = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14).iloc[-1]
        
        market_regime = 'ranging' if adx < self.config['flat_adx_threshold'] else 'trending'
        return market_regime

    def _get_single_prediction(self, strategy: str) -> (str, float):
        if strategy not in self.models or self.feature_engineer is None: return "HOLD", 0.0

        artifacts = self.models[strategy]
        model, scaler, imputer, x_sample = artifacts['model'], artifacts['scaler'], artifacts['imputer'], artifacts['x_train_sample']
        
        dfs_altcoin = {tf: self.data_cache[f"{self.symbol}_{tf}"] for tf in self.config['timeframes'] if f"{self.symbol}_{tf}" in self.data_cache}
        dfs_btc = {tf: self.data_cache[f"BTCUSDT_{tf}"] for tf in self.config['timeframes'] if f"BTCUSDT_{tf}" in self.data_cache}

        if not dfs_altcoin or not dfs_btc:
             ml_logger.warning("Нет данных для предсказания.")
             return "HOLD", 0.0

        features_df = self.feature_engineer.calculate_multi_tf_features(dfs_altcoin, dfs_btc)
        if features_df.empty: return "HOLD", 0.0
        
        features_df = features_df.reindex(columns=self.feature_engineer.feature_names_final_order, fill_value=np.nan)

        last_row_unscaled = features_df.iloc[[-1]]
        last_row_imputed = pd.DataFrame(imputer.transform(last_row_unscaled), columns=last_row_unscaled.columns, index=last_row_unscaled.index)
        last_row_scaled = scaler.transform(last_row_imputed)
        
        self.check_data_drift(last_row_scaled, x_sample)

        prediction_proba = model.predict_proba(last_row_scaled)[0]
        predicted_class = np.argmax(prediction_proba)
        confidence = prediction_proba[predicted_class]
        signal = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}[predicted_class]
        
        return signal, confidence
    
    def check_data_drift(self, X_current: np.ndarray, x_train_sample: np.ndarray) -> bool:
        try:
            cd = KSDrift(x_train_sample, p_val=0.05)
            preds = cd.predict(X_current)
            if preds['data']['is_drift']:
                ml_logger.warning(f"!!! ОБНАРУЖЕН ДРЕЙФ ДАННЫХ для {self.symbol} !!!")
                DATA_DRIFT_DETECTED.labels(symbol=self.symbol).inc()
                return True
            return False
        except Exception as e:
            ml_logger.error(f"Ошибка при проверке дрейфа данных: {e}")
            return False

    def get_signal(self) -> str:
        market_regime = self.get_market_context()
        
        if market_regime == 'ranging' and 'reversion' in self.models:
            active_strategy = 'reversion'
        else: # По умолчанию или если 'trending'
            active_strategy = 'trend'
            
        ml_logger.info(f"({self.symbol}) Контекст: {market_regime}. Активная стратегия: {active_strategy.upper()}")
        signal, confidence = self._get_single_prediction(active_strategy)
        
        confidence_threshold = self.config.get('confidence_threshold', 0.7)
        if confidence >= confidence_threshold and signal != 'HOLD':
            ml_logger.info(f"({self.symbol}) СИГНАЛ {signal} от стратегии {active_strategy.upper()} с уверенностью {confidence:.2f}")
            return signal
        else:
            if signal != 'HOLD':
                ml_logger.info(f"({self.symbol}) Сигнал {signal} отфильтрован. Уверенность {confidence:.2f} < порога {confidence_threshold}.")
            return "HOLD"
    
    def run(self):
        while True:
            if self.circuit_breaker.should_stop():
                ml_logger.critical(f"Circuit Breaker активирован для {self.symbol}! Бот остановлен.")
                time.sleep(self.circuit_breaker.reset_timeout)
                continue
            try:
                self.initial_data_load()
                signal = self.get_signal()
                ml_logger.info(f"====== ИТОГОВЫЙ СИГНАЛ для {self.symbol}: {signal} ======")
                sleep_time = int(BaseFeatureEngineer._get_interval_minutes(self.config['primary_timeframe']) * 60)
                time.sleep(max(sleep_time, 60))
            except Exception as e:
                self.circuit_breaker.record_error()
                ml_logger.error(f"Ошибка в цикле бота для {self.symbol}: {e}", exc_info=True)
                time.sleep(60)


# --- БЛОК 6: ГЛАВНЫЙ БЛОК ЗАПУСКА ---
if __name__ == "__main__":
    API_KEY = os.getenv("BINANCE_API_KEY", "YOUR_BINANCE_API_KEY")
    API_SECRET = os.getenv("BINANCE_API_SECRET", "YOUR_BINANCE_API_SECRET")

    config = {
        "api_key": API_KEY, "api_secret": API_SECRET,
        "symbols": ['NEOUSDT', 'LSKUSDT', 'NANOUSDT', 'ARKUSDT', 'STORJUSDT', 'MANAUSDT', 'SANDUSDT',
        'ENJUSDT', 'CHZUSDT', 'ONEUSDT', 'ANKRUSDT', 'CRVUSDT', 'SUSHIUSDT', 'MKRUSDT',
        'ALGOUSDT', 'FTMUSDT', 'SOLUSDT','BNBUSDT','XRPUSDT','ADAUSDT','AVAXUSDT','DOTUSDT','MATICUSDT','ATOMUSDT','NEARUSDT',
        'LINKUSDT','ICPUSDT','FILUSDT','FETUSDT','IMXUSDT','DOGEUSDT','FLOKIUSDT'],
        "primary_timeframe": '5m',
        "timeframes": ['5m', '15m', '1h'],
        "models_dir": 'models_v7_9_dual_mode', # Новая папка для моделей
        "data_start_date": "2024-01-01",
        "horizon": 6,
        "flat_adx_threshold": 22.0, 
        "confidence_threshold": 0.70
    }

    TRAIN_MODELS = True
    RUN_BOTS = False

    client = None
    try:
        client = Client(config['api_key'], config['api_secret'])
        client.ping()
        ml_logger.info("Клиент Binance успешно инициализирован.")
    except Exception as e:
        ml_logger.error(f"Ошибка инициализации клиента Binance: {e}. Проверьте API ключи.")

    if client and TRAIN_MODELS:
        for symbol in config['symbols']:
            for strategy in ['trend', 'reversion']:
                 train_and_save_model(symbol, config, strategy=strategy)
            
    if client and RUN_BOTS:
        bot_threads = []
        for symbol in config['symbols']:
            bot = AdvancedBinanceTradingBot(symbol, client, config)
            thread = threading.Thread(target=bot.run, daemon=True)
            thread.start()
            bot_threads.append(thread)
            ml_logger.info(f"Поток для мульти-стратегического бота {symbol} запущен.")
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            ml_logger.info("Программа завершается по команде пользователя...")

    if not TRAIN_MODELS and not RUN_BOTS:
        ml_logger.info("Оба режима (TRAIN_MODELS и RUN_BOTS) отключены.")