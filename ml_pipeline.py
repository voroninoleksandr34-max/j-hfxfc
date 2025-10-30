# ml_pipeline.py (ФИНАЛЬНАЯ ВЕРСИЯ)

import pandas as pd
import numpy as np
import joblib
import os
import logging
import re
from typing import Optional, Dict, Any, List, Tuple

try:
    import talib as ta
except ImportError:
    print("!!! КРИТИЧЕСКОЕ ПРЕДУПРЕЖДЕНИЕ: Библиотека 'talib' не найдена.")
    ta = None

# --- Настройка логгера ---
ml_logger = logging.getLogger("ML_Trading_Bot_v7_9_1_Dual_Mode")
if not ml_logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s [%(funcName)s:%(lineno)d]')
    handler.setFormatter(formatter)
    ml_logger.addHandler(handler)
    ml_logger.setLevel(logging.INFO)

# ==============================================================================
# 1. BaseFeatureEngineer ИЗ ВАШЕГО СКРИПТА 2.py
# ==============================================================================
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
        
        altcoin_features_by_tf = {}
        for tf in self.tfs:
            if tf in dfs_altcoin and not dfs_altcoin[tf].empty:
                altcoin_features_by_tf[tf] = self._calculate_base_tf_features(dfs_altcoin[tf], tf)

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

        merged_altcoin = altcoin_features_by_tf[primary_tf]
        for i in range(1, len(self.tfs)):
            longer_tf = self.tfs[i]
            if longer_tf in altcoin_features_by_tf:
                merged_altcoin = pd.merge_asof(
                    left=merged_altcoin.sort_index(), right=altcoin_features_by_tf[longer_tf].sort_index(),
                    left_index=True, right_index=True, direction='backward'
                )

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

        final_merged_df = pd.merge_asof(
            left=merged_altcoin, right=merged_btc,
            left_index=True, right_index=True, direction='backward'
        )
        
        ml_logger.info(f"Комбинированные признаки созданы. Итоговое кол-во: {len(final_merged_df.columns)}")
        self.feature_names_final_order = final_merged_df.columns.tolist()
        return final_merged_df

# ==============================================================================
# 2. ОБНОВЛЁННЫЙ TradingModel для работы с BTC-контекстом
# ==============================================================================
class TradingModel:
    def __init__(self, model_file_path: str, fe_file_path: str, config: Dict[str, Any]):
        self.model = None
        self.scaler = None
        self.imputer = None
        self.feature_engineer: Optional[BaseFeatureEngineer] = None
        self.is_ready = False
        self.config = config
        self._load_artifacts(model_file_path, fe_file_path)

        if self.is_ready:
            ml_logger.info(f"TradingModel для '{os.path.basename(model_file_path)}' инициализирован успешно.")
        else:
            ml_logger.critical(f"TradingModel для '{os.path.basename(model_file_path)}' НЕ удалось инициализировать!")

    def _load_artifacts(self, model_path: str, fe_path: str):
        try:
            if not os.path.exists(model_path) or not os.path.exists(fe_path):
                ml_logger.error(f"Файл модели или FE не найден: {model_path} / {fe_path}")
                return

            model_artifacts = joblib.load(model_path)
            self.model = model_artifacts['model']
            self.scaler = model_artifacts['scaler']
            self.imputer = model_artifacts.get('imputer') 
            self.feature_engineer = joblib.load(fe_path)

            if all([self.model, self.scaler, self.imputer, self.feature_engineer]):
                self.is_ready = True
                ml_logger.info("Все артефакты (модель, скейлер, imputer, FE) загружены.")
            else:
                ml_logger.error("Один из артефактов (model, scaler, imputer, fe) не был загружен корректно.")

        except Exception as e:
            ml_logger.error(f"Критическая ошибка при загрузке артефактов: {e}", exc_info=True)
            self.is_ready = False

    def process_new_data(self, dfs_altcoin: Dict[str, pd.DataFrame], dfs_btc: Dict[str, pd.DataFrame]) -> Tuple[int, float, np.ndarray]:
        if not self.is_ready:
            ml_logger.warning("Модель не готова. Возвращаю HOLD.")
            return 1, 0.0, np.array([0.0, 1.0, 0.0])
        try:
            features_df = self.feature_engineer.calculate_multi_tf_features(dfs_altcoin, dfs_btc)
            if features_df.empty:
                ml_logger.warning("Генератор признаков вернул пустой DataFrame.")
                return 1, 0.0, np.array([0.0, 1.0, 0.0])

            last_row_unscaled = features_df.iloc[[-1]].reindex(columns=self.feature_engineer.feature_names_final_order, fill_value=np.nan)
            
            last_row_imputed = self.imputer.transform(last_row_unscaled)
            last_row_scaled = self.scaler.transform(last_row_imputed)
            
            probas = self.model.predict_proba(last_row_scaled)[0]
            
            predicted_class = np.argmax(probas)
            confidence = probas[predicted_class]
            
            confidence_threshold = self.config.get('confidence_threshold', 0.70)
            
            final_signal = predicted_class if confidence >= confidence_threshold else 1

            signal_name_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
            signal_name = signal_name_map.get(final_signal, "UNKNOWN")

            ml_logger.info(
                f"Вероятности (SELL/HOLD/BUY): [{probas[0]:.2%}, {probas[1]:.2%}, {probas[2]:.2%}] -> "
                f"Итоговый сигнал: {signal_name} (Код: {final_signal})"
            )
            return final_signal, (confidence * 100.0), probas

        except Exception as e:
            ml_logger.error(f"Критическая ошибка в process_new_data: {e}", exc_info=True)
            return 1, 0.0, np.array([0.0, 1.0, 0.0])