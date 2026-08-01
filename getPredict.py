import numpy as np
import pandas as pd
import mplfinance as fplt
import pandas_ta_classic as ta
import category_encoders as ce
import scipy.stats as stats

# Обучение модели  
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn import metrics # инструменты для оценки точности модели

# Сериализация/десериализация
import joblib


#===============================================================================
class predictModels:
    def __init__(self):
        # Глобальные параметры
        self.main_period = '15min' # период сжатия минутного графика свечей и рассчета технических индикаторов
        self.main_period_int = 15
        # Параметры технических индикаторов
        self.period_EMA_fast = 9
        self.period_EMA_slow = 21
        self.period_RSI = 7
        self.period_MACD_fast = 12
        self.period_MACD_slow = 26
        self.period_MACD_signal = 9
        # Количество вычисляемых уровней расширения Фибоначчи
        self.count_FE_levels = 7
        # Значение для вычисления ближайших "круглых чисел"
        self.round_price = 5
        # Шаг цены инстумента
        self.step_price = 0.1
        # Период сохранения позиции в минутах
        self.period_holding_position = 90
        # Количество свечей, обеспечивающих ретроспективу объектов
        self.count_previous_candle = 3
        # Количество знаков для округления значений технических индикаторов, 
        # вычисляемых в масштабе, отличном от цены актива
        self.round_ta = 2
        # Загрузка моделей из файлов
        self.models = self.loadModelsPKL()


    def rnd_step(self, r, sp):
        return (r + 0.5 * sp)//sp * sp


    def fibonacci_extension_levels(self, 
                                   df: pd.DataFrame, 
                                   period: int, 
                                   long_levels: int = 7, 
                                   short_levels: int = 7) -> pd.DataFrame:
        """
        Рассчитывает уровни Фибоначчи расширения для восходящего и нисходящего тренда.
        Оптимизировано для производительности с использованием numpy и pandas.

        :param df: pd.DataFrame с колонками 'high', 'low'
        :param period: Количество свечей для расчета
        :param long_levels: Количество ближайших уровней для восходящего тренда
        :param short_levels: Количество ближайших уровней для нисходящего тренда
        :return: pd.DataFrame с уровнями Фибоначчи для каждой свечи
        """
        
        # Стандартные коэффициенты уровней расширения Фибоначчи
        extension_ratios = [1.0, 1.272, 1.414, 1.618, 2.0, 2.618, 3.618]
        
        # Ограничиваем количество уровней для каждого тренда
        long_ratios = extension_ratios[:long_levels]
        short_ratios = extension_ratios[:short_levels]
        
        # Преобразуем high и low в numpy массивы для ускорения вычислений
        high_values = df['high'].values
        low_values = df['low'].values
        
        # Создаем массивы для хранения уровней расширения для всех свечей
        long_levels_matrix = np.full((len(df), len(long_ratios)), np.nan)
        short_levels_matrix = np.full((len(df), len(short_ratios)), np.nan)
        
        # Вычисляем максимумы и минимумы для всех окон с использованием rolling
        max_high = df['high'].rolling(window=period, min_periods=period).max().values
        min_low = df['low'].rolling(window=period, min_periods=period).min().values
        
        # Векторизация расчета уровней Фибоначчи для каждого окна
        for i in range(period, len(df)):
            high = max_high[i]
            low = min_low[i]
            diff = high - low
            
            if diff == 0:
                continue  # Избегаем деления на ноль
            
            # Восходящий тренд: расширение от low вверх
            long_levels_matrix[i, :] = low + (np.array(long_ratios) - 1) * diff
            
            # Нисходящий тренд: расширение от high вниз
            short_levels_matrix[i, :] = high - (np.array(short_ratios) - 1) * diff
        
        # Собираем результаты в DataFrame
        result = pd.DataFrame(index=df.index)
        
        # Добавляем уровни для восходящего тренда (long)
        for idx, ratio in enumerate(long_ratios):
            result[f"long_ext_{ratio:.3f}"] = long_levels_matrix[:, idx]
        
        # Добавляем уровни для нисходящего тренда (short)
        for idx, ratio in enumerate(short_ratios):
            result[f"short_ext_{ratio:.3f}"] = short_levels_matrix[:, idx]

        return result


    def loadModelsPKL(self):
        # Загрузка моделей
        encoder_day_of_week = joblib.load('./models/encoder_day_of_week.pkl')
        model_to_up = joblib.load('./models/model_to_up.pkl')
        model_to_down = joblib.load('./models/model_to_down.pkl')
        model_high = joblib.load('./models/model_high.pkl')
        model_low = joblib.load('./models/model_low.pkl')
        
        return {'encoder_day_of_week': encoder_day_of_week, 
                'model_to_up': model_to_up, 
                'model_to_down': model_to_down, 
                'model_high': model_high, 
                'model_low': model_low}


    def getPredict(self,
                   df: pd.DataFrame):
        #========== Преобразование данных ==========
        # Добавление дня недели
        df['day_of_week'] = df['dt'].dt.day_name()
        df = df[['dt', 'open', 'high', 'low', 'close', 'volume', 'day_of_week']]

        df_M1 = df.copy()
        df_M1.set_index('dt', inplace=True)

        # Сворачивание данных в 15-минутные свечи
        df_main_period = df_M1.resample(self.main_period).agg({'open': 'first',   # Первая цена открытия за 15 минут
                                                        'high': 'max',     # Максимальная цена за 15 минут
                                                        'low': 'min',      # Минимальная цена за 15 минут
                                                        'close': 'last',   # Последняя цена закрытия за 15 минут
                                                        'volume': 'sum'})  # Сумма объема за 15 минут
        df_main_period.dropna(axis=0, inplace=True)

        # EMA, период 9
        df_main_period['EMA_fast'] = ta.ema(df_main_period['close'], length=self.period_EMA_fast)
        # EMA, период 21
        df_main_period['EMA_slow'] = ta.ema(df_main_period['close'], length=self.period_EMA_slow)
        # VWAP, с начала торговой сессии
        df_main_period['VWAP'] = ta.vwap(df_main_period['high'], df_main_period['low'], df_main_period['close'], df_main_period['volume'])
        # RSI, период 7
        df_main_period['RSI'] = ta.rsi(df_main_period['close'], length=self.period_RSI)
        # MACD, параметры (12, 26, 9)
        MACD = ta.macd(df_main_period['close'], fast=self.period_MACD_fast, slow=self.period_MACD_slow, signal=self.period_MACD_signal)
        added_name = '_' + str(self.period_MACD_fast) + '_' + str(self.period_MACD_slow) + '_' + str(self.period_MACD_signal)
        df_main_period['MACD'] = MACD['MACD' + added_name]
        df_main_period['MACD_signal'] = MACD['MACDs' + added_name]
        df_main_period['MACD_hist'] = MACD['MACDh' + added_name]

        # Добавление признаков описания свечей
        # Тело свечи
        df_main_period['real_body'] = np.abs(df_main_period['close'] - df_main_period['open'])
        # Является растущей свечой
        df_main_period['is_growing_candle'] = df_main_period['close'] > df_main_period['open']
        # Верхняя и нижняя тени
        df_main_period['upper_shadow'] = np.where(df_main_period['is_growing_candle'], 
                                        df_main_period['high'] - df_main_period['close'],
                                        df_main_period['high'] - df_main_period['open'])
        df_main_period['lower_shadow'] = np.where(df_main_period['is_growing_candle'], 
                                        df_main_period['high'] - df_main_period['open'],
                                        df_main_period['high'] - df_main_period['close'])

        # Вычисление уровней Фибоначчи
        df_main_period = df_main_period.join(self.fibonacci_extension_levels(df=df_main_period, period=self.count_FE_levels))
        # Добавление ближайших психологических уровней (круглые числа)
        df_main_period['round_price_up'] = df_main_period['close']//self.round_price * self.round_price + self.round_price
        df_main_period['round_price_down'] = df_main_period['close']//self.round_price * self.round_price

        # Добавление признака разрыва
        df_main_period['is_gapped_up'] = df_main_period['low'] > df_main_period['high'].shift(1)
        df_main_period['is_gapped_down'] = df_main_period['high'] < df_main_period['low'].shift(1)

        # Округление значений до шага инструмента
        list_sign_ta = ['RSI', 'MACD', 'MACD_signal', 'MACD_hist']

        for col in df_main_period.columns:
            if (pd.api.types.is_float_dtype(df_main_period[col])):
                if col in list_sign_ta:
                    df_main_period[col] = df_main_period[col].round(self.round_ta)
                else:
                    df_main_period[col] = self.rnd_step(df_main_period[col], self.step_price)
        for col in df_M1.columns:
            if (pd.api.types.is_float_dtype(df_M1[col])):
                df_M1[col] = self.rnd_step(df_M1[col], self.step_price)

        # Добавление признаков трех предыдущих свечей
        list_previous_candle_sign = ['open', 'high', 'low', 'close', 'volume', 
                                    'EMA_fast', 'EMA_slow', 'VWAP', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 
                                    'real_body', 'is_growing_candle', 'upper_shadow', 'lower_shadow', 
                                    'is_gapped_up', 'is_gapped_down']
        for i in range(1, self.count_previous_candle + 1):
            for sign in list_previous_candle_sign:
                # Pandas при использовании shift на Series типа bool возвращает object, 
                # добавляем дополнительное преобразование
                if pd.api.types.is_bool_dtype( df_main_period[sign]):
                    df_main_period[sign + '_prev_' + str(i)] = df_main_period[sign].shift(i).convert_dtypes()
                else:
                    df_main_period[sign + '_prev_' + str(i)] = df_main_period[sign].shift(i)
        df_main_period['is_growing_candle'] = df_main_period['is_growing_candle'].convert_dtypes()
        df_main_period['is_gapped_up'] = df_main_period['is_gapped_up'].convert_dtypes()
        df_main_period['is_gapped_down'] = df_main_period['is_gapped_down'].convert_dtypes()

        # Округление вверх до ближайших 15 минут
        df_M1['min_to_next'] = df_M1.index.ceil(self.main_period) - df_M1.index
        df_M1['min_to_next'] = df_M1['min_to_next'].dt.components['minutes']

        # Сокращение используемой памяти
        df_M1 = df_M1.astype({col: 'float32' for col in df_M1.select_dtypes(include='float64').columns})
        df_main_period = df_main_period.astype({col: 'float32' for col in df_main_period.select_dtypes(include='float64').columns})

        # Приведение индексов к реальному времени формирования данных
        df_M1_real_time = df_M1.copy()
        df_main_period_real_time = df_main_period.copy()

        df_M1_real_time.index = df_M1_real_time.index + pd.DateOffset(minutes=1)
        df_main_period_real_time.index = df_main_period_real_time.index + pd.DateOffset(minutes=self.main_period_int)

        # Сортировка по индексам
        df_M1_real_time = df_M1_real_time.sort_index()
        df_main_period_real_time = df_main_period_real_time.sort_index()

        # Объединение
        df_result = pd.merge_asof(df_M1_real_time, 
                                df_main_period_real_time,
                                left_index=True,
                                right_index=True,
                                direction='backward')

        # Сокращение используемой памяти
        for col in df_result.columns:
            if col.find('volume') >= 0:
                df_result[col] = df_result[col].fillna(0)
                df_result[col] = df_result[col].astype('int32')
            if col == 'min_to_next':
                df_result[col] = df_result[col].astype('int32')

        # Кодирование признаков
        day_of_week_bin = self.models['encoder_day_of_week'].transform(df_result['day_of_week'])

        df_result = pd.concat([df_result, day_of_week_bin], axis=1)
        df_result.drop(columns=['day_of_week'],
                    axis=1,
                    inplace=True)

        # Оптимизация типов
        for col in df_result.columns:
            if col.find('day_of_week_') >= 0:
                df_result[col] = df_result[col].astype('boolean')

        # Список признаков, исключаемых из масштабирования
        exclude_futures = ['volume_x', 'volume_y', 'RSI', 'is_growing_candle', 'is_gapped_up', 'is_gapped_down', 
                           'volume_prev_1', 'RSI_prev_1', 'is_growing_candle_prev_1', 'is_gapped_up_prev_1', 'is_gapped_down_prev_1',
                           'volume_prev_2', 'RSI_prev_2', 'is_growing_candle_prev_2', 'is_gapped_up_prev_2', 'is_gapped_down_prev_2',
                           'volume_prev_3', 'RSI_prev_3', 'is_growing_candle_prev_3', 'is_gapped_up_prev_3', 'is_gapped_down_prev_3', 
                           'day_of_week_Sunday', 'day_of_week_Monday', 'day_of_week_Tuesday', 'day_of_week_Wednesday', 
                           'day_of_week_Thursday', 'day_of_week_Friday', 'day_of_week_Saturday', 'min_to_next', 'close_x']
        without_div = ['MACD', 'MACD_signal', 'MACD_hist', 'real_body', 'upper_shadow', 'lower_shadow',
                       'MACD_prev_1', 'MACD_signal_prev_1', 'MACD_hist_prev_1', 'real_body_prev_1', 
                       'is_growing_candle_prev_1', 'upper_shadow_prev_1', 'lower_shadow_prev_1',
                       'MACD_prev_2', 'MACD_signal_prev_2', 'MACD_hist_prev_2', 'real_body_prev_2', 
                       'is_growing_candle_prev_2', 'upper_shadow_prev_2', 'lower_shadow_prev_2',
                       'MACD_prev_3', 'MACD_signal_prev_3', 'MACD_hist_prev_3', 'real_body_prev_3', 
                       'is_growing_candle_prev_3', 'upper_shadow_prev_3', 'lower_shadow_prev_3']

        # Масштабирование относительно цены закрытия на минутном таймфрейме
        for col in df_result.columns:
            if col not in exclude_futures:
                if col in without_div:
                    df_result[col] = (df_result[col] / df_result['close_x'] * 100).round(3)
                else:
                    df_result[col] = ((df_result[col] - df_result['close_x']) / df_result['close_x'] * 100).round(3)
        close_x = df_result.iloc[-1]['close_x']
        # Удаление признака close_x
        df_result.drop(columns=['close_x'], 
                    axis=1, 
                    inplace=True)

        columns = ['open_x', 'high_x', 'low_x', 'volume_x', 'min_to_next', 'high_y', 'low_y',
                   'volume_y', 'VWAP', 'RSI', 'MACD_signal', 'MACD_hist', 'real_body',
                   'is_growing_candle', 'upper_shadow', 'long_ext_1.000', 'long_ext_2.000',
                   'round_price_up', 'is_gapped_up', 'is_gapped_down', 'volume_prev_1',
                   'real_body_prev_1', 'is_growing_candle_prev_1', 'upper_shadow_prev_1',
                   'is_gapped_up_prev_1', 'is_gapped_down_prev_1', 'low_prev_2',
                   'volume_prev_2', 'real_body_prev_2', 'is_growing_candle_prev_2',
                   'upper_shadow_prev_2', 'is_gapped_up_prev_2', 'is_gapped_down_prev_2',
                   'volume_prev_3', 'RSI_prev_3', 'real_body_prev_3',
                   'is_growing_candle_prev_3', 'upper_shadow_prev_3',
                   'is_gapped_up_prev_3', 'is_gapped_down_prev_3', 'day_of_week_Sunday',
                   'day_of_week_Monday', 'day_of_week_Tuesday', 'day_of_week_Wednesday',
                   'day_of_week_Thursday', 'day_of_week_Friday', 'day_of_week_Saturday']
        df_result = df_result[columns]

        pred_to_up = self.models['model_to_up'].predict(df_result.tail(1))
        pred_to_down = self.models['model_to_down'].predict(df_result.tail(1))
        pred_high = (self.models['model_high'].predict(df_result.tail(1))/100 * close_x + close_x)
        pred_low = (self.models['model_low'].predict(df_result.tail(1))/100 * close_x + close_x)
        pred_high = self.rnd_step(pred_high, self.step_price)
        pred_low = self.rnd_step(pred_low, self.step_price)
        
        return pred_to_up, pred_to_down, pred_high, pred_low
    #===============================================================================


if __name__ == "__main__":
    df_test = pd.read_csv('./data/test.txt')
    # Переименование столбцов
    df_test.rename(columns={'<OPEN>': 'open',
                            '<HIGH>': 'high',
                            '<LOW>': 'low',
                            '<CLOSE>': 'close',
                            '<VOL>': 'volume'}, 
                inplace=True)
    # Преобразование времени свечей
    df_test['dt'] = df_test['<DATE>'].astype(str) + ' ' + df_test['<TIME>'].astype(str)
    df_test['dt'] = pd.to_datetime(df_test['dt'], format="%Y%m%d %H%M%S")
    
    gp = predictModels()
    pred_to_up, pred_to_down, pred_high, pred_low = gp.getPredict(df_test)
    #print(f'current close: {close_x:.2f}')
    print(f'to_up - to_down: {pred_to_up} - {pred_to_down}')
    print(f'high: {pred_high}')
    print(f'low: {pred_low}')