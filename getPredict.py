import numpy as np
import pandas as pd
import pandas_ta_classic as ta
# Получение тональности финансовых текстов
import math
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
# Сериализация/десериализация
import joblib
# Отключение сообщений Hugging Face
from transformers.utils import logging as transformers_logging
transformers_logging.set_verbosity_error()
transformers_logging.disable_progress_bar()
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

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
        # Период соранения позиции в минутах
        self.period_holding_position = 60
        # Количество свечей, обеспечивающих ретроспективность котировок
        self.count_previous_candle = 3
        # Количество свечей, обеспечивающих ретроспективность сообщений
        self.count_previous_messages = 2
        # Количество знаков для округления значений технических индикаторов, 
        # вычисляемых в масштабе, отличном от цены актива
        self.round_ta = 2
        
        # Загрузка моделей из файлов
        self.models = self.loadModelsPKL()
        # Пороги принятия решения
        self.threshold_to_up = 0.52
        self.threshold_to_down = 0.49


    def rnd_step(self, r, sp):
        return (r + 0.5 * sp)//sp * sp


    def fibonacci_extension_levels(self, df: pd.DataFrame, period: int, long_levels: int = 7, short_levels: int = 7) -> pd.DataFrame:
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
        # Загрузка кодировщиков
        encoder_day_of_week = joblib.load('./models/encoder_day_of_week.pkl')
        encoder_tags_comb = joblib.load('./models/encoder_tags_comb.pkl')
        # Загрузка моделей
        model_to_up = joblib.load('./models/model_to_up.pkl')
        model_to_down = joblib.load('./models/model_to_down.pkl')
        model_high = joblib.load('./models/model_high.pkl')
        model_low = joblib.load('./models/model_low.pkl')
        top_tags_comb = joblib.load('./models/top_tags_comb.pkl')
        
        return {'encoder_day_of_week': encoder_day_of_week, 
                'encoder_tags_comb': encoder_tags_comb, 
                'model_to_up': model_to_up, 
                'model_to_down': model_to_down, 
                'model_high': model_high, 
                'model_low': model_low,
                'top_tags_comb': top_tags_comb}


    def getPredict(self,
                   df: pd.DataFrame,
                   df_msgs: pd.DataFrame):
        '''
        Функция получения прогноза с использованием моделей.
        
        :param df: pd.DataFrame с колонками 'dt', 'open', 'high', 'low', 'close', 'volume'
                   Минутные свечи, количество свечей должно обеспечивать рассчет технических
                   индикаторов в масштабе 15-минутных свечей
        :param df_msgs: pd.DataFrame с колонкой 'message' и индексом 'dt' (дата выхода новости)
                        Список новостей канала, количество должно обеспечивать две новости после очистки.
                        Время публикации новости должно быть согласовано со временем свечей 
                        (в одном часовом поясе)
        :return: bool pred_to_up, bool pred_to_down, float pred_high, float pred_low (Прогнозируемые значения),
                 float close_x
        '''
        df = df.copy()
        df_msgs = df_msgs.copy()
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

        # Подготовка сообщений
        reg = r'#\w+'
        df_msgs['tags'] = df_msgs['message'].str.findall(reg)
        df_msgs['text'] = df_msgs['message'].str.replace(r'#\w+', '', regex=True)
        df_msgs['count_tags'] = df_msgs['tags'].apply(lambda x: len(x))
        # Очистка от сообщений с количеством тегов больше 4 и без тега
        mask = (df_msgs['count_tags'] > 0) & (df_msgs['count_tags'] <= 4)
        df_msgs = df_msgs[mask]
        # Очистка от сообщений состоящих только из тегов
        df_msgs['text'] = df_msgs['text'].str.replace(r'[^a-zA-Z0-9а-яА-ЯёЁ]', '', regex=True)
        mask = df_msgs['text'] != ''
        df_msgs = df_msgs[mask]
        df_msgs.drop(columns=['text'], inplace=True)
        # Очистка от напоминающих сообщений
        mask = df_msgs['message'].str.contains('❗️ВПЕРЕДИ', na=False)
        index_del = df_msgs[mask].index
        df_msgs.drop(index=index_del, inplace=True)
        # Выделение признаков из списка тегов
        col_tags = ['#геополитика', '#отчетность', '#прогноз', '#экономика',
                    '#дкп', '#инфляция', '#отчетности', '#золото',
                    '#россия', '#сша', '#китай', '#европа', '#украина', 
                    '#иран', '#британия', '#япония', '#индия', '#германия']
        # Формирование признаков наличия тега из списка в сообщении
        for ct in col_tags:
            df_msgs[ct] = df_msgs['tags'].apply(lambda x: ct in x)
        # Сортировка списков тегов
        df_msgs['tags'] = df_msgs['tags'].apply(lambda x: sorted(x))
        # Формирование признака комбинаций тегов
        df_msgs['tags_comb'] = df_msgs['tags'].apply(lambda x: ''.join(x))
        
        # Получение тональности сообщений
        # Настройка устройства
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("!!!GPU не найден. Работа будет идти на CPU (медленнее).")

        # Загрузка модели и токенизатора
        model_name = "ProsusAI/finbert"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                use_safetensors=True)
        # Перенс модели на устройство
        model.to(device)
        model.eval()  # Режим оценки (отключает dropout и т.д.)
        # Категории тональности сообщений
        labels = ["negative", "neutral", "positive"]

        # Подготовка данных
        texts = df_msgs['message'].tolist()
        # Сохранение индексов
        original_indices = df_msgs['message'].index.tolist()
        # Пакетная обработка (BATCHING) с GPU
        batch_size = 128  # Варианты для GPU: 16, 32, 64, 128. Если Out Of Memory - уменьшать значение.
        n_batches = math.ceil(len(df_msgs) / batch_size)
        all_probs = []

        for i in tqdm(range(n_batches), desc="FinBERT Inference", disable=True):
            batch_texts = texts[i*batch_size:(i+1)*batch_size]
            # Токенизация
            inputs = tokenizer(batch_texts,
                            return_tensors="pt",
                            truncation=True,
                            padding=True,
                            max_length=512)
            # Перенос тензоров на GPU
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
            # Получение вероятности
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs)
            
        # Объединение батчей в один тензор и перенос на CPU для работы с pandas
        all_probs_tensor = torch.cat(all_probs, dim=0)
        all_probs_np = all_probs_tensor.cpu().numpy()

        # Формирование результатов
        pred_ids = np.argmax(all_probs_np, axis=1)
        pred_labels = [labels[i] for i in pred_ids]
        pred_conf = all_probs_np[np.arange(len(pred_ids)), pred_ids]

        # Создание DataFrame с результатами
        df_sent = pd.DataFrame({"prob_neg": all_probs_np[:, 0],
                                "prob_neu": all_probs_np[:, 1],
                                "prob_pos": all_probs_np[:, 2],
                                "sentiment": pred_labels,
                                "confidence": pred_conf})
        # Восстановление исходных индексов
        df_sent.index = original_indices
        # Слияние с исходным датасетом
        df_msgs_full = pd.concat([df_msgs, df_sent], axis=1)
        
        df_msgs_full.drop(columns=['sentiment','confidence', 'message', 'tags', 'count_tags'],
                        inplace=True)
        df_msgs_full['sent_score'] = df_msgs_full['prob_pos'] - df_msgs_full['prob_neg']
        
        # Изменение значения сочетаний тегов на "other" для невошедших в список
        df_msgs_full['tags_comb'] = df_msgs_full['tags_comb'].apply(lambda x: 'other' if x not in self.models['top_tags_comb'] else x)
        # Бинарное кодирование признака
        encoded_tc = self.models['encoder_tags_comb'].transform(df_msgs_full[['tags_comb']])
        df_msgs_full = pd.concat([df_msgs_full, encoded_tc], axis=1)
        # Удаление признака tags_comb
        df_msgs_full.drop(columns=['tags_comb'], inplace=True)
        
        # вычисление факторов на базе котировок
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
        # Выделение числа и месяца
        df_M1['month_day'] = df_M1.index.day
        df_M1['month'] = df_M1.index.month
        df_M1['hour'] = df_M1.index.hour
        
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

        # Кодирование признаков
        day_of_week_bin = self.models['encoder_day_of_week'].transform(df_result['day_of_week'])

        df_result = pd.concat([df_result, day_of_week_bin], axis=1)
        df_result.drop(columns=['day_of_week'], inplace=True)

        # Приведение типов
        for col in df_result.columns:
            if col.find('day_of_week_') >= 0:
                df_result[col] = df_result[col].astype('boolean')
        # Список признаков, исключаемых из масштабирования
        exclude_futures = ['volume_x', 'volume_y', 'RSI', 'is_growing_candle', 'is_gapped_up', 'is_gapped_down', 
                        'volume_prev_1', 'RSI_prev_1', 'is_growing_candle_prev_1', 'is_gapped_up_prev_1', 'is_gapped_down_prev_1',
                        'volume_prev_2', 'RSI_prev_2', 'is_growing_candle_prev_2', 'is_gapped_up_prev_2', 'is_gapped_down_prev_2',
                        'volume_prev_3', 'RSI_prev_3', 'is_growing_candle_prev_3', 'is_gapped_up_prev_3', 'is_gapped_down_prev_3', 
                        'day_of_week_Monday', 'day_of_week_Tuesday', 'day_of_week_Wednesday', 'day_of_week_Thursday', 'day_of_week_Friday',
                        'day_of_week_Saturday', 'day_of_week_Sunday', 'min_to_next', 'close_x', 'predict_to_up', 'predict_to_down',
                        'month_day', 'month', 'hour']
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
        df_result.drop(columns=['close_x'], inplace=True)

        columns = ['open_x', 'high_x', 'low_x', 'volume_x', 'min_to_next', 'month_day',
                   'month', 'hour', 'open_y', 'high_y', 'low_y', 'close_y', 'volume_y',
                   'EMA_slow', 'VWAP', 'MACD_signal', 'MACD_hist', 'real_body',
                   'is_growing_candle', 'upper_shadow', 'long_ext_1.272', 'long_ext_2.000',
                   'long_ext_3.618', 'short_ext_2.618', 'round_price_up', 'is_gapped_up',
                   'is_gapped_down', 'volume_prev_1', 'RSI_prev_1', 'real_body_prev_1',
                   'is_growing_candle_prev_1', 'upper_shadow_prev_1',
                   'is_gapped_up_prev_1', 'is_gapped_down_prev_1', 'open_prev_2',
                   'volume_prev_2', 'real_body_prev_2', 'is_growing_candle_prev_2',
                   'upper_shadow_prev_2', 'is_gapped_up_prev_2', 'is_gapped_down_prev_2',
                   'volume_prev_3', 'RSI_prev_3', 'real_body_prev_3',
                   'is_growing_candle_prev_3', 'upper_shadow_prev_3',
                   'is_gapped_up_prev_3', 'is_gapped_down_prev_3', 'day_of_week_Monday',
                   'day_of_week_Tuesday', 'day_of_week_Wednesday', 'day_of_week_Thursday',
                   'day_of_week_Friday']
        df_result = df_result[columns]
        
        # Объединение датасетов изменения цены активов и сообщений новостного канала
        # Округление индекса датасета сообщений до минуты вниз
        df_msgs_full.index = df_msgs_full.index.floor('min')
        # Копирование времени сообщения из индекса
        df_msgs_full['dt'] = df_msgs_full.index
        
        lst_columns = df_msgs_full.columns.to_list()
        shifted_cols = {}
        # Формирование словаря с предыдущими сообщениями
        for sign in lst_columns:
            # Pandas при использовании shift на Series типа bool возвращает object,
            # добавляем дополнительное преобразование
            base_series = df_msgs_full[sign]
            is_bool = pd.api.types.is_bool_dtype(base_series)
            for i in range(1, self.count_previous_messages + 1):
                s = base_series.shift(i)
                if is_bool:
                    s = s.convert_dtypes()
                shifted_cols[f'{sign}_prev_{i}'] = s
        # Преобразование словаря в Dataframe и объединение
        df_shifts = pd.DataFrame(shifted_cols)
        df_msgs_full = pd.concat([df_msgs_full, df_shifts], axis=1)
        # Формирование признака "минут со времени публикации предыдущей новости"
        df_msgs_full['min_passed_prev_1'] = (df_msgs_full['dt'] - df_msgs_full['dt_prev_1']).dt.total_seconds() // 60
        df_msgs_full['min_passed_prev_2'] = (df_msgs_full['dt'] - df_msgs_full['dt_prev_2']).dt.total_seconds() // 60
        # 
        df_msgs_full.dropna(axis=0, inplace=True)
        df_msgs_full.drop(columns=['dt_prev_1', 'dt_prev_2'], inplace=True)
        df_msgs_full['min_passed_prev_1'].astype(int)
        df_msgs_full['min_passed_prev_2'].astype(int)
        df_msgs_full.reset_index(drop=True, inplace=True)
        # Объединение датасетов
        df_result = df_msgs_full.merge(df_result,
                                       left_on='dt',
                                       right_index=True, # использование индекса df_cleaned_strong_corr как ключ для присоединения
                                       how='left')       # оставляем все сообщения, если свечи нет — NaN
        # Очистка от пропущенных значений со сбросом индекса
        df_result = df_result.dropna(axis=0).reset_index(drop=True)
        # Удаление столбца dt
        df_result.drop(columns=['dt'], inplace=True)
        
        # Получение прогноза
        X_pred = df_result.tail(1)
        # вероятность класса True (положительного класса)
        prob_to_up = self.models['model_to_up'].predict_proba(X_pred)[:, 1]
        prob_to_down = self.models['model_to_down'].predict_proba(X_pred)[:, 1]
        # применение своего threshold
        pred_to_up = prob_to_up >= self.threshold_to_up
        pred_to_down = prob_to_down >= self.threshold_to_down
        pred_high = (np.expm1(self.models['model_high'].predict(X_pred))/100 * close_x + close_x)
        pred_low = (-np.expm1(self.models['model_low'].predict(X_pred))/100 * close_x + close_x)
        pred_high = self.rnd_step(pred_high, self.step_price)
        pred_low = self.rnd_step(pred_low, self.step_price)
        
        return bool(pred_to_up), bool(pred_to_down), pred_high[0], pred_low[0], close_x
    #===============================================================================


if __name__ == "__main__":
    # Подготовка данных для получения прогноза
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
    df_test.drop(columns=['<TICKER>', '<PER>', '<DATE>', '<TIME>'], 
                     inplace=True)
    df_test_msgs = pd.read_csv('./data/test_msg.txt', index_col='dt')
    df_test_msgs.index = pd.to_datetime(df_test_msgs.index)
    # Инициализация класса
    gp = predictModels()
    # Получение прогноза
    pred_to_up, pred_to_down, pred_high, pred_low, close_x = gp.getPredict(df_test, df_test_msgs)
    print(f'current close: {close_x:.2f}')
    print(f'to_up:   {pred_to_up}')
    print(f'to_down: {pred_to_down}')
    print(f'high:    {pred_high:.1f}')
    print(f'low:     {pred_low:.1f}')