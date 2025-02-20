import requests
import pandas as pd
import time

API_KEY = "8DTDA7A6ZLR119ED"
FROM_SYMBOL = "EUR"
TO_SYMBOL = "USD"


def get_candles():
    """Загружает дневные данные валютных пар через Alpha Vantage"""
    url = "https://www.alphavantage.co/query"

    params = {
        "function": "FX_DAILY",
        "from_symbol": FROM_SYMBOL,
        "to_symbol": TO_SYMBOL,
        "apikey": API_KEY,
        "outputsize": "compact"  # compact = 100 дней, full = вся история
    }

    response = requests.get(url, params=params)
    data = response.json()

    # Проверяем, есть ли ошибки
    time_series_key = "Time Series FX (Daily)"
    if time_series_key not in data:
        print("Ошибка загрузки данных:", data)
        return None

    # Преобразуем JSON в DataFrame
    df = pd.DataFrame.from_dict(data[time_series_key], orient="index")
    df.columns = ["Open", "High", "Low", "Close"]
    df = df.astype(float)
    df = df.iloc[::-1]  # Переворачиваем порядок (от старых к новым данным)
    return df


def signal_generator(df: pd.DataFrame) -> int:
    """Генерирует сигналы на основе дневных данных"""
    if len(df) < 2:
        return 0

    open_ = df["Open"].iloc[-1]
    close = df["Close"].iloc[-1]
    prev_open = df["Open"].iloc[-2]
    prev_close = df["Close"].iloc[-2]

    if (open_ > close) and (prev_open < prev_close) and (close < prev_open) and (open_ >= prev_close):
        return 1  # Медвежий сигнал (Sell)
    elif (open_ < close) and (prev_open > prev_close) and (close > prev_open) and (open_ <= prev_close):
        return 2  # Бычий сигнал (Buy)
    return 0


def trading_job():
    """Функция для анализа рынка и генерации торговых сигналов"""
    df = get_candles()
    if df is None:
        return

    signal = signal_generator(df)

    print(df.tail(2))  # Вывод последних двух строк
    print(f"Сигнал: {signal}")  # 0 - нет сигнала, 1 - продажа, 2 - покупка


# Запускаем анализ раз в день (24 часа)
while True:
    trading_job()
    time.sleep(86400)  # 24 часа (86400 секунд)
