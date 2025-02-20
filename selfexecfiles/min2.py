import requests
import pandas as pd
import time

# === 1. Конфигурация API и валютной пары ===
API_KEY = "8DTDA7A6ZLR119ED"  # Ключ Alpha Vantage
FROM_SYMBOL = "EUR"  # Базовая валюта
TO_SYMBOL = "USD"  # Котируемая валюта


# === 2. Функция для загрузки котировок ===
def get_candles():
    """
    Загружает дневные данные валютных пар через Alpha Vantage API.
    Возвращает DataFrame с OHLC (Open, High, Low, Close) значениями.
    """
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

    # Проверяем, есть ли данные
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


# === 3. Функция генерации торговых сигналов ===
def signal_generator(df: pd.DataFrame) -> int:
    """
    Анализирует последние две свечи и определяет торговый сигнал.
    Возвращает:
    0 - Нет сигнала
    1 - Медвежий сигнал (Sell)
    2 - Бычий сигнал (Buy)
    """
    if len(df) < 2:  # Нужно минимум 2 свечи
        return 0

    open_ = df["Open"].iloc[-1]
    close = df["Close"].iloc[-1]
    prev_open = df["Open"].iloc[-2]
    prev_close = df["Close"].iloc[-2]

    # Медвежий (Sell) сигнал
    if (open_ > close) and (prev_open < prev_close) and (close < prev_open) and (open_ >= prev_close):
        return 1

    # Бычий (Buy) сигнал
    elif (open_ < close) and (prev_open > prev_close) and (close > prev_open) and (open_ <= prev_close):
        return 2

    return 0  # Нет четкого сигнала


# === 4. Основная функция анализа рынка ===
def trading_job():
    """
    Получает котировки, анализирует рынок и выводит сигнал.
    В будущем можно добавить реальную торговлю.
    """
    df = get_candles()
    if df is None:
        return

    signal = signal_generator(df)

    # Вывод последних 2 свечей для анализа
    print(df.tail(2))

    # Вывод торгового сигнала
    print(f"Сигнал: {signal}")  # 0 - нет сигнала, 1 - Sell, 2 - Buy

    # === Форекс-комментарии (реальные покупки/продажи можно добавить тут) ===
    """
    if signal == 1:
        print("Открытие короткой позиции (Sell)")
        # Код для продажи через брокера
    elif signal == 2:
        print("Открытие длинной позиции (Buy)")
        # Код для покупки через брокера
    """


# === 5. Основной цикл работы (раз в 24 часа) ===
if __name__ == "__main__":
    while True:
        trading_job()
        time.sleep(86400)  # 24 часа (86400 секунд)
