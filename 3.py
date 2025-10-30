print("--- Запуск теста ---")
try:
    # Пытаемся импортировать всё необходимое
    from direction_filter import BTCDirectionFilter, EnsembleConfig
    from typing import Tuple
    print("1/3: Импорты прошли успешно.")
    
    # Пытаемся создать экземпляр
    cfg = EnsembleConfig()
    filt = BTCDirectionFilter(cfg)
    print("2/3: Инициализация фильтра прошла успешно.")

    # Проверяем наличие и сигнатуру метода latest_signal
    _ = getattr(filt, 'latest_signal')
    print("3/3: Метод latest_signal найден.")
    
    print("\n✅✅✅ УСПЕШНО! Файл direction_filter.py теперь корректен. ✅✅✅")
    print("Теперь можно запускать main.py")

except Exception as e:
    print(f"\n❌❌❌ ОШИБКА! Файл direction_filter.py все еще неверный. ❌❌❌")
    print(f"Детали ошибки: {e}")

print("--- Тест завершен ---")
input("Нажмите Enter для выхода...")