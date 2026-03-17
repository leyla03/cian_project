import cianparser
import pandas as pd
from datetime import datetime
import time

print("Сбор даных с Циан")


# Настройки сбора
LOCATION = "Москва"           
DEAL_TYPE = "sale"             # Продажа 
ROOMS = (1, 2, 3)              # 1, 2 и 3-комнатные квартиры
START_PAGE = 1                  # Начинаем с 1 страницы
END_PAGE = 15                   # Собираем 15 страниц 

print(f"\nПараметры сбора:")
print(f"   Город: {LOCATION}")
print(f"   Тип: продажа")
print(f"   Комнат: 1, 2, 3")
print(f"   Страницы: {START_PAGE} - {END_PAGE}")


# Создаем парсер
parser = cianparser.CianParser(location=LOCATION)
print("Начинаем сбор данных")
# Собираем данные
start_time = time.time()

data = parser.get_flats(
    deal_type=DEAL_TYPE,
    rooms=ROOMS,
    with_saving_csv=False,  
    additional_settings={
        "start_page": START_PAGE,
        "end_page": END_PAGE
    }
)

end_time = time.time()
minutes = int((end_time - start_time) / 60)
seconds = int((end_time - start_time) % 60)

print(f"\nСбор завершен за {minutes} мин {seconds} сек")
print(f"Собрано объявлений: {len(data)}")

df = pd.DataFrame(data)


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"cian_data_{timestamp}.csv"
df.to_csv(filename, index=False, encoding='utf-8-sig')

df.to_csv("data.csv", index=False, encoding='utf-8-sig')

print(f"\nФайлы сохранены:")
print(f"   {filename} (с временной меткой)")
print(f"   data.csv (основной)")


print(f"\nСтатистика по собранным данным:")
print(f"   Всего объявлений: {len(df)}")
print(f"   Диапазон цен: {df['price'].min():,} - {df['price'].max():,} руб.")


print(f"\nПример собранных данных (первые 5):")

for i, row in df.head(5).iterrows():
    print(f"\n{i+1}. {row.get('url', 'Нет ссылки')}")
    print(f"   Цена: {row.get('price', 'Нет'):,} руб.")
    print(f"   Комнат: {row.get('rooms_count', 'Нет')}")
    print(f"   Площадь: {row.get('total_meters', 'Нет')} м²")
    print(f"   Этаж: {row.get('floor', 'Нет')}/{row.get('floors_count', 'Нет')}")
    print(f"   Метро: {row.get('underground', 'Нет')}")
    print(f"   Адрес: {row.get('street', 'Нет')} {row.get('house_number', '')}")
    print(f"   Автор: {row.get('author', 'Нет')} ({row.get('author_type', 'Нет')})")

