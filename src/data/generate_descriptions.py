"""Генерация искусственных описаний для объявлений."""

import pandas as pd
from pathlib import Path


def generate_description(row):
    """Генерирует правдоподобное описание на основе данных."""
    
    # Извлекаем данные
    rooms = int(row.get('rooms_count', 1)) if pd.notna(row.get('rooms_count')) else 1
    price = float(row.get('price', 0)) if pd.notna(row.get('price')) else 0
    area = float(row.get('total_meters', 0)) if pd.notna(row.get('total_meters')) else 0
    floor = int(row.get('floor', 0)) if pd.notna(row.get('floor')) else 0
    floors_total = int(row.get('floors_total', 0)) if pd.notna(row.get('floors_total')) else 0
    metro = str(row.get('metro', 'метро')) if pd.notna(row.get('metro')) else 'метро'
    district = str(row.get('district', 'районе')) if pd.notna(row.get('district')) else 'районе'
    
    # Округляем цену до миллионов
    price_million = price / 1_000_000
    
    # Строим описание
    description = f"Продаётся {rooms}-комнатная квартира в районе {district}. "
    description += f"Общая площадь: {area:.1f} м². "
    description += f"Расположение: {floor} этаж из {floors_total}. "
    description += f"Ближайшее метро: {metro}. "
    description += f"Стоимость: {price_million:.1f} млн ₽. "
    description += "Квартира требует косметического ремонта. "
    description += "Хорошая транспортная доступность, развитая инфраструктура. "
    description += "Звоните для просмотра!"
    
    return description


def main():
    """Генерирует описания для всех объявлений."""
    
    # Загружаем данные с фото
    input_file = Path("data/raw/cian_with_photos.csv")
    output_file = Path("data/raw/cian_final.csv")
    
    df = pd.read_csv(input_file)
    print(f"Загружено {len(df)} объявлений")
    
    # Генерируем описания
    df['description'] = df.apply(generate_description, axis=1)
    
    # Сохраняем финальный датасет
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"\n✅ Сгенерировано описаний: {len(df)}")
    print(f"📁 Сохранено: {output_file}")
    
    # Показываем пример
    print("\n📝 ПРИМЕР ОПИСАНИЯ:")
    print(df['description'].iloc[0])
    print("\n📸 Фото есть у:", df['images_urls'].notna().sum(), "объявлений")


if __name__ == "__main__":
    main()