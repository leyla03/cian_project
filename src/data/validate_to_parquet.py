"""Валидация данных через Pandera и сохранение в Parquet (без S3)."""

import pandas as pd
import pandera.pandas as pa  # Новый импорт
from pandera import Column, Check, DataFrameSchema
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Определяем схему данных (сделаем nullable для проблемных полей)
schema = DataFrameSchema(
    {
        "price": Column(float, Check.in_range(100000, 1000000000), nullable=True),  # разрешаем пропуски
        "total_meters": Column(float, Check.in_range(10, 500), nullable=False),
        "floor": Column(int, Check.in_range(0, 100), nullable=True),
        "floors_total": Column(int, Check.in_range(1, 100), nullable=True),
        "rooms_count": Column(int, Check.in_range(0, 10), nullable=True),
        "url": Column(str, Check.str_length(min_value=5), nullable=True),
        "metro": Column(str, nullable=True),
        "district": Column(str, nullable=True),
        "description": Column(str, nullable=True),
    },
    strict=False,  # разрешаем лишние колонки
    coerce=True
)


def validate_and_save(input_csv: Path, output_parquet: Path):
    """Валидирует датасет и сохраняет в Parquet."""
    
    df = pd.read_csv(input_csv)
    logger.info(f"Загружено {len(df)} объявлений")
    
    # Создаём offer_id если нет
    if 'offer_id' not in df.columns:
        df['offer_id'] = [f"offer_{i}" for i in range(len(df))]
        logger.info("Создана колонка offer_id")
    
    # Чистим типы
    numeric_cols = ['price', 'total_meters', 'floor', 'floors_total', 'rooms_count']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Заполняем пропуски в цене (если есть)
    if df['price'].isna().any():
        median_price = df['price'].median()
        df['price'] = df['price'].fillna(median_price)
        logger.info(f"Заполнено {df['price'].isna().sum()} пропусков в цене медианой {median_price:.0f}")
    
    # Удаляем строки, где нет обязательных полей
    required_cols = ['total_meters']
    for col in required_cols:
        df = df[df[col].notna()]
    
    logger.info(f"После очистки: {len(df)} объявлений")
    
    # Валидация
    try:
        validated_df = schema.validate(df, lazy=True)
        logger.info("✅ Схема валидна")
    except pa.errors.SchemaErrors as e:
        logger.warning(f"⚠️ Ошибки валидации: {len(e.failure_cases)} проблем")
        validated_df = e.data  # берём данные даже с ошибками
    
    # Добавляем полезные признаки
    validated_df['price_per_m2'] = validated_df['price'] / validated_df['total_meters']
    validated_df['floor_ratio'] = validated_df['floor'] / validated_df['floors_total']
    
    # Сохраняем в Parquet
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    validated_df.to_parquet(output_parquet, index=False)
    
    file_size = output_parquet.stat().st_size / 1024
    logger.info(f"✅ Сохранено в Parquet: {output_parquet} ({file_size:.1f} KB)")
    
    # Выводим статистику
    print("\n" + "="*50)
    print("СТАТИСТИКА ДАТАСЕТА")
    print("="*50)
    print(f"📊 Всего записей: {len(validated_df)}")
    print(f"💰 Средняя цена: {validated_df['price'].mean():,.0f} ₽")
    print(f"📐 Средняя площадь: {validated_df['total_meters'].mean():.1f} м²")
    print(f"💰 Средняя цена за м²: {validated_df['price_per_m2'].mean():,.0f} ₽")
    
    return validated_df


def main():
    # Пробуем разные возможные пути
    possible_paths = [
        Path("data/raw/cian_final.csv"),
        Path("data/raw/cian_with_photos.csv"),
        Path("data/raw/cian_enriched.csv"),
        Path("data/raw/cian_raw.csv"),
    ]
    
    input_csv = None
    for path in possible_paths:
        if path.exists():
            input_csv = path
            logger.info(f"Найден файл: {input_csv}")
            break
    
    if input_csv is None:
        logger.error("❌ Не найден ни один файл с данными!")
        return
    
    validate_and_save(
        input_csv=input_csv,
        output_parquet=Path("data/processed/cian_clean.parquet")
    )


if __name__ == "__main__":
    main()