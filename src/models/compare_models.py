"""Сравнение моделей CatBoost: без эмбеддингов, с текстовыми, с фото-эмбеддингами."""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_best_available_data():
    """Загружает лучший доступный датасет."""
    
    # Приоритет: parquet с эмбеддингами > обычный parquet > csv
    candidates = [
        ("data/processed/cian_with_text_emb.parquet", "text"),
        ("data/processed/cian_with_img_emb.parquet", "image"),
        ("data/processed/cian_clean.parquet", "base"),
        ("data/raw/cian_final.csv", "base"),
        ("data/raw/cian_with_photos.csv", "base"),
    ]
    
    for path, dtype in candidates:
        full_path = Path(path)
        if full_path.exists():
            logger.info(f" Загружаем: {path}")
            if full_path.suffix == '.parquet':
                df = pd.read_parquet(full_path)
            else:
                df = pd.read_csv(full_path)
            return df, dtype, path
    
    logger.error(" Не найден ни один файл с данными!")
    return None, None, None


def get_features(df, use_text_emb=False, use_img_emb=False):
    """Возвращает список признаков в зависимости от выбора."""
    
    base_features = []
    
    # Базовые числовые признаки
    numeric_cols = ['total_meters', 'floor', 'floors_total', 'rooms_count']
    for col in numeric_cols:
        if col in df.columns:
            base_features.append(col)
    
    # Добавляем производные признаки
    if 'price' in df.columns and 'total_meters' in df.columns:
        df['price_per_m2'] = df['price'] / df['total_meters']
        base_features.append('price_per_m2')
    
    if 'floor' in df.columns and 'floors_total' in df.columns:
        df['floor_ratio'] = df['floor'] / df['floors_total']
        base_features.append('floor_ratio')
    
    features = base_features.copy()
    
    # Добавляем текстовые эмбеддинги
    if use_text_emb:
        text_cols = [c for c in df.columns if c.startswith('text_emb_')]
        if text_cols:
            features.extend(text_cols)
            logger.info(f"    Текстовые эмбеддинги: {len(text_cols)} колонок")
        else:
            logger.warning("   ⚠️ Текстовые эмбеддинги не найдены")
    
    # Добавляем фото-эмбеддинги
    if use_img_emb:
        img_cols = [c for c in df.columns if c.startswith('img_emb_')]
        if img_cols:
            features.extend(img_cols)
            logger.info(f"    Фото-эмбеддинги: {len(img_cols)} колонок")
        else:
            logger.warning("   ⚠️ Фото-эмбеддинги не найдены")
    
    return features


def train_and_evaluate(df, features, model_name: str):
    """Обучает модель и возвращает метрики."""
    
    X = df[features]
    y = df['price']
    
    # Убираем пропуски
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    if len(X) < 50:
        logger.warning(f" Слишком мало данных для {model_name}: {len(X)} записей")
        return None
    
    # Разделяем
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Параметры модели (упрощённые для небольшого датасета)
    params = {
        'iterations': 300,
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 3,
        'random_seed': 42,
        'verbose': False,
        'early_stopping_rounds': 30,
    }
    
    # Обучаем
    start_time = time.time()
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
    train_time = time.time() - start_time
    
    # Предсказания
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Метрики
    metrics = {
        'model_name': model_name,
        'n_features': len(features),
        'n_samples': len(X),
        'train_time': round(train_time, 2),
        'train_mae': float(mean_absolute_error(y_train, y_pred_train)),
        'test_mae': float(mean_absolute_error(y_test, y_pred_test)),
        'train_rmse': float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
        'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        'train_r2': float(r2_score(y_train, y_pred_train)),
        'test_r2': float(r2_score(y_test, y_pred_test)),
    }
    
    # Относительная ошибка
    avg_price = y.mean()
    metrics['test_mae_percent'] = (metrics['test_mae'] / avg_price) * 100
    
    return metrics, model


def compare_models():
    """Сравнивает три модели."""
    
    # Загружаем данные
    df, data_type, data_path = load_best_available_data()
    if df is None:
        return
    
   
    print("СРАВНЕНИЕ МОДЕЛЕЙ CATBOOST")
   
    print(f" Источник данных: {data_path}")
    print(f" Всего записей: {len(df)}")
    print(f" Средняя цена: {df['price'].mean():,.0f} ₽")
    
    # Проверяем наличие эмбеддингов
    has_text_emb = any(c.startswith('text_emb_') for c in df.columns)
    has_img_emb = any(c.startswith('img_emb_') for c in df.columns)
    
    print(f"\n📦 Доступные признаки:")
    print(f"   Базовые: total_meters, floor, floors_total, rooms_count")
    print(f"   Текстовые эмбеддинги: {'✅' if has_text_emb else '❌'}")
    print(f"   Фото-эмбеддинги: {'✅' if has_img_emb else '❌'}")
    
    results = []
    models_to_test = []
    
    # Всегда тестируем базовую модель
    models_to_test.append(("БАЗОВАЯ", False, False))
    
    # Если есть текстовые эмбеддинги
    if has_text_emb:
        models_to_test.append(("С ТЕКСТОВЫМИ ЭМБЕДДИНГАМИ", True, False))
    
    # Если есть фото-эмбеддинги
    if has_img_emb:
        models_to_test.append(("С ФОТО-ЭМБЕДДИНГАМИ", False, True))
    
    # Если есть и те, и другие
    if has_text_emb and has_img_emb:
        models_to_test.append(("СО ВСЕМИ ЭМБЕДДИНГАМИ", True, True))
    
  
    print("ОБУЧЕНИЕ МОДЕЛЕЙ")
   
    
    for name, use_text, use_img in models_to_test:
        print(f"\n🔹 {name}")
        
        features = get_features(df, use_text_emb=use_text, use_img_emb=use_img)
        print(f"   Признаков: {len(features)}")
        
        result = train_and_evaluate(df, features, name)
        
        if result:
            metrics, model = result
            results.append(metrics)
            
            # Сохраняем модель
            model_dir = Path("models")
            model_dir.mkdir(exist_ok=True)
            filename = f"catboost_{name.lower().replace(' ', '_')}.cbm"
            model.save_model(model_dir / filename)
            print(f"    Сохранена: {filename}")
    
    # Сравнительный анализ
 
    print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ")
   
    comparison = pd.DataFrame(results)
    comparison = comparison[[
        'model_name', 'n_features', 'test_mae', 'test_mae_percent', 
        'test_r2', 'train_time'
    ]]
    comparison.columns = [
        'Модель', 'Признаков', 'MAE (₽)', 'MAE (%)', 'R²', 'Время (с)'
    ]
    comparison['MAE (₽)'] = comparison['MAE (₽)'].apply(lambda x: f"{x:,.0f}")
    comparison['MAE (%)'] = comparison['MAE (%)'].apply(lambda x: f"{x:.1f}%")
    comparison['R²'] = comparison['R²'].apply(lambda x: f"{x:.3f}")
    
    print(comparison.to_string(index=False))
    
    # Вывод вердикта

    print("ВЕРДИКТ")

    
    base_mae = None
    for r in results:
        if r['model_name'] == "БАЗОВАЯ":
            base_mae = r['test_mae']
            break
    
    for r in results:
        if r['model_name'] != "БАЗОВАЯ" and base_mae:
            improvement = ((base_mae - r['test_mae']) / base_mae) * 100
            if improvement > 0:
                print(f" {r['model_name']}: ошибка МЕНЬШЕ на {improvement:.1f}%")
            elif improvement < 0:
                print(f" {r['model_name']}: ошибка БОЛЬШЕ на {abs(improvement):.1f}%")
            else:
                print(f" {r['model_name']}: ошибка не изменилась")
    
    # Сохраняем сравнение
    output_path = Path("reports/model_comparison.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)
    print(f"\n📁 Сравнение сохранено: {output_path}")
    
    return results


if __name__ == "__main__":
    compare_models()