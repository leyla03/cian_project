"""Сравнение всех моделей: базовая, с текстом, с фото, со всеми."""

import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(df, features, name):
    """Обучает модель и возвращает метрики."""
    
    X = df[features]
    y = df['price']
    
    # Убираем пропуски
    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_mask]
    y = y[valid_mask]
    
    if len(X) < 20:
        return None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    params = {
        'iterations': 200,
        'learning_rate': 0.05,
        'depth': 6,
        'random_seed': 42,
        'verbose': False
    }
    
    start = time.time()
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'model': name,
        'n_features': len(features),
        'n_samples': len(X),
        'test_mae': mean_absolute_error(y_test, y_pred),
        'test_r2': r2_score(y_test, y_pred),
        'train_time': round(train_time, 2)
    }
    
    return metrics, model


def main():
    # Загружаем данные
    df = pd.read_parquet("data/processed/cian_all_embeddings.parquet")
    
    # Определяем колонки
    base_features = ['total_meters', 'floor', 'floors_total', 'rooms_count']
    base_features = [c for c in base_features if c in df.columns]
    
    text_cols = [c for c in df.columns if c.startswith('text_emb_')]
    img_cols = [c for c in df.columns if c.startswith('img_emb_')]
    
    print("="*60)
    print("СРАВНЕНИЕ МОДЕЛЕЙ")
    print("="*60)
    print(f"Всего записей: {len(df)}")
    print(f"Средняя цена: {df['price'].mean():,.0f} ₽")
    print(f"Базовые признаки: {len(base_features)}")
    print(f"Текстовые эмбеддинги: {len(text_cols)}")
    print(f"Фото-эмбеддинги: {len(img_cols)}")
    
    results = []
    
    # 1. Базовая модель
    print("\n🔹 БАЗОВАЯ МОДЕЛЬ")
    if base_features:
        res, model = train_model(df, base_features, "Базовая")
        if res:
            results.append(res)
            print(f"   MAE: {res['test_mae']:,.0f} ₽")
            print(f"   R²: {res['test_r2']:.3f}")
            # Сохраняем
            model.save_model("models/catboost_base.cbm")
    
    # 2. Только текстовые эмбеддинги
    if text_cols:
        print("\n🔹 ТОЛЬКО ТЕКСТОВЫЕ ЭМБЕДДИНГИ")
        res, model = train_model(df, text_cols, "Только текст")
        if res:
            results.append(res)
            print(f"   MAE: {res['test_mae']:,.0f} ₽")
            print(f"   R²: {res['test_r2']:.3f}")
    
    # 3. Только фото-эмбеддинги (если есть записи с ними)
    if img_cols:
        df_with_img = df[df[img_cols[0]].notna()]
        if len(df_with_img) > 20:
            print(f"\n🔹 ТОЛЬКО ФОТО-ЭМБЕДДИНГИ ({len(df_with_img)} записей)")
            res, model = train_model(df_with_img, img_cols, "Только фото")
            if res:
                results.append(res)
                print(f"   MAE: {res['test_mae']:,.0f} ₽")
                print(f"   R²: {res['test_r2']:.3f}")
    
    # 4. Базовые + текстовые
    if base_features and text_cols:
        print("\n🔹 БАЗОВЫЕ + ТЕКСТОВЫЕ")
        res, model = train_model(df, base_features + text_cols, "База+текст")
        if res:
            results.append(res)
            print(f"   MAE: {res['test_mae']:,.0f} ₽")
            print(f"   R²: {res['test_r2']:.3f}")
            model.save_model("models/catboost_base_text.cbm")
    
    # 5. Все вместе
    if base_features and text_cols and img_cols:
        all_cols = base_features + text_cols + img_cols
        df_all = df.dropna(subset=all_cols)
        if len(df_all) > 20:
            print(f"\n🔹 ВСЕ ПРИЗНАКИ ({len(df_all)} записей)")
            res, model = train_model(df_all, all_cols, "Все признаки")
            if res:
                results.append(res)
                print(f"   MAE: {res['test_mae']:,.0f} ₽")
                print(f"   R²: {res['test_r2']:.3f}")
    
    # Сводная таблица
    print("\n" + "="*60)
    print("СВОДНАЯ ТАБЛИЦА")
    print("="*60)
    
    summary = pd.DataFrame(results)
    summary['test_mae_million'] = summary['test_mae'] / 1_000_000
    summary = summary[['model', 'n_features', 'n_samples', 'test_mae_million', 'test_r2', 'train_time']]
    summary.columns = ['Модель', 'Признаков', 'Образцов', 'MAE (млн ₽)', 'R²', 'Время (с)']
    
    print(summary.to_string(index=False))
    
    # Сохраняем
    summary.to_csv("reports/full_comparison.csv", index=False)
    print("\n✅ Сохранено: reports/full_comparison.csv")


if __name__ == "__main__":
    main()