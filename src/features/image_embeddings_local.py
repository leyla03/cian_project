"""Эмбеддинги фото через CLIP (локально, без S3)."""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sentence_transformers import SentenceTransformer
from PIL import Image
import requests
from io import BytesIO
import time
import ast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Модель CLIP через sentence-transformers
MODEL_NAME = "clip-ViT-B-32"  # размерность 512


def load_model():
    """Загружает CLIP модель (первый раз скачает ~600MB)."""
    logger.info(f"📥 Загружаем модель {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    logger.info("✅ Модель загружена")
    return model


def parse_image_urls(urls_str: str) -> list:
    """Превращает строку со списком URL в список."""
    if not urls_str or pd.isna(urls_str):
        return []
    try:
        if urls_str.startswith('['):
            return ast.literal_eval(urls_str)
        else:
            return [urls_str]
    except:
        return []


def get_image_embedding(model, image_url: str) -> np.ndarray:
    """Получает эмбеддинг для одного изображения по URL."""
    try:
        response = requests.get(image_url, timeout=10)
        img = Image.open(BytesIO(response.content))
        # CLIP ожидает RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        embedding = model.encode(img)
        return embedding
    except Exception as e:
        logger.warning(f"Ошибка загрузки {image_url[:50]}...: {e}")
        return None


def add_image_embeddings(
    input_path: Path,
    output_path: Path,
    max_offers: int = None
):
    """Добавляет эмбеддинги фото как новые колонки."""
    
    if input_path.suffix == '.parquet':
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)
    
    logger.info(f"Загружено {len(df)} объявлений")
    
    if max_offers and len(df) > max_offers:
        df = df.head(max_offers)
        logger.info(f"Обрабатываем {max_offers} объявлений")
    
    model = load_model()
    embedding_dim = 512  # для clip-ViT-B-32
    
    # Создаём колонки для эмбеддингов
    for dim in range(embedding_dim):
        df[f'img_emb_{dim}'] = np.nan
    
    success_count = 0
    
    for idx, row in df.iterrows():
        logger.info(f"[{idx+1}/{len(df)}] Обработка...")
        
        # Пробуем получить URL фото
        urls_str = row.get('images_urls', '')
        if not urls_str:
            urls_str = row.get('s3_images_uris', '')
        
        urls = parse_image_urls(urls_str)
        
        if not urls:
            continue
        
        # Берём первое фото
        first_url = urls[0]
        
        # Если это S3 URI, пробуем заменить на HTTP (если знаем эндпоинт)
        if first_url.startswith('s3://'):
            logger.warning(f"S3 URI требует MinIO, пропускаем: {first_url[:50]}")
            continue
        
        embedding = get_image_embedding(model, first_url)
        
        if embedding is not None:
            for dim, val in enumerate(embedding):
                df.at[idx, f'img_emb_{dim}'] = val
            success_count += 1
            logger.info(f"   ✅ Эмбеддинг получен")
        else:
            logger.warning(f"   ⚠️ Не удалось получить эмбеддинг")
        
        time.sleep(0.5)  # Пауза
    
    # Сохраняем
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == '.parquet':
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)
    
    logger.info(f"✅ Сохранено в {output_path}")
    logger.info(f"   Эмбеддинги получены для {success_count}/{len(df)} объявлений")
    
    return df


def main():
    add_image_embeddings(
        input_path=Path("data/processed/cian_clean.parquet"),
        output_path=Path("data/processed/cian_with_img_emb.parquet"),
        max_offers=30  # Начни с 30 для теста
    )


if __name__ == "__main__":
    main()