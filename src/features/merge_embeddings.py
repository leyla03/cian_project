"""Объединение текстовых и фото-эмбеддингов в один датасет."""

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_all_embeddings():
    """Объединяет текстовые и фото-эмбеддинги."""
    
    # Загружаем базовый датасет
    base_df = pd.read_parquet("data/processed/cian_clean.parquet")
    logger.info(f"Базовый датасет: {len(base_df)} записей")
    
    # Загружаем текстовые эмбеддинги (у них уже есть offer_id)
    text_df = pd.read_parquet("data/processed/cian_with_text_emb.parquet")
    logger.info(f"Текстовые эмбеддинги: {len(text_df)} записей")
    
    # Загружаем фото-эмбеддинги
    img_df = pd.read_parquet("data/processed/cian_with_img_emb.parquet")
    img_cols = [c for c in img_df.columns if c.startswith('img_emb_')]
    logger.info(f"Фото-эмбеддинги: {len(img_df)} записей, {len(img_cols)} колонок")
    
    # Объединяем: сначала текстовые с базовыми
    merged_df = text_df.copy()
    
    # Добавляем фото-эмбеддинги (только для тех записей, где они есть)
    for col in img_cols:
        merged_df[col] = None
    
    for idx, row in img_df.iterrows():
        offer_id = row['offer_id']
        if offer_id in merged_df['offer_id'].values:
            for col in img_cols:
                merged_df.loc[merged_df['offer_id'] == offer_id, col] = row[col]
    
    has_img = merged_df[img_cols[0]].notna().sum() if img_cols else 0
    logger.info(f"Фото-эмбеддинги есть у {has_img} записей")
    
    # Сохраняем
    output_path = Path("data/processed/cian_all_embeddings.parquet")
    merged_df.to_parquet(output_path, index=False)
    logger.info(f"✅ Сохранено: {output_path}")
    logger.info(f"   Всего колонок: {len(merged_df.columns)}")
    logger.info(f"   Размер: {output_path.stat().st_size / 1024:.1f} KB")
    
    return merged_df


if __name__ == "__main__":
    merge_all_embeddings()