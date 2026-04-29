"""Создание текстовых эмбеддингов для описаний квартир и векторной базы."""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_text_embeddings(data_path: Path, output_path: Path):
    """
    Создаёт эмбеддинги для текстовых описаний и сохраняет их как колонки.
    """
    # Загружаем данные
    if data_path.suffix == '.parquet':
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
    
    logger.info(f"Загружено {len(df)} объявлений")
    
    # Загружаем модель для текстов
    logger.info("📥 Загружаем модель для текстов (all-MiniLM-L6-v2)...")
    text_model = SentenceTransformer("all-MiniLM-L6-v2")  # размерность 384, быстро
    logger.info("✅ Модель загружена")
    
    # Подготавливаем тексты для эмбеддингов
    texts = []
    for idx, row in df.iterrows():
        # Берём описание или создаём из параметров
        desc = row.get('description', '')
        if pd.isna(desc) or len(str(desc)) < 20:
            # Генерируем описание на основе данных
            rooms = row.get('rooms_count', 1)
            area = row.get('total_meters', 50)
            floor = row.get('floor', 5)
            metro = row.get('metro', 'метро')
            desc = f"{rooms}-комнатная квартира площадью {area} м², этаж {floor}, рядом с метро {metro}"
        
        texts.append(str(desc))
    
    # Получаем эмбеддинги
    logger.info("🔢 Вычисляем эмбеддинги для текстов...")
    embeddings = text_model.encode(texts, show_progress_bar=True)
    logger.info(f"✅ Получены эмбеддинги размерности {embeddings.shape}")
    
    # Добавляем как колонки
    embedding_dim = embeddings.shape[1]  # 384
    for dim in range(embedding_dim):
        df[f'text_emb_{dim}'] = embeddings[:, dim]
    
    # Сохраняем
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == '.parquet':
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)
    
    logger.info(f"✅ Сохранено в {output_path}")
    
    return df, embeddings


def create_vector_store(data_path: Path, persist_dir: str = "data/vector_db"):
    """
    Создаёт векторную базу ChromaDB для семантического поиска.
    """
    # Загружаем данные
    if data_path.suffix == '.parquet':
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
    
    logger.info(f"Загружено {len(df)} объявлений")
    
    # Подготавливаем тексты
    texts = []
    metadatas = []
    ids = []
    
    for idx, row in df.iterrows():
        # Текст для поиска
        desc = row.get('description', '')
        if pd.isna(desc) or len(str(desc)) < 20:
            rooms = row.get('rooms_count', 1)
            area = row.get('total_meters', 50)
            desc = f"{rooms}-комнатная квартира, {area} м²"
        
        texts.append(str(desc))
        metadatas.append({
            "offer_id": str(row.get('offer_id', f'id_{idx}')),
            "price": float(row.get('price', 0)),
            "total_meters": float(row.get('total_meters', 0)),
            "rooms_count": int(row.get('rooms_count', 0)) if pd.notna(row.get('rooms_count')) else 0,
            "floor": int(row.get('floor', 0)) if pd.notna(row.get('floor')) else 0,
            "url": str(row.get('url', ''))
        })
        ids.append(str(row.get('offer_id', f'id_{idx}')))
    
    # Создаём ChromaDB
    client = chromadb.PersistentClient(path=persist_dir)
    
    # Удаляем старую коллекцию, если есть
    try:
        client.delete_collection("apartments")
    except:
        pass
    
    # Создаём новую с эмбеддингами
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.create_collection(
        name="apartments",
        embedding_function=embedding_fn
    )
    
    # Добавляем данные батчами
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i:i+batch_size],
            documents=texts[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )
        logger.info(f"Добавлено {min(i+batch_size, len(ids))}/{len(ids)}")
    
    logger.info(f"✅ Векторная база создана: {persist_dir}, записей: {collection.count()}")
    
    return client, collection


def search_similar(query: str, collection, k: int = 5):
    """Поиск похожих квартир по текстовому запросу."""
    results = collection.query(query_texts=[query], n_results=k)
    
    similar = []
    for i in range(len(results['ids'][0])):
        similar.append({
            "offer_id": results['ids'][0][i],
            "score": 1 - results['distances'][0][i],
            "price": results['metadatas'][0][i]['price'],
            "total_meters": results['metadatas'][0][i]['total_meters'],
            "url": results['metadatas'][0][i]['url'],
            "document": results['documents'][0][i][:200]
        })
    return similar


def main():
    # 1. Создаём датасет с текстовыми эмбеддингами
    print("\n🔹 Шаг 1: Создание текстовых эмбеддингов")
    df, embeddings = create_text_embeddings(
        data_path=Path("data/processed/cian_clean.parquet"),
        output_path=Path("data/processed/cian_with_text_emb.parquet")
    )
    
    # 2. Создаём векторную базу для поиска
    print("\n🔹 Шаг 2: Создание векторной базы ChromaDB")
    client, collection = create_vector_store(
        data_path=Path("data/processed/cian_clean.parquet"),
        persist_dir="data/vector_db"
    )
    
    # 3. Тестируем поиск
    print("\n🔹 Шаг 3: Тест семантического поиска")
    test_queries = [
        "светлая просторная квартира с хорошим ремонтом",
        "недорогая однокомнатная рядом с метро",
        "квартира с большой кухней и балконом"
    ]
    
    for query in test_queries:
        print(f"\n📝 Запрос: '{query}'")
        results = search_similar(query, collection, k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. Цена: {r['price']:,.0f} ₽, Площадь: {r['total_meters']} м², Сходство: {r['score']:.3f}")


if __name__ == "__main__":
    main()