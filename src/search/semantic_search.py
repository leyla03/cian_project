"""Семантический поиск похожих квартир через векторную базу."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ApartmentSearch:
    """Класс для поиска похожих квартир."""
    
    def __init__(self, persist_dir: str = "data/vector_db"):
        """Инициализация подключения к векторной базе."""
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_collection("apartments")
        logger.info(f"✅ Подключен к базе, {self.collection.count()} квартир")
    
    def search_by_text(self, query: str, k: int = 5) -> list:
        """
        Поиск похожих квартир по текстовому описанию.
        
        Args:
            query: текст запроса (например, "светлая квартира с видом на город")
            k: количество результатов
            
        Returns:
            список похожих квартир с метаданными
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        
        similar = []
        for i in range(len(results['ids'][0])):
            similar.append({
                "offer_id": results['ids'][0][i],
                "similarity_score": 1 - results['distances'][0][i],
                "price": results['metadatas'][0][i].get('price', 0),
                "total_meters": results['metadatas'][0][i].get('total_meters', 0),
                "rooms_count": results['metadatas'][0][i].get('rooms_count', 0),
                "floor": results['metadatas'][0][i].get('floor', 0),
                "url": results['metadatas'][0][i].get('url', ''),
                "description": results['documents'][0][i][:300]
            })
        
        return similar
    
    def search_by_apartment_id(self, offer_id: str, k: int = 5) -> list:
        """
        Поиск квартир, похожих на указанную по ID.
        
        Args:
            offer_id: ID квартиры из датасета
            k: количество результатов
            
        Returns:
            список похожих квартир (исключая саму себя)
        """
        # Получаем описание квартиры по ID
        result = self.collection.get(
            ids=[offer_id],
            include=["documents", "metadatas"]
        )
        
        if not result['ids']:
            logger.error(f"Квартира с ID {offer_id} не найдена")
            return []
        
        target_description = result['documents'][0]
        
        # Ищем похожие
        similar = self.search_by_text(target_description, k=k+1)
        
        # Убираем саму квартиру
        similar = [s for s in similar if s['offer_id'] != offer_id][:k]
        
        return similar
    
    def compare_with_market(self, offer_id: str) -> dict:
        """
        Сравнивает квартиру с похожими на рынке.
        
        Returns:
            словарь со статистикой сравнения
        """
        similar = self.search_by_apartment_id(offer_id, k=10)
        
        if not similar:
            return {"error": "Не найдено похожих квартир"}
        
        # Получаем целевую квартиру
        target_result = self.collection.get(
            ids=[offer_id],
            include=["metadatas"]
        )
        target = target_result['metadatas'][0]
        
        # Статистика по похожим
        prices = [s['price'] for s in similar]
        areas = [s['total_meters'] for s in similar]
        
        avg_price = sum(prices) / len(prices)
        median_price = sorted(prices)[len(prices)//2]
        min_price = min(prices)
        max_price = max(prices)
        
        target_price = target.get('price', 0)
        price_diff = target_price - avg_price
        price_diff_percent = (price_diff / avg_price) * 100
        
        # Оценка
        if price_diff_percent < -15:
            verdict = "✅ ВЫГОДНО"
            verdict_desc = "Цена значительно ниже рыночной"
        elif price_diff_percent < -5:
            verdict = "👍 ХОРОШО"
            verdict_desc = "Цена немного ниже средней"
        elif price_diff_percent < 10:
            verdict = "⚠️ НОРМАЛЬНО"
            verdict_desc = "Цена в пределах рынка"
        elif price_diff_percent < 25:
            verdict = "⚠️ ДОРОГО"
            verdict_desc = "Цена выше рыночной"
        else:
            verdict = "❌ ПЕРЕОЦЕНЕНО"
            verdict_desc = "Цена значительно выше аналогов"
        
        return {
            "target_apartment": {
                "offer_id": offer_id,
                "price": target_price,
                "total_meters": target.get('total_meters', 0),
                "rooms_count": target.get('rooms_count', 0),
                "floor": target.get('floor', 0),
                "url": target.get('url', '')
            },
            "market_stats": {
                "similar_count": len(similar),
                "avg_price": avg_price,
                "median_price": median_price,
                "min_price": min_price,
                "max_price": max_price,
                "avg_price_per_m2": avg_price / (sum(areas)/len(areas)) if areas else 0
            },
            "comparison": {
                "price_difference": price_diff,
                "price_difference_percent": price_diff_percent,
                "is_cheaper": price_diff < 0,
                "verdict": verdict,
                "verdict_description": verdict_desc
            },
            "similar_apartments": similar[:5]
        }


def main():
    """Демонстрация работы."""
    search = ApartmentSearch()
    
    # Получаем список всех ID
    all_ids = search.collection.get()['ids']
    
    if not all_ids:
        print("❌ Нет данных в базе")
        return
    
    test_id = all_ids[0]
    print(f"\n🔍 Анализ квартиры: {test_id}")
    print("="*60)
    
    # Поиск по тексту
    print("\n📝 Поиск по запросу 'просторная квартира с хорошим ремонтом':")
    results = search.search_by_text("просторная квартира с хорошим ремонтом", k=3)
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['price']:,.0f} ₽ | {r['total_meters']} м² | сходство: {r['similarity_score']:.3f}")
    
    # Сравнение с рынком
    print(f"\n📊 Сравнение квартиры {test_id} с рынком:")
    comparison = search.compare_with_market(test_id)
    
    if "error" not in comparison:
        print(f"  Целевая цена: {comparison['target_apartment']['price']:,.0f} ₽")
        print(f"  Средняя цена похожих: {comparison['market_stats']['avg_price']:,.0f} ₽")
        print(f"  Отклонение: {comparison['comparison']['price_difference_percent']:.1f}%")
        print(f"  Вердикт: {comparison['comparison']['verdict']}")
        print(f"  {comparison['comparison']['verdict_description']}")


if __name__ == "__main__":
    main()