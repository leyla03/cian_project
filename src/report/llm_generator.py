"""Генератор отчёта с использованием LLM (локальная или API)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.search.semantic_search import ApartmentSearch
import json
import os
from dotenv import load_dotenv

load_dotenv()


class ReportGenerator:
    """Генерирует отчёты о выгодности сделки."""
    
    def __init__(self, use_openai: bool = False):
        self.search = ApartmentSearch()
        self.use_openai = use_openai
        
        # Инициализируем LLM, если нужно
        self.llm = None
        if use_openai:
            try:
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.3,
                    api_key=os.getenv("OPENAI_API_KEY")
                )
                print("✅ OpenAI подключен")
            except Exception as e:
                print(f"⚠️ Ошибка подключения OpenAI: {e}")
                self.use_openai = False
    
    def _generate_report_with_llm(self, comparison: dict) -> str:
        """Генерирует отчёт через LLM."""
        if not self.llm:
            return self._generate_report_without_llm(comparison)
        
        prompt = f"""
Ты — эксперт по недвижимости. Проанализируй квартиру и дай рекомендацию.

Целевая квартира:
- Цена: {comparison['target_apartment']['price']:,.0f} ₽
- Площадь: {comparison['target_apartment']['total_meters']} м²
- Комнат: {comparison['target_apartment']['rooms_count']}

Рыночные данные по {comparison['market_stats']['similar_count']} похожим квартирам:
- Средняя цена: {comparison['market_stats']['avg_price']:,.0f} ₽
- Отклонение: {comparison['comparison']['price_difference_percent']:.1f}%

Вердикт: {comparison['comparison']['verdict_description']}

Напиши короткий отчёт (3-4 предложения) для покупателя: 
стоит ли покупать эту квартиру, какие плюсы и минусы.
"""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"Ошибка LLM: {e}")
            return self._generate_report_without_llm(comparison)
    
    def _generate_report_without_llm(self, comparison: dict) -> str:
        """Генерирует отчёт без LLM (на основе правил)."""
        target = comparison['target_apartment']
        stats = comparison['market_stats']
        comp = comparison['comparison']
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║                    ОТЧЁТ О КВАРТИРЕ                          ║
╚══════════════════════════════════════════════════════════════╝

 ЦЕЛЕВАЯ КВАРТИРА
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• ID: {target['offer_id']}
• Цена: {target['price']:,.0f} ₽
• Площадь: {target['total_meters']} м²
• Комнат: {target['rooms_count']}
• Этаж: {target['floor']}
• Цена за м²: {target['price']/target['total_meters']:,.0f} ₽

 СРАВНЕНИЕ С {stats['similar_count']} ПОХОЖИМИ КВАРТИРАМИ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Средняя цена: {stats['avg_price']:,.0f} ₽
• Медианная цена: {stats['median_price']:,.0f} ₽
• Диапазон цен: {stats['min_price']:,.0f} ₽ - {stats['max_price']:,.0f} ₽
• Отклонение: {comp['price_difference_percent']:.1f}% ({'+' if not comp['is_cheaper'] else ''}{comp['price_difference']:,.0f} ₽)

 ВЕРДИКТ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{comp['verdict']} — {comp['verdict_description']}

 ПОХОЖИЕ ВАРИАНТЫ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        for i, apt in enumerate(comparison['similar_apartments'][:3], 1):
            report += f"\n{i}. Цена: {apt['price']:,.0f} ₽ | {apt['total_meters']} м² | Сходство: {apt['similarity_score']:.2f}"
            report += f"\n   {apt['url'][:80]}..."
        
        if self.use_openai:
            report += "\n\n АНАЛИЗ LLM\n"
        
        return report
    
    def generate_report(self, offer_id: str, use_llm: bool = None) -> str:
        """
        Генерирует полный отчёт о квартире.
        
        Args:
            offer_id: ID квартиры
            use_llm: использовать ли LLM (если None, используется self.use_openai)
        """
        # Получаем сравнение с рынком
        comparison = self.search.compare_with_market(offer_id)
        
        if "error" in comparison:
            return f" {comparison['error']}"
        
        # Генерируем отчёт
        use_llm_flag = use_llm if use_llm is not None else self.use_openai
        
        if use_llm_flag:
            report = self._generate_report_with_llm(comparison)
            # Добавляем базовую статистику в начале
            base_report = self._generate_report_without_llm(comparison)
            # Берём только вердикт из базового отчёта
            base_lines = base_report.split('\n')
            stats_section = '\n'.join(base_lines[:base_lines.index("💎 ВЕРДИКТ")])
            report = f"{stats_section}\n\n LLM-АНАЛИЗ\n{report}"
        else:
            report = self._generate_report_without_llm(comparison)
        
        return report


def main():
    """Демонстрация работы генератора отчётов."""
    generator = ReportGenerator(use_openai=False)  # Пока без OpenAI
    
    # Получаем ID первой квартиры
    all_ids = generator.search.collection.get()['ids']
    
    if not all_ids:
        print(" Нет данных в базе. Сначала запусти text_embeddings.py")
        return
    
    test_id = all_ids[0]
    print(f"\n Генерируем отчёт для квартиры {test_id}")
 
    
    report = generator.generate_report(test_id)
    print(report)
    
    # Сохраняем отчёт
    report_path = Path("reports/sample_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n Отчёт сохранён: {report_path}")


if __name__ == "__main__":
    main()