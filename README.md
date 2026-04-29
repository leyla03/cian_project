
# CIAN Real Estate Price Predictor

Сервис для оценки стоимости квартир на основе данных с сайта Циан.

## Структура проекта
cian-project/
├── data/ # Данные
│ ├── raw/ # Собранные данные
│ └── processed/ # Обработанные Parquet файлы
├── models/ # Обученные модели
├── src/
│ ├── api/ # FastAPI
│ ├── data/ # Парсинг и валидация
│ ├── features/ # Эмбеддинги
│ ├── models/ # Обучение CatBoost
│ ├── report/ # Генератор отчетов
│ └── search/ # Семантический поиск
└── pyproject.toml # Зависимости

## Технологии

- **Сбор данных**: cianparser, Selenium
- **Валидация**: Pandera
- **Хранение**: Parquet, DVC
- **Эмбеддинги**: Sentence Transformers (all-MiniLM-L6-v2, CLIP)
- **Векторная БД**: ChromaDB
- **Модель**: CatBoost
- **API**: FastAPI

## Установка

```bash
git clone https://github.com/leyla03/cian_project.git
cd cian_project
pip install -e .
Запуск
1. Сбор данных
bash
python src/data/selenium_parser.py
2. Валидация и Parquet
bash
python src/data/validate_to_parquet.py
3. Текстовые эмбеддинги
bash
python src/features/text_embeddings.py
4. Обучение модели
bash
python src/models/compare_models.py
5. Запуск API
bash
python src/api/main.py
Открой http://127.0.0.1:8000/docs

API Endpoints
Метод	Эндпоинт	Описание
GET	/	Информация о сервисе
GET	/health	Проверка статуса
POST	/predict	Предсказание цены
Пример запроса
json
POST /predict
{
  "total_meters": 50,
  "floor": 5,
  "floors_total": 12,
  "rooms_count": 2
}
Пример ответа
json
{
  "price_rub": 39888484.9,
  "price_million": 39.9,
  "message": "Ориентировочная стоимость квартиры площадью 50.0 м²"
}
Результаты модели
Модель	MAE	R²
Базовая (4 признака)	24.6 млн ₽	0.28