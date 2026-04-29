# CIAN Real Estate Price Predictor

Сервис для оценки стоимости квартир на основе данных с сайта Циан.



##  Технологии

Сбор данных: cianparser, Selenium
Валидация: Pandera
Хранение: Parquet, DVC
Эмбеддинги: Sentence Transformers (all-MiniLM-L6-v2, CLIP)
Векторная БД: ChromaDB
Модель: CatBoost
API: FastAPI


##  Установка

```bash
git clone https://github.com/leyla03/cian_project.git
cd cian_project
pip install -e .
```



##  Запуск

### 1. Сбор данных

```bash
python src/data/selenium_parser.py
```

### 2. Валидация и сохранение в Parquet

```bash
python src/data/validate_to_parquet.py
```

### 3. Генерация текстовых эмбеддингов

```bash
python src/features/text_embeddings.py
```

### 4. Обучение модели

```bash
python src/models/compare_models.py
```

### 5. Запуск API сервера

```bash
python src/api/main.py
```

После запуска открой в браузере: **http://127.0.0.1:8000/docs**



##  API Endpoints

| Метод | Эндпоинт | Описание |
|-------|----------|----------|
| `GET` | `/` | Информация о сервисе |
| `GET` | `/health` | Проверка статуса сервера |
| `POST` | `/predict` | Предсказание цены квартиры |

---

##  Пример использования API

### Запрос

```json
POST /predict
Content-Type: application/json

{
  "total_meters": 50,
  "floor": 5,
  "floors_total": 12,
  "rooms_count": 2
}
```

### Ответ

```json
{
  "price_rub": 39888484.9,
  "price_million": 39.9,
  "message": "Ориентировочная стоимость квартиры площадью 50.0 м²"
}
```

---

##  Результаты модели

| Модель | Признаков | MAE | R² |
|--------|-----------|-----|-----|
| Базовая | 4 | 24.6 млн ₽ | 0.28 |
| + Текстовые эмбеддинги | 388 | 23.3 млн ₽ | 0.27 |
| + Фото-эмбеддинги | 512 | 26.1 млн ₽ | 0.05 |

**Финальная модель:** базовая (total_meters, floor, floors_total, rooms_count)


##  Ссылка на GitHub

[https://github.com/leyla03/cian_project](https://github.com/leyla03/cian_project)
```