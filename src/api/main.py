"""FastAPI сервер для предсказания цены квартиры."""

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from catboost import CatBoostRegressor
from pathlib import Path

# Загружаем модель
MODEL_PATH = Path("models/catboost_base.cbm")

if MODEL_PATH.exists():
    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)
    print(" Модель загружена")
else:
    print(" Модель не найдена")
    model = None

# Создаём приложение
app = FastAPI(
    title="Квартирный помощник",
    description="Оценка стоимости квартиры по параметрам",
    version="1.0"
)


# Описываем формат входных данных
class ApartmentInput(BaseModel):
    total_meters: float  # площадь, м²
    floor: int           # этаж
    floors_total: int    # всего этажей
    rooms_count: int     # количество комнат


# Описываем формат ответа
class PriceResponse(BaseModel):
    price_rub: float      # цена в рублях
    price_million: float  # цена в миллионах
    message: str


# Главная страница
@app.get("/")
def root():
    return {
        "service": "Квартирный помощник",
        "how_to_use": "Отправь POST запрос на /predict с параметрами квартиры"
    }


# Эндпоинт для предсказания
@app.post("/predict", response_model=PriceResponse)
def predict(apartment: ApartmentInput):
    if model is None:
        return PriceResponse(
            price_rub=0,
            price_million=0,
            message=" Модель не загружена. Сначала обучи модель."
        )
    
    # Превращаем входные данные в DataFrame
    data = pd.DataFrame([{
        'total_meters': apartment.total_meters,
        'floor': apartment.floor,
        'floors_total': apartment.floors_total,
        'rooms_count': apartment.rooms_count
    }])
    
    # Предсказываем
    price = float(model.predict(data)[0])
    price_million = price / 1_000_000
    
    return PriceResponse(
        price_rub=round(price, 2),
        price_million=round(price_million, 1),
        message=f"Ориентировочная стоимость квартиры площадью {apartment.total_meters} м²"
    )


# Эндпоинт для проверки здоровья сервера
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


# Запуск сервера (если запускаем файл напрямую)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)