"""Упрощённый парсер Циан с Selenium — только сбор фото (с перезапуском браузера)."""

import pandas as pd
from pathlib import Path
import time
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_driver():
    """Настройка Chrome драйвера (свежий экземпляр)."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Фоновый режим
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    
    # Критически важно для Windows
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-logging")
    chrome_options.add_argument("--log-level=3")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver


def collect_photos(url: str, driver) -> list:
    """Собирает только фото с одной страницы."""
    images = []
    
    try:
        driver.get(url)
        time.sleep(2.5)  # Ждём загрузку
        
        # Ищем изображения
        img_tags = driver.find_elements(By.TAG_NAME, "img")
        for img in img_tags:
            src = img.get_attribute('src')
            if src and src.startswith('http'):
                if (('thumb' not in src.lower()) and 
                    ('photo' in src.lower() or 'image' in src.lower() or 'upload' in src.lower()) and
                    ('logo' not in src.lower())):
                    images.append(src)
        
        images = list(set(images))[:15]
        logger.info(f"✅ Фото: {len(images)}")
        
    except Exception as e:
        logger.error(f"❌ Ошибка: {str(e)[:100]}")
    
    return images


def enrich_with_photos(input_csv: Path, output_csv: Path, max_offers: int = 200, restart_every: int = 15):
    """Добавляет фото, перезапуская браузер каждые N объявлений."""
    
    df = pd.read_csv(input_csv)
    logger.info(f"Загружено {len(df)} объявлений")
    
    if len(df) > max_offers:
        df = df.head(max_offers)
        logger.info(f"Обрабатываем {max_offers} объявлений")
    
    all_images = []
    total = len(df)
    
    for batch_start in range(0, total, restart_every):
        batch_end = min(batch_start + restart_every, total)
        logger.info(f"\n🔄 Запускаем браузер для объявлений {batch_start+1}-{batch_end}")
        
        driver = setup_driver()
        
        try:
            for idx in range(batch_start, batch_end):
                url = df.iloc[idx]['url']
                logger.info(f"[{idx+1}/{total}] Обработка...")
                
                photos = collect_photos(url, driver)
                all_images.append(str(photos) if photos else "")
                
                time.sleep(1.5)  # Пауза
                
        except Exception as e:
            logger.error(f"Критическая ошибка в батче: {e}")
        
        finally:
            driver.quit()
            logger.info(f"🔄 Браузер закрыт, пауза 3 секунды...")
            time.sleep(3)
    
    # Добавляем колонку
    df['images_urls'] = all_images
    
    # Сохраняем
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    
    with_photos = sum(len(str(imgs)) > 10 for imgs in all_images)
    print(f"\n✅ Готово! {with_photos}/{len(df)} объявлений с фото")
    print(f"📁 Сохранено: {output_csv}")


def main():
    enrich_with_photos(
        input_csv=Path("data/raw/cian_raw.csv"),
        output_csv=Path("data/raw/cian_with_photos.csv"),
        max_offers=200,      # 200 объявлений
        restart_every=15     # Перезапуск каждые 15 штук
    )


if __name__ == "__main__":
    main()