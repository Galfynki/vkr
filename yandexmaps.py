"""
Простой модуль для работы с API Яндекс.Карт
"""
import requests
import json
import webbrowser
from urllib.parse import quote

class YandexMaps:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.base_url = "https://static-maps.yandex.ru/1.x/"
        self.geocoder_url = "https://geocode-maps.yandex.ru/1.x/"
        
    def get_static_map(self, lat, lon, zoom=13, width=600, height=450, map_type="map"):
        """Получить статическую карту по координатам"""
        params = {
            "ll": f"{lon},{lat}",
            "z": zoom,
            "size": f"{width},{height}",
            "l": map_type,
        }
        
        if self.api_key:
            params["39047caf-5aea-428f-8489-5c45f71f55a0"] = self.api_key
            
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        map_url = f"{self.base_url}?{query_string}"
        
        return map_url
    
    def open_in_browser(self, lat, lon, zoom=13):
        """Открыть Яндекс.Карты в браузере по координатам"""
        browser_url = f"https://yandex.ru/maps/?ll={lon},{lat}&z={zoom}"
        webbrowser.open(browser_url)
        
    def geocode(self, address):
        """Геокодирование адреса (требуется API ключ)"""
        if not self.api_key:
            raise ValueError("Для геокодирования необходим API ключ Яндекс.Карт")
            
        params = {
            "apikey": self.api_key,
            "geocode": address,
            "format": "json"
        }
        
        query_string = "&".join([f"{k}={quote(str(v))}" for k, v in params.items()])
        url = f"{self.geocoder_url}?{query_string}"
        
        try:
            response = requests.get(url)
            data = response.json()
            return data
        except Exception as e:
            print(f"Ошибка при геокодировании: {e}")
            return None
    
    def get_coordinates(self, address):
        """Получить координаты по адресу (требуется API ключ)"""
        data = self.geocode(address)
        if data and 'response' in data:
            try:
                features = data['response']['GeoObjectCollection']['featureMember']
                if features:
                    coords_str = features[0]['GeoObject']['Point']['pos']
                    lon, lat = map(float, coords_str.split())
                    return lat, lon
            except (KeyError, IndexError):
                pass
        return None

# Функции для совместимости с кодом
def get_map_url(lat, lon, zoom=13):
    """Функция для обратной совместимости"""
    maps = YandexMaps()
    return maps.get_static_map(lat, lon, zoom)

def open_map_in_browser(lat, lon, zoom=13):
    """Функция для обратной совместимости"""
    maps = YandexMaps()
    maps.open_in_browser(lat, lon, zoom)
