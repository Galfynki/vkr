import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import folium
from folium.plugins import HeatMap
import json

class RoadSafetyAnalyzer:
    def __init__(self):
        self.accidents_data = None
        self.dangerous_zones = None
        self.model = None
        self.features = ['latitude', 'longitude', 'hour', 'weekday', 'weather_conditions', 
                        'road_conditions', 'light_conditions', 'speed_limit']
        self.target = 'severity'
        
    def load_data(self, filepath):
        """Загрузка данных о ДТП из файла"""
        try:
            self.accidents_data = pd.read_csv(filepath)
            print(f"Данные успешно загружены. Всего записей: {len(self.accidents_data)}")
            # Предобработка данных
            self._preprocess_data()
            return True
        except Exception as e:
            print(f"Ошибка при загрузке данных: {e}")
            return False
    
    def _preprocess_data(self):
        """Предварительная обработка данных"""
        # Удаление пропущенных значений
        self.accidents_data.dropna(subset=['latitude', 'longitude'], inplace=True)
        
        # Преобразование категориальных признаков
        self.accidents_data['weather_conditions'] = self.accidents_data['weather_conditions'].astype('category').cat.codes
        self.accidents_data['road_conditions'] = self.accidents_data['road_conditions'].astype('category').cat.codes
        self.accidents_data['light_conditions'] = self.accidents_data['light_conditions'].astype('category').cat.codes
        
        # Преобразование времени
        if 'timestamp' in self.accidents_data.columns:
            self.accidents_data['timestamp'] = pd.to_datetime(self.accidents_data['timestamp'])
            self.accidents_data['hour'] = self.accidents_data['timestamp'].dt.hour
            self.accidents_data['weekday'] = self.accidents_data['timestamp'].dt.weekday
    
    def detect_dangerous_zones(self, eps=0.01, min_samples=5):
        """Выявление опасных зон с использованием кластеризации DBSCAN"""
        if self.accidents_data is None:
            print("Данные не загружены. Сначала загрузите данные.")
            return
        
        # Нормализация координат
        coords = self.accidents_data[['latitude', 'longitude']].values
        coords = StandardScaler().fit_transform(coords)
        
        # Кластеризация
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        labels = db.labels_
        
        # Добавление меток кластеров в данные
        self.accidents_data['cluster'] = labels
        
        # Определение опасных зон (кластеры с большим количеством ДТП)
        cluster_counts = self.accidents_data['cluster'].value_counts()
        dangerous_clusters = cluster_counts[cluster_counts > min_samples].index.tolist()
        dangerous_clusters = [c for c in dangerous_clusters if c != -1]  # Исключаем шум
        
        self.dangerous_zones = self.accidents_data[self.accidents_data['cluster'].isin(dangerous_clusters)]
        print(f"Выявлено {len(dangerous_clusters)} опасных зон")
        
        return self.dangerous_zones
    
    def train_severity_model(self):
        """Обучение модели для прогнозирования тяжести ДТП"""
        if self.accidents_data is None:
            print("Данные не загружены. Сначала загрузите данные.")
            return
        
        # Подготовка данных
        X = self.accidents_data[self.features]
        y = self.accidents_data[self.target]
        
        # Разделение на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Обучение модели
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Оценка модели
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Точность модели: {accuracy:.2f}")
        print(classification_report(y_test, y_pred))
        
        return accuracy
    
    def predict_severity(self, new_data):
        """Прогнозирование тяжести ДТП для новых данных"""
        if self.model is None:
            print("Модель не обучена. Сначала обучите модель.")
            return
        
        prediction = self.model.predict(new_data)
        return prediction
    
    def visualize_dangerous_zones(self):
        """Визуализация опасных зон на карте"""
        if self.dangerous_zones is None:
            print("Опасные зоны не выявлены. Сначала выполните их обнаружение.")
            return
        
        # Создание базовой карты
        map_center = [self.accidents_data['latitude'].mean(), self.accidents_data['longitude'].mean()]
        safety_map = folium.Map(location=map_center, zoom_start=12)
        
        # Добавление тепловой карты
        heat_data = [[row['latitude'], row['longitude']] for _, row in self.dangerous_zones.iterrows()]
        HeatMap(heat_data, radius=15).add_to(safety_map)
        
        # Добавление маркеров для кластеров
        for cluster_id in self.dangerous_zones['cluster'].unique():
            cluster_data = self.dangerous_zones[self.dangerous_zones['cluster'] == cluster_id]
            cluster_center = [cluster_data['latitude'].mean(), cluster_data['longitude'].mean()]
            
            # Подсчет количества ДТП в кластере
            count = len(cluster_data)
            
            # Добавление маркера
            folium.Marker(
                location=cluster_center,
                popup=f"Опасная зона #{cluster_id}<br>Количество ДТП: {count}",
                icon=folium.Icon(color='red', icon='exclamation-triangle')
            ).add_to(safety_map)
        
        return safety_map
    
    def generate_recommendations(self):
        """Генерация рекомендаций по мерам безопасности"""
        if self.dangerous_zones is None:
            print("Опасные зоны не выявлены. Сначала выполните их обнаружение.")
            return
        
        recommendations = []
        
        for cluster_id in self.dangerous_zones['cluster'].unique():
            cluster_data = self.dangerous_zones[self.dangerous_zones['cluster'] == cluster_id]
            center = [cluster_data['latitude'].mean(), cluster_data['longitude'].mean()]
            count = len(cluster_data)
            
            # Анализ характеристик кластера
            avg_speed = cluster_data['speed_limit'].mean()
            night_ratio = (cluster_data['light_conditions'] == 0).mean()  # 0 - темное время суток
            bad_weather_ratio = (cluster_data['weather_conditions'] > 1).mean()  # 1 - ясная погода
            
            # Формирование рекомендаций
            rec = {
                'zone_id': cluster_id,
                'location': center,
                'accident_count': count,
                'recommendations': []
            }
            
            if night_ratio > 0.5:
                rec['recommendations'].append("Установить дополнительное освещение")
            
            if bad_weather_ratio > 0.4:
                rec['recommendations'].append("Установить предупреждающие знаки о плохих погодных условиях")
            
            if avg_speed > 60 and count > 10:
                rec['recommendations'].append("Установить камеры контроля скорости")
                rec['recommendations'].append("Рассмотреть возможность установки искусственной неровности")
            
            if len(rec['recommendations']) == 0:
                rec['recommendations'].append("Провести дополнительное обследование участка")
            
            recommendations.append(rec)
        
        return recommendations

# Пример использования
if __name__ == "__main__":
    analyzer = RoadSafetyAnalyzer()
    
    # Загрузка данных (предполагается, что файл содержит колонки: latitude, longitude, timestamp, 
    # weather_conditions, road_conditions, light_conditions, speed_limit, severity и др.)
    analyzer.load_data("accidents_data.csv")
    
    # Выявление опасных зон
    dangerous_zones = analyzer.detect_dangerous_zones()
    
    # Обучение модели прогнозирования тяжести ДТП
    accuracy = analyzer.train_severity_model()
    
    # Визуализация опасных зон
    safety_map = analyzer.visualize_dangerous_zones()
    #safety_map.save("dangerous_zones.html")
    
    # Генерация рекомендаций
    recommendations = analyzer.generate_recommendations()
    print("\nРекомендации по мерам безопасности:")
    