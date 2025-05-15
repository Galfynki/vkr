import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import folium
from folium.plugins import Draw
import io
import webbrowser
import os

# Импортируем наш новый модуль
import yandexmaps

class PhotoViewer:
    def __init__(self, root):  # Исправлено название метода __init__
        self.root = root
        self.frame = ttk.Frame(root)
        
        self.day_photo_path = "day.jpg"  # Замените на свой путь
        self.night_photo_path = "night.jpg"  # Замените на свой путь
        
        self.btn_day = ttk.Button(self.frame, text="День", command=self.show_day)
        self.btn_day.pack(pady=10)
        
        self.btn_night = ttk.Button(self.frame, text="Ночь", command=self.show_night)
        self.btn_night.pack(pady=10)
        
        self.back_btn = ttk.Button(self.frame, text="Назад", command=self.hide_photo)
        self.back_btn.pack(pady=10)
        self.back_btn.pack_forget()
        
        self.photo_label = ttk.Label(self.frame)
        self.photo_label.pack()
        
    def show_day(self):
        try:
            image = Image.open(self.day_photo_path)
            image = image.resize((400, 300), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.photo_label.config(image=photo)
            self.photo_label.image = photo
            self.btn_day.pack_forget()
            self.btn_night.pack_forget()
            self.back_btn.pack(pady=10)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить фото: {e}")
    
    def show_night(self):
        try:
            image = Image.open(self.night_photo_path)
            image = image.resize((400, 300), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.photo_label.config(image=photo)
            self.photo_label.image = photo
            self.btn_day.pack_forget()
            self.btn_night.pack_forget()
            self.back_btn.pack(pady=10)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить фото: {e}")
    
    def hide_photo(self):
        self.photo_label.config(image=None)
        self.photo_label.image = None
        self.back_btn.pack_forget()
        self.btn_day.pack(pady=10)
        self.btn_night.pack(pady=10)


class MapViewer:
    def __init__(self, root):  # Исправлено название метода __init__
        self.root = root
        self.frame = ttk.Frame(root)
        
        # Инициализируем Yandex Maps с помощью нашего модуля
        self.yandex_maps = yandexmaps.YandexMaps()
        
        # Текущие координаты (Москва по умолчанию)
        self.current_lat = 55.751244
        self.current_lon = 37.618423
        
        # Создаем карту Folium
        self.map = folium.Map(location=[self.current_lat, self.current_lon], zoom_start=12)
        Draw(export=True).add_to(self.map)
        
        # Файл для временной карты
        self.map_file = "temp_map.html"
        self.save_map()
        
        # Информация о карте
        self.map_label = ttk.Label(self.frame, text="Карта загружается...")
        self.map_label.pack(pady=5)
        
        # Кнопки для работы с картой
        self.btn_open_map = ttk.Button(self.frame, text="Открыть карту в браузере", 
                                       command=self.open_map_in_browser)
        self.btn_open_map.pack(pady=5)
        
        self.btn_yandex_map = ttk.Button(self.frame, text="Открыть в Яндекс.Картах", 
                                         command=self.open_yandex_map)
        self.btn_yandex_map.pack(pady=5)
        
        self.btn_route = ttk.Button(self.frame, text="Построить маршрут", 
                                    command=self.add_route)
        self.btn_route.pack(pady=5)
        
        self.btn_add_sign = ttk.Button(self.frame, text="Добавить знак ПДД", 
                                       command=self.add_traffic_sign)
        self.btn_add_sign.pack(pady=5)
        
        self.btn_add_camera = ttk.Button(self.frame, text="Добавить камеру", 
                                         command=self.add_camera)
        self.btn_add_camera.pack(pady=5)
        
        # Поле для ввода координат
        ttk.Label(self.frame, text="Координаты (широта, долгота):").pack(pady=2)
        self.coord_entry = ttk.Entry(self.frame, width=30)
        self.coord_entry.pack(pady=3)
        self.coord_entry.insert(0, f"{self.current_lat}, {self.current_lon}")
        
        self.btn_search = ttk.Button(self.frame, text="Поиск по координатам", 
                                     command=self.search_coords)
        self.btn_search.pack(pady=5)
        
    def save_map(self):
        """Сохраняем текущую карту во временный файл"""
        try:
            self.map.save(self.map_file)
            self.map_label.config(text=f"Карта координат: {self.current_lat:.6f}, {self.current_lon:.6f}")
        except Exception as e:
            self.map_label.config(text=f"Ошибка сохранения карты: {e}")
    
    def open_map_in_browser(self):
        """Открыть текущую карту в браузере"""
        try:
            # Обновляем карту перед открытием
            self.save_map()
            
            # Получаем полный путь к файлу
            map_path = os.path.abspath(self.map_file)
            
            # Открываем локальный файл в браузере
            webbrowser.open(f"file://{map_path}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть карту: {e}")
            
    def open_yandex_map(self):
        """Открыть текущие координаты в Яндекс.Картах"""
        try:
            # Используем наш модуль yandexmaps для открытия в Яндекс.Картах
            self.yandex_maps.open_in_browser(self.current_lat, self.current_lon, zoom=13)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть Яндекс.Карты: {e}")
    
    def add_route(self):
        """Добавить маршрут на карту"""
        messagebox.showinfo("Маршрут", "Функция маршрута будет реализована здесь.")
        # Можно доработать эту функцию, чтобы она использовала API Яндекс.Карт для маршрутов
    
    def add_traffic_sign(self):
        """Добавить знак ПДД на карту"""
        # Здесь можно добавить диалоговое окно выбора знака
        sign_types = ["Ограничение скорости", "Пешеходный переход", "Стоп", "Главная дорога"]
        sign = messagebox.askquestion("Знак ПДД", "Добавить знак на текущие координаты?")
        
        if sign == 'yes':
            # Добавляем маркер знака на карту folium
            folium.Marker(
                location=[self.current_lat, self.current_lon],
                popup="Знак ПДД",
                icon=folium.Icon(color="red", icon="info-sign")
            ).add_to(self.map)
            
            self.save_map()
            messagebox.showinfo("Знак ПДД", f"Знак добавлен на координаты {self.current_lat}, {self.current_lon}")
    
    def add_camera(self):
        """Добавить камеру на карту"""
        camera = messagebox.askquestion("Камера", "Добавить камеру на текущие координаты?")
        
        if camera == 'yes':
            # Добавляем маркер камеры на карту folium
            folium.Marker(
                location=[self.current_lat, self.current_lon],
                popup="Камера наблюдения",
                icon=folium.Icon(color="blue", icon="camera")
            ).add_to(self.map)
            
            self.save_map()
            messagebox.showinfo("Камера", f"Камера добавлена на координаты {self.current_lat}, {self.current_lon}")
    
    def search_coords(self):
        """Поиск по координатам"""
        coords = self.coord_entry.get()
        try:
            # Разбираем координаты
            parts = coords.replace(" ", "").split(',')
            if len(parts) >= 2:
                self.current_lat = float(parts[0])
                self.current_lon = float(parts[1])
                
                # Обновляем центр карты
                self.map = folium.Map(location=[self.current_lat, self.current_lon], zoom_start=14)
                Draw(export=True).add_to(self.map)
                
                # Добавляем маркер
                folium.Marker(
                    location=[self.current_lat, self.current_lon],
                    popup=f"Координаты: {self.current_lat}, {self.current_lon}"
                ).add_to(self.map)
                
                self.save_map()
                self.open_map_in_browser()
            else:
                raise ValueError("Неверный формат координат")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Неверный формат координат: {e}\nИспользуйте: широта, долгота")


class App:
    def __init__(self, root):  # Исправлено название метода __init__
        self.root = root
        self.root.title("Приложение с картой и фото")
        
        # Создаем вкладки
        self.tab_control = ttk.Notebook(root)
        
        # Вкладка "Фото"
        self.photo_viewer = PhotoViewer(self.tab_control)
        self.tab_control.add(self.photo_viewer.frame, text="Фото")
        
        # Вкладка "Карта"
        self.map_viewer = MapViewer(self.tab_control)
        self.tab_control.add(self.map_viewer.frame, text="Карта")
        
        self.tab_control.pack(expand=1, fill="both")


if __name__ == "__main__":  # Исправлено условие для запуска
    root = tk.Tk()
    app = App(root)
    root.geometry("600x500")
    root.mainloop()
