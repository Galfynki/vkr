import tkinter as tk
from tkinter import ttk
import folium
from folium.plugins import Draw
import io
from PIL import Image, ImageTk
import webbrowser
import requests

class MapApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Python Map Interface")
        self.root.geometry("1000x700")
        
        # Создаем базовую карту
        self.map = folium.Map(location=[55.751244, 37.618423], zoom_start=12)
        
        # Добавляем инструменты рисования
        Draw(export=True).add_to(self.map)
        
        # Создаем интерфейс
        self.create_widgets()
        
    def create_widgets(self):
        # Фрейм для элементов управления
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Поле поиска
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(control_frame, textvariable=self.search_var, width=40)
        search_entry.pack(side=tk.LEFT, padx=5)
        
        # Кнопка поиска
        search_btn = ttk.Button(control_frame, text="Поиск", command=self.search_location)
        search_btn.pack(side=tk.LEFT, padx=5)
        
        # Кнопки масштабирования
        zoom_in_btn = ttk.Button(control_frame, text="+", command=self.zoom_in, width=3)
        zoom_in_btn.pack(side=tk.RIGHT, padx=2)
        
        zoom_out_btn = ttk.Button(control_frame, text="-", command=self.zoom_out, width=3)
        zoom_out_btn.pack(side=tk.RIGHT, padx=2)
        
        # Фрейм для отображения карты
        self.map_frame = ttk.Frame(self.root)
        self.map_frame.pack(fill=tk.BOTH, expand=True)
        
        # Инициализация отображения карты
        self.update_map_display()
    
    def update_map_display(self):
        # Сохраняем карту во временный HTML файл
        self.map.save("temp_map.html")
        
        # Используем WebView для отображения (альтернатива - использовать folium в Jupyter)
        # В этом примере мы просто откроем браузер, но для реального приложения лучше использовать
        # библиотеку like pywebview или встроить WebKit в Tkinter
        
        # Для простоты откроем в браузере по умолчанию
        webbrowser.open("temp_map.html")
        
        # Альтернативный вариант - отобразить статическое изображение карты
        img_data = self.map._to_png()
        img = Image.open(io.BytesIO(img_data))
        img = img.resize((800, 600), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(img)
        
        if hasattr(self, 'map_label'):
             self.map_label.config(image=photo)
             self.map_label.image = photo
        else:
             self.map_label = ttk.Label(self.map_frame, image=photo)
             self.map_label.pack(fill=tk.BOTH, expand=True)
             self.map_label.image = photo
    
    def search_location(self):
        query = self.search_var.get()
        if not query:
            return
            
        try:
            # Используем Nominatim для геокодирования
            url = f"https://nominatim.openstreetmap.org/search?q={query}&format=json"
            response = requests.get(url).json()
            
            if response:
                lat = float(response[0]['lat'])
                lon = float(response[0]['lon'])
                
                # Обновляем карту
                self.map.location = [lat, lon]
                folium.Marker([lat, lon], popup=query).add_to(self.map)
                self.update_map_display()
                
        except Exception as e:
            print(f"Ошибка при поиске: {e}")
    
    def zoom_in(self):
        current_zoom = self.map.options['zoom']
        self.map.options['zoom'] = current_zoom + 1
        self.update_map_display()
    
    def zoom_out(self):
        current_zoom = self.map.options['zoom']
        if current_zoom > 1:
            self.map.options['zoom'] = current_zoom - 1
            self.update_map_display()

if __name__ == "__main__":
    root = tk.Tk()
    app = MapApp(root)
    root.mainloop()