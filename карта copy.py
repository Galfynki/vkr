import tkinter as tk
from tkinter import ttk
import math
from PIL import Image, ImageDraw, ImageTk

class MapApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Яндекс-подобная Карта")
        self.root.geometry("1000x800")
        
        # Исходные координаты и масштаб
        self.center = (55.838717, 36.840000)  # Центр карты
        self.zoom = 12  # Уровень масштабирования
        self.drag_start = None
        
        # Точки маршрута
        self.point_a = (55.830623, 36.860340)  # Точка A
        self.point_b = (55.846776, 36.815798)  # Точка B
        
        # Создаем основной интерфейс
        self.create_ui()
        
        # Генерируем первоначальную карту
        self.generate_map()
        
    def create_ui(self):
        # Главный контейнер
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Панель инструментов сверху
        self.toolbar = ttk.Frame(self.main_frame)
        self.toolbar.pack(fill=tk.X, padx=5, pady=5)
        
        # Поле поиска
        self.search_entry = ttk.Entry(self.toolbar, width=40)
        self.search_entry.pack(side=tk.LEFT, padx=5)
        search_btn = ttk.Button(self.toolbar, text="Найти", command=self.search_location)
        search_btn.pack(side=tk.LEFT, padx=5)
        
        # Кнопки масштаба
        zoom_frame = ttk.Frame(self.toolbar)
        zoom_frame.pack(side=tk.RIGHT, padx=5)
        ttk.Button(zoom_frame, text="+", width=3, command=self.zoom_in).pack(side=tk.TOP)
        ttk.Button(zoom_frame, text="-", width=3, command=self.zoom_out).pack(side=tk.TOP)
        
        # Контейнер для карты
        self.map_container = ttk.Frame(self.main_frame)
        self.map_container.pack(fill=tk.BOTH, expand=True)
        
        # Холст для карты
        self.canvas = tk.Canvas(self.map_container, bg="#e8e8e8")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Привязка событий мыши
        self.canvas.bind("<ButtonPress-1>", self.start_drag)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.end_drag)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        
        # Статус бар
        self.status_bar = ttk.Label(self.main_frame, text=f"Координаты: {self.center[0]:.6f}, {self.center[1]:.6f} | Масштаб: {self.zoom}")
        self.status_bar.pack(fill=tk.X, padx=5, pady=2)
    
    def generate_map(self):
        # Размеры изображения (в 2 раза больше видимой области для плавного перемещения)
        width, height = self.canvas.winfo_width() * 2, self.canvas.winfo_height() * 2
        if width < 100 or height < 100:  # Минимальный размер
            width, height = 1000, 800
        
        # Создаем изображение
        self.map_image = Image.new("RGB", (width, height), "#e8e8e8")
        draw = ImageDraw.Draw(self.map_image)
        
        # Рассчитываем границы отображаемой области
        lat_span = 180 / (2 ** (self.zoom + 1))
        lon_span = lat_span * (width / height)
        
        min_lat = self.center[0] - lat_span
        max_lat = self.center[0] + lat_span
        min_lon = self.center[1] - lon_span
        max_lon = self.center[1] + lon_span
        
        # Функция для преобразования координат в пиксели
        def coord_to_pixel(lat, lon):
            x = (lon - min_lon) / (max_lon - min_lon) * width
            y = height - (lat - min_lat) / (max_lat - min_lat) * height
            return x, y
        
        # Рисуем сетку координат
        self.draw_grid(draw, width, height, min_lat, max_lat, min_lon, max_lon)
        
        # Рисуем дороги (упрощенная схема)
        self.draw_roads(draw, coord_to_pixel)
        
        # Рисуем маршрут от A до B
        self.draw_route(draw, coord_to_pixel)
        
        # Рисуем POI (точки интереса)
        self.draw_poi(draw, coord_to_pixel)
        
        # Конвертируем изображение для Tkinter
        self.tk_image = ImageTk.PhotoImage(self.map_image)
        
        # Отображаем центральную часть карты
        self.update_canvas()
    
    def draw_grid(self, draw, width, height, min_lat, max_lat, min_lon, max_lon):
        # Вертикальные линии (долгота)
        for lon in self.frange(min_lon, max_lon, (max_lon - min_lon)/10):
            x = (lon - min_lon) / (max_lon - min_lon) * width
            draw.line([(x, 0), (x, height)], fill="#d0d0d0")
        
        # Горизонтальные линии (широта)
        for lat in self.frange(min_lat, max_lat, (max_lat - min_lat)/10):
            y = height - (lat - min_lat) / (max_lat - min_lat) * height
            draw.line([(0, y), (width, y)], fill="#d0d0d0")
    
    def draw_roads(self, draw, coord_to_pixel):
        # Основные дороги (координаты приблизительные)
        roads = [
            [(55.830623, 36.860340), (55.832, 36.858), (55.835, 36.852), (55.838, 36.845), 
             (55.840, 36.838), (55.842, 36.830), (55.844, 36.823), (55.846776, 36.815798)],
            [(55.835, 36.870), (55.835, 36.860), (55.835, 36.850), (55.835, 36.840)],
            [(55.825, 36.840), (55.830, 36.840), (55.835, 36.840), (55.840, 36.840), (55.845, 36.840)]
        ]
        
        for road in roads:
            pixels = [coord_to_pixel(*point) for point in road]
            for i in range(len(pixels)-1):
                draw.line([pixels[i], pixels[i+1]], fill="#808080", width=4)
    
    def draw_route(self, draw, coord_to_pixel):
        # Точки маршрута (следование по дорогам)
        route_points = [
            self.point_a,
            (55.832, 36.858),
            (55.835, 36.852),
            (55.838, 36.845),
            (55.840, 36.838),
            (55.842, 36.830),
            (55.844, 36.823),
            self.point_b
        ]
        
        # Рисуем маршрут
        pixels = [coord_to_pixel(*point) for point in route_points]
        if len(pixels) > 1:
            draw.line(pixels, fill="red", width=3)
        
        # Рисуем точки A и B
        a_pixel = coord_to_pixel(*self.point_a)
        b_pixel = coord_to_pixel(*self.point_b)
        
        draw.ellipse([a_pixel[0]-8, a_pixel[1]-8, a_pixel[0]+8, a_pixel[1]+8], fill="blue")
        draw.ellipse([b_pixel[0]-8, b_pixel[1]-8, b_pixel[0]+8, b_pixel[1]+8], fill="green")
        
        # Подписи
        draw.text((a_pixel[0]+10, a_pixel[1]-10), "A", fill="black")
        draw.text((b_pixel[0]+10, b_pixel[1]-10), "B", fill="black")
    
    def draw_poi(self, draw, coord_to_pixel):
        # Некоторые точки интереса (для примера)
        pois = [
            (55.835, 36.840, "Центр", "blue"),
            (55.838, 36.850, "Парк", "green"),
            (55.842, 36.830, "Магазин", "orange")
        ]
        
        for lat, lon, name, color in pois:
            x, y = coord_to_pixel(lat, lon)
            draw.ellipse([x-6, y-6, x+6, y+6], fill=color)
            draw.text((x+10, y-10), name, fill="black")
    
    def update_canvas(self):
        if not hasattr(self, 'map_image'):
            return
            
        # Отображаем центральную часть карты
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width < 10 or canvas_height < 10:
            return
            
        img_width, img_height = self.map_image.size
        x0 = (img_width - canvas_width) // 2
        y0 = (img_height - canvas_height) // 2
        x1 = x0 + canvas_width
        y1 = y0 + canvas_height
        
        # Копируем центральную часть
        visible_area = self.map_image.crop((x0, y0, x1, y1))
        self.visible_photo = ImageTk.PhotoImage(visible_area)
        
        # Обновляем холст
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.visible_photo)
        
        # Обновляем статус бар
        self.status_bar.config(text=f"Координаты: {self.center[0]:.6f}, {self.center[1]:.6f} | Масштаб: {self.zoom}")
    
    # --- Обработчики событий ---
    def start_drag(self, event):
        self.drag_start = (event.x, event.y)
    
    def on_drag(self, event):
        if self.drag_start:
            dx = event.x - self.drag_start[0]
            dy = event.y - self.drag_start[1]
            
            # Пересчитываем центр карты
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width > 0 and canvas_height > 0:
                img_width, img_height = self.map_image.size
                
                # Рассчитываем смещение в координатах
                lat_span = 180 / (2 ** (self.zoom + 1))
                lon_span = lat_span * (canvas_width / canvas_height)
                
                dlat = -dy / canvas_height * 2 * lat_span
                dlon = dx / canvas_width * 2 * lon_span
                
                self.center = (self.center[0] + dlat, self.center[1] + dlon)
                self.generate_map()
                
            self.drag_start = (event.x, event.y)
    
    def end_drag(self, event):
        self.drag_start = None
    
    def on_mousewheel(self, event):
        # Масштабирование при прокрутке колеса мыши
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()
    
    def zoom_in(self):
        if self.zoom < 18:
            self.zoom += 1
            self.generate_map()
    
    def zoom_out(self):
        if self.zoom > 1:
            self.zoom -= 1
            self.generate_map()
    
    def search_location(self):
        query = self.search_entry.get()
        if not query:
            return
            
        # Простая имитация поиска
        if "a" in query.lower() or "а" in query.lower():
            self.center = self.point_a
        elif "b" in query.lower() or "б" in query.lower():
            self.center = self.point_b
        elif "центр" in query.lower():
            self.center = (55.835, 36.840)
        
        self.zoom = 15
        self.generate_map()
    
    def frange(self, start, stop, step):
        # Генератор диапазона для float
        while start < stop:
            yield start
            start += step

if __name__ == "__main__":
    root = tk.Tk()
    app = MapApp(root)
    root.mainloop()