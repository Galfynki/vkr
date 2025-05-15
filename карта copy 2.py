import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import folium
from folium.plugins import Draw
import io
import webbrowser
import yandexmaps  # Для работы с API Яндекс.Карт (если нужно)

class PhotoViewer:
    def __init__(self, root):
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
    def __init__(self, root):
        self.root = root
        self.frame = ttk.Frame(root)
        
        # Создаем карту Folium (альтернатива, если нет API Яндекс.Карт)
        self.map = folium.Map(location=[55.751244, 37.618423], zoom_start=12)
        Draw(export=True).add_to(self.map)
        
        # Сохраняем карту во временный HTML
        self.map.save("temp_map.html")
        
        # Встроим карту в Tkinter через WebView (или браузер)
        self.map_label = ttk.Label(self.frame, text="Карта загружается...")
        self.map_label.pack()
        
        # Кнопки для работы с картой
        self.btn_route = ttk.Button(self.frame, text="Построить маршрут", command=self.add_route)
        self.btn_route.pack(pady=5)
        
        self.btn_add_sign = ttk.Button(self.frame, text="Добавить знак ПДД", command=self.add_traffic_sign)
        self.btn_add_sign.pack(pady=5)
        
        self.btn_add_camera = ttk.Button(self.frame, text="Добавить камеру", command=self.add_camera)
        self.btn_add_camera.pack(pady=5)
        
        # Поле для ввода координат
        self.coord_entry = ttk.Entry(self.frame, width=30)
        self.coord_entry.pack(pady=5)
        self.coord_entry.insert(0, "55.751244, 37.618423")
        
        self.btn_search = ttk.Button(self.frame, text="Поиск по координатам", command=self.search_coords)
        self.btn_search.pack(pady=5)
        
        # Открываем карту в браузере (альтернатива)
        webbrowser.open("temp_map.html")
    
    def add_route(self):
        messagebox.showinfo("Маршрут", "Функция маршрута будет реализована здесь.")
    
    def add_traffic_sign(self):
        messagebox.showinfo("Знак ПДД", "Добавление знака дорожного движения.")
    
    def add_camera(self):
        messagebox.showinfo("Камера", "Добавление камеры наблюдения.")
    
    def search_coords(self):
        coords = self.coord_entry.get()
        try:
            lat, lon = map(float, coords.split(','))
            self.map.location = [lat, lon]
            self.map.save("temp_map.html")
            webbrowser.open("temp_map.html")
        except:
            messagebox.showerror("Ошибка", "Неверный формат координат. Используйте: широта, долгота")


class App:
    def __init__(self, root):
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


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.geometry("600x500")
    root.mainloop()