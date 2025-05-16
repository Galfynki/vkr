import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import threading
import time
import pickle
import os
import requests
from urllib.parse import urlparse
import json
import hashlib
import uuid
from tkinter import scrolledtext
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.exceptions import ConvergenceWarning
import warnings
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, 
                            confusion_matrix, accuracy_score, precision_score, 
                            recall_score, f1_score)

# Глобальные переменные для авторизации
current_user = None
users_db = {}
DATA_FILE = "users.json"
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

# Функции для работы с пользователями
def load_users():
    global users_db
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            users_db = json.load(f)

def save_users():
    with open(DATA_FILE, "w") as f:
        json.dump(users_db, f)

def hash_password(password):
    salt = uuid.uuid4().hex
    return hashlib.sha256(salt.encode() + password.encode()).hexdigest() + ':' + salt

def check_password(hashed_password, user_password):
    password, salt = hashed_password.split(':')
    return password == hashlib.sha256(salt.encode() + user_password.encode()).hexdigest()

class VehicleTracker:
    def __init__(self):
        self.vehicles = {}
        self.next_id = 0
        self.accident_history = []
        self.model = self.init_model()
        self.features = []
        self.labels = []
        self.load_model()
        self.recording = False
        self.video_writer = None
        self.recording_start_time = None
        self.dataset_path = "accident_dataset.csv"
        self.load_dataset()
        
    def init_model(self):
        return RandomForestClassifier(n_estimators=100, random_state=42)
    
    def load_model(self):
        try:
            if os.path.exists('accident_model.pkl'):
                with open('accident_model.pkl', 'rb') as f:
                    self.model = pickle.load(f)
                print("Model loaded successfully")
            else:
                print("No saved model found, initializing new one")
                self.model = self.init_model()
        except Exception as e:
            print(f"Error loading model: {e}, initializing new one")
            self.model = self.init_model()
    
    def save_model(self):
        try:
            with open('accident_model.pkl', 'wb') as f:
                pickle.dump(self.model, f)
            print("Model saved successfully")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_dataset(self):
        try:
            if os.path.exists(self.dataset_path):
                self.dataset = pd.read_csv(self.dataset_path)
            else:
                self.dataset = pd.DataFrame(columns=[
                    'timestamp', 'vehicle1_id', 'vehicle2_id', 'distance', 
                    'speed_diff', 'lane_change', 'speed1', 'speed2', 'is_accident'
                ])
        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.dataset = pd.DataFrame(columns=[
                'timestamp', 'vehicle1_id', 'vehicle2_id', 'distance', 
                'speed_diff', 'lane_change', 'speed1', 'speed2', 'is_accident'
            ])
    
    def save_to_dataset(self, data):
        try:
            new_row = {
                'timestamp': data['time'],
                'vehicle1_id': data['vehicle1'],
                'vehicle2_id': data['vehicle2'],
                'distance': data['distance'],
                'speed_diff': data['speed_diff'],
                'lane_change': data['lane_change'],
                'speed1': self.vehicles[data['vehicle1']]['speed'],
                'speed2': self.vehicles[data['vehicle2']]['speed'],
                'is_accident': True  # Mark as confirmed accident
            }
            
            self.dataset = pd.concat([self.dataset, pd.DataFrame([new_row])], ignore_index=True)
            self.dataset.to_csv(self.dataset_path, index=False)
            print("Data saved to dataset")
        except Exception as e:
            print(f"Error saving to dataset: {e}")
    
    def start_recording(self, output_path, frame_size, fps=30):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        self.recording = True
        self.recording_start_time = datetime.now()
        print(f"Recording started at {self.recording_start_time}")
    
    def stop_recording(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.recording = False
            duration = datetime.now() - self.recording_start_time
            print(f"Recording stopped. Duration: {duration}")
            self.video_writer = None
    
    def update(self, detections, frame):
        current_vehicles = {}
        
        for detection in detections:
            x, y, w, h, confidence, class_id = detection
            center = (x + w//2, y + h//2)
            
            matched = False
            for vid, data in self.vehicles.items():
                last_center = data['positions'][-1] if data['positions'] else (0, 0)
                distance = np.sqrt((center[0]-last_center[0])**2 + (center[1]-last_center[1])**2)
                
                if distance < 50:  # Threshold for matching
                    current_vehicles[vid] = {
                        'bbox': (x, y, w, h),
                        'positions': data['positions'] + [center],
                        'class_id': class_id,
                        'speed': self.calculate_speed(data['positions'], frame),
                        'last_seen': 0,
                        'confidence': confidence,
                        'collision': False,  # Flag for collision
                        'lane': self.detect_lane(center, frame)  # Detect lane
                    }
                    matched = True
                    break
            
            if not matched:
                current_vehicles[self.next_id] = {
                    'bbox': (x, y, w, h),
                    'positions': [center],
                    'class_id': class_id,
                    'speed': 0,
                    'last_seen': 0,
                    'confidence': confidence,
                    'collision': False,
                    'lane': self.detect_lane(center, frame)
                }
                self.next_id += 1
        
        # Update vehicles and remove lost ones
        self.vehicles = current_vehicles
        
        # Analyze for potential accidents
        accident_detected = self.analyze_potential_accidents(frame)
        
        # Write frame to video if recording
        if self.recording and self.video_writer is not None:
            self.video_writer.write(frame)
            
        return accident_detected
    
    def detect_lane(self, center, frame):
        """Определение полосы движения транспортного средства"""
        height, width = frame.shape[:2]
        lane_width = width // 3  # Предполагаем 3 полосы
        
        if center[0] < lane_width:
            return 1  # Левая полоса
        elif center[0] < 2 * lane_width:
            return 2  # Средняя полоса
        else:
            return 3  # Правая полоса
    
    def calculate_speed(self, positions, frame):
        if len(positions) < 2:
            return 0
        
        # Calculate pixels per frame
        p1 = positions[-1]
        p2 = positions[-2]
        distance_pixels = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        
        # Assuming dashcam with known parameters (this would need calibration)
        # For demo purposes, we'll use arbitrary conversion
        pixels_per_meter = 20  # This should be calibrated for real use
        fps = 30  # Assuming 30 FPS
        
        distance_meters = distance_pixels / pixels_per_meter
        speed_mps = distance_meters * fps
        speed_kph = speed_mps * 3.6
        
        return speed_kph
    
    def analyze_lane_changes(self, vid, vehicle):
        if len(vehicle['positions']) < 10:
            return False
        
        # Simple lane change detection based on x-position variation
        x_positions = [p[0] for p in vehicle['positions'][-10:]]
        std_dev = np.std(x_positions)
        return std_dev > 15  # Threshold for lane change
    
    def calculate_distance(self, vid1, vid2):
        pos1 = self.vehicles[vid1]['positions'][-1]
        pos2 = self.vehicles[vid2]['positions'][-1]
        return np.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)
    
    def analyze_potential_accidents(self, frame):
        vehicle_ids = list(self.vehicles.keys())
        accident_detected = False
        
        for i in range(len(vehicle_ids)):
            vid1 = vehicle_ids[i]
            v1 = self.vehicles[vid1]
            
            for j in range(i+1, len(vehicle_ids)):
                vid2 = vehicle_ids[j]
                v2 = self.vehicles[vid2]
                
                distance = self.calculate_distance(vid1, vid2)
                speed_diff = abs(v1['speed'] - v2['speed'])
                lane_change = (self.analyze_lane_changes(vid1, v1) or self.analyze_lane_changes(vid2, v2))
                
                # Check for collision (very close distance)
                if distance < 20:  # Collision threshold
                    self.vehicles[vid1]['collision'] = True
                    self.vehicles[vid2]['collision'] = True
                    accident_detected = True
                    
                    # Draw collision marker
                    x1, y1, w1, h1 = v1['bbox']
                    x2, y2, w2, h2 = v2['bbox']
                    collision_point = (
                        (x1 + w1//2 + x2 + w2//2) // 2,
                        (y1 + h1//2 + y2 + h2//2) // 2
                    )
                    cv2.circle(frame, collision_point, 20, (0, 0, 255), -1)
                    cv2.putText(frame, "ACCIDENT!", (collision_point[0]-50, collision_point[1]-30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
                # Check for potential collision (close distance and high speed difference)
                if distance < 50 and speed_diff > 20:  # Thresholds
                    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    accident_data = {
                        'time': time_now,
                        'vehicle1': vid1,
                        'vehicle2': vid2,
                        'distance': distance,
                        'speed_diff': speed_diff,
                        'lane_change': lane_change,
                        'speed1': v1['speed'],
                        'speed2': v2['speed'],
                        'lane1': v1.get('lane', 0),
                        'lane2': v2.get('lane', 0),
                        'frame_data': frame.copy() if self.recording else None,
                        'is_confirmed': distance < 20  # Mark as confirmed if collision detected
                    }
                    
                    # Add to history
                    self.accident_history.append(accident_data)
                    
                    # Add to training data
                    features = [
                        distance,
                        speed_diff,
                        1 if lane_change else 0,
                        v1['speed'],
                        v2['speed']
                    ]
                    self.features.append(features)
                    self.labels.append(1 if distance < 20 else 0)  # 1 for accident, 0 for potential
                    
                    # Save confirmed accidents to dataset
                    if distance < 20:
                        self.save_to_dataset(accident_data)
                    
                    # Train model incrementally
                    if len(self.features) % 10 == 0:
                        self.train_model()
        
        return accident_detected
    
    def train_model(self):
        if len(self.features) < 10:
            return
            
        X = np.array(self.features)
        y = np.array(self.labels)
        
        # Add some negative examples (non-accidents)
        n_negative = min(100, len(self.features))
        negative_features = [
            [100 + np.random.rand()*50,  # distance > safe threshold
             np.random.rand()*10,        # small speed difference
             0,                          # no lane change
             50 + np.random.rand()*30,  # speed 1
             50 + np.random.rand()*30]   # speed 2
            for _ in range(n_negative)
        ]
        X_neg = np.array(negative_features)
        y_neg = np.zeros(n_negative)
        
        X = np.vstack([X, X_neg])
        y = np.hstack([y, y_neg])
        
        self.model.fit(X, y)
        self.save_model()
    
    def predict_accident_probability(self, distance, speed_diff, lane_change, speed1, speed2):
        features = [[distance, speed_diff, 1 if lane_change else 0, speed1, speed2]]
        return self.model.predict_proba(features)[0][1]  # Probability of class 1 (accident)

    def export_accident_data(self, filename):
        try:
            df = pd.DataFrame(self.accident_history)
            # Remove frame_data column as it can't be serialized to CSV
            if 'frame_data' in df.columns:
                df = df.drop(columns=['frame_data'])
            df.to_csv(filename, index=False)
            return True
        except Exception as e:
            print(f"Error exporting data: {e}")
            return False

class VideoAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Анализатор дорожного движения с поддержкой камер")
        
        # Initialize camera list before UI setup
        self.camera_list = self.get_available_cameras()
        
        # Initialize vehicle tracker
        self.tracker = VehicleTracker()
        
        # Video variables
        self.video_path = ""
        self.cap = None
        self.playing = False
        self.frame_count = 0
        self.fps = 0
        self.delay = 33  # ~30 fps
        self.camera_mode = False
        self.camera_index = 0
        self.ip_camera_url = ""
        
        # YOLO model (using tiny version for demo)
        try:
            self.net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
            with open("coco.names", "r") as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            layer_names = self.net.getLayerNames()
            unconnected = self.net.getUnconnectedOutLayers()
            if unconnected.ndim == 1:
                self.output_layers = [layer_names[i - 1] for i in unconnected]
            else:
                self.output_layers = [layer_names[i[0] - 1] for i in unconnected]
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить модель YOLO: {e}")
            self.root.destroy()
            return
        
        # Load users
        load_users()
        
        # Create admin if not exists
        if ADMIN_USERNAME not in users_db:
            users_db[ADMIN_USERNAME] = {
                'name': 'Администратор',
                'email': 'admin@example.com',
                'password': hash_password(ADMIN_PASSWORD),
                'role': 'admin'
            }
            save_users()
        
        # Start with login screen
        self.create_login_frame()
        
        # Performance monitoring
        self.last_time = time.time()
        self.frame_times = []
    
    def clear_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()
    
    def create_login_frame(self):
        self.clear_window()
        
        frame = ttk.Frame(self.root, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text="Логин:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.login_entry = ttk.Entry(frame)
        self.login_entry.grid(row=0, column=1, pady=5, padx=5, sticky=tk.EW)
        
        ttk.Label(frame, text="Пароль:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.password_entry = ttk.Entry(frame, show="*")
        self.password_entry.grid(row=1, column=1, pady=5, padx=5, sticky=tk.EW)
        
        login_btn = ttk.Button(frame, text="Войти", command=self.login)
        login_btn.grid(row=2, column=0, columnspan=2, pady=10)
        
        register_btn = ttk.Button(frame, text="Регистрация", command=self.create_register_frame)
        register_btn.grid(row=3, column=0, columnspan=2, pady=10)
        
        frame.columnconfigure(1, weight=1)
    
    def login(self):
        username = self.login_entry.get()
        password = self.password_entry.get()
        
        if not username or not password:
            messagebox.showerror("Ошибка", "Все поля обязательны для заполнения")
            return
        
        if username not in users_db:
            messagebox.showerror("Ошибка", "Пользователь не найден")
            return
        
        if check_password(users_db[username]['password'], password):
            global current_user
            current_user = username
            if username == ADMIN_USERNAME:
                self.create_admin_panel()
            else:
                self.create_user_dashboard()
        else:
            messagebox.showerror("Ошибка", "Неверный пароль")
    
    def create_register_frame(self):
        self.clear_window()
        
        frame = ttk.Frame(self.root, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text="ФИО:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.reg_name_entry = ttk.Entry(frame)
        self.reg_name_entry.grid(row=0, column=1, pady=5, padx=5, sticky=tk.EW)
        
        ttk.Label(frame, text="Email:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.reg_email_entry = ttk.Entry(frame)
        self.reg_email_entry.grid(row=1, column=1, pady=5, padx=5, sticky=tk.EW)
        
        ttk.Label(frame, text="Логин:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.reg_login_entry = ttk.Entry(frame)
        self.reg_login_entry.grid(row=2, column=1, pady=5, padx=5, sticky=tk.EW)
        
        ttk.Label(frame, text="Пароль:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.reg_password_entry = ttk.Entry(frame, show="*")
        self.reg_password_entry.grid(row=3, column=1, pady=5, padx=5, sticky=tk.EW)
        
        ttk.Label(frame, text="Подтвердите пароль:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.reg_confirm_entry = ttk.Entry(frame, show="*")
        self.reg_confirm_entry.grid(row=4, column=1, pady=5, padx=5, sticky=tk.EW)
        
        register_btn = ttk.Button(frame, text="Зарегистрироваться", command=self.register)
        register_btn.grid(row=5, column=0, columnspan=2, pady=10)
        
        back_btn = ttk.Button(frame, text="Назад", command=self.create_login_frame)
        back_btn.grid(row=6, column=0, columnspan=2, pady=10)
        
        frame.columnconfigure(1, weight=1)
    
    def register(self):
        name = self.reg_name_entry.get()
        email = self.reg_email_entry.get()
        username = self.reg_login_entry.get()
        password = self.reg_password_entry.get()
        confirm = self.reg_confirm_entry.get()
        
        if not all([name, email, username, password, confirm]):
            messagebox.showerror("Ошибка", "Все поля обязательны для заполнения")
            return
        
        if password != confirm:
            messagebox.showerror("Ошибка", "Пароли не совпадают")
            return
        
        if username in users_db:
            messagebox.showerror("Ошибка", "Пользователь с таким логином уже существует")
            return
        
        users_db[username] = {
            'name': name,
            'email': email,
            'password': hash_password(password),
            'role': 'user'
        }
        save_users()
        messagebox.showinfo("Успех", "Регистрация прошла успешно")
        self.create_login_frame()
    
    def create_admin_panel(self):
        self.clear_window()
        
        self.root.title("Панель администратора")
        
        # Создаем вкладки
        tab_control = ttk.Notebook(self.root)
        
        # Вкладка пользователей
        users_tab = ttk.Frame(tab_control)
        tab_control.add(users_tab, text="Пользователи")
        
        # Таблица пользователей
        columns = ("#1", "#2", "#3", "#4")
        self.users_tree = ttk.Treeview(users_tab, columns=columns, show="headings")
        self.users_tree.heading("#1", text="Логин")
        self.users_tree.heading("#2", text="ФИО")
        self.users_tree.heading("#3", text="Email")
        self.users_tree.heading("#4", text="Роль")
        
        self.users_tree.column("#1", width=100)
        self.users_tree.column("#2", width=150)
        self.users_tree.column("#3", width=150)
        self.users_tree.column("#4", width=80)
        
        scrollbar = ttk.Scrollbar(users_tab, orient=tk.VERTICAL, command=self.users_tree.yview)
        self.users_tree.configure(yscroll=scrollbar.set)
        
        self.users_tree.grid(row=0, column=0, sticky=tk.NSEW)
        scrollbar.grid(row=0, column=1, sticky=tk.NS)
        
        # Кнопки управления
        btn_frame = ttk.Frame(users_tab)
        btn_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        edit_btn = ttk.Button(btn_frame, text="Редактировать", command=self.edit_user)
        edit_btn.pack(side=tk.LEFT, padx=5)
        
        delete_btn = ttk.Button(btn_frame, text="Удалить", command=self.delete_user)
        delete_btn.pack(side=tk.LEFT, padx=5)
        
        change_pass_btn = ttk.Button(btn_frame, text="Сменить пароль", command=self.change_password)
        change_pass_btn.pack(side=tk.LEFT, padx=5)
        
        logout_btn = ttk.Button(btn_frame, text="Выйти", command=self.logout)
        logout_btn.pack(side=tk.RIGHT, padx=5)
        
        # Вкладка анализа дорожного движения
        analysis_tab = ttk.Frame(tab_control)
        tab_control.add(analysis_tab, text="Анализ дорожного движения")
        self.setup_analysis_tab(analysis_tab)
        
        # Вкладка анализа данных
        data_tab = ttk.Frame(tab_control)
        tab_control.add(data_tab, text="Анализ dataset.csv")
        self.setup_data_analysis_tab(data_tab)
        
        tab_control.pack(expand=1, fill="both")
        
        # Заполняем таблицу пользователей
        self.update_users_tree()
        
        users_tab.columnconfigure(0, weight=1)
        users_tab.rowconfigure(0, weight=1)
    
    def setup_data_analysis_tab(self, tab):
        # Фрейм для загрузки файла
        load_frame = ttk.LabelFrame(tab, text="Загрузить данные", padding=10)
        load_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Кнопка загрузки
        load_btn = ttk.Button(load_frame, text="Выбрать файл dataset.csv", command=self.load_dataset_file)
        load_btn.pack(pady=10)
        
        # Область для информации о данных
        self.dataset_info = scrolledtext.ScrolledText(load_frame, height=15, wrap=tk.WORD)
        self.dataset_info.pack(fill=tk.BOTH, expand=True)
        self.dataset_info.insert(tk.END, "Информация о данных появится здесь после загрузки файла.")
        
        # Кнопка анализа
        analyze_btn = ttk.Button(load_frame, text="Анализировать данные", 
                                command=self.analyze_dataset, state=tk.DISABLED)
        analyze_btn.pack(pady=10)
        self.analyze_dataset_btn = analyze_btn
        
        # Фрейм для результатов анализа
        self.results_frame = ttk.Frame(tab)
        self.results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def load_dataset_file(self):
        file_path = filedialog.askopenfilename(
            title="Выберите файл dataset.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.dataset_df = pd.read_csv(file_path)
                self.display_dataset_info()
                self.analyze_dataset_btn.config(state=tk.NORMAL)
                messagebox.showinfo("Успех", "Данные успешно загружены!")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить файл: {str(e)}")
    
    def display_dataset_info(self):
        self.dataset_info.delete(1.0, tk.END)
        
        # Основная информация
        self.dataset_info.insert(tk.END, "=== Основная информация ===\n")
        self.dataset_info.insert(tk.END, f"Количество строк: {len(self.dataset_df)}\n")
        self.dataset_info.insert(tk.END, f"Количество столбцов: {len(self.dataset_df.columns)}\n\n")
        
        # Информация о столбцах
        self.dataset_info.insert(tk.END, "=== Информация о столбцах ===\n")
        for col in self.dataset_df.columns:
            self.dataset_info.insert(tk.END, f"{col}: {self.dataset_df[col].dtype}\n")
        
        # Пример данных
        self.dataset_info.insert(tk.END, "\n=== Первые 5 строк ===\n")
        self.dataset_info.insert(tk.END, str(self.dataset_df.head()))
    
    def analyze_dataset(self):
        if not hasattr(self, 'dataset_df') or self.dataset_df is None:
            messagebox.showerror("Ошибка", "Данные не загружены!")
            return
        
        # Очищаем фрейм результатов
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        try:
            # Подготовка данных
            X = self.dataset_df[['distance', 'speed_diff', 'lane_change', 'speed1', 'speed2']].copy()
            y = self.dataset_df['is_accident']
            
            # Разделение данных
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            
            # Обучение модели
            model = LogisticRegression(max_iter=300)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Вычисление метрик
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Создаем фрейм для метрик
            metrics_frame = ttk.LabelFrame(self.results_frame, text="Метрики классификации", padding=10)
            metrics_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Отображаем метрики
            ttk.Label(metrics_frame, text=f"Accuracy (Точность): {accuracy:.2f}").pack(anchor=tk.W)
            ttk.Label(metrics_frame, text=f"Precision (Точность): {precision:.2f}").pack(anchor=tk.W)
            ttk.Label(metrics_frame, text=f"Recall (Полнота): {recall:.2f}").pack(anchor=tk.W)
            ttk.Label(metrics_frame, text=f"F1-score (F-мера): {f1:.2f}").pack(anchor=tk.W)
            
            # Матрица ошибок
            cm_frame = ttk.LabelFrame(self.results_frame, text="Матрица ошибок", padding=10)
            cm_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            cm = confusion_matrix(y_test, y_pred)
            cm_text = "\n".join([" ".join([str(cell) for cell in row]) for row in cm])
            cm_label = ttk.Label(cm_frame, text=cm_text, font=('Courier', 10))
            cm_label.pack()
            
            # Отчет о классификации
            report_frame = ttk.LabelFrame(self.results_frame, text="Отчет о классификации", padding=10)
            report_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            report = classification_report(y_test, y_pred, zero_division=0)
            report_text = scrolledtext.ScrolledText(report_frame, wrap=tk.WORD, height=10)
            report_text.insert(tk.END, report)
            report_text.pack(fill=tk.BOTH, expand=True)
            
            # График важности признаков
            fig, ax = plt.subplots(figsize=(8, 4))
            feature_importances = pd.Series(model.coef_[0], index=X.columns)
            feature_importances.plot(kind='barh', ax=ax)
            ax.set_title('Важность признаков для модели')
            
            # Встраиваем график в Tkinter
            canvas = FigureCanvasTkAgg(fig, master=self.results_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Ошибка анализа", f"Произошла ошибка: {str(e)}")
    
    def update_users_tree(self):
        for item in self.users_tree.get_children():
            self.users_tree.delete(item)
        
        for username, data in users_db.items():
            self.users_tree.insert("", tk.END, values=(
                username,
                data['name'],
                data['email'],
                data.get('role', 'user')
            ))
    
    def edit_user(self):
        selected = self.users_tree.selection()
        if not selected:
            messagebox.showwarning("Предупреждение", "Выберите пользователя")
            return
        
        username = self.users_tree.item(selected[0])['values'][0]
        
        if username == ADMIN_USERNAME:
            messagebox.showwarning("Предупреждение", "Нельзя редактировать администратора")
            return
        
        self.edit_user_window(username)
    
    def edit_user_window(self, username):
        edit_win = tk.Toplevel(self.root)
        edit_win.title("Редактирование пользователя")
        
        frame = ttk.Frame(edit_win, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        user_data = users_db[username]
        
        ttk.Label(frame, text="ФИО:").grid(row=0, column=0, sticky=tk.W, pady=5)
        name_entry = ttk.Entry(frame)
        name_entry.insert(0, user_data['name'])
        name_entry.grid(row=0, column=1, pady=5, padx=5, sticky=tk.EW)
        
        ttk.Label(frame, text="Email:").grid(row=1, column=0, sticky=tk.W, pady=5)
        email_entry = ttk.Entry(frame)
        email_entry.insert(0, user_data['email'])
        email_entry.grid(row=1, column=1, pady=5, padx=5, sticky=tk.EW)
        
        ttk.Label(frame, text="Роль:").grid(row=2, column=0, sticky=tk.W, pady=5)
        role_combobox = ttk.Combobox(frame, values=["user", "admin"], state="readonly")
        role_combobox.set(user_data.get('role', 'user'))
        role_combobox.grid(row=2, column=1, pady=5, padx=5, sticky=tk.EW)
        
        def save_changes():
            users_db[username]['name'] = name_entry.get()
            users_db[username]['email'] = email_entry.get()
            users_db[username]['role'] = role_combobox.get()
            save_users()
            self.update_users_tree()
            edit_win.destroy()
            messagebox.showinfo("Успех", "Изменения сохранены")
        
        save_btn = ttk.Button(frame, text="Сохранить", command=save_changes)
        save_btn.grid(row=3, column=0, columnspan=2, pady=10)
        
        frame.columnconfigure(1, weight=1)
    
    def delete_user(self):
        selected = self.users_tree.selection()
        if not selected:
            messagebox.showwarning("Предупреждение", "Выберите пользователя")
            return
        
        username = self.users_tree.item(selected[0])['values'][0]
        
        if username == ADMIN_USERNAME:
            messagebox.showwarning("Предупреждение", "Нельзя удалить администратора")
            return
        
        if messagebox.askyesno("Подтверждение", f"Удалить пользователя {username}?"):
            del users_db[username]
            save_users()
            self.update_users_tree()
            messagebox.showinfo("Успех", "Пользователь удален")
    
    def change_password(self):
        selected = self.users_tree.selection()
        if not selected:
            messagebox.showwarning("Предупреждение", "Выберите пользователя")
            return
        
        username = self.users_tree.item(selected[0])['values'][0]
        
        change_win = tk.Toplevel(self.root)
        change_win.title("Смена пароля")
        
        frame = ttk.Frame(change_win, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text="Новый пароль:").grid(row=0, column=0, sticky=tk.W, pady=5)
        pass_entry = ttk.Entry(frame, show="*")
        pass_entry.grid(row=0, column=1, pady=5, padx=5, sticky=tk.EW)
        
        ttk.Label(frame, text="Подтвердите пароль:").grid(row=1, column=0, sticky=tk.W, pady=5)
        confirm_entry = ttk.Entry(frame, show="*")
        confirm_entry.grid(row=1, column=1, pady=5, padx=5, sticky=tk.EW)
        
        def save_password():
            new_pass = pass_entry.get()
            confirm = confirm_entry.get()
            
            if not new_pass or not confirm:
                messagebox.showerror("Ошибка", "Введите пароль")
                return
            
            if new_pass != confirm:
                messagebox.showerror("Ошибка", "Пароли не совпадают")
                return
            
            users_db[username]['password'] = hash_password(new_pass)
            save_users()
            change_win.destroy()
            messagebox.showinfo("Успех", "Пароль изменен")
        
        save_btn = ttk.Button(frame, text="Сохранить", command=save_password)
        save_btn.grid(row=2, column=0, columnspan=2, pady=10)
        
        frame.columnconfigure(1, weight=1)
    
    def setup_analysis_tab(self, tab):
        # Video display frame
        video_frame = ttk.Frame(tab)
        video_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.canvas = tk.Canvas(video_frame, bg='black')
        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        
        self.vscroll = ttk.Scrollbar(video_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.hscroll = ttk.Scrollbar(video_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.hscroll.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.canvas.configure(yscrollcommand=self.vscroll.set, xscrollcommand=self.hscroll.set)
        self.canvas.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        
        self.video_label = ttk.Label(self.canvas)
        self.canvas.create_window((0, 0), window=self.video_label, anchor="nw")
        
        # Controls frame
        controls_frame = ttk.LabelFrame(tab, text="Управление видео", padding="10")
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Source selection
        source_frame = ttk.Frame(controls_frame)
        source_frame.pack(fill=tk.X, pady=2)
        
        ttk.Button(source_frame, text="Открыть файл", command=self.open_video).pack(side=tk.LEFT, padx=2)
        
        self.camera_btn = ttk.Button(source_frame, text="Подключить камеру", command=self.connect_camera)
        self.camera_btn.pack(side=tk.LEFT, padx=2)
        
        self.camera_menu = ttk.Combobox(source_frame, state='readonly')
        self.camera_menu.pack(side=tk.LEFT, padx=2)
        self.update_camera_menu()
        
        # IP Camera settings
        ip_camera_frame = ttk.Frame(controls_frame)
        ip_camera_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(ip_camera_frame, text="IP камера:").pack(side=tk.LEFT)
        self.ip_camera_entry = ttk.Entry(ip_camera_frame)
        self.ip_camera_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        self.ip_camera_entry.insert(0, "http://192.168.1.100:8080/video")
        
        ttk.Button(ip_camera_frame, text="Подключить", command=self.connect_ip_camera).pack(side=tk.LEFT, padx=2)
        
        # Playback controls
        playback_frame = ttk.Frame(controls_frame)
        playback_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(playback_frame, text="Старт/Стоп", command=self.toggle_play).pack(side=tk.LEFT, padx=2)
        
        # Recording controls
        record_frame = ttk.Frame(controls_frame)
        record_frame.pack(fill=tk.X, pady=2)
        
        self.record_btn = ttk.Button(record_frame, text="Начать запись", command=self.toggle_recording)
        self.record_btn.pack(side=tk.LEFT, padx=2)
        
        # Stats frame
        stats_frame = ttk.LabelFrame(tab, text="Статистика", padding="10")
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.vehicle_count_label = ttk.Label(stats_frame, text="Транспортных средств: 0")
        self.vehicle_count_label.pack(anchor=tk.W)
        
        self.accident_count_label = ttk.Label(stats_frame, text="Потенциальных ДТП: 0")
        self.accident_count_label.pack(anchor=tk.W)
        
        self.accident_risk_label = ttk.Label(stats_frame, text="Риск ДТП: Низкий")
        self.accident_risk_label.pack(anchor=tk.W)
        
        self.fps_label = ttk.Label(stats_frame, text="FPS: 0")
        self.fps_label.pack(anchor=tk.W)
        
        # Accident log
        log_frame = ttk.LabelFrame(tab, text="Журнал событий", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.log_text = tk.Text(log_frame, height=10, width=40)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        log_controls = ttk.Frame(log_frame)
        log_controls.pack(fill=tk.X, pady=5)
        
        ttk.Button(log_controls, text="Экспорт", command=self.export_log).pack(side=tk.LEFT, padx=2)
        ttk.Button(log_controls, text="Очистить", command=self.clear_log).pack(side=tk.RIGHT, padx=2)
        
        # Settings
        settings_frame = ttk.LabelFrame(tab, text="Настройки", padding="10")
        settings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(settings_frame, text="Порог риска ДТП:").pack(anchor=tk.W)
        self.risk_threshold = tk.DoubleVar(value=0.7)
        ttk.Scale(settings_frame, from_=0.1, to=1.0, variable=self.risk_threshold, 
                 orient=tk.HORIZONTAL).pack(fill=tk.X)
        
        # Model training button
        ttk.Button(settings_frame, text="Обучить модель", command=self.train_model).pack(fill=tk.X, pady=2)
    
    def create_user_dashboard(self):
        self.clear_window()
        self.root.title(f"Личный кабинет - {current_user}")
        
        # Создаем вкладки
        tab_control = ttk.Notebook(self.root)
        
        # Вкладка профиля
        profile_tab = ttk.Frame(tab_control)
        tab_control.add(profile_tab, text="Профиль")
        
        # Информация о пользователе
        user_info = users_db[current_user]
        
        ttk.Label(profile_tab, text="ФИО:").grid(row=0, column=0, sticky=tk.W, pady=5, padx=10)
        ttk.Label(profile_tab, text=user_info['name']).grid(row=0, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(profile_tab, text="Email:").grid(row=1, column=0, sticky=tk.W, pady=5, padx=10)
        ttk.Label(profile_tab, text=user_info['email']).grid(row=1, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(profile_tab, text="Логин:").grid(row=2, column=0, sticky=tk.W, pady=5, padx=10)
        ttk.Label(profile_tab, text=current_user).grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # Кнопки управления
        btn_frame = ttk.Frame(profile_tab)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=20)
        
        edit_btn = ttk.Button(btn_frame, text="Изменить данные", command=self.edit_own_profile)
        edit_btn.pack(side=tk.LEFT, padx=5)
        
        change_pass_btn = ttk.Button(btn_frame, text="Сменить пароль", command=self.change_own_password)
        change_pass_btn.pack(side=tk.LEFT, padx=5)
        
        logout_btn = ttk.Button(btn_frame, text="Выйти", command=self.logout)
        logout_btn.pack(side=tk.RIGHT, padx=5)
        
        # Вкладка анализа дорожного движения
        
        analysis_tab = ttk.Frame(tab_control)
        tab_control.add(analysis_tab, text="Анализ дорожного движения")
        self.setup_analysis_tab(analysis_tab)
        
        # Вкладка анализа данных (только для просмотра)
        data_tab = ttk.Frame(tab_control)
        tab_control.add(data_tab, text="Просмотр dataset.csv")
        self.setup_data_view_tab(data_tab)
        
        tab_control.pack(expand=1, fill="both")
        
        profile_tab.columnconfigure(1, weight=1)
    
    def setup_data_view_tab(self, tab):
        # Фрейм для загрузки файла
        load_frame = ttk.LabelFrame(tab, text="Просмотр данных", padding=10)
        load_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Кнопка загрузки
        load_btn = ttk.Button(load_frame, text="Выбрать файл dataset.csv", command=self.load_dataset_view)
        load_btn.pack(pady=10)
        
        # Таблица для отображения данных
        self.dataset_tree = ttk.Treeview(load_frame)
        self.dataset_tree.pack(fill=tk.BOTH, expand=True)
        
        # Полоса прокрутки
        scrollbar = ttk.Scrollbar(load_frame, orient=tk.VERTICAL, command=self.dataset_tree.yview)
        self.dataset_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def load_dataset_view(self):
        file_path = filedialog.askopenfilename(
            title="Выберите файл dataset.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                df = pd.read_csv(file_path)
                
                # Очищаем текущее дерево
                self.dataset_tree.delete(*self.dataset_tree.get_children())
                
                # Устанавливаем колонки
                self.dataset_tree["columns"] = list(df.columns)
                for col in df.columns:
                    self.dataset_tree.heading(col, text=col)
                    self.dataset_tree.column(col, width=100)
                
                # Заполняем данными
                for i, row in df.iterrows():
                    self.dataset_tree.insert("", tk.END, values=list(row))
                
                messagebox.showinfo("Успех", "Данные успешно загружены!")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить файл: {str(e)}")
    
    def edit_own_profile(self):
        edit_win = tk.Toplevel(self.root)
        edit_win.title("Редактирование профиля")
        
        frame = ttk.Frame(edit_win, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        user_data = users_db[current_user]
        
        ttk.Label(frame, text="ФИО:").grid(row=0, column=0, sticky=tk.W, pady=5)
        name_entry = ttk.Entry(frame)
        name_entry.insert(0, user_data['name'])
        name_entry.grid(row=0, column=1, pady=5, padx=5, sticky=tk.EW)
        
        ttk.Label(frame, text="Email:").grid(row=1, column=0, sticky=tk.W, pady=5)
        email_entry = ttk.Entry(frame)
        email_entry.insert(0, user_data['email'])
        email_entry.grid(row=1, column=1, pady=5, padx=5, sticky=tk.EW)
        
        def save_changes():
            users_db[current_user]['name'] = name_entry.get()
            users_db[current_user]['email'] = email_entry.get()
            save_users()
            edit_win.destroy()
            self.create_user_dashboard()
            messagebox.showinfo("Успех", "Изменения сохранены")
        
        save_btn = ttk.Button(frame, text="Сохранить", command=save_changes)
        save_btn.grid(row=2, column=0, columnspan=2, pady=10)
        
        frame.columnconfigure(1, weight=1)
    
    def change_own_password(self):
        change_win = tk.Toplevel(self.root)
        change_win.title("Смена пароля")
        
        frame = ttk.Frame(change_win, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text="Текущий пароль:").grid(row=0, column=0, sticky=tk.W, pady=5)
        current_pass_entry = ttk.Entry(frame, show="*")
        current_pass_entry.grid(row=0, column=1, pady=5, padx=5, sticky=tk.EW)
        
        ttk.Label(frame, text="Новый пароль:").grid(row=1, column=0, sticky=tk.W, pady=5)
        new_pass_entry = ttk.Entry(frame, show="*")
        new_pass_entry.grid(row=1, column=1, pady=5, padx=5, sticky=tk.EW)
        
        ttk.Label(frame, text="Подтвердите пароль:").grid(row=2, column=0, sticky=tk.W, pady=5)
        confirm_entry = ttk.Entry(frame, show="*")
        confirm_entry.grid(row=2, column=1, pady=5, padx=5, sticky=tk.EW)
        
        def save_password():
            current_pass = current_pass_entry.get()
            new_pass = new_pass_entry.get()
            confirm = confirm_entry.get()
            
            if not all([current_pass, new_pass, confirm]):
                messagebox.showerror("Ошибка", "Все поля обязательны для заполнения")
                return
            
            if not check_password(users_db[current_user]['password'], current_pass):
                messagebox.showerror("Ошибка", "Неверный текущий пароль")
                return
            
            if new_pass != confirm:
                messagebox.showerror("Ошибка", "Новые пароли не совпадают")
                return
            
            users_db[current_user]['password'] = hash_password(new_pass)
            save_users()
            change_win.destroy()
            messagebox.showinfo("Успех", "Пароль успешно изменен")
        
        save_btn = ttk.Button(frame, text="Сохранить", command=save_password)
        save_btn.grid(row=3, column=0, columnspan=2, pady=10)
        
        frame.columnconfigure(1, weight=1)
    
    def get_available_cameras(self, max_test=5):
        available_cameras = []
        for i in range(max_test):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        return available_cameras
    
    def update_camera_menu(self):
        if self.camera_list:
            self.camera_menu['values'] = [f"Камера {i}" for i in self.camera_list]
            self.camera_menu.current(0)
        else:
            self.camera_menu['values'] = ["Камеры не найдены"]
            self.camera_btn.config(state=tk.DISABLED)
    
    def connect_camera(self):
        if not self.camera_list:
            messagebox.showerror("Ошибка", "Не найдены доступные камеры!")
            return
            
        selected = self.camera_menu.current()
        if selected >= 0:
            self.camera_index = self.camera_list[selected]
            self.camera_mode = True
            self.video_path = f"Камера {self.camera_index}"
            
            # Release previous capture if any
            if self.cap is not None:
                self.cap.release()
                
            # Try to open camera
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                messagebox.showerror("Ошибка", f"Не удалось подключиться к камере {self.camera_index}!")
                return
                
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0:
                self.fps = 30  # Default value if camera doesn't report FPS
            self.delay = int(1000 / self.fps)
            self.frame_count = 0
            self.playing = True
            self.update_video()
    
    def connect_ip_camera(self):
        url = self.ip_camera_entry.get()
        if not url:
            messagebox.showerror("Ошибка", "Введите URL IP камеры!")
            return
            
        # Validate URL
        try:
            result = urlparse(url)
            if not all([result.scheme, result.netloc]):
                raise ValueError("Invalid URL")
        except:
            messagebox.showerror("Ошибка", "Неверный URL адрес!")
            return
            
        self.ip_camera_url = url
        self.camera_mode = False
        
        # Release previous capture if any
        if self.cap is not None:
            self.cap.release()
            
        # Try to open IP camera
        self.cap = cv2.VideoCapture(url)
        if not self.cap.isOpened():
            messagebox.showerror("Ошибка", f"Не удалось подключиться к IP камере {url}!")
            return
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30  # Default value if camera doesn't report FPS
        self.delay = int(1000 / self.fps)
        self.frame_count = 0
        self.playing = True
        self.update_video()
    
    def open_video(self):
        self.camera_mode = False
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")])
        if file_path:
            self.video_path = file_path
            
            # Release previous capture if any
            if self.cap is not None:
                self.cap.release()
                
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                messagebox.showerror("Ошибка", "Не удалось открыть видеофайл!")
                return
                
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.delay = int(1000 / self.fps)
            self.frame_count = 0
            self.playing = True
            self.update_video()
    
    def toggle_play(self):
        if self.cap:
            self.playing = not self.playing
            if self.playing:
                self.update_video()
    
    def toggle_recording(self):
        if not self.tracker.recording:
            # Start recording
            default_filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
            filename = filedialog.asksaveasfilename(
                initialfile=default_filename,
                defaultextension=".avi",
                filetypes=[("AVI files", "*.avi"), ("All files", "*.*")]
            )
            
            if filename:
                # Get frame size from current video source
                if self.cap is not None:
                    width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    self.tracker.start_recording(filename, (width, height), self.fps)
                    self.record_btn.config(text="Остановить запись")
        else:
            # Stop recording
            self.tracker.stop_recording()
            self.record_btn.config(text="Начать запись")
    
    def update_video(self):
        if not self.playing or not self.cap:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            if self.camera_mode or self.ip_camera_url:
                # For camera, just continue trying
                self.root.after(self.delay, self.update_video)
                return
            else:
                # For video file, stop at end
                self.playing = False
                self.cap.release()
                messagebox.showinfo("Информация", "Видео закончилось!")
                return
            
        # Calculate FPS
        current_time = time.time()
        elapsed = current_time - self.last_time
        self.last_time = current_time
        self.frame_times.append(elapsed)
        if len(self.frame_times) > 10:
            self.frame_times.pop(0)
        current_fps = 1.0 / (sum(self.frame_times) / len(self.frame_times)) if self.frame_times else 0
        
        self.frame_count += 1
        
        # Process every 3rd frame for performance (or adjust as needed)
        if self.frame_count % 3 == 0:
            processed_frame = self.process_frame(frame)
        else:
            processed_frame = frame
            
        # Display frame
        img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)
        
        self.video_label.img = img  # Keep reference
        self.video_label.configure(image=img)
        
        # Update canvas scroll region
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
        # Update UI
        self.update_stats(current_fps)
        
        # Schedule next update
        self.root.after(self.delay, self.update_video)
    
    def process_frame(self, frame):
        height, width, channels = frame.shape
        
        # Detect objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        
        # Process detections
        class_ids = []
        confidences = []
        boxes = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id in [2, 3, 5, 7]:  # Cars, bikes, buses, trucks
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-max suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        detections = []
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                detections.append([x, y, w, h, confidences[i], class_ids[i]])
                
                # Draw bounding box
                label = f"{self.classes[class_ids[i]]}: {confidences[i]:.2f}"
                color = (0, 255, 0)  # Default green
                
                # Check if this vehicle is in collision
                for vid, vehicle in self.tracker.vehicles.items():
                    if abs(vehicle['bbox'][0] - x) < 5 and (abs(vehicle['bbox'][1] - y) < 5):
                        if vehicle.get('collision', False):
                            color = (0, 0, 255)  # Red for collision
                        break
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Update tracker and check for accidents
        accident_detected = self.tracker.update(detections, frame)
        
        # Draw vehicle info
        for vid, vehicle in self.tracker.vehicles.items():
            x, y, w, h = vehicle['bbox']
            speed = vehicle['speed']
            lane = vehicle.get('lane', 0)
            
            # Draw ID, speed and lane
            info = f"ID:{vid} {speed:.1f}km/h Полоса:{lane}"
            color = (255, 0, 0)  # Blue for normal
            if vehicle.get('collision', False):
                color = (0, 0, 255)  # Red for collision
                info = f"ACCIDENT! ID:{vid}"
            
            cv2.putText(frame, info, (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw trajectory
            positions = vehicle['positions']
            for i in range(1, len(positions)):
                cv2.line(frame, positions[i-1], positions[i], (0, 0, 255), 2)
        
        return frame
    
    def update_stats(self, current_fps):
        # Update vehicle count
        vehicle_count = len(self.tracker.vehicles)
        self.vehicle_count_label.config(text=f"Транспортных средств: {vehicle_count}")
        
        # Update accident count
        accident_count = len(self.tracker.accident_history)
        self.accident_count_label.config(text=f"Потенциальных ДТП: {accident_count}")
        
        # Update FPS
        self.fps_label.config(text=f"FPS: {current_fps:.1f}")
        
        # Check for high risk situations
        max_risk = 0
        risk_details = ""
        
        vehicle_ids = list(self.tracker.vehicles.keys())
        for i in range(len(vehicle_ids)):
            vid1 = vehicle_ids[i]
            v1 = self.tracker.vehicles[vid1]
            
            for j in range(i+1, len(vehicle_ids)):
                vid2 = vehicle_ids[j]
                v2 = self.tracker.vehicles[vid2]
                
                distance = self.tracker.calculate_distance(vid1, vid2)
                speed_diff = abs(v1['speed'] - v2['speed'])
                lane_change = (self.tracker.analyze_lane_changes(vid1, v1) or 
                              self.tracker.analyze_lane_changes(vid2, v2))
                
                prob = self.tracker.predict_accident_probability(
                    distance, speed_diff, lane_change, v1['speed'], v2['speed'])
                
                if prob > max_risk:
                    max_risk = prob
                    risk_details = (f"ID {vid1} (скорость: {v1['speed']:.1f}km/h, полоса: {v1.get('lane', 0)}) и "
                                  f"ID {vid2} (скорость: {v2['speed']:.1f}km/h, полоса: {v2.get('lane', 0)}): "
                                  f"расстояние={distance:.1f}px, разница скорости={speed_diff:.1f}km/h")
        
        # Update risk label
        if max_risk > self.risk_threshold.get():
            self.accident_risk_label.config(text=f"Риск ДТП: ВЫСОКИЙ ({max_risk:.0%})", foreground='red')
            
            # Log high risk event
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_entry = f"{timestamp} - ВЫСОКИЙ РИСК: {risk_details}\n"
            self.log_text.insert(tk.END, log_entry)
            self.log_text.see(tk.END)
            
            # Flash warning
            self.flash_warning()
        elif max_risk > self.risk_threshold.get() / 2:
            self.accident_risk_label.config(text=f"Риск ДТП: Средний ({max_risk:.0%})", foreground='orange')
        else:
            self.accident_risk_label.config(text=f"Риск ДТП: Низкий ({max_risk:.0%})", foreground='green')
    
    def flash_warning(self):
        original_bg = self.accident_risk_label.cget('background')
        self.accident_risk_label.config(background='red')
        self.root.after(200, lambda: self.accident_risk_label.config(background=original_bg))
    
    def export_log(self):
        default_filename = f"accident_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filename = filedialog.asksaveasfilename(
            initialfile=default_filename,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            success = self.tracker.export_accident_data(filename)
            if success:
                messagebox.showinfo("Успех", f"Данные сохранены в {filename}")
            else:
                messagebox.showerror("Ошибка", "Не удалось экспортировать данные")
    
    def clear_log(self):
        self.log_text.delete(1.0, tk.END)
    
    def train_model(self):
        def train():
            try:
                self.tracker.train_model()
                messagebox.showinfo("Успех", "Модель успешно обучена!")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка при обучении модели: {e}")
        
        # Run training in a separate thread to avoid freezing the UI
        threading.Thread(target=train, daemon=True).start()
    
    def logout(self):
        global current_user
        current_user = None
        if self.tracker.recording:
            self.tracker.stop_recording()
        if self.cap is not None:
            self.cap.release()
        self.create_login_frame()
    
    def on_closing(self):
        if self.tracker.recording:
            self.tracker.stop_recording()
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()

def main():
    root = tk.Tk()
    try:
        app = VideoAnalyzerApp(root)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Критическая ошибка", f"Произошла ошибка: {str(e)}")
        root.destroy()

if __name__ == "__main__":
   main()