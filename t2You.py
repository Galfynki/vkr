import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import time
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class VehicleTracker:
    def __init__(self):
        self.vehicles = {}
        self.next_id = 0
        self.accident_history = []
        self.traffic_history = []
        self.model = self.init_model()
        self.features = []
        self.labels = []
        self.load_model()
        
    def init_model(self):
        return RandomForestClassifier(n_estimators=100, random_state=42)
    
    def load_model(self):
        try:
            with open('accident_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            print("Model loaded successfully")
        except (FileNotFoundError, EOFError, Exception) as e:
            print(f"No saved model found or error loading: {e}, initializing new one")
            self.model = self.init_model()
    
    def save_model(self):
        with open('accident_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        print("Model saved successfully")
    
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
                        'lane': self.detect_lane(center, frame)
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
                    'lane': self.detect_lane(center, frame)
                }
                self.next_id += 1
        
        # Update vehicles and remove lost ones
        self.vehicles = current_vehicles
        
        # Record traffic data
        self.record_traffic_data()
        
        # Analyze for potential accidents
        self.analyze_potential_accidents(frame)
    
    def detect_lane(self, center, frame):
        # Simple lane detection based on x-position
        height, width = frame.shape[:2]
        lane_width = width // 3  # Assuming 3 lanes
        return center[0] // lane_width + 1
    
    def record_traffic_data(self):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lane_counts = {1: 0, 2: 0, 3: 0}
        
        for vid, vehicle in self.vehicles.items():
            if 'lane' in vehicle:
                lane = vehicle['lane']
                if lane in lane_counts:
                    lane_counts[lane] += 1
        
        self.traffic_history.append({
            'time': timestamp,
            'lane1': lane_counts[1],
            'lane2': lane_counts[2],
            'lane3': lane_counts[3],
            'total': len(self.vehicles)
        })
    
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
        
        for i in range(len(vehicle_ids)):
            vid1 = vehicle_ids[i]
            v1 = self.vehicles[vid1]
            
            for j in range(i+1, len(vehicle_ids)):
                vid2 = vehicle_ids[j]
                v2 = self.vehicles[vid2]
                
                distance = self.calculate_distance(vid1, vid2)
                speed_diff = abs(v1['speed'] - v2['speed'])
                lane_change = self.analyze_lane_changes(vid1, v1) or self.analyze_lane_changes(vid2, v2)
                
                # Check for potential collision
                if distance < 50 and speed_diff > 20:  # Thresholds
                    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    accident_data = {
                        'time': time_now,
                        'vehicle1': vid1,
                        'vehicle2': vid2,
                        'distance': distance,
                        'speed_diff': speed_diff,
                        'lane_change': lane_change
                    }
                    
                    # Add to history
                    self.accident_history.append(accident_data)
                    
                    # Add to training data
                    features = [
                        distance,
                        speed_diff,
                        1 if accident_data['lane_change'] else 0,
                        v1['speed'],
                        v2['speed']
                    ]
                    self.features.append(features)
                    self.labels.append(1)  # 1 for potential accident
                    
                    # Train model incrementally
                    if len(self.features) % 10 == 0:
                        self.train_model()
    
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

class VideoAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Анализатор дорожного движения")
        
        # Initialize vehicle tracker
        self.tracker = VehicleTracker()
        
        # Video variables
        self.video_path = ""
        self.cap = None
        self.playing = False
        self.frame_count = 0
        self.fps = 0
        self.delay = 33  # ~30 fps
        self.last_frame = None
        self.stuck_frames = 0
        self.max_stuck_frames = 5  # Максимальное количество "застывших" кадров перед пропуском
        
        # YOLO model (using tiny version for demo)
        self.net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
        self.classes = []
        with open("coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Fix for different OpenCV versions
        layer_names = self.net.getLayerNames()
        unconnected = self.net.getUnconnectedOutLayers()
        if unconnected.ndim == 1:
            self.output_layers = [layer_names[i - 1] for i in unconnected]
        else:
            self.output_layers = [layer_names[i[0] - 1] for i in unconnected]
        
        # UI Setup
        self.setup_ui()
        
        # Setup traffic plot
        self.setup_traffic_plot()
        
    def setup_ui(self):
        # Main frames
        self.video_frame = ttk.Frame(self.root)
        self.video_frame.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.pack(side=tk.RIGHT, padx=5, pady=5, fill=tk.Y)
        
        # Video display
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack()
        
        # Controls
        ttk.Button(self.control_frame, text="Открыть видео", command=self.open_video).pack(fill=tk.X, pady=2)
        ttk.Button(self.control_frame, text="Старт/Стоп", command=self.toggle_play).pack(fill=tk.X, pady=2)
        
        # Stats frame
        self.stats_frame = ttk.LabelFrame(self.control_frame, text="Статистика")
        self.stats_frame.pack(fill=tk.X, pady=5)
        
        self.vehicle_count_label = ttk.Label(self.stats_frame, text="Транспортных средств: 0")
        self.vehicle_count_label.pack(anchor=tk.W)
        
        self.accident_risk_label = ttk.Label(self.stats_frame, text="Риск ДТП: Низкий")
        self.accident_risk_label.pack(anchor=tk.W)
        
        # Traffic plot frame
        self.plot_frame = ttk.LabelFrame(self.control_frame, text="Загруженность дорог")
        self.plot_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Accident log
        self.log_frame = ttk.LabelFrame(self.control_frame, text="Журнал событий")
        self.log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.log_text = tk.Text(self.log_frame, height=10, width=40)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Settings
        self.settings_frame = ttk.LabelFrame(self.control_frame, text="Настройки")
        self.settings_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(self.settings_frame, text="Порог риска ДТП:").pack(anchor=tk.W)
        self.risk_threshold = tk.DoubleVar(value=0.7)
        ttk.Scale(self.settings_frame, from_=0.1, to=1.0, variable=self.risk_threshold, 
                 orient=tk.HORIZONTAL).pack(fill=tk.X)
    
    def setup_traffic_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.ax.set_title('Загруженность полос')
        self.ax.set_xlabel('Время')
        self.ax.set_ylabel('Количество ТС')
        self.ax.grid(True)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initial empty plot
        self.lines = {
            'lane1': self.ax.plot([], [], label='Полоса 1')[0],
            'lane2': self.ax.plot([], [], label='Полоса 2')[0],
            'lane3': self.ax.plot([], [], label='Полоса 3')[0],
            'total': self.ax.plot([], [], label='Всего', linestyle='--')[0]
        }
        self.ax.legend()
    
    def update_traffic_plot(self):
        if not self.tracker.traffic_history:
            return
            
        # Get last 20 data points
        history = self.tracker.traffic_history[-20:]
        times = [datetime.strptime(h['time'], "%Y-%m-%d %H:%M:%S") for h in history]
        
        # Update each line
        self.lines['lane1'].set_data(times, [h['lane1'] for h in history])
        self.lines['lane2'].set_data(times, [h['lane2'] for h in history])
        self.lines['lane3'].set_data(times, [h['lane3'] for h in history])
        self.lines['total'].set_data(times, [h['total'] for h in history])
        
        # Adjust axes
        self.ax.relim()
        self.ax.autoscale_view()
        
        # Format x-axis
        self.ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
        self.fig.autofmt_xdate()
        
        self.canvas.draw()
    
    def open_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.delay = int(1000 / self.fps) if self.fps > 0 else 33
            self.frame_count = 0
            self.playing = True
            self.last_frame = None
            self.stuck_frames = 0
            self.update_video()
    
    def toggle_play(self):
        if self.cap:
            self.playing = not self.playing
            if self.playing:
                self.update_video()
    
    def update_video(self):
        if not self.playing or not self.cap:
            return
            
        ret, frame = self.cap.read()
        
        # Если видео закончилось, перематываем в начало
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                self.playing = False
                self.cap.release()
                return
        
        # Проверяем, изменился ли кадр
        if self.last_frame is not None:
            frame_diff = cv2.absdiff(frame, self.last_frame)
            change = np.sum(frame_diff) / (frame.shape[0] * frame.shape[1] * 255)
            
            if change < 0.01:  # Порог изменения (1%)
                self.stuck_frames += 1
                if self.stuck_frames < self.max_stuck_frames:
                    # Пропускаем кадр, если он почти не изменился
                    self.root.after(self.delay, self.update_video)
                    return
                else:
                    # Пропускаем вперед, если слишком много одинаковых кадров
                    self.stuck_frames = 0
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.cap.get(cv2.CAP_PROP_POS_FRAMES) + 10)
                    ret, frame = self.cap.read()
                    if not ret:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = self.cap.read()
                        if not ret:
                            self.playing = False
                            self.cap.release()
                            return
            else:
                self.stuck_frames = 0
        
        self.last_frame = frame.copy()
        self.frame_count += 1
        
        # Process every 5th frame for performance
        if self.frame_count % 5 == 0:
            processed_frame = self.process_frame(frame)
        else:
            processed_frame = frame
            
        # Display frame
        img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)
        
        self.video_label.img = img  # Keep reference
        self.video_label.configure(image=img)
        
        # Update UI
        self.update_stats()
        
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
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Update tracker
        self.tracker.update(detections, frame)
        
        # Draw vehicle info
        for vid, vehicle in self.tracker.vehicles.items():
            x, y, w, h = vehicle['bbox']
            speed = vehicle['speed']
            lane = vehicle.get('lane', 0)
            
            # Draw ID, speed and lane
            info = f"ID:{vid} {speed:.1f}km/h Полоса:{lane}"
            cv2.putText(frame, info, (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Draw trajectory
            positions = vehicle['positions']
            for i in range(1, len(positions)):
                cv2.line(frame, positions[i-1], positions[i], (0, 0, 255), 2)
        
        return frame
    
    def update_stats(self):
        # Update vehicle count
        vehicle_count = len(self.tracker.vehicles)
        self.vehicle_count_label.config(text=f"Транспортных средств: {vehicle_count}")
        
        # Update traffic plot
        self.update_traffic_plot()
        
        # Check for high risk situations and log vehicle movements
        max_risk = 0
        risk_details = ""
        
        vehicle_ids = list(self.tracker.vehicles.keys())
        for vid in vehicle_ids:
            vehicle = self.tracker.vehicles[vid]
            
            # Log vehicle movement every 30 frames
            if self.frame_count % 30 == 0:
                timestamp = datetime.now().strftime("%H:%M:%S")
                log_entry = (f"{timestamp} - ID:{vid} Скорость:{vehicle['speed']:.1f}km/h "
                           f"Полоса:{vehicle.get('lane', 'N/A')}\n")
                self.log_text.insert(tk.END, log_entry)
                self.log_text.see(tk.END)
        
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
                    risk_details = (f"ID {vid1} и ID {vid2}: "
                                  f"расстояние={distance:.1f}px, "
                                  f"разница скорости={speed_diff:.1f}km/h, "
                                  f"смена полосы={'да' if lane_change else 'нет'}")
        
        # Update risk label
        if max_risk > self.risk_threshold.get():
            self.accident_risk_label.config(text=f"Риск ДТП: ВЫСОКИЙ ({max_risk:.0%})", foreground='red')
            
            # Log high risk event
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_entry = f"{timestamp} - ВЫСОКИЙ РИСК: {risk_details}\n"
            self.log_text.insert(tk.END, log_entry)
            self.log_text.see(tk.END)
        elif max_risk > self.risk_threshold.get() / 2:
            self.accident_risk_label.config(text=f"Риск ДТП: Средний ({max_risk:.0%})", foreground='orange')
        else:
            self.accident_risk_label.config(text=f"Риск ДТП: Низкий ({max_risk:.0%})", foreground='green')

def main():
    root = tk.Tk()
    app = VideoAnalyzerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()