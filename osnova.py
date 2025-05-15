import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import json
import hashlib
import uuid

# Глобальные переменные
current_user = None
users_db = {}
DATA_FILE = "users.json"
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

# Загрузка/сохранение пользователей
def load_users():
    global users_db
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            users_db = json.load(f)

def save_users():
    with open(DATA_FILE, "w") as f:
        json.dump(users_db, f)

# Хеширование пароля
def hash_password(password):
    salt = uuid.uuid4().hex
    return hashlib.sha256(salt.encode() + password.encode()).hexdigest() + ':' + salt

def check_password(hashed_password, user_password):
    password, salt = hashed_password.split(':')
    return password == hashlib.sha256(salt.encode() + user_password.encode()).hexdigest()

# Класс приложения
class DataAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Анализ данных")
        self.root.geometry("800x800")
        
        self.df = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        self.create_login_frame()
        
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
        
        # Вкладка данных
        data_tab = ttk.Frame(tab_control)
        tab_control.add(data_tab, text="Анализ данных")
        self.setup_data_tab(data_tab)
        
        tab_control.pack(expand=1, fill="both")
        
        # Заполняем таблицу пользователей
        self.update_users_tree()
        
        users_tab.columnconfigure(0, weight=1)
        users_tab.rowconfigure(0, weight=1)
    
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
        
        logout_btn = ttk.Button(btn_frame, text="Выйти", command=self.logout)
        logout_btn.pack(side=tk.RIGHT, padx=5)
        
        # Вкладка данных
        data_tab = ttk.Frame(tab_control)
        tab_control.add(data_tab, text="Анализ данных")
        self.setup_data_tab(data_tab)
        
        tab_control.pack(expand=1, fill="both")
        
        profile_tab.columnconfigure(1, weight=1)
    
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
    
    def setup_data_tab(self, tab):
        # Загрузка данных
        load_frame = ttk.LabelFrame(tab, text="Загрузка данных", padding="10")
        load_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(load_frame, text="Выбрать CSV файл", command=self.load_csv).pack(side=tk.LEFT)
        
        # Предварительный просмотр данных
        preview_frame = ttk.LabelFrame(tab, text="Предварительный просмотр", padding="10")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.preview_text = tk.Text(preview_frame, height=10)
        scrollbar = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self.preview_text.yview)
        self.preview_text.configure(yscroll=scrollbar.set)
        
        self.preview_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Обработка данных
        process_frame = ttk.LabelFrame(tab, text="Обработка данных", padding="10")
        process_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(process_frame, text="Предобработать данные", command=self.preprocess_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(process_frame, text="Построить модель", command=self.build_model).pack(side=tk.LEFT, padx=5)
        
        # График
        self.graph_frame = ttk.LabelFrame(tab, text="График", padding="10")
        self.graph_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Форма для прогноза
        predict_frame = ttk.LabelFrame(tab, text="Прогноз", padding="10")
        predict_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(predict_frame, text="Дата:").grid(row=0, column=0, padx=5)
        self.date_entry = ttk.Entry(predict_frame)
        self.date_entry.grid(row=0, column=1, padx=5)
        
        ttk.Label(predict_frame, text="Округ:").grid(row=0, column=2, padx=5)
        self.district_entry = ttk.Entry(predict_frame)
        self.district_entry.grid(row=0, column=3, padx=5)
        
        ttk.Button(predict_frame, text="Построить график", command=self.show_prediction).grid(row=0, column=4, padx=5)
    
    def load_csv(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not filepath:
            return
        
        try:
            self.df = pd.read_csv(filepath)
            self.preview_text.delete(1.0, tk.END)
            self.preview_text.insert(tk.END, str(self.df.head()))
            messagebox.showinfo("Успех", "Файл успешно загружен")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить файл: {str(e)}")
    
    def preprocess_data(self):
        if self.df is None:
            messagebox.showerror("Ошибка", "Сначала загрузите данные")
            return
        
        try:
            # Заполнение пропусков медианными значениями
            numeric_cols = self.df.select_dtypes(include=['number']).columns
            imputer = SimpleImputer(strategy='median')
            self.df[numeric_cols] = imputer.fit_transform(self.df[numeric_cols])
            
            # Преобразование категориальных признаков
            cat_cols = self.df.select_dtypes(include=['object']).columns
            for col in cat_cols:
                self.df[col] = self.label_encoder.fit_transform(self.df[col])
            
            # Масштабирование признаков
            self.scaler.fit(self.df[numeric_cols])
            self.df[numeric_cols] = self.scaler.transform(self.df[numeric_cols])
            
            messagebox.showinfo("Успех", "Данные успешно предобработаны")
            self.preview_text.delete(1.0, tk.END)
            self.preview_text.insert(tk.END, str(self.df.head()))
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при предобработке данных: {str(e)}")
    
    def build_model(self):
        if self.df is None:
            messagebox.showerror("Ошибка", "Сначала загрузите данные")
            return
        
        if len(self.df.columns) < 2:
            messagebox.showerror("Ошибка", "Недостаточно столбцов для построения модели")
            return
        
        try:
            # Последний столбец - целевая переменная
            X = self.df.iloc[:, :-1]
            y = self.df.iloc[:, -1]
            
            # Разделение на обучающую и тестовую выборки
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Обучение модели
            self.model = LogisticRegression(class_weight='balanced')
            self.model.fit(X_train, y_train)
            
            # Оценка модели
            y_pred = self.model.predict(X_test)
            report = classification_report(y_test, y_pred)
            
            messagebox.showinfo("Успех", f"Модель успешно обучена\n\nClassification Report:\n{report}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при построении модели: {str(e)}")
    
    def show_prediction(self):
        if self.model is None:
            messagebox.showerror("Ошибка", "Сначала постройте модель")
            return
        
        date = self.date_entry.get()
        district = self.district_entry.get()
        
        if not date or not district:
            messagebox.showerror("Ошибка", "Заполните все поля")
            return
        
        try:
            # Здесь должна быть логика преобразования введенных данных в формат для модели
            # Для примера просто создадим случайные данные
            import numpy as np
            x = np.linspace(0, 10, 100)
            y = np.sin(x) + np.random.normal(0, 0.1, 100)
            
            # Очистка предыдущего графика
            for widget in self.graph_frame.winfo_children():
                widget.destroy()
            
            # Создание графика
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(x, y, label='Прогноз')
            ax.set_title(f"Прогноз для округа {district} на {date}")
            ax.set_xlabel("Время")
            ax.set_ylabel("Значение")
            ax.legend()
            ax.grid(True)
            
            # Встраивание графика в интерфейс
            canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Прогноз максимального значения
            max_idx = np.argmax(y)
            max_val = y[max_idx]
            max_time = x[max_idx]
            
            ttk.Label(self.graph_frame, 
                     text=f"Максимальный прогноз: {max_val:.2f} в момент времени {max_time:.2f}").pack()
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при построении графика: {str(e)}")
    
    def logout(self):
        global current_user
        current_user = None
        self.create_login_frame()

# Инициализация приложения
if __name__ == "__main__":
    load_users()
    
    # Создаем администратора, если его нет
    if ADMIN_USERNAME not in users_db:
        users_db[ADMIN_USERNAME] = {
            'name': 'Администратор',
            'email': 'admin@example.com',
            'password': hash_password(ADMIN_PASSWORD),
            'role': 'admin'
        }
        save_users()
    
    root = tk.Tk()
    app = DataAnalysisApp(root)
    root.mainloop()