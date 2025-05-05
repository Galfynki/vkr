import sys
import os
import hashlib
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox
from tkinter import *
import pandas as pd
import numpy as np
import requests
import json
import sqlite3
from functools import partial
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
import warnings
from sklearn.datasets import make_regression
from sklearn.ensemble import BaggingRegressor

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)




def main():
    print("Загрузка данных...")
    try:
        data = pd.read_csv('../../../Downloads/Telegram Desktop/dataset.csv')
        print("Данные успешно загружены!")
    except FileNotFoundError:
        print("Ошибка: Файл 'dataset.csv' не найден. Проверьте расположение файла.")
        return

    print("Информация о данных:")
    print(data.info())

    required_columns = [
        'Reference Number', 'Grid Ref: Easting', 'Grid Ref: Northing',
            'Time (24hr)', '1st Road Class', 'Road Surface',
            'Lighting Conditions', 'Weather Conditions',
            'Casualty Class', 'Casualty Severity', 'Sex of Casualty',
            'Age of Casualty', 'Type of Vehicle'
    ]

    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        print(f"Ошибка: Отсутствуют необходимые столбцы: {missing_columns}")
        return

    # Выбор признаков и целевой переменной
    try:
        X = data[[
            'Reference Number', 'Grid Ref: Easting', 'Grid Ref: Northing',
            'Time (24hr)', '1st Road Class', 'Road Surface',
            'Lighting Conditions', 'Weather Conditions',
            'Casualty Class', 'Casualty Severity', 'Sex of Casualty',
            'Age of Casualty', 'Type of Vehicle'
        ]]
        y = data['Casualty Severity']
    except KeyError as e:
        print(f"Ошибка: Неверный выбор столбцов: {e}")
        return

    print("Признаки (X):")
    print(X.head())
    print("Целевая переменная (y):")
    print(y.head())

    # Кодирование категориальных переменных
    if 'Reference Number' in data.columns:
        le = LabelEncoder()
        data['Reference Number'] = le.fit_transform(data['Reference Number'].astype(str))

    # Преобразование данных
    for column in X.columns:
        X.loc[:, column] = pd.to_numeric(X[column], errors='coerce').fillna(0)

    # Явное преобразование всех столбцов X в числа
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Проверка на бесконечности
    if np.isinf(X.to_numpy()).any():
        print("Ошибка: В данных есть бесконечные значения.")
        return

    X = X.clip(-1e6, 1e6)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("\nОбучение линейной регрессии...")
    linear_regressor = LinearRegression()
    linear_regressor.fit(X_train, y_train)
    y_pred_linear = linear_regressor.predict(X_test)
    mse_linear = mean_squared_error(y_test, y_pred_linear)
    print(f'Средняя квадратичная ошибка (линейная регрессия): {mse_linear:.2f}')

    print("\nОбучение случайного леса для регрессии...")
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train, y_train)
    y_pred_rf_reg = rf_regressor.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf_reg)
    print(f'Средняя квадратичная ошибка (случайный лес): {mse_rf:.2f}')

    y_class = data['Casualty Class']
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
        X, y_class, test_size=0.3, random_state=42
    )
    print("\nОбучение логистической регрессии...")
    logistic_model = LogisticRegression(max_iter=300)
    logistic_model.fit(X_train_class, y_train_class)
    y_pred_class = logistic_model.predict(X_test_class)
    print("\nРезультаты классификации (логистическая регрессия):")
    print("Матрица путаницы:")
    print(confusion_matrix(y_test_class, y_pred_class))
    print("Отчет о классификации:")
    print(classification_report(y_test_class, y_pred_class, zero_division=0))

    print("\nОбучение случайного леса для классификации...")
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train_class, y_train_class)
    y_pred_rf_class = rf_classifier.predict(X_test_class)
    print("\nРезультаты классификации (случайный лес):")
    print("Матрица путаницы:")
    print(confusion_matrix(y_test_class, y_pred_rf_class))
    print("Отчет о классификации:")
    print(classification_report(y_test_class, y_pred_rf_class))

if main == "main":
    main()

# Словарь для хранения пользователей и их уровней доступа
users = {
    "client": {"password": "", "role": ""},
    "admin": {"password": "adminpass", "role": "admin"},  # Фиксированный логин и пароль для администратора
}

class UserAccount:
    def __init__(self, username, password, full_name, email):
        self.username = username
        self.password = self.hash_password(password)
        self.full_name = full_name
        self.email = email

    @staticmethod
    def hash_password(password):
        return hashlib.sha256(password.encode()).hexdigest()

# Функция для инициализации базы данных
def init_db():
    if os.path.exists('clients.db'):
        os.remove('clients.db')

    conn = sqlite3.connect('clients.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS clients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            full_name TEXT,
            email TEXT,
            password TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Функция для получения данных клиента по ID
def get_client_data(client_id):
    conn = sqlite3.connect('clients.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, full_name, email FROM clients WHERE id = ?", (client_id,))
    client_data = cursor.fetchone()
    conn.close()
    return client_data

# Функция для отображения личного кабинета
def show_account_screen(username):
    for widget in root.winfo_children():
        widget.destroy()

    label_welcome = tk.Label(root, text=f"Добро пожаловать в личный кабинет, {username}!", font=("Arial", 16))
    label_welcome.pack(pady=20)

    client_data = get_client_data(1)  # Получаем данные клиента с ID 1
    label_name = tk.Label(root, text=f"Имя: {client_data[1]}")
    label_name.pack(pady=5)
    label_email = tk.Label(root, text=f"Email: {client_data[2]}")
    label_email.pack(pady=5)

    button_update_data = tk.Button(root, text="Изменить свои данные", command=show_update_form)
    button_update_data.pack(pady=10)

    button_action = tk.Button(root, text="Просмотреть свои заказы", command=lambda: messagebox.showinfo("Действие", "Здесь будут ваши заказы."))
    button_action.pack(pady=10)

    # Кнопка для выхода из аккаунта
    button_logout = tk.Button(root, text="Выйти из аккаунта", command=logout_user)
    button_logout.pack(pady=10)

# Функция для выхода из аккаунта
def logout_user():
    for widget in root.winfo_children():
        widget.destroy()
    show_login_screen()  # Возвращаемся к экрану входа

# Функция для отображения формы изменения данных клиента
def show_update_form():
    for widget in root.winfo_children():
        widget.destroy()

    client_data = get_client_data(1)  # Получаем данные клиента с ID 1

    label_id = tk.Label(root, text="ID клиента:")
    label_id.pack()
    entry_id = tk.Entry(root)
    entry_id.insert(0, client_data[0])  # Вставляем ID клиента
    entry_id.config(state='readonly')  # Делаем поле только для чтения
    entry_id.pack()

    label_name = tk.Label(root, text="Новое имя:")
    label_name.pack()
    entry_name = tk.Entry(root)
    entry_name.insert(0, client_data[1])  # Вставляем текущее имя
    entry_name.pack()

    label_email = tk.Label(root, text="Новый email:")
    label_email.pack()
    entry_email = tk.Entry(root)
    entry_email.insert(0, client_data[2])  # Вставляем текущий email
    entry_email.pack()

    button_update = tk.Button(root, text="Обновить данные", command=lambda: update_client_data(entry_id.get(), entry_name.get(), entry_email.get()))
    button_update.pack()

    button_back = tk.Button(root, text="Назад", command=lambda: show_account_screen(client_data[1]))
    button_back.pack()

# Функция для обновления данных клиента
def update_client_data(client_id, new_name, new_email):
    conn = sqlite3.connect('clients.db')
    cursor = conn.cursor()

    cursor.execute("UPDATE clients SET full_name = ?, email = ? WHERE id = ?", (new_name, new_email, client_id))
    conn.commit()
    conn.close()

    messagebox.showinfo("Успех", "Данные клиента обновлены!")
    show_account_screen(new_name)  # Обновляем экран после изменения данных

# Функция для отображения экрана входа
def show_login_screen():
    for widget in root.winfo_children():
        widget.destroy()

    tk.Label(root, text="Логин:").pack(pady=5)
    username_entry = tk.Entry(root)
    username_entry.pack(pady=5)

    tk.Label(root, text="Пароль:").pack(pady=5)
    password_entry = tk.Entry(root, show='*')
    password_entry.pack(pady=5)

    login_button = tk.Button(root, text="Войти", command=lambda: login_user(username_entry.get(), password_entry.get()))
    login_button.pack(pady=20)

    register_button = tk.Button(root, text="Регистрация", command=open_registration_window)
    register_button.pack(pady=5)

# Функция для входа пользователя
def login_user(username, password):
    # Проверка на наличие пользователя в словаре
    if username in users and users[username]["password"] == password:
        show_account_screen(username)
    else:
        # Проверка в базе данных
        conn = sqlite3.connect('clients.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM clients WHERE username = ? AND password = ?", (username, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            show_account_screen(user[1])  # user[1] - это полное имя
        else:
            messagebox.showerror("Ошибка", "Неверные учетные данные!")

    if users == "admin":
        button_action = tk.Button(root, text="Управление пользователями", command=lambda: messagebox.showinfo("Действие", "Здесь вы можете управлять пользователями."))
        button_action.pack(pady=10)
        return



# Функция для открытия окна регистрации
def open_registration_window():
    for widget in root.winfo_children():
        widget.destroy()

    tk.Label(root, text="Регистрация").pack(pady=10)

    tk.Label(root, text="Логин:").pack(pady=5)
    reg_username_entry = tk.Entry(root)
    reg_username_entry.pack(pady=5)

    tk.Label(root, text="ФИО:").pack(pady=5)
    full_name_entry = tk.Entry(root)
    full_name_entry.pack(pady=5)

    tk.Label(root, text="Почта:").pack(pady=5)
    email_entry = tk.Entry(root)
    email_entry.pack(pady=5)

    tk.Label(root, text="Пароль:").pack(pady=5)
    reg_password_entry = tk.Entry(root, show='*')
    reg_password_entry.pack(pady=5)

    register_button = tk.Button(root, text="Зарегистрироваться", command=lambda: register_user(reg_username_entry.get(), full_name_entry.get(), email_entry.get(), reg_password_entry.get()))
    register_button.pack(pady=20)

    back_button = tk.Button(root, text="Назад", command=show_login_screen)
    back_button.pack(pady=5)

# Функция для регистрации пользователя
def register_user(username, full_name, email, password):
    if not username or not full_name or not email or not password:
        messagebox.showerror("Ошибка", "Пожалуйста, заполните все поля!")
        return

    conn = sqlite3.connect('clients.db')
    cursor = conn.cursor()

    try:
        cursor.execute("INSERT INTO clients (username, full_name, email, password) VALUES (?, ?, ?, ?)", (username, full_name, email, password))
        conn.commit()
        messagebox.showinfo("Успех", "Регистрация прошла успешно!")
        show_login_screen()  # Возвращаемся к экрану входа
    except sqlite3.IntegrityError:
        messagebox.showerror("Ошибка", "Пользователь с таким логином уже существует!")
    finally:
        conn.close()

# Инициализация базы данных
init_db()

# Создаем главное окно
root = tk.Tk()
root.title("Авторизация")
root.geometry("900x000")

# Загружаем изображение
#image_file = Image.open(r"\vkr\fon1.jpg")
#image_file.show()  # Замените на путь к вашему изображению
# Изменяем размер изображения под размер окна
#image_file = image_file.resize((900, 900), Image.LANCZOS)
#bg_image = ImageTk.PhotoImage(image_file)

# Создаем метку для фона и помещаем изображение
#background_label = tk.Label(root, image=bg_image)
#background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Запускаем приложение, показывая экран входа
show_login_screen()

# Запускаем главный цикл приложения
root.mainloop()

if __name__ == "__main__":
    main()
