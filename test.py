import sys
import os
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



# Создание базы данных и таблицы, если она не существует
def create_database():
    conn = sqlite3.connect('clients.db')
    cursor = conn.cursor()
    
    # Создание таблицы clients
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS clients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL
        )
    ''')
    
    # Пример добавления клиента (можно удалить, если не нужно)
    cursor.execute("INSERT INTO clients (name, email) VALUES (?, ?)", ("Иван Иванов", "ivan@example.com"))
    
    conn.commit()
    conn.close()

# Словарь для хранения пользователей и их уровней доступа
users = {
    "client": {"password": "clientpass", "role": "client"},
    "admin": {"password": "adminpass", "role": "admin"},
}

# Функция для входа пользователя
def login_user():
    username = entry_username.get()
    password = entry_password.get()
    
    if username in users and users[username]["password"] == password:
        messagebox.showinfo("Успех", f"Вход успешен! Добро пожаловать, {username}.")
        show_account_screen(users[username]["role"])
    else:
        messagebox.showwarning("Ошибка", "Неверный логин или пароль.")

# Функция для отображения личного кабинета в зависимости от уровня доступа
def show_account_screen(role):
    for widget in root.winfo_children():
        widget.destroy()

    label_welcome = tk.Label(root, text=f"Добро пожаловать в личный кабинет, {role}!", font=("Arial", 16))
    label_welcome.pack(pady=20)

    if role == "client":
        client_data = get_client_data(1)  # Получаем данные клиента с ID 1
        label_name = tk.Label(root, text=f"Имя: {client_data[1]}")
        label_name.pack(pady=5)
        label_email = tk.Label(root, text=f"Email: {client_data[2]}")
        label_email.pack(pady=5)

        button_update_data = tk.Button(root, text="Изменить свои данные", command=show_update_form)
        button_update_data.pack(pady=10)

    button_logout = tk.Button(root, text="Выход", command=root.quit)
    button_logout.pack(pady=10)

# Функция для получения данных клиента из базы данных
def get_client_data(client_id):
    conn = sqlite3.connect('clients.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM clients WHERE id = ?", (client_id,))
    client_data = cursor.fetchone()
    conn.close()
    return client_data

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

    button_back = tk.Button(root, text="Назад", command=lambda: show_account_screen(users[entry_username.get()]["role"]))
    button_back.pack()


# Функция для обновления данных клиента
def update_client_data(client_id, new_name, new_email):
    conn = sqlite3.connect('clients.db')
    cursor = conn.cursor()

    cursor.execute("UPDATE clients SET name = ?, email = ? WHERE id = ?", (new_name, new_email, client_id))
    conn.commit()
    conn.close()

    messagebox.showinfo("Успех", "Данные клиента обновлены!")
    show_account_screen(users[entry_username.get()]["role"])  # Обновляем экран после изменения данных

# Создание основного окна
root = tk.Tk()
root.title("Регистрация и Вход")
root.geometry('400x300')

# Создание базы данных
create_database()

# Метки и поля ввода
label_username = tk.Label(root, text="Логин:")
label_username.pack(pady=5)
entry_username = tk.Entry(root)
entry_username.pack(pady=5)

label_password = tk.Label(root, text="Пароль:")
label_password.pack(pady=5)
entry_password = tk.Entry(root, show="*")
entry_password.pack(pady=5)

# Кнопки для входа
button_login = tk.Button(root, text="Вход", command=login_user)
button_login.pack(pady=10)

# Запуск главного цикла
root.mainloop()

if __name__ == "__main__":
    main()
