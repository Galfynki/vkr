import sys
import os
import tkinter as tk
from tkinter import messagebox
from tkinter import *
import pandas as pd
import numpy as np
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

# Функция для регистрации пользователя
def register():
    username = entry_username.get()
    password = entry_password.get()
    
    if username and password:
        with open("users.txt", "a") as f:
            f.write(f"{username},{password}\n")
        messagebox.showinfo("Успех", "Регистрация успешна!")
        entry_username.delete(0, tk.END)
        entry_password.delete(0, tk.END)
    else:
        messagebox.showwarning("Ошибка", "Пожалуйста, заполните все поля.")

# Функция для входа пользователя
def login():
    username = entry_username.get()
    password = entry_password.get()
    
    if username and password:
        with open("users.txt", "r") as f:
            users = f.readlines()
        
        for user in users:
            stored_username, stored_password = user.strip().split(",")
            if stored_username == username and stored_password == password:
                messagebox.showinfo("Успех", "Вход успешен!")
                show_main_interface()
                return
        messagebox.showwarning("Ошибка", "Неверный логин или пароль.")
    else:
        messagebox.showwarning("Ошибка", "Пожалуйста, заполните все поля.")

   # Функция для отображения основного интерфейса после входа
def show_main_interface():
    # Удаляем все элементы из текущего окна
    for widget in root.winfo_children():
        widget.destroy()

    # Создаем новый интерфейс
    label_welcome = tk.Label(root, text="Добро пожаловать!", font=("Arial", 16))
    label_welcome.pack(pady=20)

    button_action1 = tk.Button(root, text="Действие 1", command=lambda: messagebox.showinfo("Действие", "Вы выбрали действие 1"))
    button_action1.pack(pady=10)

    button_action2 = tk.Button(root, text="Действие 2", command=lambda: messagebox.showinfo("Действие", "Вы выбрали действие 2"))
    button_action2.pack(pady=10)

    button_logout = tk.Button(root, text="Выход", command=root.quit)
    button_logout.pack(pady=10)

# Создание основного окна
root = tk.Tk()
root.title("Регистрация и Вход")

# Метки и поля ввода
label_username = tk.Label(root, text="Логин:")
label_username.pack(pady=5)
entry_username = tk.Entry(root)
entry_username.pack(pady=5)

label_password = tk.Label(root, text="Пароль:")
label_password.pack(pady=5)
entry_password = tk.Entry(root, show="*")
entry_password.pack(pady=5)

# Кнопки для регистрации и входа
button_register = tk.Button(root, text="Регистрация", command=register)
button_register.pack(pady=10)

button_login = tk.Button(root, text="Вход", command=login)
button_login.pack(pady=10)

# Запуск главного цикла
root.mainloop()

if __name__ == "__main__":
    create_window()
