from flask import Flask
from flask_admin import Admin
from flask_sqlalchemy import SQLAlchemy
from models import User, Role, Post

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///admin_panel.db'
db = SQLAlchemy(app)

# Инициализация Flask-Admin
admin = Admin(app, name='Admin Panel', template_mode='bootstrap3')

# Регистрация моделей в админ-панели
from flask_admin.contrib.sqla import ModelView

admin.add_view(ModelView(User, db.session))
admin.add_view(ModelView(Role, db.session))
admin.add_view(ModelView(Post, db.session))

@app.route('/')
def index():
    return "Welcome to the Admin Panel!"

if __name__ == '__main__':
    db.create_all()  # Создание таблиц в базе данных
    app.run(debug=True)
