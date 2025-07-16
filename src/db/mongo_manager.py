import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from datetime import datetime
import logging

# Получаем строку подключения из переменных окружения
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB_NAME", "legal_rag_db") # Имя базы данных, по умолчанию 'legal_rag_db'

client = None
db = None

def init_mongo_db():
    """Инициализирует подключение к MongoDB."""
    global client, db
    if client and db: # Если уже инициализировано
        logging.info("MongoDB уже инициализирована.")
        return client

    if not MONGO_URI:
        logging.error("❌ Ошибка: Переменная окружения MONGO_URI не установлена. Подключение к MongoDB невозможно.")
        return None

    try:
        # Устанавливаем таймаут соединения, чтобы избежать бесконечного ожидания
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping') # Проверка соединения с базой данных
        db = client[DB_NAME]
        logging.info("✅ Подключение к MongoDB Atlas успешно установлено.")

        # Создаем индекс для session_id и message_index
        # Проверяем, существует ли индекс, чтобы избежать ошибок при повторном запуске
        # MongoDB автоматически создает коллекцию при первой вставке, если она не существует.
        # Создание индексов безопасно при повторном вызове.
        db.conversations.create_index([("session_id", 1), ("message_index", 1)], unique=False)
        logging.info("✅ Индексы MongoDB созданы/проверены.")
        return client

    except ConnectionFailure as e:
        logging.error(f"❌ Ошибка подключения к MongoDB: {e}")
        return None
    except OperationFailure as e:
        logging.error(f"❌ Ошибка операции MongoDB (возможно, проблема с аутентификацией или правами): {e}")
        return None
    except Exception as e:
        logging.error(f"❌ Неизвестная ошибка при инициализации MongoDB: {e}")
        return None

def save_message(session_id: str, role: str, content: str):
    """Сохраняет сообщение в историю беседы в MongoDB."""
    if not db:
        logging.error("MongoDB не инициализирована. Сохранение сообщения невозможно.")
        return

    try:
        # Получаем следующий message_index для данной сессии
        # Если сессии нет, начинаем с 0
        last_message = db.conversations.find_one(
            {"session_id": session_id},
            sort=[("message_index", -1)]
        )
        message_index = last_message["message_index"] + 1 if last_message else 0

        message = {
            "session_id": session_id,
            "role": role,
            "parts": [content],
            "timestamp": datetime.now(),
            "message_index": message_index
        }
        db.conversations.insert_one(message)
        logging.debug(f"Сообщение сохранено для сессии {session_id}, роль: {role}, индекс: {message_index}")
    except Exception as e:
        logging.error(f"❌ Ошибка при сохранении сообщения в MongoDB для сессии {session_id}: {e}")

def load_conversation(session_id: str):
    """Загружает историю сообщений для указанной сессии из MongoDB."""
    if not db:
        logging.error("MongoDB не инициализирована. Загрузка сообщений невозможна.")
        return []

    try:
        messages = list(db.conversations.find(
            {"session_id": session_id},
            {"_id": 0, "session_id": 0, "timestamp": 0} # Исключаем лишние поля
        ).sort("message_index", 1)) # Сортируем по индексу для правильного порядка

        formatted_history = []
        for entry in messages:
            content = entry.get('parts')[0] if isinstance(entry.get('parts'), list) and entry.get('parts') else ''
            formatted_history.append({"role": entry.get('role'), "content": content})
        
        logging.debug(f"Загружена история для сессии {session_id}: {len(formatted_history)} сообщений.")
        return formatted_history
    except Exception as e:
        logging.error(f"❌ Ошибка при загрузке истории из MongoDB для сессии {session_id}: {e}")
        return []

def delete_conversation(session_id: str):
    """Удаляет всю историю сообщений для указанной сессии из MongoDB."""
    if not db:
        logging.error("MongoDB не инициализирована. Удаление сообщений невозможно.")
        return

    try:
        result = db.conversations.delete_many({"session_id": session_id})
        logging.info(f"🗑️ Удалено {result.deleted_count} сообщений для сессии {session_id}.")
    except Exception as e:
        logging.error(f"❌ Ошибка при удалении истории из MongoDB для сессии {session_id}: {e}")

def get_all_sessions_summary_mongo():
    """
    Получает сводку всех сессий из MongoDB, включая первое пользовательское сообщение как заголовок.
    """
    if not db:
        logging.error("MongoDB не инициализирована. Получение сводки сессий невозможно.")
        return []
    try:
        pipeline = [
            # Группируем по session_id
            {"$group": {
                "_id": "$session_id",
                "first_user_message_content": {
                    "$first": {
                        "$cond": [
                            {"$eq": ["$role", "user"]}, # Если роль - user
                            {"$arrayElemAt": ["$parts", 0]}, # Берем первое значение из массива parts
                            "$$REMOVE" # Удаляем, если не user
                        ]
                    }
                }
            }},
            # Удаляем сессии без пользовательских сообщений (если таковые есть после $$REMOVE)
            {"$match": {"first_user_message_content": {"$exists": True}}},
            # Проекция для формирования нужного формата
            {"$project": {
                "id": "$_id",
                "title": {
                    "$cond": [
                        {"$ne": ["$first_user_message_content", None]},
                        # Обрезаем заголовок до 50 символов и добавляем "..." если длиннее
                        {"$concat": [
                            {"$substrCP": ["$first_user_message_content", 0, 50]},
                            {"$cond": [
                                {"$gt": [{"$strLenCP": "$first_user_message_content"}, 50]},
                                "...",
                                ""
                            ]}
                        ]},
                        "Новый чат"
                    ]
                },
                "_id": 0 # Исключаем поле _id
            }},
            {"$sort": {"id": 1}} # Сортируем по session_id для стабильного порядка
        ]

        sessions_summary = list(db.conversations.aggregate(pipeline))
        logging.info(f"✅ Получена сводка всех сессий: {len(sessions_summary)} сессий.")
        return sessions_summary
    except Exception as e:
        logging.error(f"❌ Ошибка при получении сводки сессий из MongoDB: {e}")
        return []
