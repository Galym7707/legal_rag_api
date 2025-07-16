import os
import json
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from vector_db_manager import VectorDBManager
from document_processor import process_uploaded_file
import google.generativeai as genai
from langchain_community.llms import GoogleGenerativeAI
from flask_cors import CORS # Импортируем Flask-CORS

# Загружаем переменные окружения из файла .env
load_dotenv()

app = Flask(__name__)
CORS(app) # Добавляем CORS для всех маршрутов, чтобы избежать проблем с фронтендом

# --- Инициализация компонентов ---
# Инициализация Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Переменная окружения GEMINI_API_KEY не установлена.")
genai.configure(api_key=GEMINI_API_KEY)
llm = GoogleGenerativeAI(model="gemini-pro")

# Инициализация нашей базы данных
db_manager = VectorDBManager()

# --- Вспомогательные функции ---

def get_client_id_from_auth(request):
    """
    Извлекает client_id из заголовка Authorization.
    В реальном приложении здесь должна быть более сложная логика аутентификации.
    """
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return None
    api_key = auth_header.split(' ')[1]
    # В этом MVP API ключ будет выступать в качестве client_id
    # В продакшене, API ключ должен маппиться на client_id в вашей БД
    return api_key

# --- Эндпоинты API ---

@app.route("/api/v1/ask", methods=["POST"])
def ask_legal_question():
    """
    Обрабатывает пользовательский запрос, используя RAG-подход.
    """
    data = request.json
    question = data.get("question")
    client_id = get_client_id_from_auth(request)

    if not question or not client_id:
        return jsonify({"error": "Отсутствует вопрос или ключ API."}), 400

    try:
        # 1. Поиск релевантных документов (Retrieval)
        relevant_docs = db_manager.search_documents(question, client_id)

        if not relevant_docs:
            # Если документы не найдены, отвечаем базовой фразой
            prompt = (
                f"Ответь на вопрос пользователя, используя только твои общие знания, "
                f"так как релевантной информации не найдено. Вопрос: {question}"
            )
            answer = llm.invoke(prompt)
            return jsonify({
                "answer": answer,
                "sources": [],
                "status": "success",
                "message": "Релевантные документы не найдены, ответ сгенерирован на основе общих знаний."
            })

        # 2. Формирование промпта (Augmentation)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Промпт для RAG
        rag_prompt = (
            f"Используй только предоставленный контекст для ответа на вопрос. "
            f"Не придумывай информацию. Если контекст не содержит ответа, скажи, что не можешь ответить.\n\n"
            f"Контекст:\n{context}\n\n"
            f"Вопрос: {question}"
        )

        # 3. Генерация ответа (Generation)
        answer = llm.invoke(rag_prompt)

        # Формируем список источников для ответа
        sources = [{
            "title": doc.metadata.get("title", "Неизвестный источник"),
            "document_id": doc.metadata.get("id", "Неизвестно"),
            "snippet": doc.page_content[:200] + "..." # Берем небольшой фрагмент
        } for doc in relevant_docs]

        return jsonify({
            "answer": answer,
            "sources": sources,
            "status": "success"
        })

    except Exception as e:
        print(f"❌ Ошибка при обработке запроса: {e}")
        return jsonify({"error": f"Ошибка сервера: {str(e)}"}), 500


@app.route("/api/v1/documents/upload", methods=["POST"])
def upload_document():
    """
    Обрабатывает загрузку нового документа.
    """
    client_id = get_client_id_from_auth(request)
    if not client_id:
        return jsonify({"error": "Отсутствует ключ API."}), 401

    if 'file' not in request.files:
        return jsonify({"error": "Файл не предоставлен."}), 400

    file = request.files['file']
    filename = file.filename
    metadata_str = request.form.get('metadata', '{}')

    try:
        metadata = json.loads(metadata_str)
    except json.JSONDecodeError:
        return jsonify({"error": "Неверный формат метаданных. Метаданные должны быть валидной JSON строкой."}), 400

    try:
        # 1. Обработка файла и получение чанков
        documents = process_uploaded_file(file, filename, metadata)

        if not documents:
            return jsonify({"error": "Не удалось обработать документ."}), 500

        # 2. Добавление чанков в векторную БД
        db_manager.add_documents(documents, client_id)

        # Здесь можно сгенерировать уникальный ID для документа,
        # например, на основе хэша или UUID (UUID более надежный)
        import uuid
        document_id = str(uuid.uuid4())

        return jsonify({
            "message": "Документ успешно загружен и проиндексирован.",
            "document_id": document_id,
            "status": "success"
        })

    except Exception as e:
        print(f"❌ Ошибка при загрузке документа: {e}")
        return jsonify({"error": f"Ошибка сервера: {str(e)}"}), 500

# --- Запуск приложения ---
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
