import os
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from dotenv import load_dotenv
import logging
import uuid # Для генерации session_id

# Загружаем переменные окружения из .env файла
load_dotenv()

# --- Настройка логирования ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Инициализация Flask приложения ---
app = Flask(__name__)
# Разрешаем CORS для всех источников на время разработки. В продакшене лучше ограничить.
CORS(app) 
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1 GB для загрузки документов

# --- Импорты из src ---
from src.db.mongo_manager import init_mongo_db, save_message, load_conversation, delete_conversation, get_all_sessions_summary_mongo
from src.db.chroma_manager import init_chroma_db, add_documents_to_chroma, get_retriever
from src.ingestion.document_processor import process_document_content
from src.rag.generation import get_rag_response

# --- Инициализация баз данных при старте приложения ---
# Это будет выполнено один раз при запуске `main.py`
mongo_db_client = init_mongo_db()
chroma_vector_db = init_chroma_db() # Инициализация ChromaDB


# Проверка, что LLM API ключ установлен
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logging.error("❌ Ошибка: Переменная окружения GEMINI_API_KEY не установлена.")
    # Приложение может работать без LLM, но функционал RAG будет ограничен
    # В реальном приложении можно поднять ошибку или сделать LLM опциональным

# --- API Endpoints ---

@app.route('/ask_legal_question', methods=['POST'])
def ask_legal_question_route():
    """
    Основной маршрут для получения ответа от ИИ-юриста.
    Принимает вопрос и ID сессии, возвращает ответ и использованные источники.
    """
    try:
        data = request.json
        session_id = data.get('session_id')
        question = data.get('question')

        if not question:
            return jsonify({"error": "Вопрос не может быть пустым."}), 400
        
        if not session_id:
            session_id = str(uuid.uuid4()) # Генерируем новый ID для новой сессии
            logging.info(f"🆕 Создана новая сессия с ID: {session_id}")

        # Сохраняем сообщение пользователя в историю
        save_message(session_id, "user", question)
        
        # --- RAG Logic ---
        # 1. Получаем историю чата для контекста
        chat_history = load_conversation(session_id)
        
        # 2. Получаем ретривер для поиска по ChromaDB
        retriever = get_retriever(chroma_vector_db)
        
        # 3. Генерируем ответ с помощью RAG
        # В этой функции будет происходить поиск релевантных документов и генерация ответа
        answer, sources = get_rag_response(question, chat_history, retriever)
        
        # Сохраняем ответ ИИ в историю
        save_message(session_id, "ai", answer)

        logging.info(f"✅ Вопрос '{question[:50]}...' обработан для сессии {session_id}")
        return jsonify({
            "session_id": session_id,
            "answer": answer,
            "sources": sources # sources пока будет пустым списком, но оставим место
        })

    except Exception as e:
        logging.error(f"❌ Ошибка в /ask_legal_question: {e}", exc_info=True)
        return jsonify({"error": f"Ошибка сервера при обработке запроса: {str(e)}"}), 500


@app.route('/upload_document', methods=['POST'])
def upload_document_route():
    """
    Маршрут для загрузки юридического документа.
    """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "Файл не найден в запросе"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Имя файла не может быть пустым"}), 400

        # Получаем метаданные, если они есть
        metadata_str = request.form.get('metadata', '{}')
        try:
            metadata = json.loads(metadata_str)
        except json.JSONDecodeError:
            return jsonify({"error": "Неверный формат метаданных. Должен быть JSON-строкой."}), 400

        # Обработка документа и добавление в ChromaDB
        # document_processor.py будет содержать логику извлечения текста и чанкинга
        # add_documents_to_chroma будет добавлять чанки и метаданные в ChromaDB
        doc_id = add_documents_to_chroma(file, metadata, chroma_vector_db)

        logging.info(f"✅ Документ '{file.filename}' с ID '{doc_id}' успешно загружен и обработан.")
        return jsonify({
            "message": f"Документ '{file.filename}' успешно загружен и проиндексирован.",
            "document_id": doc_id
        })

    except Exception as e:
        logging.error(f"❌ Ошибка при загрузке документа: {e}", exc_info=True)
        return jsonify({"error": f"Ошибка сервера при загрузке документа: {str(e)}"}), 500


@app.route('/get_history', methods=['GET'])
def get_history_route():
    """
    Возвращает историю сообщений для конкретной сессии.
    """
    try:
        session_id = request.args.get('session_id')
        if not session_id:
            return jsonify({"error": "Необходимо указать session_id"}), 400

        history = load_conversation(session_id)
        logging.info(f"✅ Получена история для сессии {session_id}, сообщений: {len(history)}")
        return jsonify({"session_id": session_id, "history": history})
    except Exception as e:
        logging.error(f"❌ Ошибка в /get_history: {e}", exc_info=True)
        return jsonify({"error": f"Ошибка сервера при получении истории: {str(e)}"}), 500


@app.route('/get_all_sessions_summary', methods=['GET'])
def get_all_sessions_summary_route():
    """
    Возвращает сводку всех сессий для отображения.
    """
    try:
        sessions_summary = get_all_sessions_summary_mongo()
        logging.info(f"✅ Получена сводка всех сессий: {len(sessions_summary)} сессий.")
        return jsonify({"sessions": sessions_summary})
    except Exception as e:
        logging.error(f"❌ Ошибка в /get_all_sessions_summary: {e}", exc_info=True)
        return jsonify({"error": f"Ошибка сервера при получении сводки сессий: {str(e)}"}), 500


@app.route('/clear_history', methods=['POST'])
def clear_history_route():
    """
    Удаляет историю сообщений для конкретной сессии.
    """
    try:
        data = request.json
        session_id = data.get('session_id')
        if not session_id:
            return jsonify({"error": "Необходимо указать session_id"}), 400

        delete_conversation(session_id)
        logging.info(f"✅ История для сессии {session_id} очищена.")
        return jsonify({"message": "История очищена", "session_id": session_id})
    except Exception as e:
        logging.error(f"❌ Ошибка в /clear_history: {e}", exc_info=True)
        return jsonify({"error": f"Ошибка сервера при очистке истории: {str(e)}"}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True) # debug=True только для разработки
