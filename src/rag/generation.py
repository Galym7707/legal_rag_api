import os
import logging
import google.generativeai as genai
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from typing import List, Tuple
from markdown import markdown # Для форматирования ответа

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Инициализация LLM ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    # Используем модель gemini-1.5-flash как более быструю и экономичную для большинства задач
    llm_model = genai.GenerativeModel(
        'gemini-1.5-flash',
        generation_config={"response_mime_type": "text/plain", "temperature": 0.5},
        system_instruction=(
            "Ты - ИИ-юрист, специализирующийся на законодательстве Республики Казахстан. "
            "Отвечай только на вопросы, касающиеся законодательства РК. "
            "Если вопрос не относится к юрисдикции РК или выходит за рамки юридических вопросов, "
            "вежливо откажись отвечать, предлагая задать вопрос по теме. "
            "Твоя основная задача - предоставлять точные, краткие и понятные юридические консультации, "
            "основанные на предоставленной информации. Если информации недостаточно для ответа, "
            "укажи это и попроси уточнить запрос. "
            "Форматируй свои ответы четко и структурировано, используя Markdown для списков, жирного текста и т.д. "
            "Всегда указывай статьи закона или нормативные акты, если они присутствуют в источниках."
        )
    )
    logging.info("✅ Gemini LLM модель инициализирована.")
else:
    llm_model = None
    logging.error("❌ GEMINI_API_KEY не установлен. LLM не будет функционировать.")

def _format_docs(docs: List[Document]) -> str:
    """Форматирует найденные документы для включения в промпт LLM."""
    if not docs:
        return ""
    # Объединяем содержимое документов, добавляя информацию о метаданных, если есть
    formatted_docs = []
    for i, doc in enumerate(docs):
        # Добавляем метаданные, если они есть
        meta_info = []
        if doc.metadata:
            if "original_filename" in doc.metadata:
                meta_info.append(f"ИСТОЧНИК: {doc.metadata['original_filename']}")
            if "page" in doc.metadata: # Для PDF
                meta_info.append(f"СТРАНИЦА: {doc.metadata['page']}")
            if "document_type" in doc.metadata:
                meta_info.append(f"ТИП ДОКУМЕНТА: {doc.metadata['document_type']}")
            # Можно добавить больше метаданных по мере их появления

        meta_str = " ".join(meta_info) if meta_info else f"ИСТОЧНИК {i+1}: Без названия"
        
        formatted_docs.append(f"--- НАЧАЛО ИСТОЧНИКА ---\n{meta_str}\n{doc.page_content}\n--- КОНЕЦ ИСТОЧНИКА ---")
    
    return "\n\n".join(formatted_docs)

def get_rag_response(
    question: str,
    chat_history: List[dict], # [{role: str, content: str}, ...]
    retriever: VectorStoreRetriever
) -> Tuple[str, List[dict]]:
    """
    Генерирует ответ LLM с использованием RAG.
    Возвращает сгенерированный ответ и список использованных источников.
    """
    if not llm_model:
        return "Сервис ИИ недоступен. Пожалуйста, проверьте конфигурацию API ключа.", []

    try:
        # 1. Retrieval: Поиск релевантных документов
        # Важно: поиск документов должен основываться на текущем вопросе, а не на всей истории,
        # так как это может дать нерелевантные источники.
        relevant_docs = retriever.invoke(question)
        
        # 2. Augmentation: Формирование контекста из найденных документов
        context = _format_docs(relevant_docs)

        # 3. Aggregation of chat history:
        # Формируем историю для LLM, включая предыдущие сообщения.
        # Gemini API ожидает историю в формате List[Content], где Content имеет атрибут `parts` и `role`.
        # Преобразуем chat_history в формат, подходящий для Gemini:
        gemini_chat_history = []
        for msg in chat_history:
            if msg['role'] == 'user':
                gemini_chat_history.append(genai.types.Content(parts=[msg['content']], role='user'))
            elif msg['role'] == 'ai':
                gemini_chat_history.append(genai.types.Content(parts=[msg['content']], role='model'))

        # Формируем промпт для LLM
        # Включаем историю и найденный контекст
        # Важно: промпт должен четко инструктировать LLM использовать контекст
        
        # Если есть контекст, добавляем его в начало промпта
        if context:
            full_prompt = (
                f"КОНТЕКСТ ДЛЯ ОТВЕТА:\n{context}\n\n"
                "ОТВЕТЬ НА ОСНОВЕ ПРЕДОСТАВЛЕННОГО КОНТЕКСТА И ИСТОРИИ ДИАЛОГА. "
                "Если контекст не содержит ответа, укажи это и попроси уточнить.\n\n"
                f"ВОПРОС: {question}"
            )
        else:
            full_prompt = (
                "КОНТЕКСТ ОТСУТСТВУЕТ ИЛИ НЕ НАЙДЕН. "
                "Пожалуйста, ответь на вопрос, если у тебя есть общие знания по теме. "
                "Если вопрос требует специфических юридических документов, которых нет, "
                "попроси уточнить или предоставить документ.\n\n"
                f"ВОПРОС: {question}"
            )
        
        # 4. Generation: Вызов LLM
        chat_session = llm_model.start_chat(history=gemini_chat_history)
        response = chat_session.send_message(full_prompt)
        
        answer = response.text
        
        # Преобразуем Markdown в HTML для удобства отображения, если это потребуется фронтенду
        # В B2B API мы можем просто вернуть Markdown или HTML, в зависимости от требований
        # Для начала, пусть возвращает Markdown
        # formatted_answer = markdown(answer) # Это если вы хотите HTML

        # Извлекаем информацию об источниках
        # Пока просто возвращаем название файла и первые 200 символов в качестве сниппета
        extracted_sources = []
        for doc in relevant_docs:
            source_title = doc.metadata.get("original_filename", "Неизвестный источник")
            snippet = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            extracted_sources.append({"title": source_title, "snippet": snippet})

        logging.info(f"✅ LLM сгенерировала ответ на вопрос '{question[:50]}...'")
        return answer, extracted_sources

    except genai.types.BlockedPromptException as e:
        logging.warning(f"⚠️ Запрос к LLM был заблокирован: {e.response.prompt_feedback}")
        return "Извините, ваш запрос был заблокирован системой безопасности. Пожалуйста, переформулируйте.", []
    except Exception as e:
        logging.error(f"❌ Ошибка при генерации ответа LLM: {e}", exc_info=True)
        return "Извините, произошла ошибка при получении ответа от ИИ-юриста. Попробуйте позже.", []
