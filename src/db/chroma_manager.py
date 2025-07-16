import os
import logging
from chromadb import PersistentClient, Settings
from langchain_community.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

# Получаем путь для ChromaDB из переменных окружения
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db_data")
# Название коллекции в ChromaDB
COLLECTION_NAME = "legal_documents_collection"

# Глобальные переменные для клиента и коллекции ChromaDB
chroma_client = None
chroma_collection = None
embedding_function = None

def init_chroma_db():
    """Инициализирует ChromaDB клиент и коллекцию."""
    global chroma_client, chroma_collection, embedding_function
    if chroma_client and chroma_collection and embedding_function:
        logging.info("ChromaDB уже инициализирована.")
        return chroma_client

    try:
        # Инициализируем функцию эмбеддингов
        # Используем GoogleGenerativeAIEmbeddings, так как вы работаете с Gemini
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            logging.error("❌ Ошибка: GEMINI_API_KEY не установлен. Эмбеддинги не будут работать.")
            return None
        embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)

        # Создаем персистентный клиент ChromaDB
        # Данные будут храниться в CHROMA_DB_PATH
        chroma_client = PersistentClient(path=CHROMA_DB_PATH, settings=Settings(anonymized_telemetry=False))
        logging.info(f"✅ ChromaDB клиент инициализирован по пути: {CHROMA_DB_PATH}")

        # Получаем или создаем коллекцию
        chroma_collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
        logging.info(f"✅ ChromaDB коллекция '{COLLECTION_NAME}' готова.")
        return chroma_client
    except Exception as e:
        logging.error(f"❌ Ошибка при инициализации ChromaDB: {e}")
        return None

def add_documents_to_chroma(documents: list, metadata: dict = None, chroma_db_client=None) -> list:
    """
    Добавляет список документов (текстовых чанков) в ChromaDB.
    Каждый документ должен быть словарем с 'page_content' и 'metadata'.
    Возвращает список ID добавленных документов.
    """
    if not chroma_db_client or not embedding_function:
        logging.error("ChromaDB или функция эмбеддингов не инициализированы. Документы не будут добавлены.")
        return []

    try:
        # LangChain Chroma поддерживает прямую работу с объектом коллекции.
        # Создаем объект LangChain Chroma из существующей коллекции.
        vectorstore = Chroma(
            client=chroma_db_client,
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_function
        )
        
        # Добавляем документы. `documents` ожидается список Document объектов LangChain
        # или просто список строк, если метаданные внутри.
        # Для нашей текущей схемы, process_document_content будет возвращать List[Document]
        
        # Если documents это список LangChain Document объектов, LangChain сам извлечет page_content и metadata
        doc_ids = vectorstore.add_documents(documents)
        logging.info(f"✅ Добавлено {len(doc_ids)} чанков в ChromaDB. ID: {doc_ids}")
        return doc_ids
    except Exception as e:
        logging.error(f"❌ Ошибка при добавлении документов в ChromaDB: {e}")
        return []

def get_retriever(chroma_db_client=None) -> VectorStoreRetriever:
    """Возвращает ретривер LangChain для ChromaDB."""
    if not chroma_db_client or not embedding_function:
        logging.error("ChromaDB или функция эмбеддингов не инициализированы. Ретривер не будет создан.")
        return None
    try:
        vectorstore = Chroma(
            client=chroma_db_client,
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_function
        )
        # as_retriever() создает объект, который может искать по векторной базе данных
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # Можно настроить k - количество возвращаемых документов
        return retriever
    except Exception as e:
        logging.error(f"❌ Ошибка при создании ретривера ChromaDB: {e}")
        return None
