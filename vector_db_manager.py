import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Загружаем переменные окружения
load_dotenv()

# Настройки
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db_data")
EMBEDDING_MODEL_NAME = "cointegrated/LaBSE-multilingual-sbert"

class VectorDBManager:
    """
    Класс для управления векторной базой данных ChromaDB.
    """
    def __init__(self):
        self.embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.db = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=self.embedding_function
        )
        print(f"✅ Векторная база данных инициализирована. Путь: {CHROMA_DB_PATH}")

    def add_documents(self, documents, client_id):
        """
        Добавляет документы в векторную базу данных.
        Метаданные документа расширяются client_id для мультитенантности.
        """
        if not documents:
            print("⚠️ Документы для добавления отсутствуют.")
            return

        # Добавляем client_id в метаданные каждого документа
        for doc in documents:
            if 'metadata' not in doc or not isinstance(doc['metadata'], dict):
                doc['metadata'] = {}
            doc['metadata']['client_id'] = client_id

        # Добавляем документы
        self.db.add_texts(
            texts=[doc.page_content for doc in documents],
            metadatas=[doc.metadata for doc in documents]
        )
        print(f"✅ Добавлено {len(documents)} чанков в базу данных для клиента {client_id}.")

    def search_documents(self, query: str, client_id: str, k: int = 4):
        """
        Ищет наиболее релевантные чанки по запросу.
        Обязательно фильтрует по client_id.
        """
        print(f"🔍 Поиск релевантных документов для запроса: '{query}' (клиент: {client_id})")

        # Фильтрация по client_id
        filter_dict = {"client_id": {"$eq": client_id}}

        # Используем LangChain для поиска
        results = self.db.similarity_search(query, k=k, filter=filter_dict)

        if not results:
            print(f"❌ Документы для клиента {client_id} не найдены.")
            return []

        print(f"✅ Найдено {len(results)} релевантных чанков.")
        return results
