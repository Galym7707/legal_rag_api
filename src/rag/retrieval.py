import logging
from typing import List
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def retrieve_documents(query: str, retriever: VectorStoreRetriever) -> List[Document]:
    """
    Ищет релевантные документы в векторной базе данных.
    """
    if not retriever:
        logging.error("Ретривер не инициализирован. Поиск документов невозможен.")
        return []
    try:
        # Используем ретривер для поиска документов, релевантных запросу
        relevant_docs = retriever.invoke(query)
        logging.info(f"✅ Найдено {len(relevant_docs)} релевантных документов для запроса: '{query[:50]}...'")
        return relevant_docs
    except Exception as e:
        logging.error(f"❌ Ошибка при поиске документов: {e}")
        return []
