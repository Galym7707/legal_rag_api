import io
import logging
from typing import List
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document # Для типизации

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_document_content(file_stream: io.BytesIO, filename: str, metadata: dict = None) -> List[Document]:
    """
    Обрабатывает содержимое файла, извлекает текст и разбивает его на чанки.
    Возвращает список объектов LangChain Document.
    """
    if metadata is None:
        metadata = {}

    file_extension = filename.split('.')[-1].lower()
    temp_filepath = f"/tmp/{filename}" # Временный файл для загрузчиков LangChain

    try:
        # Сохраняем поток в временный файл, так как некоторые загрузчики LangChain требуют путь к файлу
        file_stream.seek(0) # Убедимся, что указатель находится в начале файла
        with open(temp_filepath, "wb") as temp_file:
            temp_file.write(file_stream.read())

        loader = None
        if file_extension == 'txt':
            loader = TextLoader(temp_filepath, encoding='utf-8')
        elif file_extension == 'pdf':
            loader = PyPDFLoader(temp_filepath)
        elif file_extension == 'docx':
            loader = Docx2txtLoader(temp_filepath)
        # TODO: Добавить поддержку других форматов (изображения с OCR через `unstructured`)
        else:
            logging.warning(f"⚠️ Неподдерживаемый формат файла: {file_extension}. Пробуем TextLoader.")
            try:
                loader = TextLoader(temp_filepath, encoding='utf-8')
            except Exception as e:
                logging.error(f"❌ Не удалось загрузить файл {filename} как TXT: {e}")
                return []

        if loader:
            documents = loader.load()
            logging.info(f"✅ Извлечено {len(documents)} страниц/документов из файла '{filename}'.")
            
            # Добавляем переданные метаданные к каждому документу
            for doc in documents:
                doc.metadata.update(metadata)
                doc.metadata["original_filename"] = filename # Добавляем имя файла
                doc.metadata["timestamp"] = datetime.now().isoformat() # Добавляем время загрузки

            # Разбиваем документы на чанки
            # RecursiveCharacterTextSplitter - хороший универсальный сплиттер
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,      # Размер чанка в символах
                chunk_overlap=200,    # Перекрытие между чанками для сохранения контекста
                length_function=len,
                is_separator_regex=False,
            )
            chunks = text_splitter.split_documents(documents)
            logging.info(f"✅ Документ '{filename}' разбит на {len(chunks)} чанков.")
            return chunks
        else:
            logging.error(f"❌ Не удалось найти подходящий загрузчик для файла: {filename}")
            return []

    except Exception as e:
        logging.error(f"❌ Ошибка при обработке документа '{filename}': {e}", exc_info=True)
        return []
    finally:
        # Удаляем временный файл
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
            logging.debug(f"Временный файл {temp_filepath} удален.")
