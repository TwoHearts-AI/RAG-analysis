from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextChunker:
    """
    Класс для разбивки текста на чанки с использованием заданного сплиттера.
    
    Атрибуты:
        splitter (RecursiveCharacterTextSplitter): Сплиттер для разбивки текста.
    """

    def __init__(self, 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 200, 
                 separators: list = None):
        """
        Инициализирует TextChunker с указанными параметрами сплиттера.

        Args:
            chunk_size (int): Максимальный размер чанка.
            chunk_overlap (int): Перекрытие между чанками.
            separators (list, optional): Список разделителей для сплиттера.
        """
        if separators is None:
            separators = ["\n\n", "\n", " ", ""]

        self.splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

    def split_text(self, text: str) -> List[str]:
        """
        Разбивает входной текст на чанки.

        Args:
            text (str): Входной текст для разбивки.

        Returns:
            list: Список чанков текста.
        """
        return self.splitter.split_text(text)
    
chunker = TextChunker()