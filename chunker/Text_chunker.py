from datetime import datetime, timedelta
from typing import List
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

#baseline chunker
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
    

#session chunker

class Message:
    def __init__(self, date: str, author: str, text: str):
        self.date = datetime.strptime(date, "%d/%m/%Y, %H:%M:%S")
        self.author = author
        self.text = text

class SessionChunker:
    def __init__(self, session_minutes_threshold: int):
        """
        Инициализация чанкера.

        :param session_minutes_threshold: Порог времени в минутах для разделения сессий.
        """
        self.session_minutes_threshold = session_minutes_threshold

    def parse_messages_from_file(self, file_path: str) -> List[Message]:
        """
        Парсит сообщения из текстового файла.

        :param file_path: Путь к файлу с сообщениями.
        :return: Список сообщений.
        """
        messages = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                match = re.match(r"\[(\d{2}/\d{2}/\d{4}, \d{2}:\d{2}:\d{2})\] (.*?): (.*)", line.strip())
                if match:
                    date, author, text = match.groups()
                    messages.append(Message(date, author, text))
        return messages

    def chunk(self, messages: List[Message]) -> List[List[Message]]:
        """
        Разбивает список сообщений на чанки по сессиям.

        :param messages: Список сообщений.
        :return: Список чанков.
        """
        if not messages:
            return []

        sessions = []
        current_session = [messages[0]]

        for msg in messages[1:]:
            last_msg_time = current_session[-1].date
            time_diff = (msg.date - last_msg_time).total_seconds() / 60

            if time_diff <= self.session_minutes_threshold:
                current_session.append(msg)
            else:
                sessions.append(current_session)
                current_session = [msg]

        if current_session:
            sessions.append(current_session)

        return sessions


def get_chunker(use_session_chunker: bool, session_minutes_threshold: int = 20):
    """
    Возвращает экземпляр выбранного чанкера.

    :param use_session_chunker: Флаг выбора чанкера.
    :param session_minutes_threshold: Порог времени для сессий (только для SessionChunker).
    :return: Экземпляр чанкера.
    """
    if use_session_chunker:
        return SessionChunker(session_minutes_threshold)
    else:
        return TextChunker()

