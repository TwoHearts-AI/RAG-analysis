import os
from text_chunker import TextChunker

def read_file(file_path: str) -> str:
    """
    Читает содержимое файла и возвращает его как строку.

    Args:
        file_path (str): Путь к файлу для чтения.

    Returns:
        str: Содержимое файла.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл не найден: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def write_chunks(chunks: list, file_path: str) -> None:
    """
    Записывает список чанков в файл, каждый чанк на новой строке.

    Args:
        chunks (list): Список чанков текста.
        file_path (str): Путь к файлу для записи.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        for i, chunk in enumerate(chunks, 1):
            file.write(f"Чанк {i}:\n{chunk}\n\n")

def main():
    """
    Основная функция для выполнения теста:
    1. Читает текст из test_file.txt
    2. Разбивает текст на чанки
    3. Записывает чанки в chunked_test_file.txt
    """
    input_file = 'chat_example.txt'
    output_file = 'chunked_test_file.txt'
    
    try:
        # Чтение текста из входного файла
        text = read_file(input_file)
        print(f"Текст успешно прочитан из {input_file}.")

        # Инициализация TextChunker с параметрами по умолчанию
        chunker = TextChunker(chunk_size=1000, chunk_overlap=200)

        # Разбиение текста на чанки
        chunks = chunker.split_text(text)
        print(f"Текст успешно разбит на {len(chunks)} чанков.")

        # Запись чанков в выходной файл
        write_chunks(chunks, output_file)
        print(f"Чанки успешно записаны в {output_file}.")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    main()
