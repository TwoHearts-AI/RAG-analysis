from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

main = ReplyKeyboardMarkup(keyboard=[
    [KeyboardButton(text='Upload Chat')],
    [KeyboardButton(text='Search')],
    [KeyboardButton(text='RAG Analysis')],
    [KeyboardButton(text='Collections')]
], resize_keyboard=True)