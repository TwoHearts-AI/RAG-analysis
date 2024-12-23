from telegram.ext import Application, MessageHandler, filters
import httpx
import os
from config import CONFIG

API_URL = 'http://api:8000'

TELEGRAM_TOKEN = CONFIG.TELEGRAM_TOKEN


async def handle_document(update, context):
    file = await update.message.document.get_file()
    file_content = await file.download_as_bytearray()

    # Send to API
    async with httpx.AsyncClient() as client:
        files = {'file': (file.file_name, file_content)}
        response = await client.post(f"{API_URL}/upload/default", files=files)

    await update.message.reply_text("Документ обработан")


async def handle_message(update, context):
    # Make RAG request
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_URL}/rag-inference",
            json={"collection_name": "default"}
        )
        result = response.json()

    await update.message.reply_text(result["answer"])


def main():
    app = Application.builder().token(os.getenv('TELEGRAM_TOKEN')).build()

    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT, handle_message))

    app.run_polling()


if __name__ == '__main__':
    main()
