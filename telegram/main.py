import asyncio
from aiogram import Bot, Dispatcher
from router import router
from config import CONFIG

TELEGRAM_TOKEN = CONFIG.TELEGRAM_TOKEN

async def main():
    bot = Bot(token=TELEGRAM_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)
    await dp.start_polling(bot)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('Бот выключен')