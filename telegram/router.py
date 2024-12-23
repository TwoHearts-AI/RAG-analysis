from aiogram import Bot, F, Router
from aiogram.types import Message
from aiogram.filters import CommandStart
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
import keyboards as kb
import httpx

router = Router()

class UploadStates(StatesGroup):
    waiting_for_file = State()

class SearchStates(StatesGroup):
    waiting_for_query = State()

async def make_api_call(client, method, url, **kwargs):
    try:
        response = await client.request(method, url, **kwargs)
        response.raise_for_status()
        return response
    except httpx.HTTPError as e:
        return f"API Error: {str(e)}"

@router.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer('Welcome to Chat Analysis Bot!', reply_markup=kb.main)

@router.message(F.text == 'Upload Chat')
async def upload_handler(message: Message, state: FSMContext):
    await state.set_state(UploadStates.waiting_for_file)
    await message.answer('Please send me your chat file')

@router.message(UploadStates.waiting_for_file, F.document)
async def process_file(message: Message, state: FSMContext, bot: Bot):
    try:
        file = await bot.get_file(message.document.file_id)
        file_path = file.file_path
        file_content = await bot.download_file(file_path)
        
        async with httpx.AsyncClient() as client:
            files = {'file': ('chat.txt', file_content)}
            response = await make_api_call(
                client, 
                'POST',
                "http://api:8000/upload/default_collection",
                files=files
            )
            if isinstance(response, str):  # Error occurred
                await message.answer(response)
            else:
                await message.answer("File uploaded successfully!")
        await state.clear()
    except Exception as e:
        await message.answer(f"Error processing file: {str(e)}")
        await state.clear()

@router.message(F.text == 'Search')
async def search_handler(message: Message, state: FSMContext):
    await state.set_state(SearchStates.waiting_for_query)
    await message.answer('What would you like to search for?')

@router.message(SearchStates.waiting_for_query)
async def process_search(message: Message, state: FSMContext):
    async with httpx.AsyncClient() as client:
        response = await make_api_call(
            client,
            'POST',
            "http://api:8000/search",
            json={
                "text": message.text,
                "collection_name": "default_collection"
            }
        )
        if isinstance(response, str):  # Error occurred
            await message.answer(response)
        else:
            results = response.json()
            await message.answer(str(results))
    await state.clear()

@router.message(F.text == 'RAG Analysis')
async def rag_handler(message: Message):
    async with httpx.AsyncClient() as client:
        response = await make_api_call(
            client,
            'POST',
            "http://api:8000/rag-inference",
            json={
                "collection_name": "default_collection"
            }
        )
        if isinstance(response, str):  # Error occurred
            await message.answer(response)
        else:
            results = response.json()
            await message.answer(results["answer"])

@router.message(F.text == 'Collections')
async def collections_handler(message: Message):
    async with httpx.AsyncClient() as client:
        response = await make_api_call(
            client,
            'GET',
            "http://api:8000/collections"
        )
        if isinstance(response, str):  # Error occurred
            await message.answer(response)
        else:
            collections = response.json()
            await message.answer(str(collections))