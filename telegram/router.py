from aiogram import Bot, F, Router
from aiogram.types import Message
from aiogram.filters import CommandStart
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
import keyboards as kb
import httpx
from aiogram.exceptions import TelegramBadRequest
router = Router()

class UploadStates(StatesGroup):
    waiting_for_file = State()

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
        filename = message.document.file_name
        collection_name = filename.split('.')[0]  # Remove file extension
        
        file = await bot.get_file(message.document.file_id)
        file_path = file.file_path
        file_content = await bot.download_file(file_path)
        
        async with httpx.AsyncClient(timeout=180) as client:
            files = {'file': (filename, file_content)}
            response = await make_api_call(
                client, 
                'POST',
                f"http://api:8000/upload/{collection_name}",
                files=files
            )
            if isinstance(response, str):  # Error occurred
                await message.answer(response)
            else:
                await message.answer(f"File uploaded successfully to collection: {collection_name}")
        await state.clear()
    except Exception as e:
        await message.answer(f"Error processing file: {str(e)}")
        await state.clear()

class SearchStates(StatesGroup):
    waiting_for_collection = State()
    waiting_for_query = State()

@router.message(F.text == 'Search')
async def search_handler(message: Message, state: FSMContext):
    async with httpx.AsyncClient() as client:
        # Сначала получаем список коллекций
        response = await make_api_call(client, 'GET', "http://api:8000/collections")
        if isinstance(response, str):
            await message.answer(response)
            return
            
        collections = response.json()['collections']
        formatted_text = "Select collection number:\n\n"
        for i, coll in enumerate(collections, 1):
            formatted_text += f"{i}. {coll['name']}\n"
        
        # Сохраняем список коллекций в состояние
        await state.update_data(collections={str(i): coll['name'] for i, coll in enumerate(collections, 1)})
        await state.set_state(SearchStates.waiting_for_collection)
        await message.answer(formatted_text)

@router.message(SearchStates.waiting_for_collection)
async def process_collection_choice(message: Message, state: FSMContext):
    data = await state.get_data()
    collections = data.get('collections', {})
    
    if not message.text.isdigit() or message.text not in collections:
        await message.answer("Please select a valid collection number")
        return
        
    collection_name = collections[message.text]
    await state.update_data(selected_collection=collection_name)
    await state.set_state(SearchStates.waiting_for_query)
    await message.answer(f'Selected collection: {collection_name}\nWhat would you like to search for?')

@router.message(SearchStates.waiting_for_query)
async def process_search(message: Message, state: FSMContext):
    data = await state.get_data()
    collection_name = data.get('selected_collection', 'default_collection')
    
    async with httpx.AsyncClient(timeout=250) as client:
        response = await make_api_call(
            client,
            'POST',
            "http://api:8000/search",
            json={
                "text": message.text,
                "collection_name": collection_name,
                "limit": 3  # Limit to 3 results
            }
        )
        if isinstance(response, str):
            await message.answer(response)
        else:
            results = response.json()['results']
            
            # Format results nicely
            formatted_text = "🔍 Search Results:\n\n"
            for i, result in enumerate(results, 1):
                text = result['text']
                score = result['score']
                # Truncate text if too long
                if len(text) > 200:
                    text = text[:200] + "..."
                formatted_text += f"Result {i} (score: {score:.2f}):\n{text}\n\n"
            
            try:
                await message.answer(formatted_text)
            except TelegramBadRequest as e:
                if "message is too long" in str(e):
                    # Split into multiple messages if too long
                    chunks = [formatted_text[i:i+4000] for i in range(0, len(formatted_text), 4000)]
                    for chunk in chunks:
                        await message.answer(chunk)
                else:
                    raise
    await state.clear()

class RAGStates(StatesGroup):
    waiting_for_collection = State()

@router.message(F.text == 'RAG Analysis')
async def rag_handler(message: Message, state: FSMContext):
    async with httpx.AsyncClient() as client:
        # Получаем список коллекций
        response = await make_api_call(client, 'GET', "http://api:8000/collections")
        if isinstance(response, str):
            await message.answer(response)
            return
            
        collections = response.json()['collections']
        formatted_text = "Select collection for analysis:\n\n"
        for i, coll in enumerate(collections, 1):
            formatted_text += f"{i}. {coll['name']}\n"
        
        await state.update_data(collections={str(i): coll['name'] for i, coll in enumerate(collections, 1)})
        await state.set_state(RAGStates.waiting_for_collection)
        await message.answer(formatted_text)

@router.message(RAGStates.waiting_for_collection)
async def process_rag(message: Message, state: FSMContext):
    data = await state.get_data()
    collections = data.get('collections', {})
    
    if not message.text.isdigit() or message.text not in collections:
        await message.answer("Please select a valid collection number")
        return
        
    collection_name = collections[message.text]
    
    async with httpx.AsyncClient(timeout=250) as client:
        await message.answer("🔄 Analyzing messages, please wait...")
        response = await make_api_call(
            client,
            'POST',
            "http://api:8000/rag-inference",
            json={
                "collection_name": collection_name
            }
        )
        if isinstance(response, str):
            await message.answer(response)
        else:
            results = response.json()
            try:
                await message.answer(f"Analysis for {collection_name}:\n\n{results['answer']}")
            except TelegramBadRequest as e:
                if "message is too long" in str(e):
                    chunks = [results['answer'][i:i+4000] for i in range(0, len(results['answer']), 4000)]
                    for chunk in chunks:
                        await message.answer(chunk)
                else:
                    raise
    
    await state.clear()


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
            collections = response.json()['collections']
            formatted_text = "📚 Collections:\n\n"
            for coll in collections:
                formatted_text += f"📁 {coll['name']}\n"
                formatted_text += f"   Documents: {coll['vectors_count']}\n\n"
            await message.answer(formatted_text)