import asyncio
from toto_client import fetch_data  # async


semaphore = asyncio.Semaphore(100)  # max 100 requêtes simultanées

async def safe_fetch(i):
    async with semaphore:
        try:
            return await fetch_data(f"https://api.example.com/data/{i}")
        except Exception as e:
            return f"Error on {i}: {e}"

async def main():
    tasks = [asyncio.create_task(safe_fetch(i)) for i in range(10000)]
    results = await asyncio.gather(*tasks)
    return results

results = asyncio.run(main())