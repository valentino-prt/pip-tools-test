from concurrent.futures import ThreadPoolExecutor, as_completed
from toto_client import fetch_data  # bloquant
import time

def safe_fetch(i):
    try:
        return fetch_data(f"https://api.example.com/data/{i}")
    except Exception as e:
        return f"Error on {i}: {e}"

def run_batch():
    results = []
    start = time.time()

    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = [executor.submit(safe_fetch, i) for i in range(10000)]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    print(f"Done in {time.time() - start:.2f} seconds")
    return results

all_results = run_batch()