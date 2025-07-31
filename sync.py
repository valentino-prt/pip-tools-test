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

import pandas as pd
from pathlib import Path

class DataFrameManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._df = pd.DataFrame()
        return cls._instance

    def get(self):
        return self._df

    def set(self, df: pd.DataFrame):
        self._df = df

    def add_row(self, row: dict):
        self._df = pd.concat([self._df, pd.DataFrame([row])], ignore_index=True)

    def fill_from_csv(self, path: str | Path):
        self._df = pd.read_csv(path)

    def save_to_csv(self, path: str | Path, index=False):
        self._df.to_csv(path, index=index)

    def clear(self):
        self._df = pd.DataFrame()

    def shape(self):
        return self._df.shape

    def summary(self):
        return self._df.describe()
