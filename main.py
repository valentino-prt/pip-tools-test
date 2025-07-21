import pandas as pd
import requests
import numpy as np


def main():
    print("✅ All libraries imported successfully!")

    print("🔢 Example array:", np.arange(5))

    print("📦 Pandas version:", pd.__version__)
    print("🌐 Request example:", requests.get("https://httpbin.org/ip").json())


if __name__ == "__main__":
    main()
