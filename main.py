import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import sklearn
from bs4 import BeautifulSoup
import yaml

def main():
    print("✅ All libraries imported successfully!")
    print("🔢 Example array:", np.arange(5))
    print("📦 Pandas version:", pd.__version__)
    print("🌐 Request example:", requests.get("https://httpbin.org/ip").json())

if __name__ == "__main__":
    main()
