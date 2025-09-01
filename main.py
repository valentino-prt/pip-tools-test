import pandas as pd
import requests
import numpy as np

import pandas as pd

def normalize_basic_one_day(df: pd.DataFrame) -> pd.DataFrame:
    # datetime propre
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").drop_duplicates("datetime")

    # vérifier que c'est bien un seul jour (optionnel)
    if df["datetime"].dt.normalize().nunique() != 1:
        raise ValueError("La DataFrame contient plusieurs jours.")

    day = df["datetime"].dt.normalize().iloc[0]
    start = day + pd.Timedelta(hours=8)   # 08:00:00
    end   = day + pd.Timedelta(hours=22)  # 22:00:00 (INCLUS)

    idx = pd.date_range(start, end, freq="1S")  # 1 point/seconde, 22:00 inclus
    out = df.set_index("datetime").reindex(idx).ffill()
    out.index.name = "datetime"
    return out



def main():
    print("✅ All libraries imported successfully!")

    print("🔢 Example array:", np.arange(5))

    print("📦 Pandas version:", pd.__version__)
    print("🌐 Request example:", requests.get("https://httpbin.org/ip").json())


if __name__ == "__main__":
    main()
