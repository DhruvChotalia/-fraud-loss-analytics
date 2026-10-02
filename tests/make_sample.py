"""
Generate small CSVs in the same shape as the Kaggle files, for testing and CI.

  sample_clean.csv  -> every control should PASS
  sample_dirty.csv  -> planted defects the controls must catch

Usage: python tests/make_sample.py
"""
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data"
HEADER = ["", "trans_date_trans_time", "cc_num", "merchant", "category", "amt", "first", "last",
          "gender", "street", "city", "state", "zip", "lat", "long", "city_pop", "job", "dob",
          "trans_num", "unix_time", "merch_lat", "merch_long", "is_fraud"]
CATS = ["grocery_pos", "shopping_net", "misc_net", "gas_transport", "entertainment", "travel"]


def make_rows(n: int, seed: int):
    rnd = random.Random(seed)
    start = datetime(2019, 1, 1)
    for i in range(n):
        t = start + timedelta(minutes=rnd.randint(0, 60 * 24 * 365))
        fraud = 1 if rnd.random() < 0.01 else 0
        amt = round(rnd.uniform(200, 1200) if fraud else rnd.uniform(1, 150), 2)
        lat, lon = round(rnd.uniform(25, 48), 4), round(rnd.uniform(-120, -70), 4)
        yield [i, t.strftime("%Y-%m-%d %H:%M:%S"), 4000000000000000 + rnd.randint(0, 999),
               f"fraud_Merchant {rnd.randint(1, 60)}", rnd.choice(CATS), amt, "Test", "User",
               rnd.choice("MF"), "1 Main St", "Town", "ON", "10001", lat, lon, 5000, "Analyst",
               "1990-01-01", f"t{seed}-{i:07d}", int(t.timestamp()),
               round(lat + rnd.uniform(-1, 1), 4), round(lon + rnd.uniform(-1, 1), 4), fraud]


def write(name, rows):
    OUT.mkdir(exist_ok=True)
    with open(OUT / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    print(f"wrote data/{name} ({len(rows):,} rows)")


if __name__ == "__main__":
    write("sample_clean.csv", list(make_rows(5000, seed=1)))

    dirty = list(make_rows(5000, seed=2))
    dirty[10][5] = "abc"                  # amt_not_numeric
    dirty[11][5] = "-5.00"                # amt_not_positive
    dirty[12][1] = "2019/13/45"           # timestamp_invalid
    dirty[13][22] = "2"                   # is_fraud_invalid
    dirty[14][18] = ""                    # trans_num_missing
    dirty[15][18] = dirty[16][18]         # trans_num_duplicate (2 rows)
    dirty[17][13] = "123.0"               # customer_coords_out_of_range (warning)
    write("sample_dirty.csv", dirty)
