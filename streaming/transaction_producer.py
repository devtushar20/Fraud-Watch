import argparse, json, os, time, pandas as pd

def run_producer(interval=1.0, max_events=100, out_dir="data/stream_in"):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv("data/processed_transactions.csv")
    records = df.iloc[int(len(df) * 0.80):].head(max_events)      # replay the hold-out period
    for i, (_, row) in enumerate(records.iterrows()):
        payload = {k: (row[k].item() if hasattr(row[k], "item") else row[k]) for k in
                   ["transaction_id", "customer_id", "merchant_id", "amount", "merchant_category", "channel",
                    "city", "distance_from_home_km", "is_foreign", "txn_count_1h", "amount_ratio",
                    "hour_of_day", "day_of_week", "is_weekend", "is_night", "is_online", "is_fraud"]}
        with open(os.path.join(out_dir, f"evt_{i:05d}.json"), "w") as f:
            json.dump(payload, f)
        time.sleep(interval)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--interval", type=float, default=1.0); ap.add_argument("--events", type=int, default=100)
    a = ap.parse_args(); run_producer(a.interval, a.events)
