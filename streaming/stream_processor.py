import glob, json, os, time, joblib, numpy as np, pandas as pd

ALERT_THRESHOLD = 0.5
def run_processor(in_dir="data/stream_in", out_path="out/live_predictions.jsonl"):
    model = joblib.load("out/rf.joblib"); cols = list(model.feature_names_in_)
    lat, alerts, n = [], 0, 0
    with open(out_path, "w") as out:
        for fp in sorted(glob.glob(os.path.join(in_dir, "evt_*.json"))):
            ev = json.load(open(fp)); t0 = time.perf_counter()
            x = pd.DataFrame([{c: ev.get(c, 0) for c in cols if not c.startswith("cat_")}])
            for c in cols:
                if c.startswith("cat_"): x[c] = int(ev["merchant_category"] == c[4:])
            p = float(model.predict_proba(x[cols])[0, 1]); ms = (time.perf_counter() - t0) * 1000
            alert = p >= ALERT_THRESHOLD; alerts += alert; n += 1; lat.append(ms)
            out.write(json.dumps({"transaction_id": ev["transaction_id"], "fraud_probability": round(p, 4),
                                  "alert": bool(alert), "actual_fraud": ev["is_fraud"], "latency_ms": round(ms, 3)}) + "\n")
    json.dump({"events": n, "mean_ms": float(np.mean(lat)), "p95_ms": float(np.percentile(lat, 95)),
               "max_ms": float(np.max(lat)), "alerts": int(alerts)}, open("out/stream.json", "w"))

if __name__ == "__main__":
    run_processor()
