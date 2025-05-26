import os
import pandas as pd
from filelock import FileLock, Timeout
from .csv import safe_read_csv

def store_global_result(summary_csv, record, key_fields, lock_timeout=10):
    os.makedirs(os.path.dirname(summary_csv), exist_ok=True)
    
    lock_path = summary_csv + ".lock"
    lock = FileLock(lock_path, timeout=lock_timeout)

    try:
        with lock:
            if os.path.exists(summary_csv):
                df = safe_read_csv(summary_csv)
            else:
                df = pd.DataFrame()

            if not df.empty:
                mask = pd.Series(True, index=df.index)
                for k in key_fields:
                    mask &= df[k] == record[k]
            else:
                mask = pd.Series(dtype=bool)

            if df.empty or not mask.any():
                df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
                action = "Appended"
            else:
                df.loc[mask, record.keys()] = record.values()
                action = "Updated"

            df.to_csv(summary_csv, index=False)
            print(f"[global_results] {action} record in {summary_csv}")

    except Timeout:
        raise RuntimeError(f"Could not acquire lock on {lock_path} within {lock_timeout}s")
