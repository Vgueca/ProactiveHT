import os
import pandas as pd
from pandas.errors import EmptyDataError

def safe_read_csv(path, retries=3, delay=0.1):
    for attempt in range(1, retries+1):
        try:
            df = pd.read_csv(path)
            print(f"[csv_utils] Successfully read CSV: {path}")
            return df
        except EmptyDataError:
            print(f"[csv_utils] Warning: '{path}' empty on attempt {attempt}/{retries}. Retrying...")
            import time; time.sleep(delay)
    raise EmptyDataError(f"No columns to parse from '{path}' after {retries} retries.")




def update_predictions_csv(labels, predictions, approach_id, file_path):
    col = f"approach_{approach_id}"
    df_new = pd.DataFrame({'label': labels, col: predictions})
    
    df_new = df_new[df_new[col].notnull()].reset_index(drop=True)

    if os.path.exists(file_path):
        df = safe_read_csv(file_path)
        
        if 'label' not in df.columns:
            df.insert(0, 'label', df_new['label'])
            
        if len(df) != len(df_new):
            print(f"[csv_utils] Warning: length mismatch, overwriting {file_path}")
            df = df_new.copy()
        else:
            
            df[col] = df_new[col]
        df.to_csv(file_path, index=False)
        print(f"[csv_utils] Updated predictions CSV: {file_path}")
    else:
        df_new.to_csv(file_path, index=False)
        print(f"[csv_utils] Created new predictions CSV: {file_path}")




def update_memory_csv(instances, memory_usage, approach_id, file_path):
    col = f"approach_{approach_id}"
    df_new = pd.DataFrame({'instances': instances, col: memory_usage})
    if os.path.exists(file_path):
        df = safe_read_csv(file_path)
        if 'instances' not in df or len(df) != len(df_new):
            print(f"[csv_utils] Warning: inconsistent data, overwriting {file_path}")
            df = df_new
        else:
            df[col] = df_new[col]
        df.to_csv(file_path, index=False)
        print(f"[csv_utils] Updated memory CSV: {file_path}")
    else:
        df_new.to_csv(file_path, index=False)
        print(f"[csv_utils] Created new memory CSV: {file_path}")
