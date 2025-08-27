
import os
import pandas as pd
from pandas.errors import EmptyDataError
from scipy.io.arff import loadarff 



def save_stream(stream, file, size, save = True):
    if stream is None:
        return
    stream_df_x = []
    stream_df_y = []
    for i, (x, y) in enumerate(stream.take(size)):
        stream_df_x.append(x)
        stream_df_y.append(y)

    stream_df_x = pd.DataFrame(stream_df_x)
    stream_df_y = pd.DataFrame(stream_df_y)

    stream_df = pd.concat([stream_df_x, stream_df_y], axis=1, ignore_index=True)

    if save:
        stream_df.to_csv(file, index=None)
    
    return stream_df


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


class CSVStream:
    def __init__(self, csv_file, target = None):
        self.csv_file = csv_file
        self.data = pd.read_csv(self.csv_file)
        if target is None:
            self.target = self.data.columns[-1]
        self.classes = self.data[self.target].unique()
        self.n_classes = len(self.classes)
        self.n_features = self.data.shape[1] - 1
        self.n_samples = self.data.shape[0]
        self.index = 0

    def __iter__(self):
        while True:
            row = self.data.iloc[self.index, :-1]
            x = row.to_dict()
            y = self.data.iloc[self.index, -1]
            self.index += 1
            if self.index >= self.n_samples:
                break
            yield x, y

class ARFFStream:
    def __init__(self, arff_file, target = None):
        self.arff_file = arff_file
        arff_loaded = loadarff(self.arff_file)
        self.data = pd.DataFrame(arff_loaded[0])
        
        if target is None:
            self.target = self.data.columns[-1]
        codes, uniques = pd.factorize(self.data.iloc[:, -1])
        self.data.iloc[:, -1] = codes
        self.classes = self.data[self.target].unique()
        self.n_classes = len(self.classes)
        self.n_features = self.data.shape[1] - 1
        self.n_samples = self.data.shape[0]
        self.index = 0

    def __iter__(self):
        while True:
            row = self.data.iloc[self.index, :-1]
            x = row.to_dict()
            y = self.data.iloc[self.index, -1]
            self.index += 1
            if self.index >= self.n_samples:
                break
            yield x, y
