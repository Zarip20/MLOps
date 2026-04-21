import os
import pandas as pd
from src.utils import save_state, load_state

import zipfile
import os


def extract_zip(zip_path, extract_to='data', expected_csv=None):
    if expected_csv and os.path.exists(expected_csv):
        return expected_csv
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Ни файл {expected_csv}, ни архив {zip_path} не найдены.")
    print(f"Распаковка {zip_path} в {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    for root, dirs, files in os.walk(extract_to):
        for file in files:
            if file.endswith('.csv'):
                return os.path.join(root, file)
    raise RuntimeError("В архиве не найден CSV-файл.")

def split_into_batches(original_path, time_col='INSR_BEGIN', output_dir='data/raw_batches'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df = pd.read_csv(original_path, parse_dates=[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)
    df['year_month'] = df[time_col].dt.to_period('M')
    batches = []
    for period, group in df.groupby('year_month'):
        batch_name = f"batch_{period}.csv"
        batch_path = os.path.join(output_dir, batch_name)
        group.drop(columns=['year_month']).to_csv(batch_path, index=False)
        batches.append(batch_path)
    state = load_state()
    state['batches'] = batches
    state['last_processed'] = -1
    save_state(state)
    print(f"Разбито на {len(batches)} батчей")

def get_next_batch():
    state = load_state()
    next_idx = state['last_processed'] + 1
    if next_idx >= len(state['batches']):
        return None, None
    batch_path = state['batches'][next_idx]
    df = pd.read_csv(batch_path)
    return df, next_idx

def mark_batch_processed(idx):
    state = load_state()
    state['last_processed'] = idx
    save_state(state)
