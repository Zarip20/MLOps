import os
import json
import pickle
import yaml
import numpy as np
import pandas as pd

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def save_state(state):
    with open('state.json', 'w') as f:
        json.dump(state, f)

def load_state():
    if not os.path.exists('state.json'):
        return {'batches': [], 'last_processed': -1}
    with open('state.json', 'r') as f:
        return json.load(f)

def save_model(model, filename):
    model_dir = load_config()['paths']['models']
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, filename)
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    path = os.path.join(load_config()['paths']['models'], filename)
    with open(path, 'rb') as f:
        return pickle.load(f)

def _convert_to_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: _convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, np.datetime64):
        return pd.Timestamp(obj).isoformat()
    else:
        return obj

def save_metadata(prefix, data, batch_idx):
    meta_dir = load_config()['paths']['metadata']
    os.makedirs(meta_dir, exist_ok=True)
    filename = f"{prefix}_{batch_idx}.json"
    data_serializable = _convert_to_json_serializable(data)
    with open(os.path.join(meta_dir, filename), 'w') as f:
        json.dump({'batch_idx': batch_idx, **data_serializable}, f, indent=2)
