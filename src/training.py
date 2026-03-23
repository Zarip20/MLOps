import os
import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from src.utils import load_config

config = load_config()
model_params = config['models']

def train_models(X_train, y_train, incremental=False, existing_models=None):
    models = {}
    if incremental and existing_models and 'mlp' in existing_models:
        mlp = existing_models['mlp']
        mlp.partial_fit(X_train, y_train, classes=np.unique(y_train))
    else:
        mlp = MLPClassifier(**model_params['mlp'])
        mlp.fit(X_train, y_train)
    models['mlp'] = mlp
    dt = DecisionTreeClassifier(**model_params['dt'])
    dt.fit(X_train, y_train)
    models['dt'] = dt
    return models

def load_or_create_models():
    model_dir = config['paths']['models']
    models = {}
    for model_name in ['mlp', 'dt']:
        path = os.path.join(model_dir, f'{model_name}_latest.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[model_name] = pickle.load(f)
    return models if models else None
