import argparse
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from src.data_collection import split_into_batches, get_next_batch, mark_batch_processed
from src.data_quality import DataQualityEvaluator
from src.preprocessing import create_preprocessor
from src.training import train_models, load_or_create_models
from src.utils import load_config, save_metadata, save_model, load_model
from src.association import generate_association_rules
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def update():
    import os
    config = load_config()
    df_batch, idx = get_next_batch()
    if df_batch is None:
        logging.info("Нет новых батчей для обработки.")
        return True

    evaluator = DataQualityEvaluator(rules=config['quality_rules'])
    quality_metrics = {
        'completeness': evaluator.completeness(df_batch),
        'validity': evaluator.validity(df_batch),
        'timeliness': evaluator.timeliness(df_batch, time_col='INSR_BEGIN'),
        'rule_violations': evaluator.check_association_rules(df_batch)
    }
    dynamic_violations = evaluator.check_dynamic_rules(df_batch, config['categorical_cols'])
    quality_metrics['dynamic_rule_violations'] = dynamic_violations
    save_metadata('quality', quality_metrics, batch_idx=idx)

    df_clean = evaluator.clean_data(df_batch)

    df_clean['HAS_CLAIM'] = df_clean['CLAIM_PAID'].notna().astype(int)
    df_clean.drop(columns=['CLAIM_PAID'], inplace=True)

    feature_cols = config['feature_cols']
    X = df_clean[feature_cols]
    y = df_clean['HAS_CLAIM']

    preprocessor_path = os.path.join(config['paths']['models'], 'preprocessor.pkl')
    if os.path.exists(preprocessor_path):
        logging.info(f"Загрузка препроцессора из {preprocessor_path}")
        preprocessor = load_model('preprocessor.pkl')
        X_transformed = preprocessor.transform(X)
        logging.info(f"Размерность после трансформации: {X_transformed.shape}")
    else:
        logging.info("Создание препроцессора")
        preprocessor = create_preprocessor(
            categorical_cols=config['categorical_cols'],
            numerical_cols=config['numerical_cols']
        )
        X_transformed = preprocessor.fit_transform(X)
        logging.info(f"Размерность после fit_transform: {X_transformed.shape}")
        save_model(preprocessor, 'preprocessor.pkl')
        logging.info("Препроцессор сохранён")

    existing_models = load_or_create_models()
    incremental = existing_models is not None
    if not incremental:
        generate_association_rules(df_clean, config['categorical_cols'])
    models = train_models(X_transformed, y, incremental=incremental, existing_models=existing_models)

    X_tr, X_val, y_tr, y_val = train_test_split(X_transformed, y, test_size=0.2, random_state=42)
    metrics = {}
    for name, model in models.items():
        y_pred = model.predict(X_val)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_val)[:, 1]
            roc_auc = roc_auc_score(y_val, y_proba)
        else:
            roc_auc = None
        metrics[name] = {
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0),
            'f1': f1_score(y_val, y_pred, zero_division=0),
            'roc_auc': roc_auc
        }
    save_metadata('model_metrics', metrics, batch_idx=idx)

    best_model_name = max(metrics, key=lambda k: metrics[k]['f1'])
    best_model = models[best_model_name]
    save_model(best_model, 'best_model.pkl')
    for name, model in models.items():
        save_model(model, f'{name}_latest.pkl')

    mark_batch_processed(idx)
    logging.info(f"Батч {idx} обработан.")
    return True

def inference(file_path):
    config = load_config()
    df = pd.read_csv(file_path)
    preprocessor = load_model('preprocessor.pkl')
    model = load_model('best_model.pkl')
    X = df[config['feature_cols']]
    X_transformed = preprocessor.transform(X)
    df['predict'] = model.predict(X_transformed)
    output_path = file_path.replace('.csv', '_with_predict.csv')
    df.to_csv(output_path, index=False)
    logging.info(f"Результат сохранён в {output_path}")
    return output_path

def summary():
    import os
    import json
    from datetime import datetime
    config = load_config()
    meta_dir = config['paths']['metadata']
    quality_history = []
    model_history = []
    if os.path.exists(meta_dir):
        for fname in os.listdir(meta_dir):
            if fname.startswith('quality_') and fname.endswith('.json'):
                with open(os.path.join(meta_dir, fname), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    quality_history.append(data)
            elif fname.startswith('model_metrics_') and fname.endswith('.json'):
                with open(os.path.join(meta_dir, fname), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    metrics = {k: v for k, v in data.items() if k != 'batch_idx'}
                    model_history.append({
                        'batch_idx': data['batch_idx'],
                        'metrics': metrics
                    })
    quality_history.sort(key=lambda x: x['batch_idx'])
    model_history.sort(key=lambda x: x['batch_idx'])
    os.makedirs('reports', exist_ok=True)
    report_path = f"reports/summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Отчёт о работе системы\n\n")
        f.write("1. Качество данных по батчам:\n")
        for q in quality_history:
            f.write(f"Батч {q['batch_idx']}:\n")
            completeness = q.get('completeness', {})
            f.write(f"  completeness: total_missing_ratio = {completeness.get('total_missing_ratio', 'N/A')}\n")
            f.write(f"                max_col_missing_ratio = {completeness.get('max_col_missing_ratio', 'N/A')}\n")
            f.write(f"                max_row_missing_ratio = {completeness.get('max_row_missing_ratio', 'N/A')}\n")
            rule_violations = q.get('rule_violations', {})
            f.write(f"  rule_violations: {rule_violations}\n")
        f.write("\n2. Метрики моделей:\n")
        for m in model_history:
            f.write(f"Батч {m['batch_idx']}:\n")
            for name, metrics in m['metrics'].items():
                f1 = metrics.get('f1', 'N/A')
                roc = metrics.get('roc_auc', 'N/A')
                if roc != 'N/A':
                    f.write(f"  {name}: f1 = {f1:.4f}, roc_auc = {roc:.4f}\n")
                else:
                    f.write(f"  {name}: f1 = {f1:.4f}\n")
        f.write("\n3. Лучшая модель:\n")
        if model_history:
            last = model_history[-1]
            best_model_name = max(last['metrics'].items(), key=lambda x: x[1].get('f1', 0))[0]
            best_f1 = last['metrics'][best_model_name].get('f1', 'N/A')
            f.write(f"В последнем батче {last['batch_idx']} лучшая модель: {best_model_name} (f1 = {best_f1:.4f})\n")
        else:
            f.write("Нет данных о моделях.\n")
    return report_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', required=True, choices=['update', 'inference', 'summary', 'init'], help='Режим работы')
    parser.add_argument('-file', help='Путь к файлу для inference')
    args = parser.parse_args()
    if args.mode == 'init':
        split_into_batches('data/motor_data14-2018.csv')
        print("Инициализация завершена, батчи созданы.")
    elif args.mode == 'update':
        success = update()
        print(success)
    elif args.mode == 'inference':
        output = inference(args.file)
        print(output)
    elif args.mode == 'summary':
        report = summary()
        print(report)
