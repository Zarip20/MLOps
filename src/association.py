import os
import json
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def generate_association_rules(df, categorical_cols, output_path='data/rules.json'):
    cat_df = df[categorical_cols].copy()
    for col in categorical_cols:
        cat_df[col] = cat_df[col].astype(str)
    encoded = pd.get_dummies(cat_df, prefix_sep='=')
    frequent_itemsets = apriori(encoded, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    top_rules = rules.nlargest(5, 'confidence')
    rules_to_save = []
    for _, row in top_rules.iterrows():
        antecedents = [item.replace('=', ' = ') for item in row['antecedents']]
        consequents = [item.replace('=', ' = ') for item in row['consequents']]
        rules_to_save.append({
            'antecedents': antecedents,
            'consequents': consequents,
            'confidence': row['confidence'],
            'support': row['support']
        })
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(rules_to_save, f, indent=2)
    print(f"Создано {len(rules_to_save)} правил, сохранены в {output_path}")

def load_association_rules(rules_path='data/rules.json'):
    if not os.path.exists(rules_path):
        return []
    with open(rules_path, 'r') as f:
        return json.load(f)
