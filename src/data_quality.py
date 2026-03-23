import pandas as pd
import numpy as np
from typing import Dict, List
from src.association import load_association_rules

class DataQualityEvaluator:
    def __init__(self, rules: List[Dict] = None):
        self.rules = rules or []
        for rule in self.rules:
            if isinstance(rule.get('condition'), str):
                try:
                    rule['condition'] = eval(rule['condition'])
                except Exception as e:
                    print(f"Ошибка преобразования правила {rule.get('name', '?')}: {e}")

    def completeness(self, df: pd.DataFrame) -> Dict[str, float]:
        total_missing = df.isna().sum().sum()
        total_cells = np.prod(df.shape)
        col_missing = df.isna().sum().max() / df.shape[0]
        row_missing = df.isna().sum(axis=1).max() / df.shape[1]
        return {
            'total_missing_ratio': total_missing / total_cells,
            'max_col_missing_ratio': col_missing,
            'max_row_missing_ratio': row_missing
        }

    def validity(self, df: pd.DataFrame) -> Dict[str, bool]:
        numeric_cols = df.select_dtypes(include=np.number).columns
        result = {}
        for col in numeric_cols:
            result[f'{col}_non_negative'] = (df[col] >= 0).all()
        return result

    def timeliness(self, df: pd.DataFrame, time_col='INSR_BEGIN') -> Dict[str, any]:
        if time_col not in df.columns:
            return {}
        dates = pd.to_datetime(df[time_col]).sort_values()
        unique_dates = dates.unique()
        if len(unique_dates) > 1:
            max_gap = (unique_dates[1:] - unique_dates[:-1]).max().days
        else:
            max_gap = 0
        return {
            'min_date': dates.min(),
            'max_date': dates.max(),
            'max_gap_days': max_gap
        }

    def check_association_rules(self, df: pd.DataFrame) -> Dict[str, float]:
        violations = {}
        for rule in self.rules:
            name = rule['name']
            condition = rule['condition']
            expected = rule['expected']
            mask = condition(df)
            violation_mask = mask != expected
            violations[name] = violation_mask.mean()
        return violations

    def clean_data(self, df: pd.DataFrame, max_missing_row=0.5, max_violation_ratio=0.1) -> pd.DataFrame:
        missing_ratio = df.isna().sum(axis=1) / df.shape[1]
        df = df[missing_ratio <= max_missing_row].copy()
        for rule in self.rules:
            condition = rule['condition']
            expected = rule['expected']
            mask = condition(df)
            violation_ratio = (mask != expected).mean()
            if violation_ratio <= max_violation_ratio:
                df = df[mask == expected]
        return df

    def check_dynamic_rules(self, df: pd.DataFrame, categorical_cols: list) -> Dict[str, float]:
        rules = load_association_rules()
        violations = {}
        cat_df = df[categorical_cols].copy()
        for col in categorical_cols:
            cat_df[col] = cat_df[col].astype(str)
        for rule in rules:
            antecedent_mask = pd.Series(True, index=df.index)
            for ant in rule['antecedents']:
                col, val = ant.split(' = ')
                antecedent_mask &= (cat_df[col] == val)
            cons_col, cons_val = rule['consequents'][0].split(' = ')
            consequent_mask = (cat_df[cons_col] == cons_val)
            violation_mask = antecedent_mask & ~consequent_mask
            violations[rule['antecedents'][0] + ' → ' + rule['consequents'][0]] = violation_mask.mean()
        return violations
