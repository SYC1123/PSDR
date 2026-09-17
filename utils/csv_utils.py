import pandas as pd


def label_frame_to_int(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        True: 1,
        False: 0,
        "True": 1,
        "False": 0,
        "true": 1,
        "false": 0,
        "TRUE": 1,
        "FALSE": 0,
    }
    normalized = df.copy()
    for col in normalized.columns:
        normalized[col] = normalized[col].map(lambda x: mapping.get(x, x))
    return normalized.astype(int)
