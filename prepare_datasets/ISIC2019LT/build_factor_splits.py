import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from .split import check_leakage
except ImportError:
    from split import check_leakage


def _parse_factors(text):
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError("At least one factor must be provided.")
    return values


def _split_fixed_eval(df, random_seed):
    train_pool_df = pd.DataFrame()
    test_df = pd.DataFrame()
    val_df = pd.DataFrame()

    for class_name in df.columns[1:]:
        class_df = df[df[class_name] == 1]
        class_train_df, temp_df = train_test_split(
            class_df, test_size=0.3, random_state=random_seed
        )
        class_test_df, class_val_df = train_test_split(
            temp_df, test_size=1 / 3, random_state=random_seed
        )
        train_pool_df = pd.concat([train_pool_df, class_train_df])
        test_df = pd.concat([test_df, class_test_df])
        val_df = pd.concat([val_df, class_val_df])

    train_pool_df = train_pool_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    return train_pool_df, test_df, val_df


def _construct_lt_train(train_pool_df, imbalance_factor, random_seed):
    names = train_pool_df.columns[1:]
    counts = np.sum(train_pool_df.iloc[:, 1:].values, axis=0)
    idx = sorted(range(len(counts)), key=lambda k: counts[k], reverse=True)
    counts_sorted = sorted(counts, reverse=True)
    counts_lt = np.zeros_like(counts_sorted, dtype=int)
    names_lt = names[idx]

    mu = (counts_sorted[0] / (counts_sorted[-1] * imbalance_factor)) ** (1 / (len(counts) - 1))
    for i in range(len(counts_sorted)):
        counts_lt[i] = int(np.ceil(counts_sorted[i] * (mu ** i)))

    train_df = pd.DataFrame()
    for class_name, target_count in zip(names_lt, counts_lt):
        class_df = train_pool_df[train_pool_df[class_name] == 1]
        target_count = min(int(target_count), len(class_df))
        class_df = class_df.sample(n=target_count, random_state=random_seed)
        train_df = pd.concat([train_df, class_df])

    train_df = train_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    return train_df


def build_isic2019lt_factor_splits(data_root, output_root, random_seed, factors):
    csv = pd.read_csv(os.path.join(data_root, "ISIC_2019_Training_GroundTruth.csv"))
    csv = csv.iloc[:, :-1]

    train_pool_df, test_df, val_df = _split_fixed_eval(csv, random_seed)
    check_leakage(train_pool_df, test_df, val_df)

    split_dir = os.path.join(output_root, f"shared_eval_seed{int(random_seed)}")
    os.makedirs(split_dir, exist_ok=True)

    test_path = os.path.join(split_dir, "testing.csv")
    val_path = os.path.join(split_dir, "validation.csv")
    test_df.to_csv(test_path, index=False)
    val_df.to_csv(val_path, index=False)

    train_paths = []
    for factor in factors:
        train_df = _construct_lt_train(train_pool_df, int(factor), random_seed)
        train_path = os.path.join(split_dir, f"training_if{int(factor)}.csv")
        train_df.to_csv(train_path, index=False)
        train_paths.append(train_path)

    print(f"Output dir: {split_dir}")
    for train_path in train_paths:
        print(f"train: {train_path}")
    print(f"val: {val_path}")
    print(f"test: {test_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data_root/ISIC2019LT/ISIC_2019_Training_Input",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./split/ISIC2019LT",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--factors", type=str, default="100,200,500")
    args = parser.parse_args()

    build_isic2019lt_factor_splits(
        data_root=args.data_root,
        output_root=args.output_root,
        random_seed=args.seed,
        factors=_parse_factors(args.factors),
    )


if __name__ == "__main__":
    main()
