"""
data_loader.py — Load the HC3 dataset, clean, split, and save to CSV.

This module downloads the Hello-SimpleAI/HC3 dataset from HuggingFace,
flattens human and ChatGPT answers into a single text+label dataframe,
cleans the data, and produces train/val/test splits saved as CSVs.
"""

import os
import yaml
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Project root is the parent of the directory containing this file
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_config(config_path: str = None) -> dict:
    """Load the YAML configuration file.

    Args:
        config_path: Path to config.yaml. Defaults to PROJECT_ROOT/config.yaml.

    Returns:
        A dictionary of configuration values.
    """
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_hc3(config: dict) -> object:
    """Download the HC3 dataset from HuggingFace.

    The HC3 dataset uses a legacy loading script that is no longer
    supported by modern versions of the ``datasets`` library.  Instead,
    this function loads the raw JSONL files directly from the Hub.

    Args:
        config: Configuration dictionary containing dataset_name and dataset_subset.

    Returns:
        A HuggingFace Dataset object (DatasetDict with a 'train' split).
    """
    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    subset = config.get("dataset_subset", "all")
    data_file = f"{subset}.jsonl"

    print(f"Downloading HC3 dataset: {config['dataset_name']} / {data_file} ...")
    dataset = load_dataset(
        "json",
        data_files=f"hf://datasets/{config['dataset_name']}/{data_file}",
        cache_dir=raw_dir,
    )
    print(f"Download complete. Splits available: {list(dataset.keys())}")
    return dataset


def flatten_dataset(dataset) -> pd.DataFrame:
    """Flatten the HC3 dataset into a single text+label dataframe.

    HC3 stores each row as a question with lists of human_answers and
    chatgpt_answers.  This function unpacks every individual answer into
    its own row labelled 0 (human) or 1 (AI-generated).

    Args:
        dataset: A HuggingFace Dataset (typically the 'train' split of HC3).

    Returns:
        A pandas DataFrame with columns ['text', 'label'].
    """
    rows = []

    for split_name in dataset.keys():
        split = dataset[split_name]
        for example in split:
            # Human answers — label 0
            for answer in example.get("human_answers", []):
                if answer and isinstance(answer, str) and answer.strip():
                    rows.append({"text": answer.strip(), "label": 0})

            # ChatGPT answers — label 1
            for answer in example.get("chatgpt_answers", []):
                if answer and isinstance(answer, str) and answer.strip():
                    rows.append({"text": answer.strip(), "label": 1})

    df = pd.DataFrame(rows)
    print(f"Flattened dataset: {len(df)} rows  |  "
          f"Human: {(df['label'] == 0).sum()}  |  AI: {(df['label'] == 1).sum()}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop empty, null, or duplicate entries from the dataframe.

    Args:
        df: Raw dataframe with 'text' and 'label' columns.

    Returns:
        Cleaned dataframe.
    """
    initial = len(df)
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.strip().astype(bool)]
    df = df.drop_duplicates(subset=["text"])
    df = df.reset_index(drop=True)
    print(f"Cleaned: {initial} -> {len(df)} rows  (removed {initial - len(df)})")
    return df


def split_and_save(df: pd.DataFrame, seed: int = 42) -> None:
    """Split the dataframe into train/val/test (80/10/10) and save as CSVs.

    Args:
        df: Cleaned dataframe with 'text' and 'label' columns.
        seed: Random seed for reproducibility.
    """
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # 80% train, 20% temp
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df["label"]
    )
    # Split temp 50/50 -> 10% val, 10% test
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=seed, stratify=temp_df["label"]
    )

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = os.path.join(processed_dir, f"{name}.csv")
        split_df.to_csv(path, index=False)
        print(f"Saved {name}: {len(split_df)} rows -> {path}")


def main():
    """Entry-point: download -> flatten -> clean -> split -> save."""
    config = load_config()
    dataset = download_hc3(config)
    df = flatten_dataset(dataset)
    df = clean_data(df)
    split_and_save(df, seed=config.get("seed", 42))
    print("\n[OK] Data preparation complete.")


if __name__ == "__main__":
    main()
