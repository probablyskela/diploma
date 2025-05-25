import collections
import pathlib
import typing

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

from source import config

Splits = tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]


def explode_locc(locc: str) -> list[str]:  # (locc="AB123") => ["A", "AB", "AB123"]
    classes = []
    for i, c in enumerate(locc):
        if c.isalpha():
            classes.append(locc[: i + 1])
        else:
            classes.append(locc)
            break
    return classes


def explode_multiple_locc(locc: str) -> list[str]:
    return [c for cls in locc.split(";") for c in explode_locc(cls)]


def split_locc(locc: str) -> list[str]:
    return [cls for cls in locc.split(";")]


def preprocess_dataset(path: pathlib.Path, min_length: int = 10000) -> pd.DataFrame:
    def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out non-text entries and drop not needed columns, drop entries with missing LoCC classes and Etext Number.
        """

        # Filter Type column
        df = df[df["Type"] == "Text"]

        # Only select English books
        df = df[df["Language"] == "en"]

        # Drop not needed columns
        required_columns = ["Etext Number", "Title", "LoCC"]
        df = df[required_columns]

        # Drop rows with missing LoCC classes
        columns = ["LoCC", "Etext Number"]
        for column in columns:
            df = df[df[column].notna()]

        # Only select books which text is available
        missing = []
        for filename in df["Etext Number"].unique():
            filepath = path.joinpath(f"books/{filename}.txt")

            if not filepath.exists():
                missing.append(filename)
                continue

            df.loc[df["Etext Number"] == filename, "Length"] = filepath.stat().st_size

        df = df[~df["Etext Number"].isin(missing)]
        df = df[
            (
                df["Length"]
                >= max(min_length, df["Length"].mean() - 3 * df["Length"].std())
            )
            & (df["Length"] <= df["Length"].mean() + 3 * df["Length"].std())
        ]

        return df

    return preprocess_dataset


def explode_multiple_locc_classes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explode entries with multiple LoCC classes into multiple rows.
    """

    df["LoCC"] = df["LoCC"].apply(lambda x: [s.strip() for s in x.split(";")])
    df = df.explode(["LoCC"])
    return df


def merge_classes(min_samples: int) -> typing.Callable[[pd.DataFrame], pd.DataFrame]:
    """
    Merge classes with less than min_samples entries into the root class.
    """

    def merge_classes(df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge classes with less than min_samples entries into the root class.
        """

        df["LoCC"] = df["LoCC"].apply(
            lambda x: x if x[0] not in ["E", "F"] else f"DI{x[1:]}"
        )

        def explode_locc(locc: str):  # (locc="AB123") => ["A", "AB", "AB123"]
            classes = []
            for i, c in enumerate(locc):
                if c.isalpha():
                    classes.append(locc[: i + 1])
                else:
                    classes.append(locc)
                    break
            return classes

        classes = {}
        for cls in df["LoCC"].unique():
            p = explode_locc(cls)
            curr = classes
            for k in p:
                if k not in curr:
                    curr[k] = {}
                curr = curr[k]

        totals = collections.defaultdict(int)
        ld = df["LoCC"].value_counts().to_dict()

        def parse_dict(key, value):
            if len(value) == 0:
                totals[key] = ld[key]
                return
            for k, v in value.items():
                if k not in totals:
                    parse_dict(k, v)
                totals[key] += totals[k]

        for key, value in classes.items():
            parse_dict(key, value)

        to_merge = {}

        def merge_classes(key, value, target):
            for k, v in value.items():
                merge_classes(k, v, key if totals[key] >= min_samples else target)
            if totals[key] < min_samples:
                to_merge[key] = target

        for key, value in classes.items():
            merge_classes(key, value, key)

        df["LoCC"] = df["LoCC"].apply(lambda x: to_merge.get(x, x))

        to_merge = {}

        for key, value in df["LoCC"].value_counts().to_dict().items():
            if value < min_samples:
                to_merge[key] = "O"

        df["LoCC"] = df["LoCC"].apply(lambda x: to_merge.get(x, x))

        return df

    return merge_classes


def undersample_classes(
    max_samples: int, random_state: int = 25
) -> typing.Callable[[pd.DataFrame], pd.DataFrame]:
    """
    Undersample classes to have at most max_samples entries for each class.
    """

    def undersample_classes(df: pd.DataFrame) -> pd.DataFrame:
        """
        Undersample classes to have at most max_samples entries for each class.
        """

        X_full, y_full = df.drop(columns=["LoCC"]), df["LoCC"]

        undersample_strategy = {
            locc_class: max_samples
            for locc_class, count in y_full.value_counts().items()
            if count > max_samples
        }

        if len(undersample_strategy) == 0:
            return df

        rus = RandomUnderSampler(
            sampling_strategy=undersample_strategy, random_state=random_state
        )
        X_undersampled, y_undersampled = rus.fit_resample(X_full, y_full)
        df_undersampled = X_undersampled.copy()
        df_undersampled["LoCC"] = y_undersampled

        return df_undersampled

    return undersample_classes


def create_train_test_split(
    test_size: float = 0.3, random_state: int = 25
) -> typing.Callable[[pd.DataFrame], Splits]:
    """
    Create a train-test split of the dataset.
    """

    def create_train_test_split(df: pd.DataFrame) -> Splits:
        """
        Create a train-test split of the dataset.
        """

        X_full, y_full = df.drop(columns=["LoCC"]), df["LoCC"]
        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test

    return create_train_test_split


def aggregate_classes(splits: Splits) -> Splits:
    """
    Aggregate entries by Etext Number, merging multiple LoCC classes into a single entry.
    """

    X_train, X_test, y_train, y_test = splits

    X_train["LoCC"] = y_train
    X_test["LoCC"] = y_test

    def locc_agg(x):
        return ";".join(sorted(set(x) - {"O"}) or ["O"])

    X_train = X_train.groupby("Etext Number").agg({"LoCC": locc_agg}).reset_index()
    X_test = X_test.groupby("Etext Number").agg({"LoCC": locc_agg}).reset_index()

    y_train = X_train["LoCC"]
    y_test = X_test["LoCC"]

    X_train = X_train.drop(columns=["LoCC"])
    X_test = X_test.drop(columns=["LoCC"])

    return X_train, X_test, y_train, y_test


def oversample_classes(
    min_samples: int, random_state: int = 25
) -> typing.Callable[[Splits], Splits]:
    """
    Oversample classes to have at least min_samples entries for each class.
    """

    def oversample_classes(splits: Splits) -> Splits:
        """
        Oversample classes to have at least min_samples entries for each class.
        """

        X_train, X_test, y_train, y_test = splits

        oversample_strategy = {
            locc_class: min_samples
            for locc_class, count in y_train.value_counts().items()
            if ";" not in locc_class
            if count < min_samples
        }

        if len(oversample_strategy) == 0:
            return splits

        ros = RandomOverSampler(
            sampling_strategy=oversample_strategy, random_state=random_state
        )
        X_train_oversampled, y_train_oversampled = ros.fit_resample(X_train, y_train)

        return X_train_oversampled, X_test, y_train_oversampled, y_test

    return oversample_classes


def create_clean_dataset(verbose: bool = True) -> Splits:
    # path = pathlib.Path(kagglehub.dataset_download("lokeshparab/gutenberg-books-and-metadata-2025"))

    data = pd.read_csv(config.DATASET_DIR / "gutenberg_metadata.csv")

    steps = (
        preprocess_dataset(path=config.DATASET_DIR, min_length=10000),
        explode_multiple_locc_classes,
        merge_classes(min_samples=2000),
        undersample_classes(max_samples=6000),
    )

    for i, step in enumerate(steps):
        if verbose:
            print(f"Step {i+1}/{len(steps)}: {step.__doc__}")
        data = step(data)

        if verbose:
            if isinstance(data, tuple):
                print(
                    f"X_train: {data[0].shape}, X_test: {data[1].shape}, y_train: {data[2].shape}, y_test: {data[3].shape}"
                )
            else:
                print(f"Data: {data.shape}")
    return data


def create_splits(verbose: bool = True) -> Splits:
    # path = pathlib.Path(kagglehub.dataset_download("lokeshparab/gutenberg-books-and-metadata-2025"))

    data = pd.read_csv(config.DATASET_DIR / "gutenberg_metadata.csv")

    steps = (
        preprocess_dataset(path=config.DATASET_DIR, min_length=10000),
        explode_multiple_locc_classes,
        merge_classes(min_samples=2000),
        undersample_classes(max_samples=6000),
        create_train_test_split(),
        aggregate_classes,
        oversample_classes(min_samples=3000),
    )

    for i, step in enumerate(steps):
        if verbose:
            print(f"Step {i+1}/{len(steps)}: {step.__doc__}")
        data = step(data)

        if verbose:
            if isinstance(data, tuple):
                print(
                    f"X_train: {data[0].shape}, X_test: {data[1].shape}, y_train: {data[2].shape}, y_test: {data[3].shape}"
                )
            else:
                print(f"Data: {data.shape}")
    return data


def get_labels(splits: Splits) -> pd.Series:
    """
    Get the labels for the splits.
    """

    _, _, y_train, y_test = splits
    return pd.concat([y_train, y_test])


def get_label_to_index_mapping(
    splits: Splits,
) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """
    Get a mapping from labels to indices and vice versa.
    """

    _, _, y_train, y_test = splits
    labels = (
        pd.concat([y_train, y_test])
        .apply(lambda x: [c.strip() for c in x.split(";")])
        .explode()
        .unique()
        .tolist()
    )
    return (
        sorted(labels),
        {label: i for i, label in enumerate(sorted(labels))},
        {i: label for i, label in enumerate(sorted(labels))},
    )


if __name__ == "__main__":
    data = create_splits()
