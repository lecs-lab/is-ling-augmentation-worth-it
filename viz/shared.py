import typing

import pandas as pd

method_colors = {
    "Ins-Noise": "#254653",  # blue
    "Del-Excl": "#F4A261",  # teal
    "Del-Any": "#F4A261",  # light orange
    "Ins-Conj": "#254653",  # light blue
    "Upd-TAM": "#E76F51",  # dark orange
    "Dup": "#E9C46A",  # yellow
    "Ins-Intj": "#254653",  # light blue
    "Perm": "#bbbbbb",  # gray
}

linguistic_strategies = ["Ins-Conj", "Ins-Intj", "Del-Excl", "Upd-TAM", "Perm"]
non_linguistic_strategies = ["Ins-Noise", "Del-Any", "Dup"]

method_dashes = {
    **{method: "" for method in linguistic_strategies},
    **{method: (2, 2) for method in non_linguistic_strategies},
}


def create_filtered_dataframe(csv_file) -> pd.DataFrame:
    """Creates a filtered dataframe from the CSV output of the experiment runs.

    Args:
        csv_file: The CSV file to process. Must include relative path if the CSV is not stored in the data-viz directory

    Returns:
        filtered_df: A dataframe with only the relevant columns for analysis.
    """
    df = pd.read_csv(csv_file)

    # Exclude failed runs
    df = df[df["State"] == "finished"]

    filtered_df = df.filter(
        [
            "Name",
            "aug_run_delete_w_exclusions",
            "aug_run_insert_interjection",
            "aug_run_random_delete",
            "aug_run_random_duplicate",
            "aug_run_random_insert_conj",
            "aug_run_random_insert_noise",
            "aug_run_sentence_permutations",
            "aug_run_tam_update",
            "direction",
            "random-seed",
            "training_size",
            "eval/BLEU",
            "eval/chrF",
            "eval/loss",
            "test/BLEU",
            "test/chrF",
            "test/loss",
            "train/loss",
        ]
    ).copy()

    filtered_df["Method"] = ""

    for column in filtered_df:
        filtered_df[column] = filtered_df[column].fillna(  # type:ignore
            0
        )  # Fill any empty cells with 0

    for column in filtered_df:
        if column.startswith("aug"):
            filtered_df[column] = (
                filtered_df[column].astype(bool).astype(int)
            )  # Convert boolean T/F values to 1/0

    # Put categories in ascending order
    filtered_df["training_size"] = pd.Categorical(
        filtered_df["training_size"],
        ordered=True,
        categories=["100", "500", "1000", "5000", "full"],
    )

    return typing.cast(pd.DataFrame, filtered_df)


def method_names(df) -> pd.DataFrame:
    """Creates a column that lists all of the methods used in each row.

    Args:
        df: The dataframe to update

    Returns:
        df: The updated dataframe
    """

    for index, row in df.iterrows():
        methods = []
        for column in df:
            if (
                column == "aug_run_delete_w_exclusions"
                and row["aug_run_delete_w_exclusions"] == 1
            ):
                methods.append("Del-Excl")
            elif (
                column == "aug_run_random_delete" and row["aug_run_random_delete"] == 1
            ):
                methods.append("Del-Any")
            elif (
                column == "aug_run_insert_interjection"
                and row["aug_run_insert_interjection"] == 1
            ):
                methods.append("Ins-Intj")
            elif (
                column == "aug_run_random_duplicate"
                and row["aug_run_random_duplicate"] == 1
            ):
                methods.append("Dup")
            elif (
                column == "aug_run_random_insert_conj"
                and row["aug_run_random_insert_conj"] == 1
            ):
                methods.append("Ins-Conj")
            elif (
                column == "aug_run_random_insert_noise"
                and row["aug_run_random_insert_noise"] == 1
            ):
                methods.append("Ins-Noise")
            elif (
                column == "aug_run_sentence_permutations"
                and row["aug_run_sentence_permutations"] == 1
            ):
                methods.append("Perm")
            elif column == "aug_run_tam_update" and row["aug_run_tam_update"] == 1:
                methods.append("Upd-TAM")
        if not methods:
            methods.append("Baseline")
        df["Method"][index] = " + ".join(methods)
    return df
