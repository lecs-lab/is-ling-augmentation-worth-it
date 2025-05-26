import pandas as pd


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
                methods.append("Del")
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
