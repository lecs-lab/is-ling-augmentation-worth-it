import pandas as pd
from datasets import Dataset


def create_dataframe(dataset: Dataset, fraction: float) -> pd.DataFrame:
    ''' Creates a dataframe from a specified sample of the data.

    Args:
        dataset (Dataset)

    Returns:
        Dataset: A dataframe of IGT glosses. Each row contains one IGT and each of its lines is stored in its respective column.

    '''
    df = pd.DataFrame(dataset)

    segmented = df[df['is_segmented'] == 'no']
    arapaho_df = segmented[segmented['glottocode'] == 'arap1274']
    arapaho_df.drop(['glottocode', 'id', 'source', 'metalang_glottocode',
                     'is_segmented', 'language', 'metalang'], axis=1, inplace=True)

    if fraction != 1:
        arapaho_df = arapaho_df.sample(
            frac=fraction, replace=False, random_state=42)

    return arapaho_df
