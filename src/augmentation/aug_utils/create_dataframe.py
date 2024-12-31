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

    df = df.drop(columns='segmentation', errors='ignore')    # Drop the column with the segmented data
    df['transcription'] = df['transcription'].astype(str)
    df['transcription'] = df['transcription'].str.replace(',', '').str.replace('.', '').str.replace('?', '')

    if 'pos_glosses' in df.columns:
        df['pos_glosses'] = df['pos_glosses'].astype(str)

    df['glosses'] = df['glosses'].astype(str)
    df['glosses'] = df['glosses'].str.replace(',', '').str.replace('.', '').str.replace('?', '')

    df['translation'] = df['translation'].astype(str)
    df['translation'] = df['translation'].str.replace(',', '').str.replace('.', '').str.replace('?', '')

    if fraction != 1:
        df = df.sample(frac=fraction, replace = False, random_state= 42)

    return df
