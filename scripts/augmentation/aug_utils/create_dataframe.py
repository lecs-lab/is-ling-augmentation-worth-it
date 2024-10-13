import pandas as pd
from datasets import load_dataset


def create_dataframe():
    ''' Creates a dataframe from a specified sample of the data.

    Args:
        None

    Returns:
        df: A dataframe of IGT glosses. Each row contains one IGT and each of its lines is stored in its respective column.

    '''
    file_type = input("Enter one of the following: csv or parquet ")
    if file_type == "parquet":
        split = input("Enter data split. Format example: 'data/train-00000-of-00001.parquet'. Do NOT include single or double quotation marks. ")
        base_url = input("Enter base url. Format example: 'hf://datasets/lecslab/usp-igt/'. Do NOT include single or double quotation marks. ")
        df = pd.read_parquet(base_url + split)
    else:
        data_file = input("Enter file name. Include extension.")
        dataset = load_dataset(file_type, data_files=data_file)
        df = pd.DataFrame(dataset['train']) 

    df = df.drop(columns='segmentation')    # Drop the column with the segmented data
    df['transcription'] = df['transcription'].astype(str)
    df['transcription'] = df['transcription'].str.replace(',', '').str.replace('.', '').str.replace('?', '')

    df['pos_glosses'] = df['pos_glosses'].astype(str)

    df['glosses'] = df['glosses'].astype(str)
    df['glosses'] = df['glosses'].str.replace(',', '').str.replace('.', '').str.replace('?', '')

    df['translation'] = df['translation'].astype(str)
    df['translation'] = df['translation'].str.replace(',', '').str.replace('.', '').str.replace('?', '')

    fraction = float(input("What percent of the data would you like to use? Enter 1 if you want to use the entire dataset. "))
    if fraction == 1:
        pass
    else:
        df = df.sample(frac=fraction, replace = False, random_state= 42)
    return df
