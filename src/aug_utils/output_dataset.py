from datasets import Dataset

def output_dataset(output_list):
    ''' Outputs processed glosses as a dataset

    Args:
        output_list: the list of processed glosses. 

    Returns:
        dataset: A dataset object created from the list of processed glosses.
    '''
    dataset = Dataset.from_list(output_list)
    dataset = dataset.map(lambda batch: {**batch, 'segmentation': [None] * len(batch[next(iter(batch))])})
    return dataset

        
        