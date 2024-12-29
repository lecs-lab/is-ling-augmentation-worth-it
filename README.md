# AugMorph

[Data augmentation](https://arxiv.org/abs/2105.03075), where novel examples are created from existing data, has been an effective strategy in NLP, particularly in low-resource settings. Augmentation typically relies on simple perturbations, such as concatenating, permuting, and replacing substrings. However, these transformations generally fail to preserve linguistic validity, resulting in augmentated examples which are similar to valid examples, but are themselves ungrammatical or strange.

The objective of this study is to utilize linguistic resources (a grammar book) and a linguist (who is not an expert in the language) to develop intentional augmentation strategies that leverage knowledge about the language to create novel, but still valid, examples. This approach is more expensive and time-consuming, but we hypothesize it can result in higher-quality augmented data, and thereby greater performance benefits.

## Usage

Set up environment:

```bash
python -m venv .venv # Please use Python >3.12
source .venv/bin/activate
pip install -r requirements.txt
```

> [!WARNING]
> If you are trying to install this on an Apple Silicon machine, `mlconjug3` won't install its dependencies correctly.
> You can fix this by running `pip install defusedxml scikit-learn `

Run experiments:

```bash
source .venv/bin/activate
python src/train.py --direction transc->transl --sample_train_size 50 --seed 0
```

## Task

We utilize the Mayan language **Uspanteko**, which has ~10k examples of **interlinear glossed text (IGT)**, a data format that combines a transcription, segmentation, morphological glossing, and translation. For example:

```
\t o sey xtok rixoqiil              # the transcription in Uspanteko
\m o' sea x-tok r-ixóqiil           # the transcription, segmented into morphemes
\p CONJ ADV COM-VT E3S-S            # part-of-speech tags for each morpheme
\g o sea COM-buscar E3S-esposa      # interlinear glosses for each morpheme
\l O sea busca esposa.              # Spanish translation
```

The richness of IGT enables us to evaluate several tasks with the same dataset. Specifically, we use:

| Task                | Inputs -> Outputs            |
| ------------------- | ---------------------------- |
| Morpheme tagging    | segmentation -> glosses      |
| Gloss generation    | transcription -> glosses     |
| Translation         | transcription -> translation |
| Reverse translation | translation -> transcription |

## Augmentation strategies

We consider several strategies. TBD

## Experimental variations

As augmentation has been shown to primarily benefit low-resource settings, we evaluate over several training set sizes. In each case, we sample some number of training examples, created augmented examples using _only those examples_, and use the same evaluation set.
