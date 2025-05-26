# Is linguistically-motivated data augmentation worth it?

📈 [Results](https://wandb.ai/augmorph?shareProfileType=copy)

[Data augmentation](https://arxiv.org/abs/2105.03075), where novel examples are created from existing data, has been an effective strategy in NLP, particularly in low-resource settings. Augmentation typically relies on simple perturbations, such as concatenating, permuting, and replacing substrings. However, these transformations generally fail to preserve linguistic validity, resulting in augmentated examples which are similar to valid examples, but are themselves ungrammatical or strange. Other research uses linguistic knowledge to constrain the newly-created augmented examples to (hypothetically) grammatical instances. 

This study provides a **head-to-head comparison of linguistically-naive and linguistically-motivated data augmentation strategies**. We use a case study on two low-resource languages, Uspanteko and Arapaho, and study machine translation and interlinear gloss prediction. 

## Datsets 

Both IGT datasets are publicly available on Huggingface:
- [Uspanteko](https://huggingface.co/datasets/lecslab/usp-igt)
- [Arapaho](https://huggingface.co/datasets/lecslab/arp-igt)

## Usage

Set up environment:

```bash
python -m venv .venv # Please use Python >=3.12
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
| Gloss generation    | transcription -> glosses     |
| Translation         | transcription -> translation |
| Reverse translation | translation -> transcription |

## Augmentation strategies

We consider linguistically-motivated and linguistically-naive strategies for both languages. 

### Uspanteko
UPD-TAM: Updates the aspect marker of the verb \
INS-CONJ: Inserts a random conjunction at the start of the sentence  
INS-NOISE: Inserts a random word at the start of the sentence 
DEL: Randomly deletes a word by index \
DEL-EXCL: Randomly deletes a word by index, excluding verbs \
DUP: Randomly duplicates a word by index

### Arapaho

INS-INTJ: Inserts a random interjection at the start of the sentence \
INS-NOISE: Inserts a random word at the start of the sentence \
PERM: Produces up to 10 permutations of the original word order

## Experimental variations

As augmentation has been shown to primarily benefit low-resource settings, we evaluate over several training set sizes. In each case, we sample some number of training examples, created augmented examples using _only those examples_, and use the same evaluation set.
