from typing import Callable, Dict, List, Literal, Tuple

import numpy as np
from glossing.bleu import bleu_score
from sacrebleu import CHRF
from transformers import EvalPrediction, PreTrainedTokenizer

from free_word_chrf import free_word_chrf

AUGMENTATION_TYPE = Literal["baseline", "aug_m1", "aug_m2", "combo"]


def create_igt_prompt(row, use_translation: bool = False):
    """Processing function for rows in the dataset, creates an input prompt from the fields in the row."""
    transcription = " ".join((row["transcription"]).split())
    glosses = " ".join((row["glosses"]).split())
    prompt = f"Provide the glosses for the following transcription in Uspanteko.\nTranscription: {transcription}"
    if row["translation"] is not None and use_translation:
        if len(row["translation"].strip()) > 0:
            translation = " ".join((row["translation"]).split())
            prompt += f"Translation in {row['metalang']}: {translation}\n"

    prompt += "Glosses: "
    row["prompt"] = prompt
    row["glosses"] = glosses
    return row


def create_mt_prompt(
    row,
    direction: Literal[
        "transc->transl", "transl->transc", "transc->gloss", "transc->segment"
    ],
    language: Literal["usp", "arp"],
):
    """Processing function for rows in the dataset, creates an input prompt from the fields in the row."""
    if language == "usp":
        lang_name, metalang_name = "Uspanteko", "Spanish"
    elif language == "arp":
        lang_name, metalang_name = "Arapaho", "English"

    transc = " ".join((row["transcription"]).split())
    transl = " ".join((row["translation"]).split())
    if direction == "transc->transl":
        prompt = f"Translate into {metalang_name}: {transc}\nTranslation: "
        row["target"] = transl
    elif direction == "transl->transc":
        prompt = f"Translate into {lang_name}: {transl}\nTranslation: "
        row["target"] = transc
    elif direction == "transc->gloss":
        prompt = f"Output interlinear glosses for the following {lang_name}: {transc}"
        row["target"] = row["glosses"]
    elif direction == "transc->segment":
        prompt = f"Output a morphological segmentation for the following {lang_name}: {transc}"
        row["target"] = " ".join((row["segmentation"]).split())

    row["prompt"] = prompt
    return row


def tokenize(batch, tokenizer: PreTrainedTokenizer, labels_key):
    return tokenizer(
        batch["prompt"],
        text_target=batch.get(labels_key, None),
        padding=False,
    )


def decode(
    tokenizer,
    predictions: np.ndarray | Tuple[np.ndarray],
    labels: np.ndarray | Tuple[np.ndarray] | None,
) -> Tuple[List[str], List[str] | None]:
    """Decodes pred and label ids using the tokenizer"""
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    if isinstance(labels, tuple):
        labels = labels[0]

    preds = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_preds = [pred.strip() for pred in decoded_preds]

    decoded_labels = None
    if labels is not None:
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_labels = [label.strip() for label in decoded_labels]

    return decoded_preds, decoded_labels


def compute_metrics(tokenizer, metrics_fn: Callable[[List[str], List[str]], Dict]):
    """Creates the compute metrics function provided a tokenizer"""

    def _compute_metrics(eval_preds: EvalPrediction):
        preds, labels = eval_preds
        decoded_preds, decoded_labels = decode(tokenizer, preds, labels)

        if decoded_labels is None:
            raise ValueError("Need to have labels in `compute_metrics`!!")
        print("PREDS", decoded_preds[:10])
        print("LABELS", decoded_labels[:10])
        return metrics_fn(decoded_preds, decoded_labels)

    return _compute_metrics


# Just chrF, not chrF++
chrf = CHRF(word_order=0)


def mt_metrics(preds: List[str], labels: List[str]) -> Dict:
    """Computes the BLEU score (after whitespace tokenization) and chrF"""
    tokenized_preds = [pred.split() for pred in preds]
    tokenized_labels = [[label.split()] for label in labels]
    bleu = bleu_score(tokenized_preds, tokenized_labels)
    chrF_score = chrf.corpus_score(preds, [labels]).score
    free_word_chrF_score = free_word_chrf(preds, labels).score
    return {"BLEU": bleu, "chrF": chrF_score, "freeword_chrF": free_word_chrF_score}
