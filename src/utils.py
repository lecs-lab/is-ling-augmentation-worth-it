from typing import Callable, Dict, List, Literal, Tuple

import numpy as np
import torchtext
import transformers
from sacrebleu import CHRF
from transformers import EvalPrediction


class LogCallback(transformers.TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Print the logs or push them to your preferred logging framework
        print(logs)


class DelayedEarlyStoppingCallback(transformers.EarlyStoppingCallback):
    def __init__(self, *args, start_epoch=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_epoch = start_epoch

    def on_evaluate(
        self,
        args,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs,
    ):
        # Only start applying early stopping logic after start_epoch
        if state.epoch is not None and state.epoch >= self.start_epoch:
            super().on_evaluate(args, state, control, **kwargs)
        else:
            # Reset the patience if we're before the start_epoch
            self.patience = 0


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


def create_mt_prompt(row, direction: Literal["usp->esp", "esp->usp"]):
    """Processing function for rows in the dataset, creates an input prompt from the fields in the row."""
    usp_transc = " ".join((row["transcription"]).split())
    esp_transc = " ".join((row["translation"]).split())
    if direction == "usp->esp":
        prompt = f"Translate into Spanish: {usp_transc}"
    elif direction == "esp->usp":
        prompt = f"Translate into Uspanteko: {esp_transc}"

    prompt += "Translation: "
    row["prompt"] = prompt
    if direction == "usp->esp":
        row["translation"] = esp_transc
    elif direction == "esp->usp":
        row["translation"] = usp_transc

    return row


def tokenize(batch, tokenizer, labels_key, max_length: int):
    return tokenizer(
        batch["prompt"],
        text_target=batch.get(labels_key, None),
        truncation=True,
        padding=False,
        max_length=max_length,
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
        print("PREDS", decoded_preds[:5])
        print("LABELS", decoded_labels[:5])
        return metrics_fn(decoded_preds, decoded_labels)

    return _compute_metrics


# Just chrF, not chrF++
chrf = CHRF(word_order=0)


def mt_metrics(preds: List[str], labels: List[str]) -> Dict:
    """Computes the BLEU score (after whitespace tokenization) and chrF"""
    tokenized_preds = [pred.split() for pred in preds]
    tokenized_labels = [[label.split()] for label in labels]
    bleu_score = torchtext.data.metrics.bleu_score(tokenized_preds, tokenized_labels)

    print(preds[:10], [[label] for label in labels][:10])
    chrF_score = chrf.corpus_score(preds, [labels]).score

    return {"BLEU": bleu_score, "chrF": chrF_score}
