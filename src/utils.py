import transformers
from transformers import EvalPrediction
import numpy as np
from glossing import evaluate_glosses
from glossing.igt import gloss_string_to_word_glosses

class LogCallback(transformers.TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Print the logs or push them to your preferred logging framework
        print(logs)


class DelayedEarlyStoppingCallback(transformers.EarlyStoppingCallback):
    def __init__(self, *args, start_epoch=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_epoch = start_epoch

    def on_evaluate(self, args, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs):
        # Only start applying early stopping logic after start_epoch
        if state.epoch is not None and state.epoch >= self.start_epoch:
            super().on_evaluate(args, state, control, **kwargs)
        else:
            # Reset the patience if we're before the start_epoch
            self.patience = 0


def create_byt5_prompt(row, use_translation: bool = False):
    """Processing function for rows in the dataset, creates an input prompt from the fields in the row."""
    transcription = ' '.join((row['transcription']).split())
    glosses = ' '.join((row['glosses']).split())
    prompt = f"Provide the glosses for the following transcription in Uspanteko.\nTranscription: {transcription}"
    if row['translation'] is not None and use_translation:
        if len(row['translation'].strip()) > 0:
            translation = ' '.join((row['translation']).split())
            prompt += f"Translation in {row['metalang']}: {translation}\n"

    prompt += 'Glosses: '
    row['prompt'] = prompt
    row['glosses'] = glosses
    return row


def tokenize(batch, tokenizer, max_length: int):
    return tokenizer(
        batch["prompt"],
        text_target=batch.get("glosses", None),
        truncation=True,
        padding=False,
        max_length=max_length,
    )

def compute_metrics(tokenizer):
    """Creates the compute metrics function provided a tokenizer"""
    def _compute_metrics(eval_preds: EvalPrediction):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # Post-processing
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        for s in decoded_labels:
            if len(gloss_string_to_word_glosses(s)) == 0:
                print("BAD GLOSS", s)

        print("METRICS", decoded_preds, decoded_labels)
        return evaluate_glosses(decoded_preds, decoded_labels)

    return _compute_metrics
