import random
import click
import torch
import wandb
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    TrainerCallback,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    TrainerState,
    TrainerControl,
    XLMRobertaForTokenClassification,
    XLMRobertaTokenizerFast,
    DataCollatorForTokenClassification
)
from data_handling import split_line, create_vocab, prepare_dataset, load_data_file
from eval import compute_metrics

device = "cuda:0" if torch.cuda.is_available() else "mps"


class LogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Print the logs or push them to your preferred logging framework
        print(logs)


class DelayedEarlyStoppingCallback(EarlyStoppingCallback):
    def __init__(self, *args, start_epoch=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_epoch = start_epoch

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # Only start applying early stopping logic after start_epoch
        if state.epoch >= self.start_epoch:
            super().on_evaluate(args, state, control, **kwargs)
        else:
            # Reset the patience if we're before the start_epoch
            self.patience = 0


def load_aug_data(path):
    aug_data = load_data_file(path)

    aug_data_processed = []
    for row in aug_data:
        row.should_segment = False
        row_dict = row.__dict__()
        aug_data_processed.append({'transcription': row_dict['segmentation'],
                                    'glosses': row_dict['glosses'][0],
                                    'translation': row_dict['translation']})

    return Dataset.from_list(aug_data_processed)


@click.group()
def cli():
    pass


@cli.command()
@click.option('--model_type',
              type=click.Choice(['baseline', 'aug_m1', 'aug_m2']))
@click.option('--aug_mode', type=click.Choice(['mixed', 'curriculum']), default='mixed')
@click.option("--seed", help="Random seed", type=int, default=0)
@click.option("--epochs", help="Max # epochs", type=int, default=200)
@click.option("--project", type=str, default='morpheme-hallucination')
def train(model_type: str,
          aug_mode: str,
          seed: int,
          epochs: int,
          project: str):

    BATCH_SIZE = 64

    wandb.init(project=project, entity="michael-ginn", name=model_type, config={
        "random-seed": seed,
        "experimental_run": model_type,
        "training_schedule": aug_mode,
        "epochs": epochs,
    })

    random.seed(seed)

    # Load and process dataset
    dataset = load_dataset('lecslab/usp-igt')

    # Add in hallucinated data as needed
    if model_type != 'baseline':
        if model_type == 'aug_m1':
            aug_dataset = load_aug_data('../data/hallucinated/Method 1')
        elif model_type == 'aug_m2':
            aug_dataset = load_aug_data('../data/hallucinated/Method 2/method2.txt')
        else:
            raise Exception('Invalid choice!')

        if aug_mode == 'mixed':
            dataset['train'] = concatenate_datasets([dataset['train'], aug_dataset])
            dataset['train'] = dataset['train'].shuffle()
        elif aug_mode == 'curriculum':
            dataset['aug_train'] = aug_dataset

    def segment(row):
        row["morphemes"] = split_line(row["transcription"], prefix="usp_")
        row["glosses"] = split_line(row["glosses"], prefix=None)
        return row

    dataset = dataset.map(segment)
    train_vocab = create_vocab(dataset['train']['morphemes'], threshold=1)
    glosses = create_vocab(dataset['train']['glosses'], threshold=1, should_not_lower=True) + ['<unk>']
    tokenizer = XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-base', model_max_length=64)
    tokenizer.add_tokens(train_vocab)
    dataset = prepare_dataset(dataset, tokenizer, glosses)

    # Add the new tokens to the model
    model = XLMRobertaForTokenClassification.from_pretrained('xlm-roberta-base', num_labels=len(glosses))
    model.resize_token_embeddings(len(tokenizer))

    def preprocess_logits_for_metrics(logits, _):
        return logits.argmax(dim=2)

    args = TrainingArguments(
        output_dir=f"../finetune-training-checkpoints",
        evaluation_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=3,
        save_strategy="epoch",
        save_total_limit=3,
        num_train_epochs=epochs,
        load_best_model_at_end=True,
        logging_strategy='epoch',
        report_to='wandb'
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=dataset["train"] if dataset else None,
        eval_dataset=dataset["valid"] if dataset else None,
        compute_metrics=compute_metrics(glosses),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[
            LogCallback,
            DelayedEarlyStoppingCallback(early_stopping_patience=3)
        ],
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer)
    )

    trainer.train()

    trainer.save_model(f'../models/{model_type}')

    test_eval = trainer.evaluate(dataset['test'])
    test_eval = {k.replace('eval', 'test'): test_eval[k] for k in test_eval}
    wandb.log(test_eval)


if __name__ == "__main__":
    cli()
