from typing import cast 
import random, functools
import click, wandb
import datasets, transformers
import glossing

import utils

@click.group()
def cli():
    pass


@cli.command()
@click.option('--model_type',
              type=click.Choice(['baseline', 'aug_m1', 'aug_m2']))
@click.option('--aug_mode', type=click.Choice(['mixed', 'curriculum']), default='mixed')
@click.option("--seed", help="Random seed", type=int, default=0)
@click.option("--epochs", help="Max # epochs", type=int, default=200)
@click.option("--project", type=str, default='morpheme-hallucination-igt')
def train(model_type: str, aug_mode: str, seed: int, epochs: int, project: str):

    BATCH_SIZE = 64
    wandb.init(project=project, entity="michael-ginn", name=model_type, config={
        "random-seed": seed,
        "experimental_run": model_type,
        "training_schedule": aug_mode,
        "epochs": epochs,
    })
    random.seed(seed)

    dataset = cast(datasets.DatasetDict, datasets.load_dataset('lecslab/usp-igt'))

    # Add in hallucinated data as needed
    def load_aug_data(path: str) -> datasets.Dataset:
        aug_data = glossing.load_igt_file(path)
        return datasets.Dataset.from_list([ex.__dict__() for ex in aug_data])
    if model_type != 'baseline':
        if model_type == 'aug_m1':
            aug_dataset = load_aug_data('../data/hallucinated/Method 1')
        elif model_type == 'aug_m2':
            aug_dataset = load_aug_data('../data/hallucinated/method2.txt')
        else:
            raise Exception('Invalid choice!')

        if aug_mode == 'mixed':
            dataset['train'] = datasets.concatenate_datasets([dataset['train'], aug_dataset])
            dataset['train'] = dataset['train'].shuffle()
        elif aug_mode == 'curriculum':
            dataset['aug_train'] = aug_dataset

    # Preprocess dataset
    tokenizer = transformers.ByT5Tokenizer.from_pretrained("google/byt5-base", use_fast=False)
    tokenizer = cast(transformers.ByT5Tokenizer, tokenizer)
    dataset = dataset.map(utils.create_byt5_prompt)
    dataset = dataset.map(functools.partial(utils.tokenize, 
                                            tokenizer=tokenizer, 
                                            max_length=tokenizer.model_max_length),
                          batched=True)
    
    # Create the model
    model = transformers.T5ForConditionalGeneration.from_pretrained("google/byt5-small")
    model = cast(transformers.T5ForConditionalGeneration, model)

    def preprocess_logits_for_metrics(logits, _):
        return logits.argmax(dim=2)

    args = transformers.Seq2SeqTrainingArguments(
        output_dir=f"../finetune-training-checkpoints",
        evaluation_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=3,
        save_strategy="epoch",
        save_total_limit=3,
        num_train_epochs=epochs,
        load_best_model_at_end=True,
        predict_with_generate=True,
        generation_max_length=1024,
        fp16=True,
        logging_strategy='epoch',
        report_to='wandb',
        # dataloader_num_workers=2,
    )

    trainer = transformers.Seq2SeqTrainer(
        model,
        args,
        tokenizer=tokenizer,
        train_dataset=dataset["train"], # type: ignore
        eval_dataset=dataset["valid"], # type: ignore
        compute_metrics=utils.compute_metrics(tokenizer=tokenizer),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[
            utils.LogCallback(),
            utils.DelayedEarlyStoppingCallback(early_stopping_patience=3)
        ],
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer=tokenizer, 
                                                          model=model,
                                                          label_pad_token_id=tokenizer.pad_token_id or -100)
    )

    trainer.train()
    trainer.save_model(f'../models/{model_type}')

    test_eval = trainer.evaluate(dataset['test']) # type: ignore
    test_eval = {k.replace('eval', 'test'): test_eval[k] for k in test_eval}
    wandb.log(test_eval)


if __name__ == "__main__":
    cli()
