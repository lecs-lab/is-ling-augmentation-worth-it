import functools
import os
import random
from typing import List, Literal, Optional, cast
from tqdm import tqdm


import datasets
import click
import glossing
import torch
from torch.utils.data import DataLoader
import transformers
from transformers.models.t5.modeling_t5 import Seq2SeqLMOutput

from data_handling import create_dataset
import utils
import wandb

os.environ["WANDB_LOG_MODEL"] = "end"
os.environ["NEPTUNE_PROJECT"] = "lecslab/aug-ling"


@click.group()
def cli():
    pass


@cli.command()
@click.option("--model_type", type=click.Choice(["baseline", "aug_m1", "aug_m2"]))
@click.option("--direction", type=click.Choice(["usp->esp", "esp->usp"]))
@click.option("--sample_train_size", type=int, default=None)
@click.option("--seed", help="Random seed", type=int, default=0)
@click.option("--epochs", help="Max # epochs", type=int, default=250)
def train(
    model_type: utils.AUGMENTATION_TYPE,
    direction: Literal["usp->esp", "esp->usp"],
    sample_train_size: Optional[int],
    seed: int,
    epochs: int,
):
    if direction not in ["usp->esp", "esp->usp"]:
        raise ValueError("Must be one of 'usp->esp' | 'esp->usp'")

    project = f"morpheme-hallucination-mt-{direction}"

    BATCH_SIZE = 64
    AUG_STEPS = 500
    TRAIN_STEPS = 1000

    wandb.init(
        project=project,
        entity="lecslab",
        config={
            "random-seed": seed,
            "experimental_run": model_type,
            "training_schedule": "curriculum",
            "epochs": epochs,
            "training_size": sample_train_size or "full",
            "direction": direction,
        }
    )
    random.seed(seed)

    dataset = create_dataset(model_type=model_type, sample_train_size=sample_train_size, seed=seed)

    # Preprocess dataset
    model_key = "google/byt5-small"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_key)
    dataset = dataset.map(
        functools.partial(utils.create_mt_prompt, direction=direction)
    )
    dataset = dataset.map(
        functools.partial(
            utils.tokenize,
            tokenizer=tokenizer,
            labels_key="translation",
            max_length=tokenizer.model_max_length,
        ),
        batched=True,
        remove_columns=['transcription', 'translation', 'segmentation', 'glosses', 'pos_glosses', 'prompt']
    )

    # Create the model
    model = transformers.T5ForConditionalGeneration.from_pretrained(model_key)
    model = cast(transformers.T5ForConditionalGeneration, model)

    print(
        f"Found {model.num_parameters()} parameters. Training with {len(dataset['train'])} examples."
    )

    # Collation
    collator=transformers.DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                model=model,
                label_pad_token_id=tokenizer.pad_token_id or -100,
            )
    train_dataloader = DataLoader(dataset['train'], BATCH_SIZE, collate_fn=collator)
    aug_dataloader = train_dataloader if model_type == "baseline" else DataLoader(dataset['aug_train'], BATCH_SIZE, collate_fn=collator)
    eval_dataloader = DataLoader(dataset['eval'], BATCH_SIZE, collate_fn=collator)
    test_dataloader = DataLoader(dataset['test'], BATCH_SIZE, collate_fn=collator)

    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.5)
    stage: Literal["aug", "train"] = "aug"

    progress = tqdm(total=AUG_STEPS + TRAIN_STEPS)
    total_steps = 0
    epoch = 0
    while total_steps < AUG_STEPS + TRAIN_STEPS:
        model.train()
        train_loss = 0
        train_epoch_steps = 0 # Track the number of steps for the current epoch

        for batch in aug_dataloader if stage == "aug" else train_dataloader:
            optimizer.zero_grad()
            loss = model(batch['input_ids'], labels=batch['labels']).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.detach().item()

            # Count step, and switch mode if needed
            total_steps += 1
            train_epoch_steps += 1
            progress.update(1)
            if total_steps >= AUG_STEPS and stage == "aug":
                stage = "train"
                break
            if total_steps >= AUG_STEPS + TRAIN_STEPS:
                break


        # Eval
        eval_loss = 0
        model.eval()
        for batch in eval_dataloader:
            out = cast(Seq2SeqLMOutput, model.forward(batch['input_ids'], labels=batch['label_ids']))
            eval_loss += out.loss.detach().item()

        print(f"Epoch {epoch}\tLoss: {train_loss / train_epoch_steps}\tEval loss: {eval_loss / len(eval_dataloader)}")

        wandb.log({
            "train/loss": train_loss / train_epoch_steps,
            "eval/loss": eval_loss / len(eval_dataloader),
            "train/stage": stage,
        })

        epoch += 1


    # for key in training_keys:
    #     args = transformers.Seq2SeqTrainingArguments(
    #         output_dir=f"/scratch/alpine/migi8081/augmorph/{wandb.run.name}-checkpoints",
    #         evaluation_strategy="epoch",
    #         per_device_train_batch_size=BATCH_SIZE,
    #         per_device_eval_batch_size=BATCH_SIZE,
    #         gradient_accumulation_steps=1,
    #         save_strategy="epoch",
    #         save_total_limit=3,
    #         # num_train_epochs=epochs,
    #         learning_rate=0.0001,
    #         max_steps=4000 if key == "train" else 2000,  # Less steps for initial phase
    #         weight_decay=0.5,
    #         load_best_model_at_end=False,
    #         # metric_for_best_model="bleu_score",
    #         predict_with_generate=True,
    #         generation_max_length=1024,
    #         # fp16=True,
    #         logging_strategy="epoch",
    #         report_to=["wandb", "neptune"],
    #         save_safetensors=False,
    #         # dataloader_num_workers=2,
    #         log_on_each_node=False,
    #     )

    #     trainer = transformers.Seq2SeqTrainer(
    #         model,
    #         args,
    #         tokenizer=tokenizer,
    #         train_dataset=dataset[key],  # type: ignore
    #         eval_dataset=dataset["eval"],  # type: ignore
    #         compute_metrics=utils.compute_metrics(
    #             tokenizer=tokenizer, metrics_fn=utils.mt_metrics
    #         ),
    #         data_collator=transformers.DataCollatorForSeq2Seq(
    #             tokenizer=tokenizer,
    #             model=model,
    #             label_pad_token_id=tokenizer.pad_token_id or -100,
    #         ),
    #     )

    #     trainer.train()

    # trainer.save_model(f"../models/{model_type}")

    # Testing
    test_preds = trainer.predict(dataset["test"])  # type: ignore
    test_eval = test_preds.metrics
    test_eval = {k.replace("eval", "test"): test_eval[k] for k in test_eval}  # type: ignore
    wandb.log(test_eval)

    # Decode preds and log to wandb
    predictions, labels = utils.decode(
        tokenizer, test_preds.predictions, test_preds.label_ids
    )
    preds_table = wandb.Table(
        columns=["predicted", "label"],
        data=[[p, lab] for p, lab in zip(predictions, cast(List[str], labels))],
    )
    wandb.log({"test_predictions": preds_table})


if __name__ == "__main__":
    cli()
