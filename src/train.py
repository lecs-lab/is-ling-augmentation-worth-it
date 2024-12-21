import functools
import random
import sys
from dataclasses import asdict
from typing import Literal, Optional, cast

import click
import torch
import transformers
from dataclass_click import dataclass_click
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.models.t5.modeling_t5 import Seq2SeqLMOutput

import utils
import wandb
from augmentation.aug_generation import AugmentationParameters
from data_handling import create_dataset

device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)


@click.command()
@click.option("--direction", type=click.Choice(["usp->esp", "esp->usp", "usp->gloss", "usp->segment"]))
@click.option("--sample_train_size", type=int, default=None)
@click.option("--seed", help="Random seed", type=int, default=0)
@click.option("--epochs", help="Max # epochs", type=int, default=250)
@dataclass_click(AugmentationParameters, kw_name="params")
def train(
    params: AugmentationParameters,
    direction: Literal["usp->esp", "esp->usp", "usp->gloss", "usp->segment"],
    sample_train_size: Optional[int],
    seed: int,
    epochs: int,
):
    if direction not in ["usp->esp", "esp->usp", "usp->gloss", "usp->segment"]:
        raise ValueError("Must be one of 'usp->esp' | 'esp->usp' | 'usp->gloss' | 'usp->segment'")

    project = f"augmorph-mt-{direction}"

    BATCH_SIZE = 32
    AUG_STEPS = 500
    TRAIN_STEPS = 1000

    config = {
        "random-seed": seed,
        "training_schedule": "curriculum",
        "epochs": epochs,
        "training_size": sample_train_size or "full",
        "direction": direction,
        "reset_optimizer_between_stages": True,
        "aug": asdict(params),
    }

    # Check if this run is a duplicate
    try:
        runs = wandb.Api().runs(
            path=f"augmorph/{project}",
            filters={f"config.{key}": value for key, value in config.items()},
        )
        if len(runs) > 0 and any(r._state == "finished" for r in runs):
            print("Skipping run, identical run already found!!", file=sys.stderr)
            return
    except:
        print("Project does not exist yet")

    wandb.init(entity="augmorph", project=project, config=config)
    random.seed(seed)

    dataset = create_dataset(
        augmentation_type="combo",
        params=params,
        sample_train_size=sample_train_size,
        seed=seed,
    )

    # Preprocess dataset
    model_key = "google/byt5-small"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_key)
    dataset = dataset.map(
        functools.partial(utils.create_mt_prompt, direction=direction)
    )
    dataset = dataset.map(
        lambda batch: utils.tokenize(
            batch,
            tokenizer=tokenizer,
            labels_key="target",
            max_length=tokenizer.model_max_length,
        ),
        batched=True,
        remove_columns=[
            "transcription",
            "translation",
            "segmentation",
            "glosses",
            "pos_glosses",
            "prompt",
            "target",
        ],
    )

    # Create the model
    model = transformers.T5ForConditionalGeneration.from_pretrained(model_key)
    model = cast(transformers.T5ForConditionalGeneration, model)
    model = model.to(device)  # type:ignore

    print(
        f"Found {model.num_parameters()} parameters. Training with {len(dataset['train'])} examples on {device}."
    )

    # Collation
    collator = transformers.DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=tokenizer.pad_token_id or -100,
    )
    train_dataloader = DataLoader(dataset["train"], BATCH_SIZE, collate_fn=collator)  # type:ignore
    aug_dataloader = DataLoader(dataset["aug_train"], BATCH_SIZE, collate_fn=collator)  # type:ignore
    eval_dataloader = DataLoader(dataset["eval"], BATCH_SIZE, collate_fn=collator)  # type:ignore

    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.5)
    stage: Literal["aug", "train"] = "aug"

    progress = tqdm(total=AUG_STEPS + TRAIN_STEPS, desc="Training")
    total_steps = 0
    epoch = 0
    while total_steps < AUG_STEPS + TRAIN_STEPS:
        model.train()
        train_loss = 0
        train_epoch_steps = 0  # Track the number of steps for the current epoch
        for batch in aug_dataloader if stage == "aug" else train_dataloader:
            optimizer.zero_grad()
            loss = model(
                batch["input_ids"].to(device), labels=batch["labels"].to(device)
            ).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.detach().item()

            # Count step, and switch mode if needed
            total_steps += 1
            train_epoch_steps += 1
            progress.update(1)
            if total_steps >= AUG_STEPS and stage == "aug":
                # Next stage! Reset optimizer
                stage = "train"
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=0.0001, weight_decay=0.5
                )
                break
            if total_steps >= AUG_STEPS + TRAIN_STEPS:
                break

            wandb.log(
                {
                    "train/loss": train_loss / train_epoch_steps,
                    "stage": 0 if stage == "aug" else 1,
                },
                step=total_steps,
            )

        eval_loss = 0
        model.eval()
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            out = cast(
                Seq2SeqLMOutput,
                model.forward(
                    batch["input_ids"].to(device), labels=batch["labels"].to(device)
                ),
            )
            eval_loss += out.loss.detach().item() # type: ignore

        print(
            f"Epoch {epoch}\tLoss: {train_loss / train_epoch_steps}\tEval loss: {eval_loss / len(eval_dataloader)}"
        )

        wandb.log({"eval/loss": eval_loss / len(eval_dataloader)}, step=total_steps)
        epoch += 1

    # Use a Trainer just for prediction
    args = transformers.Seq2SeqTrainingArguments(
        output_dir="..", #"/scratch/alpine/migi8081/augmorph/",
        predict_with_generate=True,
        generation_max_length=1024,
        report_to=None,
    )

    trainer = transformers.Seq2SeqTrainer(
        model,
        args=args,
        compute_metrics=utils.compute_metrics(
            tokenizer=tokenizer, metrics_fn=utils.mt_metrics
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=tokenizer.pad_token_id or -100,
        ),
    )

    # Testing
    print("Running last eval on eval set...")
    eval_metrics = trainer.evaluate(dataset["eval"]) # type: ignore
    wandb.log(eval_metrics)


    print("Running evaluation on test set...")
    test_preds = trainer.predict(dataset["test"])  # type: ignore
    test_eval = test_preds.metrics
    test_eval = {k.replace("eval/", ""): test_eval[k] for k in test_eval}  # type: ignore
    wandb.log(test_eval)

    # # Decode preds and log to wandb
    # predictions, labels = utils.decode(
    #     tokenizer, test_preds.predictions, test_preds.label_ids
    # )
    # preds_table = wandb.Table(
    #     columns=["predicted", "label"],
    #     data=[[p, lab] for p, lab in zip(predictions, cast(List[str], labels))],
    # )
    # wandb.log({"test_predictions": preds_table})


if __name__ == "__main__":
    train()
