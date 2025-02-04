"""Maps improperly named aug keys to use underscores"""

import wandb
from tqdm import tqdm

project = "augmorph/augmorph-usp-transl-transc"
runs = wandb.Api().runs(path=project)

mapped = 0
for run in tqdm(runs):
    config = run.config
    if "aug" in run.config:
        mapped += 1
        for key in run.config["aug"]:
            run.config["aug_" + key] = run.config["aug"][key]
        del run.config["aug"]

    for key in run.config:
        if key.startswith("aug."):
            mapped += 1
            run.config[key.replace("aug.", "aug_")] = run.config[key]
            del run.config[key]

    run.update()

print(f"Successfully mapped {mapped} runs")
