"""Maps improperly named aug keys to use underscores"""

import wandb
from tqdm import tqdm

total_seconds = 0

for project in ["augmorph/augmorph-usp-transl-transc", "augmorph/augmorph-usp-transc-transl", "augmorph/augmorph-usp-transc-gloss"]:
    runs = wandb.Api().runs(path=project)
    for r in runs:
        total_seconds += r.summary["_runtime"]

print(total_seconds)
