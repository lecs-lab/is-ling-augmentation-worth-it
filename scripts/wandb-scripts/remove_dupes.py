from itertools import groupby
from operator import itemgetter

import wandb

project = "augmorph/augmorph-arp-transc-gloss-v2"
runs = wandb.Api().runs(path=project)

aug_keys = [k for k in runs[0].config.keys() if k.startswith("aug_")]
keys_getter = itemgetter("random-seed", "training_size", *aug_keys)  # type:ignore

dupes = 0
runs_to_delete = []
sorted_runs = sorted(runs, key=lambda r: tuple(str(x) for x in keys_getter(r.config)))
for key, grp in groupby(sorted_runs, lambda r: keys_getter(r.config)):
    grp = list(grp)
    if len(grp) > 1:
        dupes += len(grp) - 1

        # Keep the oldest run in the group
        grp = sorted(grp, key=lambda r: r.createdAt)
        runs_to_delete.extend(grp[1:])

print(f"Found {dupes} duplicate runs from {len(runs)} total")

for run in runs_to_delete:
    run.delete()

runs = wandb.Api().runs(path=project)
print(f"{len(runs)} runs remaining")
