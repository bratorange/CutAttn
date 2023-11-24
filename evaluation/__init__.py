import re
from pathlib import Path


def get_epochs(experiments, args):
    experiment = experiments[args.experiment_id]
    epochs = [re.sub(r"([0-9]*|latest)_.*", r"\1", checkpoint.name) for checkpoint in
              (Path("checkpoints") / experiment.kvs["name"]).glob("*net_G.pth")]
    epochs.sort(key=lambda x: (int(x) if x.isdigit() else float('inf'), x))
    return experiment, epochs