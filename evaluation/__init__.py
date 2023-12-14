import argparse
import copy
import re
from pathlib import Path

from typing import List, Tuple

from experiments.tmux_launcher import Options


def get_score_file(name):
    return Path("results") / name / "scores.npz"


def get_eval_file(name):
    return Path("results") / name / "eval.npz"


def get_experiment(experiments, args) -> (Options, list, str):
    experiment = experiments[args.experiment_id]
    epochs = [re.sub(r"([0-9]*).*_.*", r"\1", checkpoint.name) for checkpoint in
              (Path("checkpoints") / experiment.kvs["name"]).glob("*net_G.pth")]
    epochs = [int(epoch) for epoch in epochs if epoch.isdigit()]
    epochs.sort()
    name = experiment.kvs['name']
    return experiment, epochs, name


def get_experiments(experiments, args) -> List[Tuple[Options, list, str, argparse.Namespace]]:
    experiment_ids = [int(i) for i in args.experiment_id.split(',')]
    res = []
    for experiment_id in experiment_ids:
        args = copy.deepcopy(args)
        args.experiment_id = int(experiment_id)
        res.append(get_experiment(experiments, args) + (args,))
    return res
