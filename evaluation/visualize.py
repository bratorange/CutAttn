from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

name = "resnet_atn_23_from_imagenet_multiple_atn_spectral_norm"

def get_eval_file(name):
    return Path("../results") / name / "scores.npz"

def visualize(name):
    loaded_data = np.load(get_eval_file(name))
    metric_names = loaded_data['metric_names']
    scores = loaded_data['scores']
    for score_name, score in zip(metric_names, scores):
        plt.plot(score)
    plt.show()

if __name__ == '__main__':
    visualize(name)