from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

name = "resnet_atn_23_from_imagenet_multiple_atn_spectral_norm"

def get_eval_file(name):
    return Path("results") / name / "scores.npz"

def visualize(name):
    loaded_data = np.load(get_eval_file(name))
    metric_names = loaded_data['metric_names']
    scores = loaded_data['scores']
    epochs = loaded_data['epochs']

    markers = ['o', 'v', 's', '*']
    colors = ['red', 'blue', 'yellow', 'green']


    for i, (score_name, score, marker, color) in enumerate(zip(metric_names, scores, markers, colors)):
        if i == 0:
            fig, axes = plt.subplots()
            axes.set_xlabel("epoch")
        else:
            axes = axes.twinx()
        axes.plot(epochs, score, marker=marker, color=color)
        axes.set_ylabel(score_name)
    plt.title(name)
    plt.show()

if __name__ == '__main__':
    visualize(name)