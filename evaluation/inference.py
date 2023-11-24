import copy
import os

from evaluation import get_epochs


def test_all(experiments, args):
    experiment, epochs = get_epochs(experiments, args)
    for epoch in epochs:
        command = "python test.py " + str(
            copy.deepcopy(experiment).set(epoch=epoch, num_test=args.num_test).remove('continue_train', "lr",
                                                                                      "pretrained_name", ))
        print(f"Infering epoch {epoch}...")
        print(command)
        if not args.dry:
            if os.system(command) != 0:
                exit()


def test(experiments, args):
    command = "python test.py " + str(
        experiments[args.experiment_id].set(epoch=args.epoch, num_test=args.num_test).remove('continue_train', "lr",
                                                                                             "pretrained_name", ))
    print(command)
    if not args.dry:
        os.system(command)
