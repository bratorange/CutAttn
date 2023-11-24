#!/usr/bin/env python
import argparse
import os

from evaluation import eval_simple
from evaluation.eval_simple import eval_all
from evaluation.inference import test_all, test
from experiments.tmux_launcher import Options

experiments = {
    2: Options(
        name="resnet_atn_02_continue_from_10",
        netG="resnet_atn",
        continue_train="",
    ),
    3: Options(
        name="resnet_atn_03_from_start_trans_at_16",
        netG="resnet_atn",
    ),
    4: Options(
        name="resnet_atn_04_start",
        netG="resnet_atn",
    ),
    5: Options(
        name="resnet_atn_05_from_5_trans_at_14",
        netG="resnet_atn",
        continue_train="",
        no_random_mask="",
    ),
    6: Options(
        name="resnet_atn_06_start_adain",
        netG="resnet_adain",
        no_random_mask="",
    ),
    7: Options(
        name="resnet_atn_07_from_05_adain",
        netG="resnet_adain",
        no_random_mask="",
        continue_train="",
    ),
    8: Options(
        name="resnet_atn_08_start_spectral_norm",
        netG="resnet_atn",
        netD="basic_spectral_norm",
    ),
    9: Options(
        name="resnet_atn_09_from_05_multiple_atn",
        netG="resnet_atn",
        continue_train="",
    ),
    10: Options(
        name="resnet_atn_10_from_05_spectral_norm",
        netG="resnet_atn",
        netD="basic_spectral_norm",
        continue_train="",
        ada_norm_layers="12",
    ),
    11: Options(
        name="resnet_atn_11_from_02_spectral_norm",
        netG="resnet_atn",
        netD="basic_spectral_norm",
        continue_train="",
        ada_norm_layers="12",
    ),
    12: Options(
        name="resnet_atn_12_start_multiple_atn_spectral_norm",
        netG="resnet_atn",
        netD="basic_spectral_norm",
        ada_norm_layers="12,13",
    ),
    13: Options(
        name="resnet_atn_13_from_02_adain",
        netG="resnet_adain",
        netD="basic",
        ada_norm_layers="12",
        continue_train="",
    ),
    14: Options(
        name="baseline_14_start_spectral_norm",
        netG="resnet_9blocks",
        netD="basic_spectral_norm",
    ),
    15: Options(
        name="resnet_atn_15_start_multiple_adain",
        netG="resnet_adain",
        netD="basic",
        ada_norm_layers="12,13,14",
    ),
    16: Options(
        name="resnet_atn_16_from_02_multiple_atn_spectral_norm",
        netG="resnet_atn",
        netD="basic_spectral_norm",
        ada_norm_layers="12,13",
        continue_train="",
        pretrained_name="baseline_14_start_spectral_norm",
        epoch=2,
    ),
    17: Options(
        name="resnet_atn_17_start_spectral_norm_low_lr",
        netG="resnet_atn",
        netD="basic_spectral_norm",
        lr=0.0001,
    ),
    18: Options(
        name="baseline_18_start_spectral_norm_fast_cut",
        netG="resnet_9blocks",
        netD="basic_spectral_norm",
        CUT="FastCUT",
    ),
    19: Options(
        name="resnet_atn_19_start_spectral_norm_low_lr_multiple_atn",
        netG="resnet_atn",
        netD="basic_spectral_norm",
        lr=0.00002,
        ada_norm_layers="12,13",
    ),
    20: Options(
        name="baseline_20_start_spectral_norm_fast_cut_less_nce",
        netG="resnet_9blocks",
        netD="basic_spectral_norm",
        CUT="FastCUT",
        lambda_NCE=1,
    ),
    21: Options(
        name="baseline_21_high_GAN",
        netG="resnet_9blocks",
        netD="basic_spectral_norm",
        lambda_NCE=.1,
    ),
    22: Options(
        name="pretrain_22",
        netG="resnet_9blocks",
        netD="basic_spectral_norm",
        lambda_NCE=0.00001,
        dataset_mode="pretrain",
        dataroot="imagenet",
    ),
    23: Options(
        name="resnet_atn_23_from_imagenet_multiple_atn_spectral_norm",
        netG="resnet_atn",
        netD="basic_spectral_norm",
        ada_norm_layers="12,13",
        continue_train="",
        pretrained_name="pretrain_22",
        epoch=15,
        lr=0.00002
    ),
}

experiments = {k: opt if "dataroot" in opt.kvs else opt.set(dataroot="dataset") for k, opt in experiments.items()}

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest='subcommand')

parser_train = subparsers.add_parser('train')
parser_train.add_argument('experiment_id', type=int)
parser_train.add_argument("--dry", action='store_true')

parser_test = subparsers.add_parser('test')
parser_test.add_argument('experiment_id', type=int)
parser_test.add_argument('epoch', default='latest')
parser_test.add_argument("--num_test", type=int, default=50)
parser_test.add_argument("--dry", action='store_true')

parser_list = subparsers.add_parser('list')

parser_eval = subparsers.add_parser('eval')
parser_eval.add_argument('experiment_id', type=int)
parser_eval.add_argument('epoch', default='latest')
parser_eval.add_argument('--batch_size', type=int, default=4)
parser_eval.add_argument("--dry", action='store_true')

parser_eval_all = subparsers.add_parser('eval_all')
parser_eval_all.add_argument('experiment_id', type=int)
parser_eval_all.add_argument('--batch_size', type=int, default=4)
parser_eval_all.add_argument("--dry", action='store_true')

parser_test_all = subparsers.add_parser('test_all')
parser_test_all.add_argument('experiment_id', type=int)
parser_test_all.add_argument("--num_test", type=int, default=50)
parser_test_all.add_argument("--dry", action='store_true')

args = parser.parse_args()
for k, v in sorted(vars(args).items()):
    print(f"{k}: {v}")

if args.subcommand == "train":
    command = "python train.py " + str(experiments[args.experiment_id]
                                       .set(n_epochs=15, n_epochs_decay=0, save_epoch_freq=1, update_html_freq=10))
    print(command)
    if not args.dry:
        os.system(command)
elif args.subcommand == "test":
    test(experiments, args)
elif args.subcommand == "eval":
    print("Calculating metrics...")
    eval_simple.eval(experiments[args.experiment_id].kvs['name'], args)
elif args.subcommand == "eval_all":
    eval_all(experiments, args)
elif args.subcommand == "list":
    for k, v in experiments.items():
        print(f"{k}: {v.kvs['name']}")
elif args.subcommand == "test_all":
    test_all(experiments, args)
else:
    print("Please provide a command")
    parser.print_help()
