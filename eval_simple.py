import torch
from torch.utils.data import DataLoader
from piq import FID

from data import create_dataset
from options.test_options import TestOptions

opt = TestOptions().parse()
dataset = create_dataset(opt)

exit()

first_dl, second_dl = DataLoader(), DataLoader()
fid_metric = FID()
first_feats = fid_metric.compute_feats(first_dl)
second_feats = fid_metric.compute_feats(second_dl)
fid: torch.Tensor = fid_metric(first_feats, second_feats)