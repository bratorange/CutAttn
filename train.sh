python train.py \
  --name baseline_14_start_spectral_norm \
  --CUT_mode CUT \
  --dataroot dataset \
  --netG resnet_9blocks \
  --netD basic_spectral_norm \
  --n_epochs 15 \
  --n_epochs_decay 0 \
  --save_epoch_freq 1 \
  --update_html_freq 10 \
  # --ada_norm_layers 12 \
  # --continue_train \
