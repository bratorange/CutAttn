python train.py \
  --name resnet_atn_09_start_spectral_norm \
  --CUT_mode CUT \
  --dataroot dataset \
  --netG resnet_adain \
  --netD basic_spectral_norm \
  --n_epochs 15 \
  --n_epochs_decay 0 \
