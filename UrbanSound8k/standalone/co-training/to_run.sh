# python co-training-AugAsAdv.py --model cnn03 --nb_epoch 200 --augment ffd_2 --augment noise_snr20
# python co-training-AugAsAdv.py --model cnn03 --nb_epoch 200 --augment ffd_3 --augment noise_snr15
# python co-training-AugAsAdv.py --model cnn03 --nb_epoch 200 --augment noise_snr10 --augment s_psr_5
# python co-training-AugAsAdv.py --model cnn03 --nb_epoch 200 --augment rfd_1 --augment s_psr_4
# python co-training-AugAsAdv.py --model cnn03 --nb_epoch 200 --augment ffs_54 --augment s_psr_1

# python co-training-AugAsAdv.py --model cnn03 --nb_epoch 200 --augment rfd_3 --augment rfd_3
# python co-training-AugAsAdv.py --model cnn03 --nb_epoch 200 --augment usn_28 --augment usn_28
# python co-training-AugAsAdv.py --model cnn03 --nb_epoch 200 --augment usn_29 --augment usn_29
# python co-training-AugAsAdv.py --model cnn03 --nb_epoch 200 --augment rfd_2 --augment rfd_2
# python co-training-AugAsAdv.py --model cnn03 --nb_epoch 200 --augment noise_snr20 --augment noise_snr20
# python co-training-AugAsAdv.py --model cnn03 --nb_epoch 80 --augment noise_snr10 --augment noise_snr10
# python co-training-AugAsAdv.py --model cnn03 --nb_epoch 80 --augment noise_snr15 --augment noise_snr15
# python co-training-AugAsAdv.py --model cnn03 --nb_epoch 80 --augment noise_snr25 --augment noise_snr25
# python co-training-AugAsAdv.py --model cnn03 --nb_epoch 80 --augment s_n_10 --augment s_n_10
# python co-training-AugAsAdv.py --model cnn03 --nb_epoch 80 --augment s_n_15 --augment s_n_15
# python co-training-AugAsAdv.py --model cnn03 --nb_epoch 80 --augment s_n_20 --augment s_n_20

python co-training_aug4adv.future.py --model cnn03 --nb_epoch 100 --augment_adv n_15_05 --augment_S --augment n_15_05
python co-training_aug4adv.future.py --model cnn03 --nb_epoch 100 --augment_adv n_20_05 --augment_S --augment n_20_05
