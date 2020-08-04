# python co-training-AugAsAdv.py --model cnn0 --supervised_ratio 0.1 --nb_epoch 250 --augment noise_snr15 --augment rtd_3 --tensorboard_sufix simple_run --resume
# 
# python co-training-AugAsAdv.py --model cnn0 --supervised_ratio 0.1 --nb_epoch 250 --augment fts_5 --augment flip_lr --tensorboard_sufix simple_run --resume
# 
# python co-training-AugAsAdv.py --model cnn0 --supervised_ratio 0.1 --nb_epoch 250 --augment noise_snr20 --augment ftd_2 --tensorboard_sufix simple_run --resume
# 
python co-training-AugAsAdv.py --model cnn0 --supervised_ratio 0.1 --nb_epoch 250 --augment flip_up --augment rtd_8 --tensorboard_sufix simple_run --resume
