#!/bin/bash

# common_param="--dataset esc10 --learning_rate 0.001 --epoch 200 --ratio 1.0 --num_classes 10"
# bash full_supervised_esc.sh --model cnn03 --batch_size 128 ${common_param}
# bash full_supervised_esc.sh --model resnet18 --batch_size 128 ${common_param}
# bash full_supervised_esc.sh --model resnet34 --batch_size 128 ${common_param}
# bash full_supervised_esc.sh --model resnet50 --batch_size 128 ${common_param}
# bash full_supervised_esc.sh --model esc_wideresnet28_2 --batch_size 128 ${common_param}
# bash full_supervised_esc.sh --model esc_wideresnet28_4 --batch_size 128 ${common_param}
# bash full_supervised_esc.sh --model esc_wideresnet28_8 --batch_size 64 ${common_param}
# 
# common_param="--dataset esc10 --learning_rate 0.001 --epoch 100 --ratio 0.1 --num_classes 10"
# bash full_supervised_esc.sh --model cnn03 --batch_size 128 ${common_param}
# bash full_supervised_esc.sh --model resnet18 --batch_size 128 ${common_param}
# bash full_supervised_esc.sh --model resnet34 --batch_size 128 ${common_param}
# bash full_supervised_esc.sh --model resnet50 --batch_size 128 ${common_param}
# bash full_supervised_esc.sh --model esc_wideresnet28_2 --batch_size 128 ${common_param}
# bash full_supervised_esc.sh --model esc_wideresnet28_4 --batch_size 128 ${common_param}
# bash full_supervised_esc.sh --model esc_wideresnet28_8 --batch_size 64 ${common_param}

common_param="--dataset esc50 --learning_rate 0.001 --epoch 200 --ratio 0.1 --num_classes 50"
bash full_supervised_esc.sh --model cnn03 --batch_size 128 ${common_param}
bash full_supervised_esc.sh --model resnet18 --batch_size 128 ${common_param}
bash full_supervised_esc.sh --model resnet34 --batch_size 128 ${common_param}
bash full_supervised_esc.sh --model resnet50 --batch_size 128 ${common_param}
bash full_supervised_esc.sh --model esc_wideresnet28_2 --batch_size 128 ${common_param}
bash full_supervised_esc.sh --model esc_wideresnet28_4 --batch_size 128 ${common_param}
bash full_supervised_esc.sh --model esc_wideresnet28_8 --batch_size 64 ${common_param}

common_param="--dataset ubs8k --learning_rate 0.001 --epoch 200 --ratio 1.0 --num_classes 10"
# bash full_supervised.sh --model cnn03 --batch_size 128 ${common_param}
# bash full_supervised.sh --model resnet18 --batch_size 128 ${common_param}
# bash full_supervised.sh --model resnet34 --batch_size 128 ${common_param}
# bash full_supervised.sh --model resnet50 --batch_size 128 ${common_param}
# bash full_supervised.sh --model esc_wideresnet28_2 --batch_size 128 ${common_param}
# bash full_supervised.sh --model esc_wideresnet28_4 --batch_size 128 ${common_param}
# bash full_supervised.sh --model esc_wideresnet28_8 --batch_size 64 ${common_param}