#!/bin/bash

common_param="--dataset SpeechCommand --learning_rate 0.003 --epoch 100 --batch_size 100 --ratio 1.0 --num_classes 35"
bash full_supervised.sh --model cnn03 ${common_param}
bash full_supervised.sh --model resnet18 ${common_param}
bash full_supervised.sh --model resnet34 ${common_param}
bash full_supervised.sh --model resnet50 ${common_param}
bash full_supervised.sh --model esc_wideresnet28_2 ${common_param}
bash full_supervised.sh --model esc_wideresnet28_4 ${common_param}
bash full_supervised.sh --model esc_wideresnet28_8 ${common_param}

common_param="--dataset SpeechCommand --learning_rate 0.003 --epoch 200 --batch_size 100 --ratio 0.1 --num_classes 35"
bash full_supervised.sh --model cnn03 ${common_param}
bash full_supervised.sh --model resnet18 ${common_param}
bash full_supervised.sh --model resnet34 ${common_param}
bash full_supervised.sh --model resnet50 ${common_param}
bash full_supervised.sh --model esc_wideresnet28_2 ${common_param}
bash full_supervised.sh --model esc_wideresnet28_4 ${common_param}
bash full_supervised.sh --model esc_wideresnet28_8 ${common_param}