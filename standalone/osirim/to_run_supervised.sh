# === === === === === === === === === === === === === === === === === === === === ===
# SPEECHCOMMAND
# === === === === === === === === === === === === === === === === === === === === ===
# 100% supervised
# bash full_supervised.sh cnn03 speechcommand -r 1.0
# bash full_supervised.sh resnet18 speechcommand -r 1.0
# bash full_supervised.sh resnet34 speechcommand -r 1.0
# bash full_supervised.sh resnet50 speechcommand -r 1.0
# bash full_supervised.sh wideresnet28_2 speechcommand -r 1.0
# bash full_supervised.sh wideresnet28_4 speechcommand -r 1.0
# bash full_supervised.sh wideresnet28_8 speechcommand -r 1.0

# 10% supervised
# bash full_supervised.sh cnn03 speechcommand -r 0.1
# bash full_supervised.sh resnet18 speechcommand -r 0.1
# bash full_supervised.sh resnet34 speechcommand -r 0.1
# bash full_supervised.sh resnet50 speechcommand -r 0.1
# bash full_supervised.sh wideresnet28_2 speechcommand -r 0.1
# bash full_supervised.sh wideresnet28_4 speechcommand -r 0.1
# bash full_supervised.sh wideresnet28_8 speechcommand -r 0.1

# Custom
common_args="--batch_size 64 --epoch 100 --learning_rate 0.003"
# bash full_supervised.sh wideresnet28_2 ubs8k ${common_args} -r 0.1 -C
# bash full_supervised.sh wideresnet28_2 ubs8k ${common_args} -r 0.25 -C
# bash full_supervised.sh wideresnet28_2 ubs8k ${common_args} -r 0.50 -C
# bash full_supervised.sh wideresnet28_2 ubs8k ${common_args} -r 1.0 -C
# 
# bash full_supervised.sh wideresnet28_2 esc10 ${common_args} -r 0.1 -C
# bash full_supervised.sh wideresnet28_2 esc10 ${common_args} -r 0.25 -C
# bash full_supervised.sh wideresnet28_2 esc10 ${common_args} -r 0.50 -C
# bash full_supervised.sh wideresnet28_2 esc10 ${common_args} -r 1.0 -C

common_args="--batch_size 64 --epoch 100 --learning_rate 0.003"
#bash full_supervised.sh wideresnet28_2 SpeechCommand ${common_args} -r 1.0

for i in 1234; do 
    bash full_supervised.sh wideresnet28_2 SpeechCommand ${common_args} --seed $i -r 0.10
    bash full_supervised.sh wideresnet28_2 SpeechCommand ${common_args} --seed $i -r 0.25
    bash full_supervised.sh wideresnet28_2 SpeechCommand ${common_args} --seed $i -r 0.50
    bash full_supervised.sh wideresnet28_2 SpeechCommand ${common_args} --seed $i -r 1.00
done
