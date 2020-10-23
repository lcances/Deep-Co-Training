# === === === === === === === === === === === === === === === === === === === === ===
# Urbansound8k
# === === === === === === === === === === === === === === === === === === === === ===
# bash deep-co-training.sh cnn03 ubs8k
# bash deep-co-training.sh resnet18 ubs8k
# bash deep-co-training.sh resnet34 ubs8k
# bash deep-co-training.sh resnet50 ubs8k
# bash deep-co-training.sh wideresnet28_2 ubs8k -p RTX6000Node
# bash deep-co-training.sh wideresnet28_2 ubs8k -p RTX6000Node
# bash deep-co-training.sh wideresnet28_8 ubs8k -p RTX6000Node -g 2

# === === === === === === === === === === === === === === === === === === === === ===
# ESC10
# === === === === === === === === === === === === === === === === === === === === ===
# bash deep-co-training.sh cnn03 esc10
# bash deep-co-training.sh resnet18 esc10
# bash deep-co-training.sh resnet34 esc10
# bash deep-co-training.sh resnet50 esc10
# bash deep-co-training.sh wideresnet28_2 esc10 -p RTX6000Node
# bash deep-co-training.sh wideresnet28_2 esc10 -p RTX6000Node
# bash deep-co-training.sh wideresnet28_8 esc10 -p RTX6000Node -g 2

# === === === === === === === === === === === === === === === === === === === === ===
# ESC50
# === === === === === === === === === === === === === === === === === === === === ===
# bash deep-co-training.sh cnn03 esc50
# bash deep-co-training.sh resnet18 esc50
# bash deep-co-training.sh resnet34 esc50
# bash deep-co-training.sh resnet50 esc50
# bash deep-co-training.sh wideresnet28_2 esc50 -p RTX6000Node
# bash deep-co-training.sh wideresnet28_4 esc50 -p RTX6000Node
# bash deep-co-training.sh wideresnet28_8 esc50 -p RTX6000Node -g 2


# === === === === === === === === === === === === === === === === === === === === ===
# SPEECHCOMMAND
# === === === === === === === === === === === === === === === === === === === === ===
# bash deep-co-training.sh cnn03 speechcommand
# bash deep-co-training.sh resnet18 speechcommand
# bash deep-co-training.sh resnet34 speechcommand
# bash deep-co-training.sh resnet50 speechcommand
# bash deep-co-training.sh wideresnet28_2 speechcommand
# bash deep-co-training.sh wideresnet28_4 speechcommand
# bash deep-co-training.sh wideresnet28_8 speechcommand -p RTX6000Node






# === === === === === === === === === === === === === === === === === === === === ===
# CUSTOM
# === === === === === === === === === === === === === === === === === === === === ===
c_args="--lambda_cot_max 1 --lambda_diff_max 0.5 --warmup_length 160 --learning_rate 0.0005 --epoch 300"
# 
d_args="--batch_size 100"
bash deep-co-training.sh wideresnet28_2 esc10 -p RTX6000Node ${c_args} ${d_args} --ratio 0.10 -C
bash deep-co-training.sh wideresnet28_2 esc10 -p RTX6000Node ${c_args} ${d_args} --ratio 0.25 -C
bash deep-co-training.sh wideresnet28_2 esc10 -p RTX6000Node ${c_args} ${d_args} --ratio 0.50 -C
# 
d_args="--batch_size 100"
bash deep-co-training.sh wideresnet28_2 ubs8k -p RTX6000Node ${c_args} ${d_args} --ratio 0.10 -C
bash deep-co-training.sh wideresnet28_2 ubs8k -p RTX6000Node ${c_args} ${d_args} --ratio 0.25 -C
bash deep-co-training.sh wideresnet28_2 ubs8k -p RTX6000Node ${c_args} ${d_args} --ratio 0.50 -C
# 
d_args="--batch_size 256"
# bash deep-co-training.sh wideresnet28_2 SpeechCommand ${c_args} ${d_args} --seed $i --ratio 0.10 -p RTX6000Node
# bash deep-co-training.sh wideresnet28_2 SpeechCommand ${c_args} ${d_args} --seed $i --ratio 0.25 -p RTX6000Node
# bash deep-co-training.sh wideresnet28_2 SpeechCommand ${c_args} ${d_args} --seed $i --ratio 0.50 -p RTX6000Node

