# === === === === === === === === === === === === === === === === === === === === ===
# Urbansound8k
# === === === === === === === === === === === === === === === === === === === === ===
# bash student-teacher.sh cnn03 ubs8k
# bash student-teacher.sh resnet18 ubs8k
# bash student-teacher.sh resnet34 ubs8k
# bash student-teacher.sh resnet50 ubs8k
# bash student-teacher.sh wideresnet28_2 ubs8k
# bash student-teacher.sh wideresnet28_2 ubs8k
# bash student-teacher.sh wideresnet28_8 ubs8k

# === === === === === === === === === === === === === === === === === === === === ===
# ESC10
# === === === === === === === === === === === === === === === === === === === === ===
# bash student-teacher.sh cnn03 esc10
# bash student-teacher.sh resnet18 esc10
# bash student-teacher.sh resnet34 esc10
# bash student-teacher.sh resnet50 esc10
# bash student-teacher.sh wideresnet28_2 esc10
# bash student-teacher.sh wideresnet28_2 esc10
# bash student-teacher.sh wideresnet28_8 esc10

# === === === === === === === === === === === === === === === === === === === === ===
# ESC50
# === === === === === === === === === === === === === === === === === === === === ===
# bash student-teacher.sh cnn03 esc50
# bash student-teacher.sh resnet18 esc50
# bash student-teacher.sh resnet34 esc50
# bash student-teacher.sh resnet50 esc50
# bash student-teacher.sh wideresnet28_2 esc50
# bash student-teacher.sh wideresnet28_4 esc50
# bash student-teacher.sh wideresnet28_8 esc50


# === === === === === === === === === === === === === === === === === === === === ===
# SPEECHCOMMAND
# === === === === === === === === === === === === === === === === === === === === ===
# bash student-teacher.sh cnn03 speechcommand
# bash student-teacher.sh resnet18 speechcommand
# bash student-teacher.sh resnet34 speechcommand
# bash student-teacher.sh resnet50 speechcommand
# bash student-teacher.sh wideresnet28_2 speechcommand
# bash student-teacher.sh wideresnet28_4 speechcommand
# bash student-teacher.sh wideresnet28_8 speechcommand






# === === === === === === === === === === === === === === === === === === === === ===
# CUSTOM
# === === === === === === === === === === === === === === === === === === === === ===
c_args="--alpha 0.999 --warmup_length 50 --lambda_ccost_max 1 --learning_rate 0.003"
d_args="--batch_size 64"
bash student-teacher.sh wideresnet28_2 esc10 ${d_args} ${c_args} --ratio 0.10 -C
bash student-teacher.sh wideresnet28_2 esc10 ${d_args} ${c_args} --ratio 0.25 -C
bash student-teacher.sh wideresnet28_2 esc10 ${d_args} ${c_args} --ratio 0.50 -C

bash student-teacher.sh wideresnet28_2 ubs8k ${d_args} ${c_args} --ratio 0.10 -C
bash student-teacher.sh wideresnet28_2 ubs8k ${d_args} ${c_args} --ratio 0.25 -C
bash student-teacher.sh wideresnet28_2 ubs8k ${d_args} ${c_args} --ratio 0.50 -C

d_args="--batch_size 256"
# bash student-teacher.sh wideresnet28_2 SpeechCommand ${d_args} ${c_args} --seed $i --ratio 0.10
# bash student-teacher.sh wideresnet28_2 SpeechCommand ${d_args} ${c_args} --seed $i --ratio 0.25     
# bash student-teacher.sh wideresnet28_2 SpeechCommand ${d_args} ${c_args} --seed $i --ratio 0.50
