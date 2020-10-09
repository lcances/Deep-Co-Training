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
bash full_supervised.sh cnn03 speechcommand -r 0.1
bash full_supervised.sh resnet18 speechcommand -r 0.1
bash full_supervised.sh resnet34 speechcommand -r 0.1
bash full_supervised.sh resnet50 speechcommand -r 0.1
bash full_supervised.sh wideresnet28_2 speechcommand -r 0.1
bash full_supervised.sh wideresnet28_4 speechcommand -r 0.1
bash full_supervised.sh wideresnet28_8 speechcommand -r 0.1
