# === === === === === === === === === === === === === === === === === === === === ===
# Urbansound8k
# === === === === === === === === === === === === === === === === === === === === ===
# bash deep-co-training_uniloss.sh cnn03 ubs8k
# bash deep-co-training_uniloss.sh resnet18 ubs8k
# bash deep-co-training_uniloss.sh resnet34 ubs8k
# bash deep-co-training_uniloss.sh resnet50 ubs8k
# bash deep-co-training_uniloss.sh wideresnet28_2 ubs8k -p RTX6000Node
# bash deep-co-training_uniloss.sh wideresnet28_2 ubs8k -p RTX6000Node
# bash deep-co-training_uniloss.sh wideresnet28_8 ubs8k -p RTX6000Node -g 2

# === === === === === === === === === === === === === === === === === === === === ===
# ESC10
# === === === === === === === === === === === === === === === === === === === === ===
# bash deep-co-training_uniloss.sh cnn03 esc10
# bash deep-co-training_uniloss.sh resnet18 esc10
# bash deep-co-training_uniloss.sh resnet34 esc10
# bash deep-co-training_uniloss.sh resnet50 esc10
# bash deep-co-training_uniloss.sh wideresnet28_2 esc10 -p RTX6000Node
# bash deep-co-training_uniloss.sh wideresnet28_2 esc10 -p RTX6000Node
# bash deep-co-training_uniloss.sh wideresnet28_8 esc10 -p RTX6000Node -g 2

# === === === === === === === === === === === === === === === === === === === === ===
# ESC50
# === === === === === === === === === === === === === === === === === === === === ===
# bash deep-co-training_uniloss.sh cnn03 esc50
# bash deep-co-training_uniloss.sh resnet18 esc50
# bash deep-co-training_uniloss.sh resnet34 esc50
# bash deep-co-training_uniloss.sh resnet50 esc50
# bash deep-co-training_uniloss.sh wideresnet28_2 esc50 -p RTX6000Node
#bash deep-co-training_uniloss.sh wideresnet28_4 esc50 -p RTX6000Node
# bash deep-co-training_uniloss.sh wideresnet28_8 esc50 -p RTX6000Node -g 2


# === === === === === === === === === === === === === === === === === === === === ===
# SPEECHCOMMAND
# === === === === === === === === === === === === === === === === === === === === ===
# bash deep-co-training_uniloss.sh cnn03 speechcommand
# bash deep-co-training_uniloss.sh resnet18 speechcommand
# bash deep-co-training_uniloss.sh resnet34 speechcommand
# bash deep-co-training_uniloss.sh resnet50 speechcommand
# bash deep-co-training_uniloss.sh wideresnet28_2 speechcommand
# bash deep-co-training_uniloss.sh wideresnet28_4 speechcommand
# bash deep-co-training_uniloss.sh wideresnet28_8 speechcommand -p RTX6000Node


# === === === === === === === === === === === === === === === === === === === === ===
# CUSTOM


bash deep-co-training_uniloss.sh wideresnet28_2 speechcommand -bs 256 -lr 0.0005 -e 8000 -r 0.1 $@
