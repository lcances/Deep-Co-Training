# cifar 10 Co-Training classic
bash simple_co-training.sh \
    --dataset cifar10 \
    --model Pmodel \
    --ratio 0.08 \
    --epoch 600 \
    --learning_rate 0.05 \
    --lambda_cot_max 10 \
    --lambda_diff_max 0.5 \
    --lambda_sup_max 1 \
