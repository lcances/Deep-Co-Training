c_args="wideresnet28_2 ubs8k"
c_args="${c_args} --batch_size 64 --epoch 200 --noise"

LR=(0.003)
WL=(50)
ALPHA=(0.999 0.99)
LCM=(1 1.5 2)

for l in ${!LR[*]}; do
for w in ${!WL[*]}; do
for a in ${!ALPHA[*]}; do
for c in ${!LCM[*]}; do

args="--learning_rate ${LR[$l]}"
args="${args} --warmup_length ${WL[$w]}"
args="${args} --alpha ${ALPHA[$a]}"
args="${args} --lambda_ccost_max ${LCM[$c]}"
args="${args} --noise"
args="${args} --tensorboard_sufix ${LR[$l]}lr_${WL[$w]}wl_${ALPHA[$a]}a_${LCM[$c]}lccm"

bash student-teacher.sh ${c_args} ${args}

done
done
done
done
