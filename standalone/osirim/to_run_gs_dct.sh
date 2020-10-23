# === === === === === === === === === === === === === === === === === === === === ===
# Urbansound8k
# === === === === === === === === === === === === === === === === === === === === ===
LCM=(1 5 10)
LDM=(1 0.25 0.5)
WL=(80 160)
LR=(0.003 0.001 0.0005)


for l in ${!LR[*]}; do
for w in ${!WL[*]}; do
for d in ${!LDM[*]}; do
for c in ${!LCM[*]}; do

bash deep-co-training_gridSearch.sh wideresnet28_2 ubs8k --learning_rate ${LR[$l]} --warmup_length ${WL[$w]} --lambda_diff_max ${LDM[$d]} --lambda_cot_max ${LCM[$c]} --tensorboard_sufix ${LR[$l]}lr_${WL[$w]}wl_${LCM[$c]}lcm_${LDM[$d]}ldm

done
done
done
done
