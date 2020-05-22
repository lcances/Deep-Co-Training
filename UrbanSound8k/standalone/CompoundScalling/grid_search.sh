#!/usr/bin/env bash

job_name="--job_name grid_search"
alpha_list=(1.0 1.2 1.4 1.6 1.8 2.0)
beta_list=(1.0 1.2 1.4 1.6 1.8 2.0)
gamma_list=(1.0)

for alpha in ${alpha_list}
do
    for beta in ${beta_list}
    do
        for gamma in ${gamma_list}
        do
            factors="-a ${alpha} -b ${beta} -g ${gamma}"
            python CompoundScalling1.py ${job_name} -t 1 2 3 4 5 6 7 8 9 -v 10 ${factors}
        done
    done
done