#!/bin/bash

# manual optimization, use --optuna for automatic optimization

datasets=("idmt" "mimii")
features=("mel" "reassigned")
losses=("mse" "ccc" "mae" "mape")

for dataset in ${datasets[@]}; do
    for feature in ${features[@]}; do
        for loss in ${losses[@]}; do
            for ((i=1; i<=10; i++)); do
                echo "Running $dataset $feature $loss #$i"
                python baseline5.py --dataset $dataset --feature $feature --loss $loss
            done
        done
    done
done
