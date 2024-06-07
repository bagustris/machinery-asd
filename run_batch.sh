#!/bin/bash
datasets=("imdt" "mimii")
features=("mel" "reassigned")
losses=("mse" "ccc" "mae" "mape")

for dataset in ${datasets[@]}; do
    for feature in ${features[@]}; do
        for loss in ${losses[@]}; do
            for ((i=1; i<=5; i++)); do
                echo "Running $dataset $feature $loss #$i"
                python baseline4.py --dataset $dataset --feature $feature --loss $loss
        done
    done
