#!/bin/bash
datasets=("idmt" "mimii")
features=("mel" "reassigned")
losses=("mse" "ccc" "mae" "mape")

#
# normalizes=("True" "False")

for dataset in ${datasets[@]}; do
    for feature in ${features[@]}; do
        for loss in ${losses[@]}; do
            for ((i=1; i<=10; i++)); do
                echo "Running $dataset $feature $loss #$i"
                python baseline4.py --dataset $dataset --feature $feature --loss $loss --normalize
            done
        done
    done
done
