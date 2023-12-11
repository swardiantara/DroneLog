#!/bin/bash
datasets=( filtered unfiltered )
word_embeds=( bert )
encoders=( transformer )
weight_class=( uniform balanced inverse )
losses=( cross_entropy focal )
poolings=( cls max min mean )
bidirectionals=( true false )
n_layers=( 2 )
n_heads=( 2 )

for dataset in "${datasets[@]}"; do
    echo "dataset: "$dataset""
    for word_embed in "${word_embeds[@]}"; do
        for encoder in "${encoders[@]}"; do
            for weight in "${weight_class[@]}"; do
                for loss in "${losses[@]}"; do
                    if [ "$encoder" = "linear" ]; then
                        for pooling in "${poolings[@]}"; do
                            python baseline.py --dataset "$dataset" --word_embed "$word_embed" --encoder linear --pooling "$pooling" --class_weight "$weight" --loss "$loss" --output_dir transsentlog
                        done
                    else
                        for n_layer in "${n_layers[@]}"; do
                            if [ "$encoder" = "transformer" ]; then
                                for n_head in "${n_heads[@]}"; do
                                    for pooling in "${poolings[@]}"; do
                                        python baseline.py --dataset "$dataset" --word_embed "$word_embed" --encoder "$encoder" --n_layers "$n_layer" --n_heads "$n_head" --pooling "$pooling" --class_weight "$weight" --loss "$loss" --output_dir transsentlog
                                    done
                                done
                            else
                                for bidirectional in "${bidirectionals[@]}"; do
                                    if [ "$bidirectional" = true ]; then
                                        python baseline.py --dataset "$dataset" --word_embed "$word_embed" --encoder "$encoder" --n_layers "$n_layer" --bidirectional --class_weight "$weight" --loss "$loss" --output_dir transsentlog
                                    else
                                        python baseline.py --dataset "$dataset" --word_embed "$word_embed" --encoder "$encoder" --n_layers "$n_layer" --class_weight "$weight" --loss "$loss" --output_dir transsentlog
                                    fi
                                done
                            fi
                        done
                    fi
                done
            done
        done
    done
done