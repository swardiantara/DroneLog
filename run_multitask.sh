#!/bin/bash
datasets=( filtered unfiltered )
word_embeds=( bert )
encoders=( transformer lstm gru none )
weight_class=( uniform balanced inverse )
losses=( logloss )
label_schemas=( 101 111 )
poolings=( cls max avg last )
bidirectionals=( true false )
n_layers=( 1 2 3 )
n_heads=( 4 6 8 )

for dataset in "${datasets[@]}"; do
    echo "dataset: "$dataset""
    for word_embed in "${word_embeds[@]}"; do
        for encoder in "${encoders[@]}"; do
            for weight in "${weight_class[@]}"; do
                for loss in "${losses[@]}"; do
                    for label_schema in "${label_schemas[@]}"; do
                        for pooling in "${poolings[@]}"; do
                            if [ "$encoder" = "none" ]; then
                                python multitask.py --dataset "$dataset" --word_embed "$word_embed" --encoder "$encoder" --pooling "$pooling" --class_weight "$weight" --loss "$loss" --label_schema "$label_schema" --viz_projection --output_dir multitask
                            else
                                for n_layer in "${n_layers[@]}"; do
                                    if [ "$encoder" = "transformer" ]; then
                                        for n_head in "${n_heads[@]}"; do
                                            # for pooling in "${poolings[@]}"; do
                                            python multitask.py --dataset "$dataset" --word_embed "$word_embed" --encoder "$encoder" --n_layers "$n_layer" --n_heads "$n_head" --pooling "$pooling" --class_weight "$weight" --loss "$loss" --label_schema "$label_schema" --viz_projection --output_dir multitask
                                            # done
                                        done
                                    else
                                        for bidirectional in "${bidirectionals[@]}"; do
                                            if [ "$bidirectional" = true ]; then
                                                python multitask.py --dataset "$dataset" --word_embed "$word_embed" --encoder "$encoder" --n_layers "$n_layer" --bidirectional --pooling "$pooling" --class_weight "$weight" --loss "$loss" --label_schema "$label_schema" --viz_projection --output_dir multitask
                                            else
                                                python multitask.py --dataset "$dataset" --word_embed "$word_embed" --encoder "$encoder" --n_layers "$n_layer" --pooling "$pooling" --class_weight "$weight" --loss "$loss" --label_schema "$label_schema" --viz_projection --output_dir multitask
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
    done
done