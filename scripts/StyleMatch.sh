#!/bin/bash

DATA=/data

DATASET=$1

TRAINER=StyleMatch
NET=resnet18


for EX_PER_CLASS in 10 5
    do
    for CFG in v1 v4
        do
        if [ ${DATASET} == ssdg_pacs ]; then
            if [ ${EX_PER_CLASS} == 5 ]; then
                NLAB=105
            elif [ ${EX_PER_CLASS} == 10 ]; then
                NLAB=210
            fi
            PREFIX=20epoch
            D1=art_painting
            D2=cartoon
            D3=photo
            D4=sketch
        elif [ ${DATASET} == ssdg_officehome ]; then
            if [ ${EX_PER_CLASS} == 5 ]; then
                NLAB=975
            elif [ ${EX_PER_CLASS} == 10 ]; then
                NLAB=1950
            fi
            PREFIX=20epoch
            D1=art
            D2=clipart
            D3=product
            D4=real_world
        elif [ ${DATASET} == ssdg_digitdg ]; then
            if [ ${EX_PER_CLASS} == 5 ]; then
                NLAB=150
            elif [ ${EX_PER_CLASS} == 10 ]; then
                NLAB=300
            fi
            PREFIX=20epoch
            D1=mnist
            D2=mnist_m
            D3=svhn
            D4=syn
        elif [ ${DATASET} == ssdg_minidomainnet ]; then
            if [ ${EX_PER_CLASS} == 5 ]; then
                NLAB=1890
            elif [ ${EX_PER_CLASS} == 10 ]; then
                NLAB=3780
            fi
            PREFIX=10epoch
            D1=clipart
            D2=painting
            D3=real
            D4=sketch
        fi

        for SEED in $(seq 1 5)
        do
            for SETUP in $(seq 1 4)
            do
                if [ ${SETUP} == 1 ]; then
                    S1=${D2}
                    S2=${D3}
                    S3=${D4}
                    T=${D1}
                elif [ ${SETUP} == 2 ]; then
                    S1=${D1}
                    S2=${D3}
                    S3=${D4}
                    T=${D2}
                elif [ ${SETUP} == 3 ]; then
                    S1=${D1}
                    S2=${D2}
                    S3=${D4}
                    T=${D3}
                elif [ ${SETUP} == 4 ]; then
                    S1=${D1}
                    S2=${D2}
                    S3=${D3}
                    T=${D4}
                fi
                
                python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --source-domains ${S1} ${S2} ${S3} \
                --target-domains ${T} \
                --dataset-config-file configs/datasets/${DATASET}_${NLAB}.yaml \
                --config-file configs/trainers/${TRAINER}/${PREFIX}_${CFG}.yaml \
                --output-dir output/${DATASET}/nlab_${NLAB}/${TRAINER}/${NET}/${CFG}/${T}/seed${SEED} \
                MODEL.BACKBONE.NAME ${NET} \
                DATASET.NUM_LABELED ${NLAB}
            done
        done
    done
done
