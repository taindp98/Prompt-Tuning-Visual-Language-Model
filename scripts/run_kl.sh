#!/bin/bash
# custom config
DATA=/netscratch/duynguyen/Research/ACML_TMLR_2024/DATA_PLOT
TRAINER=TCP
WEIGHT=1.0 

CFG=vit_b16_ep100_ctxv1    ## vit_b16_ep100_ctxv1 rn50
CTP=end  # class token position (end or middle)
NCTX=4  # number of context tokens  NOTE: NCTX=16
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)
FOLDER=output

for DATASET in imagenet
do
for SEED in 1
do
    DIR=${FOLDER}/${DATASET}/fewshot_exp_${NCTX}/shots_${SHOTS}_${WEIGHT}/${TRAINER}/${CFG}/seed${SEED}
    # if [ -d "$DIR" ]; then
    #     echo "Results are available in ${DIR}. Skip this job"
    # else
    # rm -rf ${DIR}
        echo "Run this job and save the output to ${DIR}"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.W ${WEIGHT} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES all
    # fi
done
done