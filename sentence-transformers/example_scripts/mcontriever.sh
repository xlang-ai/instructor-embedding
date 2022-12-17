#!/bin/bash
#SBATCH --cpus-per-task=5
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --job-name=mcontriever
#SBATCH --output=/private/home/gizacard/contriever/logtrain/%A
#SBATCH --partition=learnlab
#SBATCH --mem=450GB
#SBATCH --signal=USR1@140
#SBATCH --open-mode=append


port=$(shuf -i 15000-16000 -n 1)

TDIR=/private/home/gizacard/contriever/encoded-data/bert-base-multilingual-cased/
TRAINDATASETS="${TDIR}fr_XX ${TDIR}en_XX ${TDIR}ar_AR ${TDIR}bn_IN ${TDIR}fi_FI ${TDIR}id_ID ${TDIR}ja_XX ${TDIR}ko_KR ${TDIR}ru_RU ${TDIR}sw_KE ${TDIR}hu_HU ${TDIR}he_IL ${TDIR}it_IT ${TDIR}km_KM ${TDIR}ms_MY ${TDIR}nl_XX ${TDIR}no_XX ${TDIR}pl_PL ${TDIR}pt_XX ${TDIR}sv_SE ${TDIR}te_IN ${TDIR}th_TH ${TDIR}tr_TR ${TDIR}vi_VN ${TDIR}zh_CN ${TDIR}zh_TW ${TDIR}es_XX ${TDIR}de_DE ${TDIR}da_DK"

rmin=0.1
rmax=0.5
T=0.05
QSIZE=32768
MOM=0.999
POOL=average
AUG=none
PAUG=0.
LC=0.
mo=bert-base-multilingual-cased
mp=none

name=$SLURM_JOB_ID-$POOL-rmin$rmin-rmax$rmax-T$T-$QSIZE-$MOM-$mo-$AUG-$PAUG

srun ~gizacard/anaconda3/envs/pytorch10/bin/python3 ~gizacard/contriever/train.py \
        --model_path $mp \
        --sampling_coefficient $LC \
        --augmentation $AUG --prob_augmentation $PAUG \
        --retriever_model_id $mo --pooling $POOL \
        --train_data $TRAINDATASETS --loading_mode split \
        --ratio_min $rmin --ratio_max $rmax --chunk_length 256 \
        --momentum $MOM --queue_size $QSIZE --temperature $T \
        --warmup_steps 20000 --total_steps 500000 --lr 0.00005 \
        --name $name \
        --scheduler linear \
        --optim adamw \
        --per_gpu_batch_size 64 \
        --output_dir /checkpoint/gizacard/contriever/xling/$name \
        --main_port $port \
