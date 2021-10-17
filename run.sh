#! /bin/bash
eval "$(conda shell.bash hook)"
read -p 'Which conda env to use: '  CENV
conda activate $CENV
read -p 'Which GPU ID to use: '  GPUID
export CUDA_VISIBLE_DEVICES=$GPUID
read -p 'Which config file to use: '  CFG
python dgcnn_segmentation.py --cfg configs/$CFG.yaml