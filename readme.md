# Usage

## bash
./script/sam2_kits.sh

## command line
python ./train/sam2_kits.py \
        --data_dir=./dataset/kits19/ \
        --resolution=256 \
        --patch_size=16 \
        --pretrain=sam2-t \
        --epoch=1000 \
        --batch_size=16 \
        --savefile=./sam2-kits-re

# Explain

sam2-kits: where pretrained model saved *.pth

./train/sam2_kits.py: training code for sam2 on kits.