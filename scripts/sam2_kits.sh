# exec
export CUDA_VISIBLE_DEVICES=0
python ./train/sam2_kits.py \
        --data_dir=./dataset/kits19/ \
        --resolution=256 \
        --patch_size=16 \
        --pretrain=sam2-t \
        --epoch=1000 \
        --batch_size=32 \
        --savefile=./sam2-kits