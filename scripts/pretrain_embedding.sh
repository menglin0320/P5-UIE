# Run with $ bash scripts/pretrain_P5_small_beauty.sh 4

#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

name=pretrain_item

output=snap/$name

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_port 12345 \
    src/pretrain.py \
        --distributed --multiGPU \
        --seed 2022 \
        --train beauty \
        --valid beauty \
        --batch_size 32 \
        --optim adamw \
        --warmup_ratio 0.05 \
        --lr 5 \
        --weight_decay 0.000002\
        --num_workers 4 \
        --clip_grad_norm 1.0 \
        --losses 'relate_inout_embeds,learn_title,learn_category,learn_brand,learn_price,learn_user' \
        --backbone 't5-small' \
        --output $output ${@:2} \
        --epoch 30 \
        --max_text_length 512 \
        --gen_max_length 64 \
        --continuous_embed \
        --pretrain_embed \
        --mean_regularize \
        --whole_word_embed > $name.log