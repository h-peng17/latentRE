source ~/.bashrc
conda activate torch
module load cuda/10.0.130
uid=$RANDOM
echo $uid
python main.py --cuda "0,1,2,3" \
    --batch_size 144 \
    --gen_loss_scale 0.1 \
    --ce_loss_scale 0 \
    --lr 5e-5 \
    --info 01gen0cebetweengpt2 \
    --latent \
    --mask_mode between \
    --max_epoch 3

