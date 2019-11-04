source ~/.bashrc
conda activate torch
module load cuda/10.0.130
uid=$RANDOM
echo $uid
python main.py --cuda "0,1,2,3" \
    --batch_size 320 \
    --ce_loss_scale 0.5 \
    --lr 5e-5 \
    --info 1kl05ce \
    --max_epoch 3

