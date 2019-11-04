"""
# This file is all config for model.
"""
# from torch.utils.tensorboard import SummaryWriter
import os 

class Config():
    """
    # Be careful! All paras are class vars.
    """
    word_tot = 0 # This para is set in dataloader
    rel_num = 0 # This para is set in dataloader
    word_embeeding_dim = 0 # This para is set in dataloader
    pos_embedding_dim = 5
    sen_len = 96
    pos_num = sen_len * 2
    batch_size = 96
    hidden_size = 768
    max_epoch = 3
    dev_step = 1
    dropout = 0.5
    save_epoch = 2
    save_path = "../ckpt"
    training = True
    dataset = 'nyt'
    
    # neg sample
    down_size = False
    neg_samples = 1 

    # mode
    train_bag = False
    eval_bag = False
    
    # train info
    info = ""

    # bert
    model_type = 'bert'
    model_name_or_path = "bert-base-uncased"
    gumbel_temperature = 0.5 #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    mlm_probability = 0.15
    mask_mode = "none"
    seed = 42


    # for pre test
    latent = False
    
    # optimize
    lr = 3e-5
    weight_decay = 0.0
    adam_epsilon = 1e-8
    gradient_accumulation_steps = 1
    max_grad_norm = 1
    warmup_steps = 500
    gen_loss_scale = 1.0
    ce_loss_scale = 1.0
    kl_loss_scale = 1.0