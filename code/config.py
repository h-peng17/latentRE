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
    sen_len = 120
    pos_num = sen_len * 2
    neg_samples = 1 
    batch_size = 8
    hidden_size = 256
    max_epoch = 60
    dev_step = 5
    dropout = 0.5
    save_epoch = 5
    save_path = "../ckpt"
    loss_func = ""
    training = True
    down_size = False
    train_bag = False
    eval_bag = True
    info = ""

    # bert
    max_seq_length = 384
    model_type = 'bert'
    model_name_or_path = "bert-base-uncased"
    gumbal_temperature = 0.5
    

    # optimize
    lr = 3e-5
    weight_decay = 0.0
    adam_epsilon = 1e-8
    gradient_accumulation_steps = 4
    max_grad_norm = 1
    warmup_steps = 0