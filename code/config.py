"""
# This file is all config for model.
"""

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
    neg_samples = 5 
    batch_size = 160
    hidden_size = 256
    lr = 1e-3
    max_epoch = 100
    dev_step = 5
    dropout = 0.5