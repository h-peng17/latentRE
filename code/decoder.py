
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 
from config import Config
from transformers import (WEIGHTS_NAME, BertConfig,
                            BertForMaskedLM, BertTokenizer)


class BertDecoder(nn.Module):
    def __init__(self):
        super(BertDecoder, self).__init__()
        
        # pre-train model dict
        self.MODEL_CLASSES = {
            'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
        }

        # load pretrained model
        config_class, model_class, tokenizer_class = self.MODEL_CLASSES[Config.model_type]
        config = config_class.from_pretrained(Config.model_name_or_path)
        self.tokenizer = tokenizer_class.from_pretrained(Config.model_name_or_path, do_lower_case=True)
        self.model = model_class.from_pretrained(Config.model_name_or_path, from_tf=bool('.ckpt' in Config.model_name_or_path), config=config)

        self.linear_for_label_info = nn.Linear(Config.hidden_size, 768)
        self.linear_for_latent = nn.Linear(Config.hidden_size, 768)
    
    def forward(self, input_ids, latent, label_info, attention_mask):
        """input_ids shape is `(batch_size, sequence_length)`
           latent shape is `(batch_size, hidden_size)`
           label shape is `(batch_size, hidden_size)`
        """
        inputs = {
            'input_ids':F.relu(input_ids),
            'latent':self.linear_for_latent(latent),
            'label_info':self.linear_for_label_info(label_info),
            'attention_mask':attention_mask,
            'masked_lm_labels':input_ids
        }
        ouputs = self.model(**inputs)
        loss = ouputs[0]
        return loss 
        