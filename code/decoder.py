
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
        torch.cuda.manual_seed_all(42)

        self.MASK_MODE = {
            "origin": self.mask_tokens,
            "entity": self.mask_not_entity_tokens,
            "none": self.not_mask,
        }
    
    
    
    def mask_tokens(self, inputs, tokenizer, attention_mask, token_mask=None):
        inputs = inputs.cpu()
        padding = ((1 - attention_mask) * 100).cpu()
        inputs = inputs + padding
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability Config.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        masked_indices = torch.bernoulli(torch.full(labels.shape, Config.mlm_probability)).bool()
        labels[~masked_indices] = -1  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs.cuda(), labels.cuda()
    
    def mask_not_entity_tokens(self, inputs, tokenizer, attention_mask, token_mask):
        """prepare masked tokens"""
        labels = inputs.clone()
        # mask not entity tokens
        token_mask_indices = token_mask.bool()
        attention_mask_indices = (~(attention_mask.bool())) | (~token_mask_indices)
        inputs[token_mask_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        labels[attention_mask_indices] = -1

        # we only compute loss for masked token && not padding token
        return inputs.cuda(), labels.cuda()

    def not_mask(self, inputs, tokenizer, attention_mask, token_mask=None):
        """prepare not masked tokens"""
        labels = inputs.clone()
        # mask padding tokens 
        attention_mask = ~(attention_mask.bool())
        labels[attention_mask] = -1

        # we only compute loss for not padding token
        return inputs.cuda(), labels.cuda()
        


    def forward(self, input_ids, attention_mask, token_mask=None, latent=None):
        """input_ids shape is `(batch_size, sequence_length)`
           latent shape is `(batch_size, hidden_size)`
           label shape is `(batch_size, hidden_size)`
        """
        mask_func = self.MASK_MODE[Config.mask_mode]
        input_ids, labels = mask_func(input_ids, self.tokenizer, attention_mask, token_mask)
        
        inputs = {
            'input_ids':input_ids,
            'latent':latent,
            'attention_mask':attention_mask,
            'masked_lm_labels':labels
        }
        ouputs = self.model(**inputs)
        loss = ouputs[0]
        return loss 
        