
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 
from config import Config
from transformers import (WEIGHTS_NAME, BertConfig, BertForMaskedLM, BertTokenizer, \
                                        GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)


class BertDecoder(nn.Module):
    def __init__(self):
        super(BertDecoder, self).__init__()
        
        # pre-train model dict
        self.MODEL_CLASSES = {
            'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
        }

        # set seed
        torch.manual_seed(Config.seed)

        # load pretrained model
        config_class, model_class, tokenizer_class = self.MODEL_CLASSES[Config.model_type]
        config = config_class.from_pretrained(Config.model_name_or_path)
        self.tokenizer = tokenizer_class.from_pretrained(Config.model_name_or_path, do_lower_case=True)
        self.model = model_class.from_pretrained(Config.model_name_or_path, from_tf=bool('.ckpt' in Config.model_name_or_path), config=config)

        self.MASK_MODE = {
            "entity": self.mask_not_entity_tokens,
            "between": self.mask_between_entity,
            "governor": self.governor_mask,
            # "none": self.not_mask,
            "origin":self.mask_tokens
        }

    def mask_tokens(self, inputs, tokenizer, token_mask):
        inputs = inputs.cpu()
        token_mask = token_mask.cpu()
        labels = inputs.clone()
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
        # We sample a few tokens in each sequence for masked-LM training (with probability Config.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        masked_indices = torch.bernoulli(torch.full(labels.shape, Config.mlm_probability)).bool() & (token_mask.bool())
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
    
    def mask_not_entity_tokens(self, inputs, tokenizer, token_mask):
        inputs = inputs.cpu()
        token_mask = token_mask.cpu()
        labels = inputs.clone()
        token_mask_indices = token_mask.bool()
        inputs[token_mask_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        labels[~token_mask_indices] = -1

        # we only compute loss for masked token && not padding token
        return inputs.cuda(), labels.cuda()
    
    def governor_mask(self, inputs, tokenizer, governor_mask):
        inputs = inputs.cpu()
        governor_mask = governor_mask.cpu()
        labels = inputs.clone()
        governor_mask_indices = governor_mask.bool()
        inputs[governor_mask_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        labels[~governor_mask_indices] = -1

        return inputs.cuda(), labels.cuda()
    
    def mask_between_entity(self, inputs, tokenizer, between_mask):
        inputs = inputs.cpu()
        between_mask = between_mask.cpu()
        labels = inputs.clone()
        between_entity_mask_indices = between_mask.bool()
        inputs[between_entity_mask_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        labels[~between_entity_mask_indices] = -1
        # we only compute loss for masked token && not padding token
        return inputs.cuda(), labels.cuda()

    def not_mask(self, inputs, tokenizer, attention_mask):
        """prepare not masked tokens"""
        labels = inputs.clone()
        # mask padding tokens 
        attention_mask = ~(attention_mask.bool())
        labels[attention_mask] = -1

        # we only compute loss for not padding token
        return inputs, labels
        

    def forward(self, input_ids, attention_mask, mask, latent=None):
        """input_ids shape is `(batch_size, sequence_length)`
           latent shape is `(batch_size, hidden_size)`
           label shape is `(batch_size, hidden_size)`
        """        
        mask_func = self.MASK_MODE[Config.mask_mode]
        input_ids, labels = mask_func(input_ids, self.tokenizer, mask)

        inputs = {
            'input_ids':input_ids,
            'attention_mask':attention_mask,
            'mask':mask,
            'latent':latent,
            'masked_lm_labels':labels
        }
        ouputs = self.model(**inputs)
        loss = ouputs[0]
        return loss
        
# class GPT2Decoder(nn.Module):
#     def __init__(self):
#         super(GPT2Decoder, self).__init__()
        
#         # pre-train model dict
#         self.MODEL_CLASSES = {
#             'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
#         }

#         # load pretrained model
#         config_class, model_class, tokenizer_class = self.MODEL_CLASSES["gpt2"]
#         config = config_class.from_pretrained(Config.gpt2)
#         self.tokenizer = tokenizer_class.from_pretrained(Config.gpt2)
#         self.model = model_class.from_pretrained(Config.gpt2, from_tf=bool('.ckpt' in Config.gpt2), config=config)

#         self.MASK_MODE = {
#             "between": self.mask_between_entity,
#         }
    
#     def mask_between_entity(self, inputs, between_mask):
#         labels = inputs.clone()
#         between_entity_mask_indices = between_mask.bool()
#         labels[~between_entity_mask_indices] = -1
#         # we only compute loss for masked token && not padding token
#         return labels

#     def forward(self, input_ids, attention_mask, mask, latent=None, labels=None):
#         """input_ids shape is `(batch_size, sequence_length)`
#            latent shape is `(batch_size, hidden_size)`
#            label shape is `(batch_size, hidden_size)`
#         """        
#         inputs = {
#             'input_ids':input_ids,
#             'attention_mask':attention_mask,
#             'mask':mask,
#             'latent':latent,
#             'labels':labels
#         }
#         ouputs = self.model(**inputs)
#         loss = ouputs[0]
#         return loss, torch.argmax(ouputs[1], 2)
        
