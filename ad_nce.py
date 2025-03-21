import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
__all__ = ['ADNCE', 'ad_nce']


class ADNCE(nn.Module):
    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return ad_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)


def ad_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)   #[bs*few,1]

        if negative_mode == 'unpaired':
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)  #[bs*few,1,num_neg]
            negative_logits = negative_logits.squeeze(1)

        # adjust weights
        if negative_logits.shape[-1] != 1:
            mean = negative_logits.mean(dim=-1, keepdim=True)
            std = negative_logits.std(dim=-1, keepdim=True) + 1e-8
            z_scores = (negative_logits - mean) / std
            neg_weights = torch.exp(-0.5 * z_scores**2) / (std * np.sqrt(2 * np.pi))  
            neg_weights = neg_weights / neg_weights.sum(dim=-1, keepdim=True)  
        else:
            neg_weights = torch.ones_like(negative_logits) 

        negative_logits=negative_logits*neg_weights
        logits = torch.cat([positive_logit, negative_logits], dim=1)    #[bs*few,1+num_neg]
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)    #[bs*few]
    else:
        logits = query @ transpose(positive_key)
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]