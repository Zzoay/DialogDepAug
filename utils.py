
from typing import *
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def length_to_mask(length, max_len=None, dtype=None):
    """
    >>> lens = [3, 5, 4]
    >>> length_to_mask(length)
    >>> [[1, 1, 1, 0, 0],\
        [1, 1, 1, 1, 1], \
        [1, 1, 1, 1, 0]]
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    if max_len is None:
        max_len = max_len or torch.max(length)[0].item()
    mask = torch.arange(max_len, device=length.device, dtype=length.dtype). \
               expand(length.shape[0], max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = mask.to(dtype)
    return mask


def seq_mask_by_lens(lengths:torch.Tensor, 
                     max_len=None, 
                     dtype=torch.bool,
                     device='cpu'):
    """
    giving sequence lengths, return mask of the sequence.
    example:
        input: 
        lengths = torch.tensor([4, 5, 1, 3])
        output:
        tensor([[ True,  True,  True,  True, False],
                [ True,  True,  True,  True,  True],
                [ True, False, False, False, False],
                [ True,  True,  True, False, False]])
    """
    if max_len is None:
        max_len = lengths.max()
    row_vector = torch.arange(start=0, end=max_len, step=1).to(device)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix

    mask.type(dtype)
    return mask


def to_cuda(data):
    if isinstance(data, tuple):
        return [d.cuda() for d in data]
    elif isinstance(data, torch.Tensor):
        return data.cuda()
    raise RuntimeError
    

def arc_rel_loss(arc_logits: torch.Tensor, 
                 rel_logits: torch.Tensor, 
                 arc_gt: torch.Tensor,  # ground truth
                 rel_gt: torch.Tensor, 
                 mask: torch.Tensor) -> torch.Tensor:
    flip_mask = mask.eq(0)  

    def one_loss(logits, gt):
        tmp1 = logits.view(-1, logits.size(-1))
        tmp2 = gt.masked_fill(flip_mask, -1).view(-1)
        return F.cross_entropy(tmp1, tmp2, ignore_index=-1)

    if len(arc_gt.shape) == 1:  # batch_size == 1
        arc_gt = arc_gt.unsqueeze(0)
    index = arc_gt.unsqueeze(2).unsqueeze(3).expand(-1, -1, -1, rel_logits.shape[-1]) 
    rel_logits = torch.gather(rel_logits, dim=2, index=index).squeeze(2)

    arc_loss = one_loss(arc_logits, arc_gt)
    rel_loss = one_loss(rel_logits, rel_gt)

    return arc_loss + rel_loss


def inter_uttr_loss(sem_role_logit, sem_roles):
    loss = F.cross_entropy(sem_role_logit, sem_roles.squeeze(0))
    return loss


def uas_las(arc_logits: torch.Tensor,
            rel_logits: torch.Tensor,
            arc_gt: torch.Tensor,  # ground truth
            rel_gt: torch.Tensor,
            mask: torch.Tensor) -> Dict:
    """
    CoNLL:
    LAS(labeled attachment score): the proportion of “scoring” tokens that are assigned both the correct head and the correct dependency relation label.
    Punctuation tokens are non-scoring. In very exceptional cases, and depending on the original treebank annotation, some additional types of tokens might also be non-scoring.
    The overall score of a system is its labeled attachment score on all test sets taken together.

    UAS(Unlabeled attachment score): the proportion of “scoring” tokens that are assigned the correct head (regardless of the dependency relation label).
    """
    arc_preds = arc_logits.max(2)[1]

    index = arc_preds.unsqueeze(2).unsqueeze(3).expand(-1, -1, -1, rel_logits.shape[-1]) 
    rel_logits = torch.gather(rel_logits, dim=2, index=index).squeeze(2)

    rel_preds = rel_logits.max(2)[1]

    arc_logits_correct = (arc_preds == arc_gt).long() * mask
    rel_logits_correct = (rel_preds == rel_gt).long() * arc_logits_correct
    arc = arc_logits_correct.sum().item()
    rel = rel_logits_correct.sum().item()
    num = mask.sum().item()

    if num == 0:
        return {'UAS': 0.0, 'LAS': 0.0} 
    return {'UAS': float(arc) / float(num), 'LAS': float(rel) / float(num)}


def arc_rel_loss_inter(arc_logits: torch.Tensor,
                        rel_logits: torch.Tensor,
                        arc_gt: torch.Tensor,  # ground truth
                        rel_gt: torch.Tensor,
                        mask: torch.Tensor) -> Dict:
    flip_mask = mask.eq(0)  

    def one_loss(logits, gt):
        tmp1 = logits.view(-1, logits.size(-1))
        tmp2 = gt.masked_fill(flip_mask, -1).view(-1)
        return F.cross_entropy(tmp1, tmp2, ignore_index=-1)

    # if len(arc_gt.shape) == 1:  # batch_size == 1
    #     arc_gt = arc_gt.unsqueeze(0)
    # index = arc_gt.unsqueeze(2).unsqueeze(3).expand(-1, -1, -1, rel_logits.shape[-1]) 
    # rel_logits = torch.gather(rel_logits, dim=2, index=index).squeeze(2)

    arc_loss = one_loss(arc_logits, arc_gt)
    rel_loss = one_loss(rel_logits, rel_gt) * rel_gt.shape[0]

    return arc_loss + rel_loss


def uas_las_inter(arc_logits: torch.Tensor,
                rel_logits: torch.Tensor,
                arc_gt: torch.Tensor,  # ground truth
                rel_gt: torch.Tensor,
                mask: torch.Tensor) -> Dict:
    arc_preds = arc_logits.max(2)[1]

    # index = arc_preds.unsqueeze(2).unsqueeze(3).expand(-1, -1, -1, rel_logits.shape[-1]) 
    # rel_logits = torch.gather(rel_logits, dim=2, index=index).squeeze(2)

    rel_preds = rel_logits.max(2)[1]

    arc_logits_correct = (arc_preds == arc_gt).long() * mask
    rel_logits_correct = (rel_preds == rel_gt).long() * arc_logits_correct
    arc = arc_logits_correct.sum().item()
    rel = rel_logits_correct.sum().item()
    num = mask.sum().item()

    if num == 0:
        return {'UAS': 0.0, 'LAS': 0.0} 
    return {'UAS': float(arc) / float(num), 'LAS': float(rel) / float(num)}

def arc_rel_loss_split(arc_logits: torch.Tensor, 
                 rel_logits: torch.Tensor, 
                 arc_gt: torch.Tensor,  # ground truth
                 rel_gt: torch.Tensor, 
                 mask: torch.Tensor) -> torch.Tensor:
    batch_size, num_uttrs, seq_len = arc_gt.size()
    arc_gt = arc_gt.view(batch_size, -1)
    rel_gt = rel_gt.view(batch_size, -1)
    mask = mask.view(batch_size, -1)
    
    flip_mask = mask.eq(0)  

    def one_loss(logits, gt):
        tmp1 = logits.view(-1, logits.size(-1))
        tmp2 = gt.masked_fill(flip_mask, -1).view(-1)
        return F.cross_entropy(tmp1, tmp2, ignore_index=-1)

    arc_loss = one_loss(arc_logits, arc_gt)
    rel_loss = one_loss(rel_logits, rel_gt)

    return arc_loss + rel_loss


def uas_las_split(arc_logits: torch.Tensor,
                    rel_logits: torch.Tensor,
                    arc_gt: torch.Tensor,  # ground truth
                    rel_gt: torch.Tensor,
                    mask: torch.Tensor) -> Dict:
    batch_size, num_uttrs, seq_len = arc_gt.size()
    arc_gt = arc_gt.view(batch_size, -1)
    rel_gt = rel_gt.view(batch_size, -1)
    mask = mask.view(batch_size, -1)
    
    if len(arc_logits.shape) > len(arc_gt.shape):
        pred_dim, indices_dim = 2, 1
        arc_logits = arc_logits.max(pred_dim)[indices_dim]
    
    if len(rel_logits.shape) > len(rel_gt.shape):
        pred_dim, indices_dim = 2, 1
        rel_logits = rel_logits.max(pred_dim)[indices_dim]

    arc_logits_correct = (arc_logits == arc_gt).long() * mask
    rel_logits_correct = (rel_logits == rel_gt).long() * arc_logits_correct
    arc = arc_logits_correct.sum().item()
    rel = rel_logits_correct.sum().item()
    num = mask.sum().item()

    return {'UAS': float(arc) / float(num), 'LAS': float(rel) / float(num)}

def seed_everything(seed):
    np.random.seed(seed%(2**32-1))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic =True
    torch.backends.cudnn.benchmark = False
