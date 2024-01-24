
from typing import *

import os
import yaml
from tqdm import tqdm
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoConfig, AutoModel, AutoTokenizer

from model import DepParser, InterParser
from utils import arc_rel_loss, uas_las, to_cuda, seq_mask_by_lens, uas_las_inter
from config import DataClassUnpack, CFG
from data_helper import load_signal, load_annoted, DialogEvalDataset, DialogUttrDataset, InterDataset, DialogUttrInterDataset
from chuliu_edmonds import mst


def eval_eduwise(model, CFG, test_iter):
    if CFG.cuda and torch.cuda.is_available():
        model.cuda()
    model.eval()

    signal_cnt, seq_accum = 0, 0
    signals_lst = load_signal('data/mlm/diag_test.conll')

    heads_whole, rels_whole, masks_whole = torch.Tensor(), torch.Tensor(), torch.Tensor()
    inner_head_preds, inner_rel_preds = torch.Tensor(), torch.Tensor()
    inter_head_preds, inter_rel_preds = torch.Tensor(), torch.Tensor()
    for batch in test_iter:
        inputs, offsets, heads, rels, masks, roles = batch
        if CFG.cuda and torch.cuda.is_available():
            inputs_cuda = {}
            for key, value in inputs.items():
                inputs_cuda[key] = value.cuda()
            inputs = inputs_cuda
            offsets, heads, rels, masks = to_cuda(data=(offsets, heads, rels, masks))
            inputs_squeeze= {}
            for key, value in inputs.items():  
                inputs_squeeze[key] = value.squeeze(0) # rst's batch size is always 1
            inputs = inputs_squeeze
            offsets, heads, rels, masks = offsets.squeeze(0), heads.squeeze(0), rels.squeeze(0), masks.squeeze(0)
        
        with torch.no_grad():
            inner_outputs, inter_outputs = model.predict(inputs, offsets)
        inner_head_logit, inner_rel_logit = [x.cpu() for x in inner_outputs]

        inner_head_logit[:, 0, 1:] = float('-inf')
        inner_head_logit.diagonal(0, 1, 2)[1:].fill_(float('-inf'))

        seq_lens = (offsets != 0).sum(1) + 1
        seq_masks = seq_mask_by_lens(seq_lens, max_len=CFG.dialog_max_length, device='cuda') 
        seq_masks = torch.cat([torch.zeros(seq_masks.shape[0], 1).cuda(), seq_masks], dim=1)[:, :-1].bool().cpu()

        in_head_pred = mst(inner_head_logit, seq_masks)
        index = in_head_pred.unsqueeze(2).unsqueeze(3).expand(-1, -1, -1, inner_rel_logit.shape[-1]) 
        inner_rel_logit = torch.gather(inner_rel_logit, dim=2, index=index).squeeze(2)
        in_rel_pred = inner_rel_logit.argmax(-1)
        in_rel_pred.masked_fill_(in_head_pred == 0, 0)

        inner_head_pred, inner_rel_pred = torch.zeros_like(heads).long(), torch.zeros_like(heads).long()
        subroot_ids = torch.Tensor()
        accum = 0
        for i, seq_len in enumerate(seq_lens):  
            head_pred, rel_pred = in_head_pred[i, :seq_len+1], in_rel_pred[i, :seq_len+1]

            subroot_id = ((rel_pred == 0) * seq_masks[i, :seq_len+1]).nonzero().view(-1).cpu() + accum
            if len(subroot_id) > 1 :
                subroot_id = subroot_id[0].unsqueeze(0)
            elif len(subroot_id) == 0:
                subroot_id = torch.tensor([accum])
            subroot_ids = torch.cat([subroot_ids, subroot_id], dim=0).long() # subroot ids

            # assert ((rel_pred == 0) * seq_masks[i, :seq_len+1]).sum() == 1
            assert len(subroot_id) == 1

            inner_head_pred[accum+1:len(head_pred)+accum] = head_pred[1:] + accum
            inner_rel_pred[accum+1:len(rel_pred)+accum] = rel_pred[1:]

            accum += seq_len.cpu().item()
        
        # inter
        # inter_head_logit, inter_rel_logit = [x.cpu() for x in inter_outputs]
        # mask_ids = [[1, 1, 2, 2, 3, 3], [0, 2, 0, 1, 1, 2]]
        # for i in range(4, inter_head_logit.shape[-1]-1):
        #     mask_ids[0].extend([i, i])
        #     mask_ids[1].extend([i-2, i-1])
        # mask_ids[0].extend([i+1, i+1])
        # mask_ids[1].extend([i-1, i])

        # window_masks = torch.zeros_like(inter_head_logit)
        # window_masks[:, mask_ids[0], mask_ids[1]] = 1

        # mst_mask = torch.ones(inter_head_logit.shape[:2]).bool()
        # mst_mask[:, 0] = False
        # tmp_head_pred = mst((inter_head_logit.softmax(-1) * window_masks), mst_mask).view(-1)
        # index = tmp_head_pred.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(-1, -1, -1, inter_rel_logit.shape[-1]) 
        # inter_rel_logit = torch.gather(inter_rel_logit, dim=2, index=index).squeeze(2)
        # tmp_rel_pred = inter_rel_logit.argmax(-1).squeeze(0)
        # tmp_rel_pred.masked_fill_(tmp_head_pred == 0, 0)

        # root_dct = {key: value.item() for key, value in zip(range(1, tmp_head_pred.shape[-1]), subroot_ids)}
        # root_dct[0] = 0
        # inter_head_pred, inter_rel_pred = torch.zeros_like(heads).cpu(), torch.zeros_like(heads).cpu()
        # inter_head_pred[subroot_ids] = torch.tensor([root_dct[x.item()] for x in tmp_head_pred[1:]])
        
        # signal-based
        num_turn = inputs['input_ids'].shape[0]
        signals = []
        for i, item in enumerate(signals_lst[signal_cnt:signal_cnt+num_turn]):
            word, signal1, signal2, left, right = item[0]
            left, right = left - seq_accum, right - seq_accum
            signals.append([word, signal1, signal2, left, right])
        if len(signals) != num_turn:  # last turn bug
            signals = []
            for i, item in enumerate(signals_lst[-num_turn:]):
                word, signal1, signal2, left, right = item[0]
                left, right = left - seq_accum, right - seq_accum
                signals.append([word, signal1, signal2, left, right])
        signal_cnt += num_turn
        seq_accum += len(signals)
        
        relpred_by_signal = []
        for i in range(1, num_turn):
            word, signal1, signal2, left, right = signals[i]
            if roles[:, i] != roles[:, i-1]:
                relpred_by_signal.append(signal2)
            else:
                relpred_by_signal.append(signal1)

        # inter-utterance
        # inner_head_pred[subroot_ids][1:] = subroot_ids[:-1]
        # inner_rel_pred[subroot_ids][1:] = torch.tensor(relpred_by_signal)

        heads_whole = torch.cat([heads_whole, heads.unsqueeze(0).cpu()])
        rels_whole = torch.cat([rels_whole, rels.unsqueeze(0).cpu()])
        masks_whole = torch.cat([masks_whole, masks.unsqueeze(0).cpu()])

        inner_head_preds = torch.cat([inner_head_preds, inner_head_pred.cpu().unsqueeze(0)])
        inner_rel_preds = torch.cat([inner_rel_preds, inner_rel_pred.cpu().unsqueeze(0)])

        # inter_head_preds = torch.cat([inter_head_preds, inter_head_pred.unsqueeze(0)])
        # inter_rel_preds = torch.cat([inter_rel_preds, inter_rel_pred.unsqueeze(0)])

    arc_logits_correct_inner = (inner_head_preds == heads_whole).long() * masks_whole * (rels_whole < 21).long() * (rels_whole != 0).long()
    rel_logits_correct_inner = (inner_rel_preds == rels_whole).long() * arc_logits_correct_inner
    subroot_correct = (((inner_rel_preds == 0) * (rels_whole >= 21)) * masks_whole).sum()

    inner_uas = arc_logits_correct_inner.sum() / (masks_whole * (rels_whole < 21).long() * (rels_whole != 0).long()).sum()
    inner_las = rel_logits_correct_inner.sum() / (masks_whole * (rels_whole < 21).long() * (rels_whole != 0).long()).sum()
    print(subroot_correct / ((rels_whole >= 21) * masks_whole).sum())
    
    arc_logits_correct_inter =  (inner_head_preds == heads_whole.cpu()) * (rels_whole.cpu() >= 21)
    rel_logits_correct_inter = (inner_rel_preds == rels_whole.cpu()) * arc_logits_correct_inter
    inter_uas = arc_logits_correct_inter.sum() / (rels_whole.cpu() >= 21).sum()
    inter_las = rel_logits_correct_inter.sum() / (rels_whole.cpu() >= 21).sum()

    return (inner_uas, inner_las), (inter_uas, inter_las)

def eval_uttrwise(model, CFG, test_iter):
    if CFG.cuda and torch.cuda.is_available():
        model.cuda()
    model.eval()

    heads_whole, rels_whole, masks_whole = torch.Tensor(), torch.Tensor(), torch.Tensor()
    inner_head_preds, inner_rel_preds = torch.Tensor(), torch.Tensor()
    for batch in test_iter:
        inputs, offsets, heads, rels, masks = batch
        if CFG.cuda and torch.cuda.is_available():
            inputs_cuda = {}
            for key, value in inputs.items():
                inputs_cuda[key] = value.cuda()
            inputs = inputs_cuda
            offsets, heads, rels, masks = to_cuda(data=(offsets, heads, rels, masks))
            
        with torch.no_grad():
            inner_outputs, inter_outputs = model.predict(inputs, offsets)
        
        inner_head_logit, inner_rel_logit = [x.cpu() for x in inner_outputs]
        inner_head_logit[:, torch.arange(inner_head_logit.size()[1]), torch.arange(inner_head_logit.size()[2])] = -1e4

        index = inner_head_logit.argmax(-1).unsqueeze(2).unsqueeze(3).expand(-1, -1, -1, inner_rel_logit.shape[-1]) 
        inner_rel_logit = torch.gather(inner_rel_logit, dim=2, index=index).squeeze(2)

        inner_head_preds = torch.cat([inner_head_preds, inner_head_logit.argmax(-1).cpu()])
        inner_rel_preds = torch.cat([inner_rel_preds, inner_rel_logit.argmax(-1).cpu()])

        heads_whole = torch.cat([heads_whole, heads.cpu()])
        rels_whole = torch.cat([rels_whole, rels.cpu()])
        masks_whole = torch.cat([masks_whole, masks.cpu()])

    arc_logits_correct_inner = (inner_head_preds == heads_whole).long() * masks_whole * (rels_whole < 21).long() * (rels_whole != 0).long()
    rel_logits_correct_inter = (inner_rel_preds == rels_whole).long() * arc_logits_correct_inner

    uas = arc_logits_correct_inner.sum() / (masks_whole * (rels_whole < 21).long() * (rels_whole != 0).long()).sum()
    las = rel_logits_correct_inter.sum() / (masks_whole * (rels_whole < 21).long() * (rels_whole != 0).long()).sum()

    return uas, las

def postproc(head_preds, rel_preds, masks_whole):
    from constant import rel2id, punct_lst
    origin4change = [rel2id[x] for x in ['root', 'dfsubj', 'sasubj']]
    # origin4change.extend([i for i in range(21, 35)])

    max_len = 160

    signals_new_whole = torch.Tensor()
    heads_new_whole, rels_new_whole = torch.Tensor(), torch.Tensor()
    for sample_idx, (deps, pred_signals) in tqdm(enumerate(zip(load_annoted('data/dialog/test.json'), load_signal('data/mlm/diag_test.conll')))):
        seq_len = len(deps)
        if seq_len == 0:
            continue

        signals = torch.full(size=(max_len,), fill_value=rel2id['elbr']).int()
        heads, rels = torch.full(size=(max_len,), fill_value=-2).int(), torch.zeros(max_len).int()
        split, splits, signal, word_lst  = 1, [1], rel2id['elbr'], ['root']
        for i, dep in enumerate(deps[:-1]):
            if i + 2 >= max_len:
                break

            word = dep.word
            word_lst.append(word)

            # if word in signal_dct.keys():
            #     signal = signal_dct[word]
            # if f'{word} {deps[i+1].word}' in signal_dct.keys():
            #     signal = signal_dct[f'{word} {deps[i+1].word}']

            try:
                signal = pred_signals[i]
            except IndexError:
                signal = pred_signals[len(pred_signals) - 1]

            if word in punct_lst and deps[i+1].word not in punct_lst:
                if i + 2 - split > 2:  # set 2 to the min length of edu
                    signals[split:i+2] = signal
                    # signal = None
                split = i + 2
                splits.append(split)

        splits.append(len(deps))

        # add the last data
        if i + 1 < max_len:
            signal = pred_signals[-1]
            word_lst.append(word)

        heads = head_preds[sample_idx]
        heads.masked_fill_(mask=~masks_whole[sample_idx].bool(), value=-2)

        rels = rel_preds[sample_idx]
        rels.masked_fill_(mask=~masks_whole[sample_idx].bool(), value=-2)

        cnt, attr, = -1, False
        for idx, head in enumerate(heads[1:]):
            if head == -2:
                break
            if head == -1:
                continue

            if len(splits) > 2 and idx + 1 >= splits[cnt+1] and cnt < len(splits) - 2:
                cnt += 1

            if ((len(splits) > 2 and (head < splits[cnt] or head >= splits[cnt+1])) or idx - head > 0) and rels[idx + 1] in origin4change:  # cross 'edu'

                rels[idx+1] = signals[idx+1]

                if rels[idx + 1] in [rel2id['cond']]:  # reverse
                    tmp_heads = heads.clone()
                    tmp_heads[:splits[cnt+1]] = 0
                    head_idx = [idx + 1]
                    tail_idx = (tmp_heads == idx + 1).nonzero()  # find tail index
                    if len(tail_idx) == 0:  # ring or fail
                        # unchange
                        tail_idx = [idx + 1]
                        head_idx = (heads == idx + 1).nonzero() if head_idx == tail_idx else head_idx
                    elif len(head_idx) != 0:
                        heads[tail_idx[0]] = 0
                        heads[head_idx[0]] = tail_idx[0]

                # special cases
                if word_lst[idx+1] == '好' and word_lst[idx] in ['你', '您']:  # reverse
                    tmp_heads = heads.clone()
                    tmp_heads[:splits[cnt+1]] = 0
                    tail_idx = (tmp_heads == idx + 1).nonzero()  # find tail index
                    if len(tail_idx) != 0:  
                        heads[tail_idx[0]] = 0
                        heads[idx + 1] = tail_idx[0]
                        rels[idx + 1] = rel2id['elbr']

            if not attr and rels[idx + 1] in [rel2id['obj']] and signals[idx+1] == rel2id['attr']:
                rels[idx+1] = signals[idx+1]
                attr = True

        rels.masked_fill_(heads == 0, 0)  # root
        heads[0] = 0
        heads[1:].masked_fill_(heads[1:] == -2, 0)

        heads_new_whole = torch.cat([heads_new_whole, heads.unsqueeze(0)])
        rels_new_whole = torch.cat([rels_new_whole, rels.unsqueeze(0)])
        signals_new_whole = torch.cat([signals_new_whole, signals.unsqueeze(0)])

    return heads_new_whole, rels_new_whole


def get_root(rel_preds, masks_whole):
    root_ids = []
    # for rel_pred, mask in zip(rel_preds, masks_whole):
    for rel_pred, mask in zip(rel_preds, masks_whole):
        try:
            root_idx = (((rel_pred == 0) * mask) != 0).nonzero()[0].item()
        except IndexError: # no root
            root_idx = 2
        root_ids.append(root_idx)

    return root_ids


# def eval_inter(inter_dataloader, masks_whole, root_ids, role_preds):
def eval_inter(inter_dataloader, masks_whole, root_ids):
    cnt = 0

    inter_heads_whole, inter_rels_whole, inter_masks_whole = torch.Tensor(), torch.Tensor(), torch.Tensor()
    inter_heads_preds, inter_rels_preds = torch.Tensor(), torch.Tensor()
    for turn, batch in enumerate(inter_dataloader):
        inputs, offsets, heads, rels, masks, speakers, signs = batch
        inter_head_preds = torch.zeros_like(heads, dtype=int)
        inter_rel_preds = torch.zeros_like(rels, dtype=int)

        inter_heads_whole = torch.cat([inter_heads_whole, heads])
        inter_rels_whole = torch.cat([inter_rels_whole, rels])
        inter_masks_whole = torch.cat([inter_masks_whole, masks])

        accum = 1
        for i, speakr in enumerate(speakers[1:]):
            seq_len = masks_whole[cnt].sum().item() + 1

            if speakr == speakers[i]:
                rel = signs[i][0]
            else:
                rel = signs[i][1]
            
            # rel = int(role_preds[cnt-turn].item() + 21)
            
            head_idx = int(root_ids[cnt] + accum) if i > 0 else root_ids[cnt]
            tail_idx = int(root_ids[cnt+1] + accum + seq_len)
            
            inter_head_preds[0][tail_idx] = head_idx
            inter_rel_preds[0][tail_idx] = rel

            cnt += 1
            accum += seq_len

        cnt += 1

        inter_heads_preds = torch.cat([inter_heads_preds, inter_head_preds])
        inter_rels_preds = torch.cat([inter_rels_preds, inter_rel_preds])

    arc_logits_correct = (inter_heads_preds == inter_heads_whole).long() * inter_masks_whole
    rel_logits_correct = (inter_rels_preds == inter_rels_whole).long() * arc_logits_correct
    # print(rel_logits_correct.sum() / inter_masks_whole.long().sum())
    # print(arc_logits_correct.sum() / inter_masks_whole.long().sum())
    return arc_logits_correct, rel_logits_correct, inter_masks_whole

def eval_all(model, test_iter, inter_iter):
    head_preds, rel_preds = torch.Tensor(), torch.Tensor()
    role_preds = torch.Tensor()
    heads_whole, rels_whole, masks_whole = torch.Tensor(), torch.Tensor(), torch.Tensor()
    for batch in test_iter:
        inputs, offsets, heads, rels, masks = batch
        
        inputs_cuda = {}
        for key, value in inputs.items():
            inputs_cuda[key] = value.cuda()
        inputs = inputs_cuda

        offsets, heads, rels, masks = to_cuda(data=(offsets, heads, rels, masks))

        # inputs, offsets, heads, rels, sem_roles, masks, speakers = batch
        # inputs_cuda = {}
        # for key, value in inputs.items():  
        #     inputs_cuda[key] = value.squeeze(0).cuda()  # rst's batch size is always 1
        # inputs = inputs_cuda

        # offsets, heads, rels, sem_roles, masks, speakers = to_cuda(data=(offsets, heads, rels, sem_roles, masks, speakers))
        # offsets, heads, rels, masks, speakers = offsets.squeeze(0), heads.squeeze(0), rels.squeeze(0), masks.squeeze(0), speakers.squeeze(0)

        with torch.no_grad():
            model.eval()
            inner_outputs, inter_outputs = model.predict(inputs, offsets)
        
        inner_head_logit, inner_rel_logit = [x.cpu() for x in inner_outputs]
        inner_head_logit[:, torch.arange(inner_head_logit.size()[1]), torch.arange(inner_head_logit.size()[2])] = -1e4

        index = inner_head_logit.argmax(-1).unsqueeze(2).unsqueeze(3).expand(-1, -1, -1, inner_rel_logit.shape[-1]) 
        inner_rel_logit = torch.gather(inner_rel_logit, dim=2, index=index).squeeze(2)

        head_preds = torch.cat([head_preds, inner_head_logit.argmax(-1).cpu()])
        rel_preds = torch.cat([rel_preds, inner_rel_logit.argmax(-1).cpu()])

        # role_preds = torch.cat([role_preds, inter_outputs.argmax(-1).cpu()])

        heads_whole = torch.cat([heads_whole, heads.cpu()])
        rels_whole = torch.cat([rels_whole, rels.cpu()])
        masks_whole = torch.cat([masks_whole, masks.cpu()])
    
    head_preds, rel_preds = postproc(head_preds, rel_preds, masks_whole)
    root_ids = get_root(rel_preds, masks_whole)
    # head_correct_inter, rel_correct_inter, inter_masks_whole = eval_inter(inter_iter, masks_whole, root_ids, role_preds)
    head_correct_inter, rel_correct_inter, inter_masks_whole = eval_inter(inter_iter, masks_whole, root_ids)
    
    # inner-EDU
    head_correct_syn = (head_preds == heads_whole).long() * masks_whole * (rels_whole < 21).long()
    rel_correct_syn = (rel_preds == rels_whole).long() * head_correct_syn
    inner_uas = head_correct_syn.sum() / (masks_whole * (rels_whole < 21).long()).sum()
    inner_las = rel_correct_syn.sum() / (masks_whole * (rels_whole < 21).long()).sum()
    # print(inner_uas, inner_las)

    # inter-EDU, including inner-utterance and inter-utterance
    head_correct_inner = (head_preds == heads_whole).long() * masks_whole * (rels_whole >= 21).long()
    rel_correct_inner = (rel_preds == rels_whole).long() * head_correct_inner

    inter_uas = (head_correct_inner.sum() + head_correct_inter.sum()) / ((rels_whole >= 21).long().sum() + inter_masks_whole.long().sum())
    inter_las = (rel_correct_inner.sum() + rel_correct_inter.sum()) / ((rels_whole >= 21).long().sum() + inter_masks_whole.long().sum())
    # print(inter_uas, inter_las)
    return (inner_uas, inner_las), (inter_uas, inter_las)

def eval_all_new(model, test_iter, inter_iter):
    head_preds, rel_preds = torch.Tensor(), torch.Tensor()
    heads_whole, rels_whole, masks_whole = torch.Tensor(), torch.Tensor(), torch.Tensor()
    # inter_arc_logits, inter_rel_logits = torch.Tensor(), torch.Tensor()
    # inter_heads_whole, inter_rels_whole, inter_masks_whole = torch.Tensor(), torch.Tensor(), torch.Tensor()
    h_inter_cnt, r_inter_cnt, inter_cnt = 0, 0, 0
    for batch in test_iter:
        inputs, offsets, heads, rels, masks, inter_heads, inter_rels, inter_masks, speakers = batch
        if torch.cuda.is_available():
            inputs_cuda = {}
            for key, value in inputs.items():  
                inputs_cuda[key] = value.squeeze(0).cuda()  # rst's batch size is always 1
            inputs = inputs_cuda

            offsets, heads, rels, masks, speakers = to_cuda(data=(offsets, heads, rels, masks, speakers))
            inter_heads, inter_rels, inter_masks = to_cuda(data=(inter_heads, inter_rels, inter_masks))
        offsets, heads, rels, masks, speakers = offsets.squeeze(0), heads.squeeze(0), rels.squeeze(0), masks.squeeze(0), speakers.squeeze(0)
        inter_heads, inter_rels, inter_masks = inter_heads.squeeze(0), inter_rels.squeeze(0), inter_masks.squeeze(0)

        with torch.no_grad():
            model.eval()
            inner_outputs, inter_outputs = model.predict(inputs, offsets)
        
        # inner
        inner_head_logit, inner_rel_logit = [x.cpu() for x in inner_outputs]
        inner_head_logit[:, torch.arange(inner_head_logit.size()[1]), torch.arange(inner_head_logit.size()[2])] = -1e4
        index = inner_head_logit.argmax(-1).unsqueeze(2).unsqueeze(3).expand(-1, -1, -1, inner_rel_logit.shape[-1]) 
        inner_rel_logit = torch.gather(inner_rel_logit, dim=2, index=index).squeeze(2)

        head_preds = torch.cat([head_preds, inner_head_logit.argmax(-1).cpu()])
        rel_preds = torch.cat([rel_preds, inner_rel_logit.argmax(-1).cpu()])

        heads_whole = torch.cat([heads_whole, heads.cpu()])
        rels_whole = torch.cat([rels_whole, rels.cpu()])
        masks_whole = torch.cat([masks_whole, masks.cpu()])

        # inter
        inter_arc_logit, inter_rel_logit = [x.cpu() for x in inter_outputs]
        inter_arc_logit[:, torch.arange(inter_arc_logit.size()[1]), torch.arange(inter_arc_logit.size()[2])] = -1e4
        h_correct_inter = (inter_arc_logit.argmax(-1) == inter_heads.cpu()).long() * inter_masks.cpu()
        r_correct_inter = (inter_rel_logit.argmax(-1) == inter_rels.cpu()).long() * h_correct_inter
        h_inter_cnt += h_correct_inter.cpu().sum().item()
        r_inter_cnt += r_correct_inter.cpu().sum().item()
        inter_cnt += inter_masks.cpu().long().sum()
        # inter_arc_logits = torch.cat([inter_arc_logits, inter_arc_logit.cpu()])
        # inter_rel_logits = torch.cat([inter_rel_logits, inter_rel_logit.cpu()])

        # inter_heads_whole =  torch.cat([inter_heads_whole, inter_heads.cpu()])
        # inter_rels_whole =  torch.cat([inter_rels_whole, inter_rels.cpu()])
        # inter_masks_whole = torch.cat([inter_masks_whole, inter_masks.cpu()])
    
    # inner-EDU
    head_correct_syn = (head_preds == heads_whole).long() * masks_whole * (rels_whole < 21).long()
    rel_correct_syn = (rel_preds == rels_whole).long() * head_correct_syn
    inner_uas = head_correct_syn.sum() / (masks_whole * (rels_whole < 21).long()).sum()
    inner_las = rel_correct_syn.sum() / (masks_whole * (rels_whole < 21).long()).sum()

    # inter-EDU, including inner-utterance and inter-utterance
    head_correct_inner = (head_preds == heads_whole).long() * masks_whole * (rels_whole >= 21).long()
    rel_correct_inner = (rel_preds == rels_whole).long() * head_correct_inner

    # head_correct_inter = (inter_arc_logits.argmax(-1) == inter_heads_whole).long() * inter_masks_whole
    # rel_correct_inter = (inter_rel_logits.argmax(-1) == inter_rels_whole).long() * head_correct_inter

    inter_uas = (head_correct_inner.sum() + h_inter_cnt) / ((rels_whole >= 21).long().sum() + inter_cnt)
    inter_las = (rel_correct_inner.sum() + r_inter_cnt) / ((rels_whole >= 21).long().sum() + inter_cnt)
    return (inner_uas, inner_las), (inter_uas, inter_las)

if __name__ == '__main__':
    # transformers.logging.set_verbosity_error() # only report errors.

    # load_config
    with open('config/inter.yaml', 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    CFG = DataClassUnpack.instantiate(CFG, config)

    tokenizer = AutoTokenizer.from_pretrained(CFG.plm)
    special_tokens_dict = {'additional_special_tokens': ['[Q]', '[A]']}
    num_added_tokens = tokenizer.add_tokens(special_tokens_dict['additional_special_tokens'])
    print('token added', num_added_tokens, 'token')
    CFG.tokenizer = tokenizer

    # model = DepParser(CFG)
    model = InterParser(CFG)
    model.load_state_dict(torch.load('ckpt/inter_par_59.pt'))
    model = model.cuda()

    # diagdataset_test = DialogEvalDataset(CFG, CFG.test_file, tokenizer)
    # test_iter = DataLoader(diagdataset_test, batch_size=1, shuffle=False)

    # uas, las = eval_eduwise(model, CFG, test_iter)
    # print(uas, las)

    # diagdataset_test = DialogUttrDataset(CFG, CFG.test_file, tokenizer, test=True)
    diagdataset_test = DialogUttrInterDataset(CFG, CFG.test_file, tokenizer, test=True)
    test_iter = DataLoader(diagdataset_test, batch_size=1, shuffle=False)
    # uas, las = eval_uttrwise(model, CFG, test_iter)
    # print(uas, las)

    inter_dataset = InterDataset(CFG)
    inter_iter = DataLoader(inter_dataset, batch_size=1)

    (inner_uas, inner_las), (inter_uas, inter_las) = eval_all_new(model, test_iter, inter_iter)
    print((inner_uas, inner_las), (inter_uas, inter_las))