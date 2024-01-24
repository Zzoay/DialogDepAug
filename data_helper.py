
from typing import *
import json
from itertools import chain

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset

from constant import rel2id, punct_lst, relsyn2id


class Dependency():
    def __init__(self, idx, word, head, rel):
        self.id = idx
        self.word = word
        self.head = head
        self.rel = rel

    def __str__(self):
        # example:  1	上海	_	NR	NR	_	2	nn	_	_
        values = [str(self.idx), self.word, "_", "_", "_", "_", str(self.head), self.rel, "_", "_"]
        return '\t'.join(values)

    def __repr__(self):
        return f"({self.word}, {self.head}, {self.rel})"
    

def load_annoted(data_file, sparse=False):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    sample_lst:List[List[Dependency]] = []
    
    for i, d in enumerate(data):
        rel_dct = {}
        for tripple in d['relationship']:
            head, rel, tail = tripple
            head_uttr_idx, head_word_idx = [int(x) for x in head.split('-')]
            tail_uttr_idx, tail_word_idx = [int(x) for x in tail.split('-')]
            if head_uttr_idx != tail_uttr_idx:
                continue
            
            if not rel_dct.get(head_uttr_idx, None):
                rel_dct[head_uttr_idx] = {tail_word_idx: [head_word_idx, rel]}
            else:
                rel_dct[head_uttr_idx][tail_word_idx] = [head_word_idx, rel]
            
        for item in d['dialog']:
            turn = item['turn']
            utterance = item['utterance']
            dep_lst:List[Dependency] = []
            
            for word_idx, word in enumerate(utterance.split(' ')):
                head_word_idx, rel = rel_dct[turn].get(word_idx + 1, [word_idx, 'adjct'])  # some word annoted missed, padded with last word and 'adjct'
                if sparse and rel != 'root' and rel in relsyn2id.keys(): 
                    head_word_idx = -1
                dep_lst.append(Dependency(word_idx + 1, word, head_word_idx, rel))  # start from 1
            
            sample_lst.append(dep_lst)
        
    return sample_lst


class DialogUttrDataset(Dataset):
    def __init__(self, cfg, data_file, tokenizer, test=False):
        self.cfg = cfg
        self.test = test
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.inputs, self.offsets, self.heads, self.rels, self.masks = self.read_data()

    def read_data(self):
        inputs, offsets = [], []
        tags, heads, rels, masks = [], [], [], []
        
        if self.test:
            gen = load_annoted(self.data_file)
        else:
            gen = chain(load_annoted(self.data_file), load_annoted('aug-data/aug_rpl_utterance.json', sparse=True))
        for deps in gen:
            seq_len = len(deps)
            max_len = self.cfg.dialog_max_length

            word_lst = [] 
            head_tokens = np.zeros(max_len, dtype=np.int64)  # same as root index is 0, constrainting by mask 
            rel_tokens = np.zeros(max_len, dtype=np.int64)
            mask_tokens = np.zeros(max_len, dtype=np.int64)
            for i, dep in enumerate(deps):
                if i == seq_len or i + 1== max_len:
                    break

                word_lst.append(dep.word)

                if dep.head != -1 and dep.rel in rel2id.keys() and dep.head + 1 < max_len:
                    head_tokens[i+1] = dep.head
                    mask_tokens[i+1] = 1
                    rel_tokens[i+1] = rel2id[dep.rel]

            tokenized = self.tokenizer.encode_plus(word_lst, 
                                              padding='max_length', 
                                              truncation=True,
                                              max_length=max_len, 
                                              return_offsets_mapping=True, 
                                              return_tensors='pt',
                                              is_split_into_words=True)
            inputs.append({"input_ids": tokenized['input_ids'][0],
                           "token_type_ids": tokenized['token_type_ids'][0],
                           "attention_mask": tokenized['attention_mask'][0]
                          })

            sentence_word_idx = []
            for idx, (start, end) in enumerate(tokenized.offset_mapping[0][1:]):
                if start == 0 and end != 0:
                    sentence_word_idx.append(idx)

            if len(sentence_word_idx) < max_len - 1:
                sentence_word_idx.extend([0]* (max_len - 1 - len(sentence_word_idx)))
            offsets.append(torch.as_tensor(sentence_word_idx))

            heads.append(head_tokens)
            rels.append(rel_tokens)
            masks.append(mask_tokens)
                    
        return inputs, offsets, heads, rels, masks

    def __getitem__(self, idx):
        return self.inputs[idx], self.offsets[idx], self.heads[idx], self.rels[idx], self.masks[idx]
    
    def __len__(self):
        return len(self.rels)
    
def load_annoted_inter(data_file, sparse=False):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    sample_lst:List[List[Dependency]] = []
    speakers_lst, inters_lst = [], []
    
    for d in data:
        rel_dct = {}
        speakers = []
        inters = []
        for tripple in d['relationship']:
            head, rel, tail = tripple
            head_uttr_idx, head_word_idx = [int(x) for x in head.split('-')]
            tail_uttr_idx, tail_word_idx = [int(x) for x in tail.split('-')]
            if head_uttr_idx != tail_uttr_idx:
                inters.append([head_word_idx + head_uttr_idx * 80, rel, tail_word_idx + tail_uttr_idx * 80])
                continue

            if not rel_dct.get(head_uttr_idx, None):
                rel_dct[head_uttr_idx] = {tail_word_idx: [head_word_idx, rel]}
            else:
                rel_dct[head_uttr_idx][tail_word_idx] = [head_word_idx, rel]
        
        sample = []
        for item in d['dialog']:
            turn = item['turn']
            if turn > 25:
                break
            utterance = item['utterance']
            dep_lst:List[Dependency] = []

            speakers.append(item['speaker'])
            
            for word_idx, word in enumerate(utterance.split(' ')):
                head_word_idx, rel = rel_dct[turn].get(word_idx + 1, [word_idx, 'adjct'])  # some word annoted missed, padded with last word and 'adjct'
                if sparse and rel != 'root' and rel in relsyn2id.keys(): 
                    head_word_idx = -1
                dep_lst.append(Dependency(word_idx + 1, word, head_word_idx, rel))  # start from 1
            sample.append(dep_lst)
            
        sample_lst.append(sample)
        speakers_lst.append(speakers)
        inters_lst.append(inters)
        
    return sample_lst, inters_lst, speakers_lst


class DialogUttrInterDataset(Dataset):
    def __init__(self, cfg, data_file, tokenizer, test=False):
        self.cfg = cfg
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.test = test
        self.inputs, self.offsets, self.heads, self.rels, self.masks, \
            self.inter_heads, self.inter_rels, self.inter_masks, self.speakers = self.read_data()
        
    def read_data(self):
        inputs, offsets = [], []
        heads, rels, masks = [], [], []
        inter_heads, inter_rels, inter_masks = [], [], []
        sem_roles = []
        speakers = []
        max_len = self.cfg.dialog_max_length

        sample_lst, inters_lst, speakers_lst = load_annoted_inter(self.data_file)

        sample_lst_aug, inters_lst_aug, speakers_lst_aug = [], [], []
        data_files = self.cfg.aug_files
        for file_path in data_files:
            sa_lst, in_lst, sp_lst = load_annoted_inter(file_path, sparse=True)
            sample_lst_aug.extend(sa_lst)
            inters_lst_aug.extend(in_lst)
            speakers_lst_aug.extend(sp_lst)

        for i, inters in enumerate(inters_lst):
            num_turn = len(speakers_lst[i])
            h, r, t = [x[0] for x in inters], [x[1] for x in inters], [x[2] for x in inters]
            inter_head = torch.zeros(num_turn*max_len).long()
            inter_rel = torch.zeros(num_turn*max_len).long()
            inter_mask = torch.zeros(num_turn*max_len).long()
            for i in range(len(h)):
                if h[i] < num_turn * max_len and t[i] < num_turn * max_len:
                    inter_head[t[i]] = h[i]
                    inter_rel[t[i]] = rel2id[r[i]] - 21
                    inter_mask[t[i]] = 1
            inter_heads.append(inter_head); inter_rels.append(inter_rel); inter_masks.append(inter_mask)
        
        for sps in speakers_lst:
            speaker_tokens = np.zeros(len(sps), dtype=np.int64)
            for i, s in enumerate(sps):
                speaker_tokens[i] = 1 if s == '[A]' else 0
            speakers.append(speaker_tokens)
        
        for sample, sps in zip(sample_lst, speakers_lst):
            input, offset = {"input_ids": torch.Tensor(), "token_type_ids": torch.Tensor(), "attention_mask":torch.Tensor()}, []
            head, rel, mask = [], [], []
            for i, deps in enumerate(sample):
                seq_len = len(deps)

                word_lst = [f'[{sps[i]}]'] 
                head_tokens = torch.zeros(max_len).int()  # same as root index is 0, constrainting by mask 
                rel_tokens = torch.zeros(max_len).int()
                mask_tokens = torch.zeros(max_len).int()
                
                for i, dep in enumerate(deps):
                    if i == seq_len or i + 1== max_len:
                        break

                    word_lst.append(dep.word)

                    if dep.head != -1 and dep.rel in rel2id.keys() and dep.head + 1 < max_len:
                        head_tokens[i+1] = dep.head
                        mask_tokens[i+1] = 1
                        rel_tokens[i+1] = rel2id[dep.rel]

                tokenized = self.tokenizer.encode_plus(word_lst, 
                                                padding='max_length', 
                                                truncation=True,
                                                max_length=max_len, 
                                                return_offsets_mapping=True, 
                                                return_tensors='pt',
                                                is_split_into_words=True)
                input["input_ids"] = torch.cat([input['input_ids'], tokenized['input_ids']], dim=0).long()
                input["token_type_ids"] = torch.cat([input['token_type_ids'], tokenized['token_type_ids']], dim=0).long()
                input["attention_mask"] = torch.cat([input['attention_mask'], tokenized['attention_mask']], dim=0).long()

                sentence_word_idx = []
                for idx, (start, end) in enumerate(tokenized.offset_mapping[0][2:]):
                    if start == 0 and end != 0:
                        sentence_word_idx.append(idx)

                if len(sentence_word_idx) < max_len - 1:
                    sentence_word_idx.extend([0]* (max_len - 1 - len(sentence_word_idx)))

                offset.append(torch.as_tensor(sentence_word_idx))

                head.append(head_tokens)
                rel.append(rel_tokens)
                mask.append(mask_tokens)
            
            offset = torch.cat([h.unsqueeze(0) for h in offset], dim=0).long()
            head = torch.cat([h.unsqueeze(0) for h in head], dim=0).long()
            rel = torch.cat([h.unsqueeze(0) for h in rel], dim=0).long()
            mask = torch.cat([h.unsqueeze(0) for h in mask], dim=0).long()
            
            inputs.append(input);offsets.append(offset);heads.append(head);rels.append(rel);masks.append(mask)
                    
        return inputs, offsets, heads, rels, masks, inter_heads, inter_rels, inter_masks, speakers

    def __getitem__(self, idx):
        return self.inputs[idx], self.offsets[idx], self.heads[idx], self.rels[idx], self.masks[idx], \
            self.inter_heads[idx], self.inter_rels[idx], self.inter_masks[idx], self.speakers[idx]
    
    def __len__(self):
        return len(self.rels)
    

def load_conll(data_file: str):
    # sentence:List[Dependency] = [Dependency(0, '[root]', -1, '_')]
    sentence = []

    with open(data_file, 'r', encoding='utf-8') as f:
        # data example: 1	上海	_	NR	NR	_	2	nn	_	_
        for line in f.readlines():
            toks = line.strip().split('\t')
            if len(toks) == 0 or line == '\n':
                yield sentence
                # sentence = [Dependency(0, '[root]', -1, '_')]
                sentence = []
            else:
                dep = Dependency(toks[0], toks[1], int(toks[6]), toks[7])
                sentence.append(dep)
        yield sentence


class SynDataset(Dataset):
    def __init__(self, cfg, tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.inputs, self.offsets, self.heads, self.rels, self.masks = self.read_data()
        
    def read_data(self):
        inputs, offsets = [], []
        tags, heads, rels, masks = [], [], [], []
        
        for domain in self.cfg.syn_domains:
            file = f'{self.cfg.syn_train_path}/{domain}-Train-full.conll'
            
            for deps in tqdm(load_conll(file)):
                seq_len = len(deps)
                
                word_lst = [] 
                head_tokens = np.zeros(self.cfg.syn_max_length, dtype=np.int64)  # same as root index is 0, constrainting by mask 
                rel_tokens = np.zeros(self.cfg.syn_max_length, dtype=np.int64)
                mask_tokens = np.zeros(self.cfg.syn_max_length, dtype=np.int64)
                for i, dep in enumerate(deps):
                    if i == self.cfg.syn_max_length - 1:
                        break
                        
                    word_lst.append(dep.word)
                      
                    if dep.head != -1 and dep.rel in rel2id.keys() and dep.head + 1 < self.cfg.syn_max_length:
                        head_tokens[i+1] = dep.head
                        mask_tokens[i+1] = 1
                        rel_tokens[i+1] = rel2id[dep.rel]

                tokenized = self.tokenizer.encode_plus(word_lst, 
                                                       padding='max_length', 
                                                       truncation=True,
                                                       max_length=self.cfg.syn_max_length, 
                                                       return_offsets_mapping=True, 
                                                       return_tensors='pt',
                                                       is_split_into_words=True)
                inputs.append({"input_ids": tokenized['input_ids'][0],
                               "token_type_ids": tokenized['token_type_ids'][0],
                               "attention_mask": tokenized['attention_mask'][0]
                              })
                
                sentence_word_idx = []
                for idx, (start, end) in enumerate(tokenized.offset_mapping[0][1:]):
                    if start == 0 and end != 0:
                        sentence_word_idx.append(idx)
                if len(sentence_word_idx) < self.cfg.syn_max_length - 1:
                    sentence_word_idx.extend([0]* (self.cfg.syn_max_length - 1 - len(sentence_word_idx)))
                offsets.append(torch.as_tensor(sentence_word_idx))
                
                heads.append(head_tokens)
                rels.append(rel_tokens)
                masks.append(mask_tokens)

                if self.cfg.debug:
                    return inputs, offsets, heads, rels, masks
                    
        return inputs, offsets, heads, rels, masks

    def __getitem__(self, idx):
        return self.inputs[idx], self.offsets[idx], self.heads[idx], self.rels[idx], self.masks[idx]
    
    def __len__(self):
        return len(self.rels)


def load_diag4eval(data_file, data_ids=None):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if data_ids is None:
        data_ids = list(range(len(data)))
        
    sample_lst:List = []
    
    for i, d in enumerate(data):
        if i not in data_ids:
            continue

        rel_dct = {}
        for tripple in d['relationship']:
            head, rel, tail = tripple
            head_uttr_idx, head_word_idx = [int(x) for x in head.split('-')]
            tail_uttr_idx, tail_word_idx = [int(x) for x in tail.split('-')]
            
            if rel == 'root' and head_uttr_idx != 0: # ignore root
                continue
                 
            if not rel_dct.get(tail_uttr_idx, None):
                rel_dct[tail_uttr_idx] = {tail_word_idx: [head, rel]}
            else:
                rel_dct[tail_uttr_idx][tail_word_idx] = [head, rel]
        
        uttr_lst, role_lst = [], []
        for item in d['dialog']:
            turn = item['turn']
            utterance = item['utterance']
            role = '[ans]' if item['speaker'] == 'A' else '[qst]'
            word_lst = utterance.split(' ')

            uttr_lst.append(word_lst)
            role_lst.append(role)

        sent_lens_accum = [0]
        for i, item in enumerate(d['dialog']):
            utterance = item['utterance']
            sent_lens_accum.append(sent_lens_accum[i] + len(utterance.split(' ')))
        
        heads, rels = [], []
        for item in d['dialog']:
            turn = item['turn']
            utterance = item['utterance']
            
            for word_idx, word in enumerate(utterance.split(' ')):
                tail2head = rel_dct.get(turn, {1: [f'{turn}-{word_idx}', 'adjct']})
                head, rel = tail2head.get(word_idx + 1, [f'{turn}-{word_idx}', 'adjct'])  # some word annoted missed, padded with last word and 'adjct'
                head_uttr_idx, head_word_idx = [int(x) for x in head.split('-')]
                
                heads.append(head_word_idx + sent_lens_accum[head_uttr_idx])   # add with accumulated length
                rels.append(rel)
         
        sample_lst.append([uttr_lst, heads, rels, role_lst])
        
    return sample_lst


class DialogEvalDataset(Dataset):
    def __init__(self, cfg, data_file, tokenizer, data_ids=None):
        self.cfg = cfg
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.data_ids = data_ids
        self.inputs, self.offsets, self.heads, self.rels, self.masks, self.roles = self.read_data()
        
    def read_data(self):
        inputs, offsets = [], []
        roles, heads, rels, masks = [], [], [], []
        
        for edu_lst, head_lst, rel_lst, role_lst in tqdm(load_diag4eval(self.data_file, self.data_ids)):
            whole_max_len = 1024
            head_tokens = np.zeros(whole_max_len + 1, dtype=np.int64)
            rel_tokens = np.zeros(whole_max_len + 1, dtype=np.int64)
            mask_tokens = np.zeros(whole_max_len + 1, dtype=np.int64)
            role_tokens = np.zeros(len(role_lst) + 1, dtype=np.int64)
            for i, (head, rel) in enumerate(zip(head_lst, rel_lst)):
                if i == whole_max_len:
                    break

                if head != -1 and rel in rel2id.keys() and head + 1 < whole_max_len:
                    head_tokens[i+1] = head
                    mask_tokens[i+1] = 1
                    rel_tokens[i+1] = rel2id[rel]
                    
            for i, role in enumerate(role_lst):
                role_tokens[i+1] = 1 if role == '[ans]' else 2

            tokenized = self.tokenizer(edu_lst, 
                                        padding='max_length', 
                                        truncation=True,
                                        max_length=self.cfg.dialog_max_length, 
                                        return_offsets_mapping=True, 
                                        return_tensors='pt',
                                        is_split_into_words=True)
            
            one_offsets = []
            for offset_mapping in tokenized.offset_mapping:
                sentence_word_idx = []
                for idx, (start, end) in enumerate(offset_mapping[1:]):
                    if start == 0 and end != 0:
                        sentence_word_idx.append(idx)
                if len(sentence_word_idx) < self.cfg.dialog_max_length - 1:
                    sentence_word_idx.extend([0]* (self.cfg.dialog_max_length - 1 - len(sentence_word_idx)))
                one_offsets.append(sentence_word_idx)
            
            inputs.append({"input_ids": tokenized['input_ids'],
                           "token_type_ids": tokenized['token_type_ids'],
                           "attention_mask": tokenized['attention_mask']
                            })
            offsets.append(torch.tensor(one_offsets))
            heads.append(head_tokens)
            rels.append(rel_tokens)
            masks.append(mask_tokens)
            roles.append(role_tokens)
                    
        return inputs, offsets, heads, rels, masks, roles

    def __getitem__(self, idx):
        return self.inputs[idx], self.offsets[idx], self.heads[idx], self.rels[idx], self.masks[idx], self.roles[idx]
    
    def __len__(self):
        return len(self.rels)

def load_signal(data_file: str, return_two=False):
    sentence:List[Dependency] = []

    with open(data_file, 'r', encoding='utf-8') as f:
        # data example: 1	上海	_	NR	NR	_	2	nn	_	_
        for line in f.readlines():
            toks = line.split()
            if len(toks) == 0 and len(sentence) != 0:
                yield sentence
                sentence = []
            elif len(toks) == 10:                
                if return_two:
                    sentence.append([int(toks[2]), int(toks[3])])
                else:
                    sentence.append(int(toks[2]))

def load_inter(data_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    signal_iter = load_signal('data/mlm/diag_test.conll', return_two=True)

    sample_lst:List[List[Dependency]] = []
    # for d, pred_signals in tqdm(zip(data, load_codt_signal('../prompt_based/diag_test.conll', idx=3))):
    for d in data:
        rel_dct = {}
        for tripple in d['relationship']:
            head, rel, tail = tripple
            head_uttr_idx, head_word_idx = [int(x) for x in head.split('-')]
            tail_uttr_idx, tail_word_idx = [int(x) for x in tail.split('-')]
            
            if rel == 'root' and head_uttr_idx != 0: # ignore root
                continue
                 
            if not rel_dct.get(tail_uttr_idx, None):
                rel_dct[tail_uttr_idx] = {tail_word_idx: [head, rel]}
            else:
                rel_dct[tail_uttr_idx][tail_word_idx] = [head, rel]
                
        sent_lens_accum = [1]
        for i, item in enumerate(d['dialog']):
            utterance = item['utterance']
            sent_lens_accum.append(sent_lens_accum[i] + len(utterance.split(' ')) + 1)
        sent_lens_accum[0] = 0
        
        dep_lst:List[Dependency] = []
        role_lst:List[str] = []
        weak_signal = []
        for item in d['dialog']:
            turn = item['turn']
            utterance = item['utterance']

            pred_signals = next(signal_iter)

            role = '[ans]' if item['speaker'] == 'A' else '[qst]'
            dep_lst.append(Dependency(sent_lens_accum[turn], role, -1, '_'))  
            
            tmp_signal = []
            for word_idx, word in enumerate(utterance.split(' ')):
                tail2head = rel_dct.get(turn, {1: [f'{turn}-{word_idx}', 'adjct']})
                head, rel = tail2head.get(word_idx + 1, [f'{turn}-{word_idx}', 'adjct'])  # some word annoted missed, padded with last word and 'adjct'
                head_uttr_idx, head_word_idx = [int(x) for x in head.split('-')]
                
                # only parse cross-utterance
                if turn != head_uttr_idx:
                    dep_lst.append(Dependency(sent_lens_accum[turn] + word_idx + 1, word, sent_lens_accum[head_uttr_idx] + head_word_idx, rel))  # add with accumulated length
                else:
                    dep_lst.append(Dependency(sent_lens_accum[turn] + word_idx + 1, word, -1, '_')) 

                try:
                    signal1, signal2 = pred_signals[i]
                except IndexError:
                    signal1, signal2 = pred_signals[len(pred_signals) - 1]
                
                tmp_signal = [signal1, signal2]
                
                # if word in weak_signal_dct.keys():
                #     tmp_signal.append(weak_signal_dct[word])

            if len(tmp_signal) != 0:
                # weak_signal.append(tmp_signal[-1])  # choose the last
                weak_signal.append(tmp_signal)  # choose the last
            else:
                weak_signal.append(-1)
            role_lst.append(item['speaker'])        
        sample_lst.append([dep_lst, role_lst, weak_signal])
        
    return sample_lst


class InterDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.inputs, self.offsets, self.heads, self.rels, self.masks, self.speakers, self.signs = self.read_data()
        
    def read_data(self):
        inputs, offsets = [], []
        tags, heads, rels, masks, speakers, signs = [], [], [], [], [], []
                
        for idx, (deps, roles, sign) in enumerate(load_inter(self.cfg.test_file)):
            seq_len = len(deps)
            signs.append(sign)

            word_lst = [] 
            head_tokens = np.zeros(1024, dtype=np.int64)  # same as root index is 0, constrainting by mask 
            rel_tokens = np.zeros(1024, dtype=np.int64)
            mask_tokens = np.zeros(1024, dtype=np.int64)
            for i, dep in enumerate(deps):
                if i == seq_len or i + 1== 1024:
                    break

                word_lst.append(dep.word)
                
                if int(dep.head) == -1 or int(dep.head) + 1 >= 1024:
                    head_tokens[i+1] = 0
                    mask_tokens[i+1] = 0
                else:
                    head_tokens[i+1] = int(dep.head)
                    mask_tokens[i+1] = 1
#                     head_tokens[i] = dep.head if dep.head != '_' else 0
                rel_tokens[i+1] = rel2id.get(dep.rel, 0)

            tokenized = self.cfg.tokenizer.encode_plus(word_lst, 
                                              padding='max_length', 
                                              truncation=True,
                                              max_length=1024, 
                                              return_offsets_mapping=True, 
                                              return_tensors='pt',
                                              is_split_into_words=True)
            inputs.append({"input_ids": tokenized['input_ids'][0],
                          "token_type_ids": tokenized['token_type_ids'][0],
                           "attention_mask": tokenized['attention_mask'][0]
                          })

            sentence_word_idx = []
            for idx, (start, end) in enumerate(tokenized.offset_mapping[0][1:]):
                if start == 0 and end != 0:
                    sentence_word_idx.append(idx)

            if len(sentence_word_idx) < 1024 - 1:
                sentence_word_idx.extend([0]* (1024 - 1 - len(sentence_word_idx)))
            offsets.append(torch.as_tensor(sentence_word_idx))

            heads.append(head_tokens)
            rels.append(rel_tokens)
            masks.append(mask_tokens)
            speakers.append(roles)
                    
        return inputs, offsets, heads, rels, masks, speakers, signs

    def __getitem__(self, idx):
        return self.inputs[idx], self.offsets[idx], self.heads[idx], self.rels[idx], self.masks[idx], self.speakers[idx], self.signs[idx]
    
    def __len__(self):
        return len(self.rels)

