
from typing import *
import random

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel

from utils import arc_rel_loss, inter_uttr_loss, arc_rel_loss_inter


class DepParser(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.cfg = cfg
        self.device = 'cuda' if cfg.cuda else 'cpu'

        arc_hidden_size:int = cfg.hidden_size
        rel_hidden_size:int = cfg.hidden_size
        self.total_num = int((arc_hidden_size + rel_hidden_size) / 100)
        self.arc_num = int(arc_hidden_size / 100)
        self.rel_num = int(rel_hidden_size / 100)

        self.encoder = AutoModel.from_pretrained(cfg.plm)
        self.encoder.resize_token_embeddings(len(cfg.tokenizer))

        self.gru = nn.GRU(input_size=self.encoder.config.hidden_size, hidden_size=arc_hidden_size,
                                      bidirectional=True, batch_first=True)     

        self.mlp_arc_dep =  NonLinear(in_features=arc_hidden_size * 2, 
                                            out_features=arc_hidden_size, 
                                            activation=nn.LeakyReLU(0.1))
        self.mlp_arc_head = NonLinear(in_features=arc_hidden_size * 2, 
                                            out_features=arc_hidden_size, 
                                            activation=nn.LeakyReLU(0.1))
        self.mlp_rel_dep = NonLinear(in_features=rel_hidden_size * 2, 
                                            out_features=rel_hidden_size, 
                                            activation=nn.LeakyReLU(0.1))     
        self.mlp_rel_head = NonLinear(in_features=rel_hidden_size * 2, 
                                            out_features=rel_hidden_size, 
                                            activation=nn.LeakyReLU(0.1))

        self.arc_biaffine = Biaffine(arc_hidden_size, arc_hidden_size, 1)
        self.rel_biaffine = Biaffine(rel_hidden_size, rel_hidden_size, 35)

        self.dropout = nn.Dropout(cfg.dropout)

    def feat(self, inputs):
        length = torch.sum(inputs["attention_mask"], dim=-1) - 2
        
        feats, *_ = self.encoder(**inputs, return_dict=False)   # batch_size, seq_len (tokenized), plm_hidden_size
           
        # remove [SEP]
        word_cls = feats[:, :1]
        char_input = torch.narrow(feats, 1, 1, feats.size(1) - 2)
        return word_cls, char_input, length
        
    def char2word(self, char_feat, offsets):
        word_idx = offsets.unsqueeze(-1).expand(-1, -1, char_feat.shape[-1])  # expand to the size of char feat
        word_feat = torch.gather(char_feat, dim=1, index=word_idx)  # embeddings of first char in each word
        return word_feat

    def parse_dep(self, input_pckg):
        inputs, offsets, heads, rels, masks = input_pckg
        cls_feat, char_feat, word_len = self.feat(inputs)
        
        word_feat = self.char2word(char_feat, offsets)
        word_cls_feat = torch.cat([cls_feat, word_feat], dim=1)
        feats = self.dropout(word_cls_feat)  # syn_batch_size, seq_len, hidden_size 

        feats = self.dropout(self.gru(feats)[0])

        arc_dep = self.dropout(self.mlp_arc_dep(feats))
        arc_head = self.dropout(self.mlp_arc_dep(feats))

        rel_dep = self.dropout(self.mlp_rel_dep(feats))
        rel_head = self.dropout(self.mlp_rel_head(feats))

        arc_logit = self.arc_biaffine(arc_dep, arc_head)  
        arc_logit = arc_logit.squeeze(3)    # batch_size, seq_len, seq_len

        rel_logit = self.rel_biaffine(rel_dep, rel_head)  # batch_size, seq_len, seq_len, num_rels

        loss = arc_rel_loss(arc_logit, rel_logit, heads, rels, masks)
        return arc_logit, rel_logit, loss

    # training
    def forward(self, dialog_pckg, syn_pckg, epoch):
        dialog_arc_logit, dialog_rel_logit, dialog_loss = self.parse_dep(dialog_pckg)
        if not syn_pckg:
            return ((dialog_arc_logit, dialog_rel_logit), (None, None)), dialog_loss

        # with syntax treebank
        syn_arc_logit, syn_rel_logit, syn_loss = self.parse_dep(syn_pckg)
        if epoch > 2:
            # loss = self.cfg.alpha * dialog_loss + (1 - self.cfg.alpha) * syn_loss
            loss = dialog_loss**2
        else:
            # loss = 0 * dialog_loss + 1 * syn_loss
            loss = syn_loss**2
        # loss = self.cfg.alpha* dialog_loss + (1 - self.cfg.alpha) * syn_loss
        return ((dialog_arc_logit, dialog_rel_logit), (syn_arc_logit, syn_rel_logit)), loss
        # return ((None, None), (syn_arc_logit, syn_rel_logit)), syn_loss

    def predict(self, inputs, offsets):
        cls_feat, char_feat, word_len = self.feat(inputs)

        word_feat = self.char2word(char_feat, offsets)
        word_cls_feat = torch.cat([cls_feat, word_feat], dim=1)
        inner_feats = word_cls_feat     # turns, seq_len, hidden_size 

        # inner
        inner_feats = self.gru(inner_feats)[0]

        arc_dep = self.mlp_arc_dep(inner_feats)
        arc_head = self.mlp_arc_dep(inner_feats)
        rel_dep = self.mlp_rel_dep(inner_feats)
        rel_head = self.mlp_rel_head(inner_feats)

        arc_logit = self.arc_biaffine(arc_dep, arc_head)  
        inner_arc_logit = arc_logit.squeeze(3)    # batch_size, seq_len, seq_len
        inner_rel_logit = self.rel_biaffine(rel_dep, rel_head)  # batch_size, seq_len, seq_len, num_rels

        return (inner_arc_logit, inner_rel_logit), (None, None)


class InterParser(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.cfg = cfg
        self.device = 'cuda' if cfg.cuda else 'cpu'

        arc_hidden_size:int = cfg.hidden_size
        rel_hidden_size:int = cfg.hidden_size
        self.total_num = int((arc_hidden_size + rel_hidden_size) / 100)
        self.arc_num = int(arc_hidden_size / 100)
        self.rel_num = int(rel_hidden_size / 100)

        self.encoder = AutoModel.from_pretrained(cfg.plm)
        self.encoder.resize_token_embeddings(len(cfg.tokenizer))

        self.gru = nn.GRU(input_size=self.encoder.config.hidden_size, hidden_size=arc_hidden_size,
                                      bidirectional=True, batch_first=True)     

        self.mlp_arc_dep =  NonLinear(in_features=arc_hidden_size * 2, 
                                            out_features=arc_hidden_size, 
                                            activation=nn.LeakyReLU(0.1))
        self.mlp_arc_head = NonLinear(in_features=arc_hidden_size * 2, 
                                            out_features=arc_hidden_size, 
                                            activation=nn.LeakyReLU(0.1))
        self.mlp_rel_dep = NonLinear(in_features=rel_hidden_size * 2, 
                                            out_features=rel_hidden_size, 
                                            activation=nn.LeakyReLU(0.1))     
        self.mlp_rel_head = NonLinear(in_features=rel_hidden_size * 2, 
                                            out_features=rel_hidden_size, 
                                            activation=nn.LeakyReLU(0.1))

        self.arc_biaffine = Biaffine(arc_hidden_size, arc_hidden_size, 1)
        self.rel_biaffine = Biaffine(rel_hidden_size, rel_hidden_size, 35)
        
        self.inter_arc_biaffine = Biaffine(arc_hidden_size, arc_hidden_size, 1)
        self.inter_rel_biaffine = Biaffine(rel_hidden_size, rel_hidden_size, 19)

        self.speaker_arc_dep = nn.Linear(arc_hidden_size * 2, arc_hidden_size)
        self.speaker_rel_head = nn.Linear(arc_hidden_size * 2, arc_hidden_size)
        self.speaker_rel_dep = nn.Linear(arc_hidden_size * 2, arc_hidden_size)
        # self.inter_classifer = nn.Linear(arc_hidden_size, 19)

        self.dropout = nn.Dropout(cfg.dropout)

    def feat(self, inputs):
        length = torch.sum(inputs["attention_mask"], dim=-1) - 2
        
        feats, *_ = self.encoder(**inputs, return_dict=False)   # batch_size, seq_len (tokenized), plm_hidden_size
           
        # remove [SEP]
        word_cls = feats[:, :1]
        char_input = torch.narrow(feats, 1, 1, feats.size(1) - 2)
        return word_cls, char_input, length
        
    def char2word(self, char_feat, offsets):
        word_idx = offsets.unsqueeze(-1).expand(-1, -1, char_feat.shape[-1])  # expand to the size of char feat
        word_feat = torch.gather(char_feat, dim=1, index=word_idx)  # embeddings of first char in each word
        return word_feat

    def parse_dep(self, input_pckg):
        inputs, offsets, heads, rels, masks = input_pckg
        cls_feat, char_feat, word_len = self.feat(inputs)
        
        word_feat = self.char2word(char_feat, offsets)
        word_cls_feat = torch.cat([cls_feat, word_feat], dim=1)
        feats = self.dropout(word_cls_feat)  # syn_batch_size, seq_len, hidden_size 

        feats = self.dropout(self.gru(feats)[0])

        arc_dep = self.dropout(self.mlp_arc_dep(feats))
        arc_head = self.dropout(self.mlp_arc_dep(feats))

        rel_dep = self.dropout(self.mlp_rel_dep(feats))
        rel_head = self.dropout(self.mlp_rel_head(feats))

        arc_logit = self.arc_biaffine(arc_dep, arc_head)  
        arc_logit = arc_logit.squeeze(3)    # batch_size, seq_len, seq_len

        rel_logit = self.rel_biaffine(rel_dep, rel_head)  # batch_size, seq_len, seq_len, num_rels

        loss = arc_rel_loss(arc_logit, rel_logit, heads, rels, masks)
        return arc_logit, rel_logit, loss
    
    def get_root(self, inter_arc_logit, inter_rel_logit, head_assigns, rel_assigns, head_preds, speakers, masks):
        masks[:, 0] = 0
        root_ids = [2]
        for i, (head_pred, mask) in enumerate(zip(head_preds, masks)):
            try:
                root_idx = (((head_pred == 0) * mask) != 0).nonzero()[0].item()
            except IndexError: # no root
                root_idx = 2
            root_ids.append(root_idx)
            inter_arc_logit[i, root_idx] = head_assigns[i]
            # if i > 0 and speakers[i] != speakers[i-1]:
                # inter_rel_logit[i, root_idx, 15:] = rel_assigns[i, 15:]
            # else:
                # inter_rel_logit[i, root_idx, :15] = rel_assigns[i, :15]
            inter_rel_logit[i, root_idx] = rel_assigns[i]

        return inter_arc_logit, inter_rel_logit

    def parse_dep_inter(self, input_pckg, test=False):
        inputs, offsets, heads, rels, masks, inter_heads, inter_rels, inter_masks = input_pckg
        speakers = inputs['input_ids'][:, 1]
        cls_feat, char_feat, word_len = self.feat(inputs)
        char_feat, speaker_feat = char_feat[:, 1:, :], char_feat[:, 0, :].unsqueeze(1)

        word_feat = self.char2word(char_feat, offsets)
        word_cls_feat = torch.cat([cls_feat, speaker_feat, word_feat], dim=1)
        inner_feats = word_cls_feat     # turns, seq_len, hidden_size 

        # inner
        inner_feats = self.dropout(self.gru(inner_feats)[0])
        feats, speaker_feat = torch.cat([inner_feats[:, 0, :].unsqueeze(1), inner_feats[:, 2:, :]], dim=1), \
            inner_feats[:, 1, :].unsqueeze(1)
        
        speaker_arc_dep = self.dropout(self.speaker_arc_dep(speaker_feat))
        speaker_rel_head = self.dropout(self.speaker_rel_head(speaker_feat))
        speaker_rel_dep = self.dropout(self.speaker_rel_dep(speaker_feat))

        arc_dep = self.dropout(self.mlp_arc_dep(feats))
        arc_head = self.dropout(self.mlp_arc_dep(feats))

        rel_dep = self.dropout(self.mlp_rel_dep(feats))
        rel_head = self.dropout(self.mlp_rel_head(feats))

        arc_logit = self.arc_biaffine(arc_dep, arc_head)  
        arc_logit = arc_logit.squeeze(3)    # batch_size, seq_len, seq_len
        rel_logit = self.rel_biaffine(rel_dep, rel_head)  # batch_size, seq_len, seq_len, num_rels
        
        batch_size, seq_len, hidden_size = arc_dep.shape
        inter_arc_logit = torch.ones(batch_size, seq_len, batch_size*seq_len).to(self.device) * -1e4
        inter_rel_logit = torch.ones(batch_size, seq_len, 19).to(self.device) * -1e4

        if inter_heads.sum() == 0:
            inter_arc_logit = inter_arc_logit.view(-1, batch_size*seq_len) 
            inter_rel_logit = inter_rel_logit.view(batch_size * seq_len, -1)
            inter_arc_logit, inter_rel_logit = inter_arc_logit.unsqueeze(0), inter_rel_logit.unsqueeze(0)
            loss = arc_rel_loss(arc_logit, rel_logit, heads, rels, masks)
            return (arc_logit, rel_logit), (inter_arc_logit, inter_rel_logit), loss

        dummy = torch.zeros(1, seq_len, hidden_size).to(self.device)
        pre_one_arc = torch.cat([dummy, arc_head[:-1, :, :]], dim=0)
        po_arc_logit = self.inter_arc_biaffine(pre_one_arc, speaker_arc_dep).squeeze(-1)   # batch_size, seq_len, seq_len

        dummy = torch.zeros(1, 1, hidden_size).to(self.device)
        pre_one_rel = torch.cat([dummy, speaker_rel_head[:-1, :, :]], dim=0)
        po_rel_logit = self.inter_rel_biaffine(pre_one_rel, speaker_rel_dep).squeeze(1).squeeze(1)   # batch_size, num_rels
        
        po_arc_diag = torch.block_diag(*po_arc_logit)
        po_arc_shift = torch.cat([po_arc_diag[:, 1*seq_len:], torch.zeros(batch_size, 1*seq_len).to(self.device)], dim=1)

        head_preds = arc_logit.softmax(-1).argmax(-1)
        inter_arc_logit, inter_rel_logit = \
            self.get_root(inter_arc_logit, inter_rel_logit, po_arc_shift, po_rel_logit, head_preds, speakers, masks) 
        inter_arc_logit = inter_arc_logit.view(-1, batch_size*seq_len) 

        inter_rel_logit = inter_rel_logit.view(batch_size * seq_len, -1)

        inter_arc_logit, inter_rel_logit = inter_arc_logit.unsqueeze(0), inter_rel_logit.unsqueeze(0)
        if test:
            return (arc_logit, rel_logit), (inter_arc_logit, inter_rel_logit)
        
        inter_loss = arc_rel_loss_inter(inter_arc_logit, inter_rel_logit, inter_heads, inter_rels, inter_masks)
        loss = arc_rel_loss(arc_logit, rel_logit, heads, rels, masks)
        loss += inter_loss * batch_size

        return (arc_logit, rel_logit), (inter_arc_logit, inter_rel_logit), loss

    # training
    def forward(self, dialog_pckg, syn_pckg, epoch):
        (inner_arc_logit, inner_rel_logit), (inter_arc_logit, inter_rel_logit), dialog_loss = self.parse_dep_inter(dialog_pckg)
        if not syn_pckg:
            return ((inner_arc_logit, inner_rel_logit), (None, None)), dialog_loss

        # with syntax treebank
        syn_arc_logit, syn_rel_logit, syn_loss = self.parse_dep(syn_pckg)
        if epoch > 4:
            # loss = self.cfg.alpha * dialog_loss + (1 - self.cfg.alpha) * syn_loss
            loss = dialog_loss + 0.01 * syn_loss
        else:
            # loss = 0 * dialog_loss + 1 * syn_loss
            loss = syn_loss
        # loss = self.cfg.alpha* dialog_loss + (1 - self.cfg.alpha) * syn_loss
        return ((inner_arc_logit, inner_rel_logit), (inter_arc_logit, inter_rel_logit), (syn_arc_logit, syn_rel_logit)), loss

    def predict(self, inputs, offsets):
        cls_feat, char_feat, word_len = self.feat(inputs)
        speakers = inputs['input_ids'][:, 1]
        char_feat, speaker_feat = char_feat[:, 1:, :], char_feat[:, 0, :].unsqueeze(1)

        word_feat = self.char2word(char_feat, offsets)
        word_cls_feat = torch.cat([cls_feat, speaker_feat, word_feat], dim=1)
        inner_feats = word_cls_feat     # turns, seq_len, hidden_size 

        # inner
        inner_feats = self.dropout(self.gru(inner_feats)[0])
        feats, speaker_feat = torch.cat([inner_feats[:, 0, :].unsqueeze(1), inner_feats[:, 2:, :]], dim=1), \
            inner_feats[:, 1, :].unsqueeze(1)
        
        speaker_arc_dep = self.dropout(self.speaker_arc_dep(speaker_feat))
        speaker_rel_head = self.dropout(self.speaker_rel_head(speaker_feat))
        speaker_rel_dep = self.dropout(self.speaker_rel_dep(speaker_feat))

        arc_dep = self.dropout(self.mlp_arc_dep(feats))
        arc_head = self.dropout(self.mlp_arc_dep(feats))

        rel_dep = self.dropout(self.mlp_rel_dep(feats))
        rel_head = self.dropout(self.mlp_rel_head(feats))

        arc_logit = self.arc_biaffine(arc_dep, arc_head)  
        arc_logit = arc_logit.squeeze(3)    # batch_size, seq_len, seq_len
        rel_logit = self.rel_biaffine(rel_dep, rel_head)  # batch_size, seq_len, seq_len, num_rels

        batch_size, seq_len, hidden_size = arc_dep.shape
        dummy = torch.zeros(1, seq_len, hidden_size).to(self.device)
        pre_one_arc = torch.cat([dummy, arc_head[:-1, :, :]], dim=0)
        po_arc_logit = self.inter_arc_biaffine(pre_one_arc, speaker_arc_dep).squeeze(-1)   # batch_size, seq_len, seq_len

        dummy = torch.zeros(1, 1, hidden_size).to(self.device)
        pre_one_rel = torch.cat([dummy, speaker_rel_head[:-1, :, :]], dim=0)
        po_rel_logit = self.inter_rel_biaffine(pre_one_rel, speaker_rel_dep).squeeze(1).squeeze(1)   # batch_size, num_rels
        
        po_arc_diag = torch.block_diag(*po_arc_logit)
        po_arc_shift = torch.cat([po_arc_diag[:, 1*seq_len:], torch.zeros(batch_size, 1*seq_len).to(self.device)], dim=1)

        inter_arc_logit = torch.ones(batch_size, seq_len, batch_size*seq_len).to(self.device) * -1e4
        inter_rel_logit = torch.ones(batch_size, seq_len, 19).to(self.device) * -1e4
        head_preds = arc_logit.softmax(-1).argmax(-1)
        inter_arc_logit, inter_rel_logit = \
            self.get_root(inter_arc_logit, inter_rel_logit, po_arc_shift, po_rel_logit, head_preds, speakers, inputs['attention_mask']) 
        inter_arc_logit = inter_arc_logit.view(-1, batch_size*seq_len) 

        inter_rel_logit = inter_rel_logit.view(batch_size * seq_len, -1)
        inter_arc_logit, inter_rel_logit = inter_arc_logit.unsqueeze(0), inter_rel_logit.unsqueeze(0)

        return (arc_logit, rel_logit), (inter_arc_logit, inter_rel_logit)
        

class NonLinear(nn.Module):
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 activation: Optional[Callable] = None, 
                 init_func: Optional[Callable] = None) -> None: 
        super(NonLinear, self).__init__()
        self._linear = nn.Linear(in_features, out_features)
        self._activation = activation

        self.reset_parameters(init_func=init_func)
    
    def reset_parameters(self, init_func: Optional[Callable] = None) -> None:
        if init_func:
            init_func(self._linear.weight)

    def forward(self, x):
        if self._activation:
            return self._activation(self._linear(x))
        return self._linear(x)


class Biaffine(nn.Module):
    def __init__(self, 
                 in1_features: int, 
                 in2_features: int, 
                 out_features: int,
                 init_func: Optional[Callable] = None) -> None:
        super(Biaffine, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features

        self.linear_in_features = in1_features 
        self.linear_out_features = out_features * in2_features

        # with bias default
        self._linear = nn.Linear(in_features=self.linear_in_features,
                                out_features=self.linear_out_features)

        self.reset_parameters(init_func=init_func)

    def reset_parameters(self, init_func: Optional[Callable] = None) -> None:
        if init_func:
            init_func(self._linear.weight)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor):
        batch_size, len1, dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()

        affine = self._linear(input1)

        affine = affine.view(batch_size, len1*self.out_features, dim2)
        input2 = torch.transpose(input2, 1, 2)

        biaffine = torch.transpose(torch.bmm(affine, input2), 1, 2)

        biaffine = biaffine.contiguous().view(batch_size, len2, len1, self.out_features)
        return biaffine