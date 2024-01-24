
from typing import *

import wandb
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from utils import to_cuda, uas_las_inter, arc_rel_loss_inter


class MyTrainer():
    def __init__(self, 
                 model,
                 one_epoch_steps,
                 loss_fn: Callable, 
                 metrics_fn: Callable, 
                 config: Dict) -> None:
        self.loss_fn = loss_fn
        self.metrics_fn = metrics_fn

        plm_params = [p for n,p in model.named_parameters() if 'encoder' in n]
        head_params = [p for n,p in model.named_parameters() if 'encoder' not in n]
        self.optim = AdamW([{'params': plm_params, 'lr':config.plm_lr}, 
                            {'params': head_params, 'lr':config.head_lr}], 
                            lr=config.plm_lr,
                            weight_decay=config.weight_decay
                          )
        
        training_step = int(config.num_epochs * one_epoch_steps)
        warmup_step = int(config.warmup_ratio * training_step)  
        self.optim_schedule = get_linear_schedule_with_warmup(optimizer=self.optim, 
                                                              num_warmup_steps=warmup_step, 
                                                              num_training_steps=training_step)
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)
        
        self.print_every = int(config.print_every_ratio * one_epoch_steps)

        self.config = config

    def train(self, 
              model: nn.Module,
              train_iter: DataLoader, 
              syn_iter: DataLoader,
              val_iter: DataLoader = None):
        model.train()
        if self.config.cuda and torch.cuda.is_available():
            model.cuda()
        
        best_res = [0, 0, 0]
        early_stop_cnt = 0
        best_state_dict = None
        step = 0
        for epoch in tqdm(range(self.config.num_epochs)):
            for (dialog_batch, syn_batch) in zip(train_iter, syn_iter):
            # for syn_batch in syn_iter:
                inputs, offsets, heads, rels, masks = syn_batch
                if self.config.cuda and torch.cuda.is_available():
                    inputs_cuda = {}
                    for key, value in inputs.items():
                        inputs_cuda[key] = value.cuda()
                    inputs = inputs_cuda
    
                    offsets, heads, rels, masks = to_cuda(data=(offsets, heads, rels, masks))
                syn_batch = (inputs, offsets, heads, rels, masks)

                inputs, offsets, heads, rels, masks = dialog_batch
                if self.config.cuda and torch.cuda.is_available():
                    inputs_cuda = {}
                    for key, value in inputs.items():  
                        inputs_cuda[key] = value.cuda()  # rst's batch size is always 1
                    inputs = inputs_cuda
    
                    offsets, heads, rels, masks = to_cuda(data=(offsets, heads, rels, masks))
                inputs_cuda = {}
                for key, value in inputs.items():  
                    inputs_cuda[key] = value.squeeze(0) # rst's batch size is always 1
                inputs = inputs_cuda
                offsets, heads, rels, masks = offsets.squeeze(0), heads.squeeze(0), rels.squeeze(0), masks.squeeze(0)
                dialog_batch = (inputs, offsets, heads, rels, masks)
                # inputs, offsets, heads, rels, sem_roles, masks, speakers = dialog_batch
                # if self.config.cuda and torch.cuda.is_available():
                #     inputs_cuda = {}
                #     for key, value in inputs.items():  
                #         inputs_cuda[key] = value.squeeze(0).cuda()  # rst's batch size is always 1
                #     inputs = inputs_cuda
    
                #     offsets, heads, rels, sem_roles, masks, speakers = to_cuda(data=(offsets, heads, rels, sem_roles, masks, speakers))
                # offsets, heads, rels, masks, speakers = offsets.squeeze(0), heads.squeeze(0), rels.squeeze(0), masks.squeeze(0), speakers.squeeze(0)
                # dialog_batch = (inputs, offsets, heads, rels, sem_roles, masks, speakers)
                
                outputs, loss = model(dialog_batch, syn_batch, epoch)
                
                self.optim.zero_grad()
                if self.config.cuda and self.config.fp16:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optim)
                else:
                    loss.backward()

                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=self.config.grad_clip)

                if self.config.cuda and self.config.fp16:
                    self.scaler.step(self.optim)
                    self.scaler.update()
                else:
                    self.optim.step()
                self.optim_schedule.step()

                with torch.no_grad():
                    (dialog_arc_logit, dialog_rel_logit), (syn_arc_logit, syn_rel_logit) = outputs
                    dialog_metrics = self.metrics_fn(dialog_arc_logit, dialog_rel_logit, dialog_batch[2], dialog_batch[3], dialog_batch[4])
                    # dialog_metrics = self.metrics_fn(dialog_arc_logit, dialog_rel_logit, dialog_batch[2], dialog_batch[3], dialog_batch[5])
                    syn_metrics = self.metrics_fn(syn_arc_logit, syn_rel_logit, syn_batch[2], syn_batch[3], syn_batch[4])
                    
                if step % self.print_every == 0:
                    wandb.log({
                        'epoch': epoch,
                        'step': step,
                        'loss': loss,
                        'dialog_metrics': dialog_metrics,
                        'syn_metrics': syn_metrics,
                    })

                    if dialog_metrics["LAS"] > 0.99:
                        return 0.0, model.state_dict()

                if val_iter is not None and self.config.eval_strategy == 'step' and (step + 1) % self.config.eval_every == 0:
                    avg_loss, uas, las = self.eval(model, val_iter)
                    res = [avg_loss, uas, las]
                    if las > best_res[2]:  # las
                        best_res = res
                        best_state_dict = model.state_dict()
                        early_stop_cnt = 0
                    else:
                        early_stop_cnt += 1
                    # back to train mode
                    model.train()
                
                step += 1
                    
            if val_iter is not None and self.config.eval_strategy == 'epoch':
                avg_loss, uas, las = self.eval(model, val_iter)
                res = [avg_loss, uas, las]
                if las > best_res[2]:  # las
                    best_res = res
                    best_state_dict = model.state_dict()
                    early_stop_cnt = 0
                else:
                    early_stop_cnt += 1
                # back to train mode
                model.train()
                
            if early_stop_cnt >= self.config.num_early_stop:
                return best_res, best_state_dict
            
            # save each epoch
            if self.config.eval_strategy == 'epoch':
                torch.save(best_state_dict, 'ckpt/base_par.pt')

        if best_state_dict is None:
            return 0.0, model.state_dict()
        return best_res, best_state_dict

    # eval func
    def eval(self, model: nn.Module, eval_iter: DataLoader, save_file: str = "", save_title: str = ""):
        model.eval()

        head_whole, rel_whole, mask_whole = torch.Tensor(), torch.Tensor(), torch.Tensor()
        arc_logit_whole, rel_logit_whole = torch.Tensor(), torch.Tensor()
        avg_loss = 0.0
        for step, batch in enumerate(eval_iter):
            inputs, offsets, heads, rels, masks = batch

            if self.config.cuda and torch.cuda.is_available():
                inputs_cuda = {}
                for key, value in inputs.items():
                    inputs_cuda[key] = value.cuda()
                inputs = inputs_cuda

                offsets, heads, rels, masks = to_cuda(data=(offsets, heads, rels, masks))
            
            with torch.no_grad():
                arc_logits, rel_logits = model(inputs, offsets, heads, rels, masks, evaluate=True)
                
            loss = self.loss_fn(arc_logits, rel_logits, heads, rels, masks)

            arc_logit_whole = torch.cat([arc_logit_whole, arc_logits.cpu()], dim=0)
            rel_logit_whole = torch.cat([rel_logit_whole, rel_logits.cpu()], dim=0)

            head_whole, rel_whole = torch.cat([head_whole, heads.cpu()], dim=0), torch.cat([rel_whole, rels.cpu()], dim=0)
            mask_whole = torch.cat([mask_whole, masks.cpu()], dim=0)

            avg_loss += loss.item() * len(heads)  # times the batch size of data

        metrics = self.metrics_fn(arc_logit_whole, rel_logit_whole, head_whole, rel_whole, mask_whole)
        uas, las = metrics['UAS'], metrics['LAS']

        avg_loss /= len(eval_iter.dataset)  # type: ignore

        if save_file != "":
            results = [save_title, avg_loss, uas, las]  # type: ignore
            results = [str(x) for x in results]
            with open(save_file, "a+") as f:
                f.write(",".join(results) + "\n")  # type: ignore

        return avg_loss, uas, las  # type: ignore
    
    def save_results(self, save_file, save_title, results):
        saves = [save_title] + results
        saves = [str(x) for x in saves]
        with open(save_file, "a+") as f:
            f.write(",".join(saves) + "\n")  # type: ignore


class InterTrainer():
    def __init__(self, 
                 model,
                 one_epoch_steps,
                 loss_fn: Callable, 
                 metrics_fn: Callable, 
                 config: Dict) -> None:
        self.loss_fn = loss_fn
        self.metrics_fn = metrics_fn

        plm_params = [p for n,p in model.named_parameters() if 'encoder' in n]
        head_params = [p for n,p in model.named_parameters() if 'encoder' not in n]
        self.optim = AdamW([{'params': plm_params, 'lr':config.plm_lr}, 
                            {'params': head_params, 'lr':config.head_lr}], 
                            lr=config.plm_lr,
                            weight_decay=config.weight_decay
                          )
        
        training_step = int(config.num_epochs * one_epoch_steps)
        warmup_step = int(config.warmup_ratio * training_step)  
        self.optim_schedule = get_linear_schedule_with_warmup(optimizer=self.optim, 
                                                              num_warmup_steps=warmup_step, 
                                                              num_training_steps=training_step)
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)
        
        self.print_every = int(config.print_every_ratio * one_epoch_steps)

        self.config = config

    def train(self, 
              model: nn.Module,
              train_iter: DataLoader, 
              syn_iter: DataLoader,
              val_iter: DataLoader = None):
        model.train()
        if self.config.cuda and torch.cuda.is_available():
            model.cuda()
        
        best_res = [0, 0, 0]
        early_stop_cnt = 0
        best_state_dict = None
        step = 0
        for epoch in tqdm(range(self.config.num_epochs)):
            for (dialog_batch, syn_batch) in zip(train_iter, syn_iter):
            # for syn_batch in syn_iter:
                inputs, offsets, heads, rels, masks = syn_batch
                if self.config.cuda and torch.cuda.is_available():
                    inputs_cuda = {}
                    for key, value in inputs.items():
                        inputs_cuda[key] = value.cuda()
                    inputs = inputs_cuda
    
                    offsets, heads, rels, masks = to_cuda(data=(offsets, heads, rels, masks))
                syn_batch = (inputs, offsets, heads, rels, masks)

                inputs, offsets, heads, rels, masks, inter_heads, inter_rels, inter_masks, speakers = dialog_batch
                if self.config.cuda and torch.cuda.is_available():
                    inputs_cuda = {}
                    for key, value in inputs.items():  
                        inputs_cuda[key] = value.squeeze(0).cuda()  # rst's batch size is always 1
                    inputs = inputs_cuda
    
                    offsets, heads, rels, masks, speakers = to_cuda(data=(offsets, heads, rels, masks, speakers))
                    inter_heads, inter_rels, inter_masks = to_cuda(data=(inter_heads, inter_rels, inter_masks))
                offsets, heads, rels, masks, speakers = offsets.squeeze(0), heads.squeeze(0), rels.squeeze(0), masks.squeeze(0), speakers.squeeze(0)
                inter_heads, inter_rels, inter_masks = inter_heads.squeeze(0), inter_rels.squeeze(0), inter_masks.squeeze(0)

                dialog_batch = (inputs, offsets, heads, rels, masks, inter_heads, inter_rels, inter_masks)
            
                outputs, loss = model(dialog_batch, syn_batch, epoch)
                
                self.optim.zero_grad()
                if self.config.cuda and self.config.fp16:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optim)
                else:
                    loss.backward()

                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=self.config.grad_clip)

                if self.config.cuda and self.config.fp16:
                    self.scaler.step(self.optim)
                    self.scaler.update()
                else:
                    self.optim.step()
                self.optim_schedule.step()

                with torch.no_grad():
                    (inner_arc_logit, inner_rel_logit), (inter_arc_logit, inter_rel_logit), (syn_arc_logit, syn_rel_logit) = outputs
                    dialog_metrics = self.metrics_fn(inner_arc_logit, inner_rel_logit, dialog_batch[2], dialog_batch[3], dialog_batch[4])
                    inter_metrics = uas_las_inter(inter_arc_logit, inter_rel_logit, inter_heads, inter_rels, inter_masks)
                    syn_metrics = self.metrics_fn(syn_arc_logit, syn_rel_logit, syn_batch[2], syn_batch[3], syn_batch[4])
                    
                if step % self.print_every == 0:
                    wandb.log({
                        'epoch': epoch,
                        'step': step,
                        'loss': loss,
                        'dialog_metrics': dialog_metrics,
                        'inter_metrics': inter_metrics,
                        'syn_metrics': syn_metrics,
                    })
                    # print(arc_rel_loss_inter(inter_arc_logit, inter_rel_logit, inter_heads, inter_rels, inter_masks))
                    # print(uas_las_inter(inter_arc_logit, inter_rel_logit, inter_heads, inter_rels, inter_masks))
                    # print(dialog_metrics)
                    # print(syn_metrics)
                    # print('------------------------')

                if val_iter is not None and self.config.eval_strategy == 'step' and (step + 1) % self.config.eval_every == 0:
                    avg_loss, uas, las = self.eval(model, val_iter)
                    res = [avg_loss, uas, las]
                    if las > best_res[2]:  # las
                        best_res = res
                        best_state_dict = model.state_dict()
                        early_stop_cnt = 0
                    else:
                        early_stop_cnt += 1
                    # back to train mode
                    model.train()
                
                step += 1
                    
            if val_iter is not None and self.config.eval_strategy == 'epoch':
                avg_loss, uas, las = self.eval(model, val_iter)
                res = [avg_loss, uas, las]
                if las > best_res[2]:  # las
                    best_res = res
                    best_state_dict = model.state_dict()
                    early_stop_cnt = 0
                else:
                    early_stop_cnt += 1
                # back to train mode
                model.train()
                
            if early_stop_cnt >= self.config.num_early_stop:
                return best_res, best_state_dict
            
            # save each epoch
            if self.config.eval_strategy == 'epoch':
                torch.save(best_state_dict, 'ckpt/base_par.pt')

        if best_state_dict is None:
            return 0.0, model.state_dict()
        return best_res, best_state_dict

    # eval func
    def eval(self, model: nn.Module, eval_iter: DataLoader, save_file: str = "", save_title: str = ""):
        model.eval()

        head_whole, rel_whole, mask_whole = torch.Tensor(), torch.Tensor(), torch.Tensor()
        arc_logit_whole, rel_logit_whole = torch.Tensor(), torch.Tensor()
        avg_loss = 0.0
        for step, batch in enumerate(eval_iter):
            inputs, offsets, heads, rels, masks = batch

            if self.config.cuda and torch.cuda.is_available():
                inputs_cuda = {}
                for key, value in inputs.items():
                    inputs_cuda[key] = value.cuda()
                inputs = inputs_cuda

                offsets, heads, rels, masks = to_cuda(data=(offsets, heads, rels, masks))
            
            with torch.no_grad():
                arc_logits, rel_logits = model(inputs, offsets, heads, rels, masks, evaluate=True)
                
            loss = self.loss_fn(arc_logits, rel_logits, heads, rels, masks)

            arc_logit_whole = torch.cat([arc_logit_whole, arc_logits.cpu()], dim=0)
            rel_logit_whole = torch.cat([rel_logit_whole, rel_logits.cpu()], dim=0)

            head_whole, rel_whole = torch.cat([head_whole, heads.cpu()], dim=0), torch.cat([rel_whole, rels.cpu()], dim=0)
            mask_whole = torch.cat([mask_whole, masks.cpu()], dim=0)

            avg_loss += loss.item() * len(heads)  # times the batch size of data

        metrics = self.metrics_fn(arc_logit_whole, rel_logit_whole, head_whole, rel_whole, mask_whole)
        uas, las = metrics['UAS'], metrics['LAS']

        avg_loss /= len(eval_iter.dataset)  # type: ignore

        if save_file != "":
            results = [save_title, avg_loss, uas, las]  # type: ignore
            results = [str(x) for x in results]
            with open(save_file, "a+") as f:
                f.write(",".join(results) + "\n")  # type: ignore

        return avg_loss, uas, las  # type: ignore
    
    def save_results(self, save_file, save_title, results):
        saves = [save_title] + results
        saves = [str(x) for x in saves]
        with open(save_file, "a+") as f:
            f.write(",".join(saves) + "\n")  # type: ignore