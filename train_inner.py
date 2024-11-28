
from typing import *
import random
import datetime

import os
import yaml
import wandb
import torch
from torch.utils.data import DataLoader, Subset
import transformers
from transformers import AutoTokenizer

from trainer import MyTrainer
from model import DepParser, InterParser
from utils import arc_rel_loss, uas_las, to_cuda, seed_everything

from data_helper import SynDataset, DialogUttrDataset, InterDataset, DialogUttrInterDataset
from evaluation import eval_eduwise, eval_uttrwise, eval_all


def run():
    from config import DataClassUnpack, CFG
    # load_config
    with open('config/base.yaml', 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    CFG = DataClassUnpack.instantiate(CFG, config)
    CFG.debug = False

    wandb.init(project="Dialogue-Dep-Aug", entity="zzoay")
    if 'alpha' in wandb.config:
        del CFG.alpha
        print('Duplicate parameters deleted.')
    wandb.config.update(CFG)

    tokenizer = AutoTokenizer.from_pretrained(CFG.plm)
    CFG.tokenizer = tokenizer

    traindataset = DialogUttrDataset(CFG, CFG.train_file, tokenizer)
    # traindataset = DialogUttrInterDataset(CFG, CFG.train_file, tokenizer)
    syndataset = SynDataset(CFG, tokenizer)

    syn_iter = DataLoader(syndataset, batch_size=CFG.syn_batch_size, shuffle=True, drop_last=True)

    one_epoch_steps = int(len(syndataset) / CFG.syn_batch_size)

    # alignment
    if not CFG.debug:
        n = int(one_epoch_steps / len(traindataset) * CFG.dialog_batch_size)
        rst_ids = list(range(len(traindataset)))
        tmp_ids = []
        for i in range(n):
            tmp_ids.extend(rst_ids)

        remain_num = one_epoch_steps - int(n * len(traindataset) / CFG.dialog_batch_size)
        random.shuffle(rst_ids)
        tmp_ids.extend(rst_ids[:remain_num])
        traindataset_extend = Subset(traindataset, tmp_ids)
        train_iter = DataLoader(traindataset_extend, batch_size=CFG.dialog_batch_size, shuffle=False)

    model = DepParser(CFG)
    # model = InterParser(CFG)
    wandb.watch(model)

    trainer = MyTrainer(model, one_epoch_steps=one_epoch_steps, loss_fn=arc_rel_loss, metrics_fn=uas_las, config=CFG)

    best_res, best_state_dict = trainer.train(model, train_iter, syn_iter)

    torch.save(best_state_dict, 'ckpt/base_par.pt')

    diagdataset_test = DialogUttrDataset(CFG, CFG.test_file, tokenizer, test=True)
    # # diagdataset_test = DialogUttrInterDataset(CFG, CFG.test_file, tokenizer)
    test_iter = DataLoader(diagdataset_test, batch_size=64, shuffle=False)

    inter_dataset = InterDataset(CFG)
    inter_iter = DataLoader(inter_dataset, batch_size=1)
    (inner_uas, inner_las), (inter_uas, inter_las) = eval_all(model, test_iter, inter_iter)
    wandb.log({"inner-EDU": {"uas": inner_uas, "las": inner_las}, 
               "inter-EDU": {"uas": inter_uas, "las": inter_las}})
    print((inner_uas, inner_las), (inter_uas, inter_las))
    

if __name__ == '__main__':
    transformers.logging.set_verbosity_error() # only report errors.

    seed_everything(42)

    debug = False
    use_wandb = True
    sweep = False

    # wandb
    if debug:
        os.environ['WANDB_MODE'] = 'disabled'  # offline / disabled

    if not use_wandb:
        os.environ['WANDB_MODE'] = 'disabled'  # offline / disabled
    
    time_now = datetime.datetime.now().isoformat()
    print(f'=-=-=-=-=-=-=-=-={time_now}=-=-=-=-=-=-=-=-=-=')

    if sweep:
        sweep_configuration = {
            "name": "alpha-sweep",
            "metric": {"name": "inter-EDU.las", "goal": "maximize"},
            "method": "grid",
            "parameters": {
                "alpha": {
                    "values": [0.9, 0.8, 0.7, 0.6, 0.5]
                },
            }
        }
        sweep_id = wandb.sweep(sweep_configuration, project="Dialogue-Dep-Aug", entity="zzoay")
        # run the sweep
        wandb.agent(sweep_id, function=(run))
    else:
        run()

    print('=================End=================')
    print(datetime.datetime.now().isoformat())
    print('=====================================')
